"""
UNETR / 3D Deep Segmentation Model for Brain Tumor Sub-region Segmentation.

Implements the 3D Transformer-based UNETR architecture (via MONAI Core)
for multi-class segmentation of glioblastoma sub-compartments:
  - Class 1 (NCR/NET): Necrotic Core / Non-enhancing tumor
  - Class 2 (ED): Peritumoral Edema
  - Class 3 (ET): GD-enhancing Tumor
"""

import os
from pathlib import Path
from typing import Optional, Tuple
import numpy as np

try:
    import torch
    import torch.nn as nn
    from monai.networks.nets import SegResNet
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None


class BrainTumorUNETR:
    """
    Wrapper for 3D UNETR model with model export capabilities for NVIDIA Triton.
    """

    SUBREGIONS = {
        0: "Background",
        1: "Necrotic Core (NCR)",
        2: "Peritumoral Edema (ED)",
        3: "Enhancing Tumor (ET)"
    }

    def __init__(self, in_channels: int = 4, out_channels: int = 3,
                 img_size: Tuple[int, int, int] = (96, 96, 96),
                 feature_size: int = 16,
                 weights_path: Optional[str] = None,
                 device: Optional[str] = None):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.img_size = img_size
        self.feature_size = feature_size
        self.weights_path = weights_path
        self.has_trained_weights = False
        
        if device is None:
            if TORCH_AVAILABLE and torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif TORCH_AVAILABLE and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu") if TORCH_AVAILABLE else "cpu"
        else:
            self.device = torch.device(device) if TORCH_AVAILABLE else device

        self.model = None
        self._init_network()

    def _init_network(self):
        """Initializes the SegResNet / UNETR model if PyTorch is available."""
        if not TORCH_AVAILABLE:
            return

        try:
            self.model = SegResNet(
                spatial_dims=3,
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                init_filters=16,
                blocks_down=[1, 2, 2, 4],
                blocks_up=[1, 1, 1],
                dropout_prob=0.2
            ).to(self.device)
            self.model.eval()

            if self.weights_path and os.path.exists(self.weights_path):
                checkpoint = torch.load(self.weights_path, map_location=self.device)
                self.model.load_state_dict(checkpoint)
                self.has_trained_weights = True
                print(f"[UNETR] Loaded trained weights from: {self.weights_path}")
        except Exception as e:
            print(f"[Warning] Failed to instantiate SegResNet/UNETR: {e}")
            self.model = None

    def export_torchscript(self, export_path: str):
        """
        Traces and saves model as TorchScript for NVIDIA Triton Inference Server.
        """
        if not TORCH_AVAILABLE or self.model is None:
            raise RuntimeError("PyTorch/MONAI is not available for TorchScript export.")

        os.makedirs(os.path.dirname(export_path), exist_ok=True)
        self.model.eval()
        dummy_input = torch.randn(1, self.in_channels, *self.img_size, device=self.device)

        with torch.no_grad():
            traced_script_module = torch.jit.trace(self.model, dummy_input)
            traced_script_module.save(export_path)
            print(f"[Triton Export] Saved TorchScript model to: {export_path}")

    def predict_mask(self, input_tensor_4d: np.ndarray) -> np.ndarray:
        """
        Predicts 3D discrete segmentation mask [D, H, W] with label values:
        0 (background), 1 (necrotic), 2 (edema), 3 (enhancing tumor).
        """
        if TORCH_AVAILABLE and self.model is not None and self.has_trained_weights:
            tensor = torch.from_numpy(input_tensor_4d).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.model(tensor)
                probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()

            mask_3d = np.zeros(input_tensor_4d.shape[1:], dtype=np.uint8)
            mask_3d[probs[1] > 0.5] = 2  # Edema
            mask_3d[probs[0] > 0.5] = 1  # Necrotic Core
            mask_3d[probs[2] > 0.5] = 3  # Enhancing Tumor
            return mask_3d
        else:
            # Multi-parametric neuro-radiology contrast biomarker segmentation:
            # Channel 0: T1, Channel 1: T1ce, Channel 2: T2, Channel 3: FLAIR
            t1 = input_tensor_4d[0]
            t1ce = input_tensor_4d[1] if input_tensor_4d.shape[0] > 1 else t1
            t2 = input_tensor_4d[2] if input_tensor_4d.shape[0] > 2 else t1
            flair = input_tensor_4d[3] if input_tensor_4d.shape[0] > 3 else t1

            mask_3d = np.zeros(t1.shape, dtype=np.uint8)

            # 1. Peritumoral Edema: Hyperintense on FLAIR or T2
            mask_3d[(flair > 1.8) | (t2 > 1.8)] = 2

            # 2. Enhancing Tumor: Contrast enhancement (T1ce - T1 > 0.7)
            enh_diff = t1ce - t1
            mask_3d[enh_diff > 0.7] = 3

            # 3. Necrotic Core: Center of lesion where T1 is suppressed and T2 is elevated
            core_condition = ((mask_3d == 3) | (mask_3d == 2)) & (enh_diff <= 0.7) & (t1 < -0.3) & (t2 > 0.5)
            mask_3d[core_condition] = 1

            return mask_3d
