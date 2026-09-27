"""MONAI Model Zoo BraTS MRI segmentation inference.

Input channel order follows the MONAI BraTS model: T1c, T1, T2, FLAIR.
Outputs follow the model zoo's independent sigmoid heads: tumor core (TC),
whole tumor (WT), enhancing tumor (ET). BraTS-derived labels are 1=necrotic
core/non-enhancing core, 2=edema, 3=enhancing tumor.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    from monai.networks.nets import SegResNet
    from monai.inferers import sliding_window_inference
except ImportError:
    torch = None
    F = None
    SegResNet = None
    sliding_window_inference = None


WEIGHTS_PATH = Path(os.environ.get(
    "NEUROSIGHT_MONAI_WEIGHTS",
    Path(__file__).resolve().parent.parent / "models" / "brats_mri_segmentation.pt",
))


class BrainTumorSegmenter:
    """Run the pretrained MONAI SegResNet BraTS model locally."""

    def __init__(self):
        if torch is None or SegResNet is None:
            raise RuntimeError("PyTorch and MONAI are required for MRI segmentation.")
        if not WEIGHTS_PATH.is_file():
            raise RuntimeError(f"Pretrained MONAI BraTS weights are missing: {WEIGHTS_PATH}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SegResNet(
            spatial_dims=3, blocks_down=(1, 2, 2, 4), blocks_up=(1, 1, 1),
            init_filters=16, in_channels=4, out_channels=3, dropout_prob=0.2,
        ).to(self.device)
        checkpoint = torch.load(str(WEIGHTS_PATH), map_location=self.device, weights_only=False)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        # MONAI bundle checkpoints may prefix parameters with the network name.
        if isinstance(checkpoint, dict) and checkpoint and all(k.startswith("model.") for k in checkpoint):
            checkpoint = {k[len("model."):]: v for k, v in checkpoint.items()}
        self.model.load_state_dict(checkpoint, strict=True)
        self.model.eval()
        self.triton_client = None
        self.triton_model_name = os.environ.get("NEUROSIGHT_TRITON_MODEL", "brats_mri_segmentation")
        self.triton_url = os.environ.get("NEUROSIGHT_TRITON_URL", "localhost:8003")
        if os.environ.get("NEUROSIGHT_USE_TRITON", "1").lower() not in ("0", "false", "no"):
            try:
                import tritonclient.http as httpclient
                client = httpclient.InferenceServerClient(url=self.triton_url, network_timeout=1.0)
                if client.is_server_ready() and client.is_model_ready(self.triton_model_name):
                    self.triton_client = client
            except Exception as exc:
                print(f"[SegModel] Triton unavailable ({exc}); using the same trained model locally.")

    @property
    def backend_name(self) -> str:
        return (f"Triton MONAI BraTS SegResNet ({self.triton_url})" if self.triton_client
                else f"MONAI BraTS SegResNet ({self.device.type}; Triton offline)")

    def is_deep_learning(self) -> bool:
        return True

    @property
    def triton_connected(self) -> bool:
        return self.triton_client is not None

    def segment(self, tensor_4ch: np.ndarray) -> Tuple[np.ndarray, dict]:
        if tensor_4ch.ndim != 4 or tensor_4ch.shape[0] != 4:
            raise ValueError("Expected MRI tensor [4,D,H,W] ordered T1c,T1,T2,FLAIR.")
        # Model was trained at ~1mm voxels and uses a sliding window of 240x240x160.
        x = torch.as_tensor(np.ascontiguousarray(tensor_4ch), dtype=torch.float32,
                            device=self.device).unsqueeze(0)
        original_shape = tuple(x.shape[2:])
        model_roi = (240, 240, 160)
        if self.triton_client is not None:
            # Triton is configured for the MONAI model-zoo inference window.
            d, h, w = original_shape
            x = F.pad(x, (0, max(0, model_roi[2]-w), 0, max(0, model_roi[1]-h),
                          0, max(0, model_roi[0]-d)))
            roi = model_roi
        else:
            roi = tuple(min(n, r) for n, r in zip(original_shape, model_roi))
        def predict_windows(windows):
            if self.triton_client is not None:
                try:
                    import tritonclient.http as httpclient
                    batch = windows.detach().cpu().numpy().astype(np.float32, copy=False)
                    inp = httpclient.InferInput("INPUT__0", batch.shape, "FP32")
                    inp.set_data_from_numpy(batch)
                    out = self.triton_client.infer(self.triton_model_name, [inp],
                                                   outputs=[httpclient.InferRequestedOutput("OUTPUT__0")])
                    return torch.from_numpy(out.as_numpy("OUTPUT__0")).to(windows.device)
                except Exception as exc:
                    print(f"[SegModel] Triton request failed ({exc}); retrying this window locally.")
                    self.triton_client = None
            return self.model(windows)

        with torch.inference_mode():
            logits = sliding_window_inference(x, roi, sw_batch_size=1,
                                               predictor=predict_windows, overlap=0.5)
            probs = torch.sigmoid(logits)[0, :, :original_shape[0], :original_shape[1], :original_shape[2]].cpu().numpy()
        tc, wt, et = probs >= 0.5
        # Enforce anatomical label nesting and disjoint labels for display/volumes.
        wt |= tc | et
        edema = wt & ~tc
        necrotic = tc & ~et
        labels = np.zeros(wt.shape, dtype=np.uint8)
        labels[edema] = 2
        labels[necrotic] = 1
        labels[et] = 3
        return labels, {
            "backend": self.backend_name,
            "model": "MONAI Model Zoo brats_mri_segmentation (BraTS 2018)",
            "confidence": None,
            "triton_used": self.triton_client is not None,
            "labels": {0: "BG", 1: "NCR/NET", 2: "ED", 3: "ET"},
        }
