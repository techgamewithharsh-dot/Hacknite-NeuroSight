"""
MONAI-Powered Medical Imaging Preprocessing Pipeline for Multi-Modal Brain MRI.

Standardizes input sequences (T1, T1ce, T2, FLAIR) to isotropic 1mm^3 voxel spacing,
converts to RAS canonical orientation, applies intensity normalization, and prepares
4D/5D tensors for 3D UNETR segmentation inference.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import nibabel as nib

# Check if MONAI & PyTorch are installed in the active environment
try:
    import torch
    from monai.transforms import (
        Compose,
        EnsureChannelFirstd,
        EnsureTyped,
        Orientationd,
        Spacingd,
        NormalizeIntensityd,
        CropForegroundd,
        CenterSpatialCropd,
    )
    MONAI_AVAILABLE = True
except ImportError:
    MONAI_AVAILABLE = False
    torch = None


class BrainMRIPipeline:
    """
    Handles preprocessing and alignment of multi-modal brain MRI scans
    (T1, T1ce, T2, FLAIR) using MONAI transforms.
    """

    # Official MONAI Model Zoo BraTS channel order.
    SEQUENCE_KEYS = ["t1ce", "t1", "t2", "flair"]

    def __init__(self, target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
                 spatial_shape: Tuple[int, int, int] = (128, 128, 128),
                 crop_foreground: bool = False):
        self.target_spacing = target_spacing
        self.spatial_shape = spatial_shape
        self.crop_foreground = crop_foreground
        self._build_monai_transforms()

    def _build_monai_transforms(self):
        """Constructs MONAI Compose pipeline if monai is available."""
        if not MONAI_AVAILABLE:
            self.transform = None
            return

        transforms = [
            EnsureChannelFirstd(keys=self.SEQUENCE_KEYS, channel_dim="no_channel"),
            EnsureTyped(keys=self.SEQUENCE_KEYS, dtype=torch.float32),
            Orientationd(keys=self.SEQUENCE_KEYS, axcodes="RAS"),
            Spacingd(keys=self.SEQUENCE_KEYS, pixdim=self.target_spacing, mode=("bilinear", "bilinear", "bilinear", "bilinear")),
            NormalizeIntensityd(keys=self.SEQUENCE_KEYS, nonzero=True, channel_wise=True),
        ]
        if self.crop_foreground:
            transforms.append(
                CropForegroundd(keys=self.SEQUENCE_KEYS, source_key="t1ce", margin=4, allow_smaller=True)
            )

        self.transform = Compose(transforms)

    def load_nifti(self, file_path_or_obj) -> Tuple[np.ndarray, np.ndarray]:
        """Loads a NIfTI image and returns voxel array and affine matrix."""
        if hasattr(file_path_or_obj, "read"):
            # Streamlit UploadedFile object or file-like stream
            import tempfile
            import os
            with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp:
                tmp.write(file_path_or_obj.getvalue())
                tmp_path = tmp.name
            try:
                img = nib.load(tmp_path)
                data = img.get_fdata(dtype=np.float32)
                affine = img.affine
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
        else:
            img = nib.load(file_path_or_obj)
            data = img.get_fdata(dtype=np.float32)
            affine = img.affine
        return data, affine

    def preprocess_multimodal_dict(self, modal_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Preprocesses a dictionary of 3D modal arrays:
        {'t1': ndarray, 't1ce': ndarray, 't2': ndarray, 'flair': ndarray}
        Returns stacked 4D tensor shape [4, D, H, W] normalized.
        """
        if MONAI_AVAILABLE and self.transform is not None:
            # Run through official MONAI pipeline
            transformed = self.transform(modal_dict)
            stacked = torch.cat([transformed[k] for k in self.SEQUENCE_KEYS], dim=0) # [4, D, H, W]
            return stacked.cpu().numpy()
        else:
            # Fallback pure NumPy/SciPy preprocessing (for environment without full MONAI setup)
            processed_channels = []
            for key in self.SEQUENCE_KEYS:
                arr = modal_dict.get(key)
                if arr is None:
                    # If specific modality not provided, synthesize or zero-fill
                    sample_key = next(iter(modal_dict.keys()))
                    arr = np.zeros_like(modal_dict[sample_key])

                # Nonzero z-score normalization
                mask = arr > 0
                if np.any(mask):
                    mean = arr[mask].mean()
                    std = arr[mask].std()
                    norm_arr = np.zeros_like(arr, dtype=np.float32)
                    norm_arr[mask] = (arr[mask] - mean) / (std + 1e-7)
                else:
                    norm_arr = arr.astype(np.float32)
                processed_channels.append(norm_arr)

            stacked = np.stack(processed_channels, axis=0) # [4, D, H, W]
            return stacked

    def crop_or_pad_spatial(self, tensor_4d: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
        """
        Crops or zero-pads spatial dimensions [4, D, H, W] to target_shape.
        """
        channels, d, h, w = tensor_4d.shape
        td, th, tw = target_shape

        out = np.zeros((channels, td, th, tw), dtype=tensor_4d.dtype)

        # Calculate bounding boxes
        sd, ed = max(0, (d - td) // 2), min(d, (d + td) // 2)
        sh, eh = max(0, (h - th) // 2), min(h, (h + th) // 2)
        sw, ew = max(0, (w - tw) // 2), min(w, (w + tw) // 2)

        osd = max(0, (td - d) // 2)
        osh = max(0, (th - h) // 2)
        osw = max(0, (tw - w) // 2)

        cropped = tensor_4d[:, sd:ed, sh:eh, sw:ew]
        out[:, osd:osd+cropped.shape[1], osh:osh+cropped.shape[2], osw:osw+cropped.shape[3]] = cropped
        return out
