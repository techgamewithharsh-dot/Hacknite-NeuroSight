"""
NVIDIA Triton Inference Server Client Bridge.

Provides a unified interface for submitting 3D brain MRI tensors to Triton
Inference Server via gRPC or HTTP. Seamlessly falls back to local PyTorch/MONAI
inference if Triton Server container is not reachable.
"""

import json
import logging
from typing import Dict, Any, Optional, Tuple
import numpy as np

# Configure logger
logger = logging.getLogger("TritonClient")
logger.setLevel(logging.INFO)

# Try importing tritonclient
try:
    import tritonclient.http as httpclient
    TRITON_HTTP_AVAILABLE = True
except ImportError:
    TRITON_HTTP_AVAILABLE = False
    httpclient = None

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


class TritonInferenceBridge:
    """
    Manages communication with NVIDIA Triton Server with automated
    local PyTorch/MONAI failover.
    """

    def __init__(self,
                 triton_url: str = "localhost:8000",
                 model_name: str = "brain_tumor_unetr",
                 model_version: str = "1",
                 prefer_triton: bool = True):
        self.triton_url = triton_url
        self.model_name = model_name
        self.model_version = model_version
        self.prefer_triton = prefer_triton
        self._local_unetr = None
        self._triton_client = None

    def is_triton_live(self) -> bool:
        """Checks if NVIDIA Triton Server is responding and ready."""
        if not self.prefer_triton:
            return False

        if REQUESTS_AVAILABLE:
            try:
                url = f"http://{self.triton_url}/v2/health/ready"
                resp = requests.get(url, timeout=0.8)
                return resp.status_code == 200
            except Exception:
                return False
        return False

    def _get_local_model(self):
        """Lazy loader for local PyTorch/MONAI model engine."""
        if self._local_unetr is None:
            from core.unetr_model import BrainTumorUNETR
            self._local_unetr = BrainTumorUNETR()
        return self._local_unetr

    def infer(self, tensor_4d: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Runs 3D tumor sub-region segmentation inference.
        Returns:
            mask_3d (np.ndarray): 3D volume label map [D, H, W]
            metadata (dict): Execution details (backend, latency, status)
        """
        import time
        start_time = time.time()

        triton_active = self.is_triton_live()

        if triton_active and TRITON_HTTP_AVAILABLE:
            try:
                # Triton HTTP client inference
                client = httpclient.InferenceServerClient(url=self.triton_url)
                
                # Reshape to batch dimension: [1, 4, D, H, W]
                batch_input = np.expand_dims(tensor_4d.astype(np.float32), axis=0)

                inputs = [
                    httpclient.InferInput("INPUT__0", batch_input.shape, "FP32")
                ]
                inputs[0].set_data_from_numpy(batch_input)

                outputs = [
                    httpclient.InferRequestedOutput("OUTPUT__0")
                ]

                response = client.infer(
                    model_name=self.model_name,
                    inputs=inputs,
                    outputs=outputs
                )

                output_arr = response.as_numpy("OUTPUT__0") # [1, 3, D, H, W]
                probs = output_arr[0]
                
                mask_3d = np.zeros(tensor_4d.shape[1:], dtype=np.uint8)
                mask_3d[probs[1] > 0.5] = 2  # Edema
                mask_3d[probs[0] > 0.5] = 1  # Necrotic Core
                mask_3d[probs[2] > 0.5] = 3  # Enhancing Tumor

                latency = round((time.time() - start_time) * 1000, 2)
                return mask_3d, {
                    "backend": "NVIDIA Triton Inference Server (Docker)",
                    "url": self.triton_url,
                    "model": self.model_name,
                    "latency_ms": latency,
                    "is_fallback": False
                }
            except Exception as e:
                logger.warning(f"Triton inference failed, falling back to local: {e}")

        # Local PyTorch / Heuristic Failover
        local_model = self._get_local_model()
        mask_3d = local_model.predict_mask(tensor_4d)
        latency = round((time.time() - start_time) * 1000, 2)

        backend_name = "Local PyTorch / MONAI Engine" if getattr(local_model, "model", None) else "Local Biomarker Heuristic Engine"

        return mask_3d, {
            "backend": backend_name,
            "url": "local://in-process",
            "model": "SegResNet / UNETR 3D",
            "latency_ms": latency,
            "is_fallback": True,
            "triton_status": "Offline (Local Engine Active)"
        }
