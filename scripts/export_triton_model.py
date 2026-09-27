"""
Exports 3D UNETR/SegResNet model to TorchScript for NVIDIA Triton Inference Server.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.unetr_model import BrainTumorUNETR

def main():
    target_path = PROJECT_ROOT / "triton/model_repository/brain_tumor_unetr/1/model.pt"
    target_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Exporting model to {target_path}...")
    model_wrapper = BrainTumorUNETR()
    try:
        model_wrapper.export_torchscript(str(target_path))
        print("✅ Triton TorchScript model successfully exported!")
    except Exception as e:
        print(f"Note: Model export error: {e}")

if __name__ == "__main__":
    main()
