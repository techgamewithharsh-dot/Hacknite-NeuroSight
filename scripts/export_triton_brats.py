"""Export the repository's trained MONAI BraTS model for Triton PyTorch backend."""
from pathlib import Path
import torch
from monai.networks.nets import SegResNet

ROOT = Path(__file__).resolve().parents[1]
weights = ROOT / "models" / "brats_mri_segmentation.pt"
output = ROOT / "triton" / "model_repository" / "brats_mri_segmentation" / "1" / "model.pt"
output.parent.mkdir(parents=True, exist_ok=True)

model = SegResNet(spatial_dims=3, blocks_down=(1, 2, 2, 4), blocks_up=(1, 1, 1),
                  init_filters=16, in_channels=4, out_channels=3, dropout_prob=0.2)
model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=False), strict=True)
model.eval()
example = torch.zeros((1, 4, 240, 240, 160), dtype=torch.float32)
with torch.inference_mode():
    traced = torch.jit.trace(model, example, strict=False)
    traced.save(str(output))
print(f"Exported {output} ({output.stat().st_size / 1024 / 1024:.1f} MiB)")
