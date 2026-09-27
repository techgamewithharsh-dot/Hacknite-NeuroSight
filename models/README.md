# MONAI BraTS segmentation model

`brats_mri_segmentation.pt` is the pretrained SegResNet checkpoint from the
[MONAI Model Zoo BraTS MRI segmentation bundle](https://github.com/Project-MONAI/model-zoo/tree/dev/models/brats_mri_segmentation).
Its published MD5 is `870e677b782a5184cbc48db1456b78e8`.

The expected channels are T1c, T1, T2, and FLAIR. The model predicts tumor
core (TC), whole tumor (WT), and enhancing tumor (ET); the app derives the
edema and non-enhancing/core labels from those overlapping masks. MONAI Model
Zoo metadata describes this as an example and says it is not for diagnostic
use. Treat all results as research segmentation requiring expert MRI review.

## Optional Triton serving

On a Linux host with Docker and Compose installed, from the repository root:

```sh
.venv/bin/python scripts/export_triton_brats.py
docker compose -f triton/docker-compose.triton.yml up -d
```

The FastAPI service uses Triton HTTP on `localhost:8003` when the Python
`tritonclient[http]` package is installed and the model is ready. Otherwise it
runs the same checkpoint locally with MONAI. Set `NEUROSIGHT_TRITON_URL` to
change the Triton address or `NEUROSIGHT_USE_TRITON=0` to force local inference.
