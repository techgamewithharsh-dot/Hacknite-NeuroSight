# NeuroSight — Brain MRI Segmentation & Progression Analysis Prototype

A Hacknite 2026 research prototype for comparing baseline and follow-up brain MRI scans. NeuroSight combines Python, MONAI/PyTorch segmentation, a FastAPI backend, a browser interface, and a Streamlit viewer to explore longitudinal imaging and 3D visualization.

**Research and demonstration use only.** The repository does not establish clinical validation. Displayed verdicts and confidence scores must not be treated as diagnoses or measured clinical accuracy. See the [model documentation](models/README.md) for model provenance and limitations.

## Explore the project

- **MRI segmentation:** four-channel T1c, T1, T2, and FLAIR input using a MONAI BraTS SegResNet model.
- **Longitudinal comparison:** volumetric changes and progression-related outputs.
- **Visualization:** browser and Streamlit interfaces, with mesh generation and atlas-registration code.
- **Synthetic cases:** a generator for exploring the demonstration workflow.
- **Optional inference serving:** NVIDIA Triton, with local MONAI inference when Triton is unavailable and the checkpoint is present.

[Setup](#local-setup) · [Repository map](#repository-map) · [API](#api-and-example-data) · [Tests](#verification) · [Model details](models/README.md)

## Local setup

Use a Python 3.12 environment, as targeted by the original project documentation. Start from the repository root:

```sh
git clone https://github.com/techgamewithharsh-dot/Hacknite-NeuroSight.git
cd Hacknite-NeuroSight
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate`.

### Model weights are required for segmentation

The segmentation implementation expects a pretrained checkpoint at `models/brats_mri_segmentation.pt`, or the path supplied through `NEUROSIGHT_MONAI_WEIGHTS`. This checkpoint is not bundled in the public checkout.

Read [models/README.md](models/README.md) for the upstream MONAI bundle and expected channels. The helper below downloads the bundle and copies its checkpoint to a different location:

```sh
python scripts/download_models.py --bundle brats_mri_segmentation
export NEUROSIGHT_MONAI_WEIGHTS="$PWD/triton/model_repository/brain_tumor_segresnet/weights.pth"
```

Confirm that the download succeeded and that this file exists before starting segmentation. The current segmenter raises an error if the checkpoint is missing; Triton being optional does not make the checkpoint optional.

### Launch the browser interface

```sh
python -m uvicorn api.server:app --host 127.0.0.1 --port 8000
```

Open **http://localhost:8000** for the frontend and **http://localhost:8000/docs** for the generated API reference. The API has a synthetic-case generation path when sample cases are absent.

### Launch the Streamlit interface

In the activated environment:

```sh
streamlit run app.py --server.port=8501 --server.address=127.0.0.1
```

Open **http://localhost:8501**.

Optional Triton deployment instructions are in [models/README.md](models/README.md). Set `NEUROSIGHT_USE_TRITON=0` to force local model inference.

## Repository map

| Path | Purpose |
| --- | --- |
| [api/server.py](api/server.py) | FastAPI routes and frontend serving |
| [api/schemas.py](api/schemas.py) | Request and response models |
| [core/](core/) | Segmentation, progression analysis, registration, and mesh generation |
| [frontend/](frontend/) | Built browser frontend |
| [neurosight-src/](neurosight-src/) | Frontend and related TypeScript source workspace |
| [app.py](app.py) and [ui/](ui/) | Streamlit application and visualization |
| [data_generator/](data_generator/) | Synthetic demonstration case generation |
| [models/README.md](models/README.md) | Model provenance and serving setup |
| [tests/](tests/) | API, progression, atlas, and GLB checks |

## API and example data

Inspect **/docs** on the running local server for the endpoint-specific request formats. The preset analysis route is `POST /api/v1/analyze/preset`; upload routes have their own input requirements.

Example preset request, using an ID that must exist in your generated cases:

```json
{
  "preset_case_id": "case_001_true_progression",
  "patient_id": "DEMO-001",
  "scan_interval_days": 90,
  "radiation_completion_interval": "90 Days post-radiation"
}
```

The response schema includes patient metadata, a verdict card, volumetric metrics, an expansion vector, telemetry, and serving-backend information. Refer to [api/schemas.py](api/schemas.py) for current fields.

The included case generator creates synthetic examples. Values shown by those examples, including confidence percentages and timings, are demonstration outputs rather than validation on a clinical cohort.

## Verification

After installing dependencies and any required model/data assets:

```sh
python -m unittest discover -s tests -p "test_*.py"
```

## Troubleshooting

- **Missing pretrained weights:** check `NEUROSIGHT_MONAI_WEIGHTS` and the file created by the download helper.
- **Triton unavailable:** local MONAI inference still requires the same trained checkpoint.
- **Missing sample cases:** inspect startup output or run `python data_generator/generate_synthetic_cases.py`.
- **Installation failure:** include the Python version, platform, and failing dependency when reporting it. Dependencies currently use minimum-version constraints, so installations can change over time.

## Contributing and feedback

[Open an issue](https://github.com/techgamewithharsh-dot/Hacknite-NeuroSight/issues) with reproduction steps, expected behavior, and a synthetic example. Useful contributions include reproducible setup instructions, dependency compatibility fixes, and tests for failure cases. Never include identifiable patient data in a public issue.
