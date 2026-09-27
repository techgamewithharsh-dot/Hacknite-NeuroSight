# 🧠 Real-Time Multi-Modal Brain Tumor Progression & Pseudo-Progression Analyzer
### *Hacknite 2026 High-Impact Healthcare AI Project*

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![NVIDIA MONAI](https://img.shields.io/badge/NVIDIA-MONAI_Core-76B900.svg)](https://monai.io/)
[![NVIDIA Triton](https://img.shields.io/badge/NVIDIA-Triton_Inference_Server-76B900.svg)](https://developer.nvidia.com/nvidia-triton-inference-server)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit_Terminal-FF4B4B.svg)](https://streamlit.io/)

---

## 1. The Core Clinical Problem
In neuro-oncology, post-chemoradiation brain MRI scans for glioblastoma patients are notoriously difficult to interpret. When a lesion expands on follow-up imaging, clinicians face a critical dilemma:
* **True Tumor Progression (TP):** Aggressive cancer recurrence requiring immediate surgical re-intervention or second-line oncology regimens.
* **Pseudo-Progression (PsP) / Radiation Necrosis:** A benign, transient inflammatory cascade with blood-brain barrier disruption caused by radiation therapy that will stabilize or regress without changing therapy.

**The Clinical Consequence:** Misdiagnosis leads to either unnecessary invasive brain re-resections or fatal delays in switching cancer therapies.

---

## 2. Front-End Web Applications

We provide two complementary interfaces:

### A. Dark-Mode Clinical Diagnostic Web Terminal (`frontend/` + FastAPI)
* **Aesthetic:** Dark-mode clinical diagnostic terminal (deep obsidian `#070A11`, glowing neon indicators).
* **Core Workflow:**
  1. **Dual Drag-and-Drop Landing Zone:** Ingests Timepoint A (Baseline `.nii.gz`) and Timepoint B (Follow-up `.nii.gz`) with file size validation.
  2. **Live Pipeline Telemetry Console:** Monospace console terminal streaming step-by-step progress from MONAI transforms (`Spacingd`, `Orientationd`) to Triton dynamic batching and inference.
  3. **High-Impact Verdict Dashboard:**
     - Massive color-coded banner (`🔴 CRITICAL: TRUE TUMOR PROGRESSION` vs. `🟢 SAFE: RADIATION NECROSIS / PSEUDO-PROGRESSION`).
     - Diagnostic Confidence percentage (e.g., 98.0%).
     - Volumetric Delta metrics & Growth Velocity rate ($\text{cm}^3/\text{month}$).
     - Visual comparative slice viewer displaying the volumetric expansion vector.

Launch with:
```bash
./scripts/run_api.sh
```
Open in browser: 👉 **`http://localhost:8000`**

### B. Integrated Streamlit Diagnostic HUD (`app.py`)
Launch with:
```bash
./scripts/run_app.sh
```
Open in browser: 👉 **`http://localhost:8501`**

---

## 3. Input & Output API Contracts (`api/schemas.py`)

The FastAPI backend enforces structured Pydantic contracts:

### Input Schema (`POST /api/v1/analyze/preset` or `/upload`)
```json
{
  "preset_case_id": "case_001_true_progression",
  "patient_id": "PT-84920",
  "scan_interval_days": 90,
  "radiation_completion_interval": "90 Days post-radiation"
}
```

### Output Schema (`ProgressionAnalysisResponse`)
```json
{
  "success": true,
  "patient_metadata": {
    "patient_id": "PT-84920",
    "scan_interval_days": 90
  },
  "verdict_card": {
    "verdict": "TRUE_TUMOR_PROGRESSION",
    "verdict_display_title": "CRITICAL: TRUE TUMOR PROGRESSION",
    "confidence_score": 98.0,
    "ppri_score": 2.0,
    "banner_theme": "alert-red",
    "alert_badge": "CRITICAL: ACTIVE GLIOBLASTOMA RECURRENCE",
    "rano_category": "Progressive Disease (PD)",
    "clinical_rationale": "Aggressive nodular enhancing tumor expansion...",
    "actionable_recommendation": "Urgent neuro-oncology tumor board review."
  },
  "volumetric_metrics": {
    "absolute_change_cm3": 18.35,
    "relative_change_pct": 742.0,
    "growth_velocity_cm3_per_month": 6.12,
    "enhancing_tumor_delta_cm3": 12.11,
    "edema_delta_cm3": 6.25,
    "edema_to_enhancing_ratio_delta": -1.2
  },
  "expansion_vector": {
    "centroid_displacement_mm": 10.4,
    "expansion_direction": "Anterior-Lateral",
    "infiltrative_spread_score": 8.3
  },
  "telemetry_logs": [ ... ],
  "serving_backend": "NVIDIA Triton Inference Server",
  "total_pipeline_latency_ms": 32.4
}
```

---

## 4. Benchmark Clinical Cases

* **Case 001 — True Glioblastoma Progression (Recurrence):**
  - **Verdict:** `TRUE_TUMOR_PROGRESSION` (`PPRI = 2.0%`, Confidence: `98.0%`)
  - **Volumetric Delta:** +12.11 cm³ Enhancing Tumor (+953.2%), Growth Velocity: +6.12 cm³/month
  - **Expansion Vector:** 10.4 mm invasive centroid shift
* **Case 002 — Radiation Necrosis (Pseudo-Progression):**
  - **Verdict:** `RADIATION_NECROSIS_PSEUDOPROGRESSION` (`PPRI = 98.0%`, Confidence: `98.0%`)
  - **Volumetric Delta:** +42.36 cm³ Edema Flare (+602.7%), Enhancing Tumor stable (+3.2%)
  - **Expansion Vector:** Stationary centroid, low infiltrative spread

---

## 5. Automated Verification Tests

Run the full verification test suite:
```bash
python -m unittest discover -s tests -p "test_*.py"
```
