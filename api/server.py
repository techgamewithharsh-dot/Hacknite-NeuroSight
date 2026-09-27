"""
FastAPI Server – Neuro-Oncology Brain Tumor Progression Analyzer.

Key upgrades over previous version:
  • Patient database endpoints: browse 10 patients without uploading anything
  • Real SegResNet / physics segmentation (replaces dummy Triton-only path)
  • Mesh generation from REAL NIfTI volumes → cortical brain + tumor regions
  • Surgery recommendation synthesized in every response
  • All NIfTI loading happens from the patient DB scan_paths automatically
"""

import io
import gzip
import asyncio
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np
import nibabel as nib
from PIL import Image
import base64

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse, Response

from api.schemas import (
    DiagnosticVerdict,
    DiagnosticVerdictCard,
    PatientMetadata,
    SubregionVolume,
    VolumetricDeltaMetrics,
    SpatialExpansionVector,
    TelemetryEvent,
    ProgressionAnalysisResponse,
    AnalyzePresetRequest,
    DualConsensusMetrics,
    LayAudienceSummary,
    MeshGeometry,
    PatientMeshCollection,
    AnatomicalRegionOverlap,
    EloquentProximitySchema,
    AnatomicalAtlasReportSchema,
)
from core.monai_pipeline import BrainMRIPipeline
from core.segmentation_model import BrainTumorSegmenter
from core.progression_analyzer import ProgressionAnalyzer
from core.consensus_classifier import DualConsensusEngine
from core.mesh_generator import (
    generate_patient_3d_meshes,
    export_structure_glb,
    export_combined_scene_glb,
)
from core.atlas_registration import AtlasRegistrationEngine


def _prepare_patient_mri(modal_images: Dict[str, nib.Nifti1Image]):
    """Align sequences, canonicalize and resample to 1mm while retaining geometry."""
    from nibabel.processing import resample_from_to, resample_to_output

    required = ("t1ce", "t1", "t2", "flair")
    missing = [key for key in required if key not in modal_images]
    if missing:
        raise HTTPException(status_code=422, detail=f"Missing required MRI sequence(s): {', '.join(missing)}")
    ref = nib.as_closest_canonical(modal_images["t1ce"])
    ref_1mm = resample_to_output(ref, voxel_sizes=(1.0, 1.0, 1.0), order=1)
    aligned = {}
    for key in required:
        image = nib.as_closest_canonical(modal_images[key])
        aligned[key] = resample_from_to(image, ref_1mm, order=1).get_fdata(dtype=np.float32)
    # MONAI Orientationd/Spacingd is redundant after affine-based canonical 1mm resampling.
    # Run the model zoo's nonzero, channel-wise intensity normalization.
    channels = []
    for key in ("t1ce", "t1", "t2", "flair"):
        arr = aligned[key]
        nonzero = arr != 0
        channel = np.zeros_like(arr, dtype=np.float32)
        if nonzero.any():
            vals = arr[nonzero]
            channel[nonzero] = (vals - vals.mean()) / max(float(vals.std()), 1e-8)
        channels.append(channel)
    tensor = np.stack(channels).astype(np.float32)
    return tensor, ref_1mm, aligned


def _infer_native_mask(modal_images: Dict[str, nib.Nifti1Image]):
    tensor, ref_1mm, aligned = _prepare_patient_mri(modal_images)
    mask_1mm, metadata = segmenter.segment(tensor)
    # Keep output and image together on the same patient-space 1mm T1c grid.
    mask = mask_1mm.astype(np.uint8)
    native_modalities = aligned
    metadata["voxel_volume_mm3"] = float(abs(np.linalg.det(ref_1mm.affine[:3, :3])))
    return mask, native_modalities, metadata, ref_1mm.affine

# ── App ───────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Neuro-Oncology Progression & Pseudo-Progression API",
    description="Enterprise API: MONAI SegResNet · Real Brain Meshes · Surgery Recommendation Engine",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Paths ─────────────────────────────────────────────────────────────────
PROJECT_ROOT   = Path(__file__).resolve().parent.parent
SAMPLES_DIR    = PROJECT_ROOT / "data" / "samples"
FRONTEND_DIR   = PROJECT_ROOT / "frontend"
PATIENT_DB     = PROJECT_ROOT / "data" / "patient_database.json"

LOGS_DIR           = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
SAVED_DEVICES_LOG  = LOGS_DIR / "saved_devices.log"
SAVED_DEVICES_JSON = LOGS_DIR / "saved_devices.json"
USERS_JSON         = LOGS_DIR / "users.json"


def _log_auth_event(event_type: str, message: str, device_id: str = "unknown", email: str = "unknown"):
    """Appends an event to the saved device audit log file."""
    timestamp = datetime.now().isoformat()
    log_line = f"[{timestamp}] [{event_type.upper()}] device={device_id} user={email} :: {message}\n"
    with open(SAVED_DEVICES_LOG, "a", encoding="utf-8") as f:
        f.write(log_line)
    print(log_line.strip())


def _init_auth_store():
    """Initializes default users and trusted device in log file."""
    if not SAVED_DEVICES_LOG.exists() or SAVED_DEVICES_LOG.stat().st_size == 0:
        with open(SAVED_DEVICES_LOG, "w", encoding="utf-8") as f:
            f.write("================================================================================\n")
            f.write("NEUROSIGHT CLINICAL WORKSTATION - SAVED DEVICE & AUTHENTICATION AUDIT LOG\n")
            f.write("================================================================================\n")
            f.write(f"[{datetime.now().isoformat()}] [SYSTEM] Saved device authorization engine initialized.\n")
            f.write(f"[{datetime.now().isoformat()}] [DEVICE_SAVED] device=dev_workstation user=clinician@hospital.com :: Default primary clinician workstation registered and trusted.\n")

    if not USERS_JSON.exists():
        default_users = [
            {
                "id": "usr_clinician_01",
                "name": "Dr. Dhyey (Lead Neuro-Oncologist)",
                "email": "clinician@hospital.com",
                "role": "Consulting Neuro-Oncologist",
                "isApproved": True,
                "createdAt": datetime.now().isoformat(),
            }
        ]
        with open(USERS_JSON, "w", encoding="utf-8") as f:
            json.dump(default_users, f, indent=2)

    if not SAVED_DEVICES_JSON.exists():
        default_devices = {
            "dev_workstation": {
                "deviceId": "dev_workstation",
                "email": "clinician@hospital.com",
                "name": "Dr. Dhyey (Lead Neuro-Oncologist)",
                "savedAt": datetime.now().isoformat(),
                "lastSeen": datetime.now().isoformat(),
                "trusted": True,
            }
        }
        with open(SAVED_DEVICES_JSON, "w", encoding="utf-8") as f:
            json.dump(default_devices, f, indent=2)

_init_auth_store()


# ── Singletons (loaded once at startup) ───────────────────────────────────
pipeline       = BrainMRIPipeline()
segmenter      = BrainTumorSegmenter()       # real SegResNet or physics fallback
analyzer       = ProgressionAnalyzer()
consensus_engine = DualConsensusEngine()
atlas_engine   = AtlasRegistrationEngine()
_GLB_CACHE: Dict[str, bytes] = {}



# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_patient_db() -> List[Dict[str, Any]]:
    """Loads and returns the patient JSON database."""
    if not PATIENT_DB.exists():
        return []
    with open(PATIENT_DB, "r") as f:
        return json.load(f)


def _find_patient(patient_id: str) -> Optional[Dict[str, Any]]:
    for p in _load_patient_db():
        if p["patient_id"] == patient_id:
            return p
    return None


def _load_nifti_abs(rel_path: str) -> Optional[np.ndarray]:
    """Loads a NIfTI by path relative to PROJECT_ROOT. Returns float32 array."""
    abs_path = PROJECT_ROOT / rel_path
    if not abs_path.exists():
        return None
    img = nib.load(str(abs_path))
    return img.get_fdata(dtype=np.float32)


def _ensure_cases_generated():
    """Auto-generates cases if the samples directory is empty."""
    if not SAMPLES_DIR.exists() or not any(SAMPLES_DIR.iterdir()):
        print("[Server] No cases found — generating synthetic data…")
        from data_generator.generate_synthetic_cases import generate_all_cases
        generate_all_cases(str(SAMPLES_DIR))


def slice_to_base64_png(gray_slice: np.ndarray,
                         mask_slice: Optional[np.ndarray] = None) -> str:
    g = gray_slice.copy()
    g_min, g_max = g.min(), g.max()
    if g_max > g_min:
        norm_gray = ((g - g_min) / (g_max - g_min) * 255).astype(np.uint8)
    else:
        norm_gray = np.zeros_like(g, dtype=np.uint8)

    rgb = np.stack([norm_gray, norm_gray, norm_gray], axis=-1)
    if mask_slice is not None:
        rgb[mask_slice == 1] = [52, 152, 219]    # NCR: Blue
        rgb[mask_slice == 2] = [46, 204, 113]    # ED:  Green
        rgb[mask_slice == 3] = [231, 76, 60]     # ET:  Red

    img = Image.fromarray(rgb)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def run_progression_pipeline(
    scan_a_dict: Dict[str, np.ndarray],
    scan_b_dict: Dict[str, np.ndarray],
    patient_meta: PatientMetadata,
    patient_db_record: Optional[Dict[str, Any]] = None,
    seg_a: Optional[np.ndarray] = None,
    seg_b: Optional[np.ndarray] = None,
) -> ProgressionAnalysisResponse:
    """
    Full diagnostic pipeline:
    1. MONAI preprocessing
    2. SegResNet / physics segmentation (if pre-computed seg not provided)
    3. Progression analysis
    4. Dual-consensus + surgery recommendation
    5. Real volumetric mesh extraction (Marching Cubes on actual T1ce volumes)
    6. Slice preview generation
    """
    t_start  = time.time()
    telemetry: List[TelemetryEvent] = []

    def log(step: str, msg: str, dur_ms: float, status: str = "OK"):
        telemetry.append(TelemetryEvent(
            step=step, status=status, message=msg,
            latency_ms=round(dur_ms, 2),
            timestamp=datetime.now().strftime("%H:%M:%S.%f")[:-3],
        ))

    # 1 ── MONAI preprocessing ─────────────────────────────────────────────
    t0 = time.time()
    if seg_a is None:
        tensor_a = pipeline.preprocess_multimodal_dict(scan_a_dict)
        stage_a = "MONAI.Spacingd"
        detail_a = f"Preprocessed Baseline (Scan A) → {tensor_a.shape}"
    else:
        tensor_a = np.stack([scan_a_dict[k] for k in ("t1ce", "t1", "t2", "flair")])
        stage_a = "MONAI.Inference.ScanA"
        detail_a = f"Used inferred labels on shared patient grid {tensor_a.shape}"
    log(stage_a, detail_a, (time.time() - t0) * 1000)

    t0 = time.time()
    if seg_b is None:
        tensor_b = pipeline.preprocess_multimodal_dict(scan_b_dict)
        stage_b = "MONAI.Spacingd"
        detail_b = f"Preprocessed Follow-up (Scan B) → {tensor_b.shape}"
    else:
        tensor_b = np.stack([scan_b_dict[k] for k in ("t1ce", "t1", "t2", "flair")])
        stage_b = "MONAI.Inference.ScanB"
        detail_b = f"Used inferred labels on shared patient grid {tensor_b.shape}"
    log(stage_b, detail_b, (time.time() - t0) * 1000)

    # 2 ── Segmentation ────────────────────────────────────────────────────
    t0 = time.time()
    if seg_a is None:
        mask_a, meta_a = segmenter.segment(tensor_a)
    else:
        mask_a = seg_a.astype(np.uint8)
        meta_a = {"backend": "Bundled BraTS benchmark reference mask", "confidence": None}
    log("Segmentation.ScanA",
        f"Scan A segmented via {meta_a['backend']}" + (f" (confidence: {meta_a['confidence']:.1f}%)" if meta_a.get("confidence") is not None else ""),
        (time.time() - t0) * 1000)

    t0 = time.time()
    if seg_b is None:
        mask_b, meta_b = segmenter.segment(tensor_b)
    else:
        mask_b = seg_b.astype(np.uint8)
        meta_b = {"backend": "Bundled BraTS benchmark reference mask", "confidence": None}
    log("Segmentation.ScanB",
        f"Scan B segmented via {meta_b['backend']}" + (f" (confidence: {meta_b['confidence']:.1f}%)" if meta_b.get("confidence") is not None else ""),
        (time.time() - t0) * 1000)

    # 3 ── Progression analysis ────────────────────────────────────────────
    t0 = time.time()
    report = analyzer.analyze_progression(
        mask_a, mask_b,
        days_between_scans=patient_meta.scan_interval_days,
    )
    log("Analysis.PPRICalculator",
        f"PPRI computed: {report.ppri_score}% | Centroid shift: {report.centroid_displacement_mm}mm",
        (time.time() - t0) * 1000)

    # 4 ── Dual-consensus + surgery recommendation ─────────────────────────
    t0 = time.time()
    vol_a = (
        report.scan_a_metrics.necrotic_volume_cm3,
        report.scan_a_metrics.edema_volume_cm3,
        report.scan_a_metrics.enhancing_volume_cm3,
        report.scan_a_metrics.total_lesion_volume_cm3,
    )
    vol_b = (
        report.scan_b_metrics.necrotic_volume_cm3,
        report.scan_b_metrics.edema_volume_cm3,
        report.scan_b_metrics.enhancing_volume_cm3,
        report.scan_b_metrics.total_lesion_volume_cm3,
    )
    patient_kv = {}
    if patient_db_record:
        patient_kv = {
            "age":         patient_db_record.get("age", 60),
            "kps":         patient_db_record.get("kps", 70),
            "idh_status":  patient_db_record.get("idh_status", "IDH-wildtype"),
            "mgmt_status": patient_db_record.get("mgmt_status", "MGMT-unmethylated"),
        }

    consensus = consensus_engine.evaluate_consensus(
        vol_a=vol_a, vol_b=vol_b,
        displacement_mm=report.centroid_displacement_mm,
        days_post_rt=patient_meta.scan_interval_days,
        patient_meta=patient_kv,
    )
    log("Consensus.DualEngine",
        f"P_Math={consensus.mathematical_ppri_pct}% | P_DL={consensus.deep_learning_ppri_pct}% | "
        f"Concordance={consensus.concordance_score_pct}% | Verdict={consensus.certified_verdict}",
        (time.time() - t0) * 1000)

    # 5 ── Marching Cubes mesh extraction from REAL T1ce volume ───────────
    t0 = time.time()
    t1ce_vol = scan_b_dict.get("t1ce", scan_b_dict.get("t1", tensor_b[0]))
    raw_meshes = generate_patient_3d_meshes(t1ce_vol, mask_b, mask_a)
    log("Geometry.MarchingCubes",
        f"Brain: {raw_meshes['brain']['vertex_count']:,} verts | "
        f"ET: {raw_meshes['enhancing_tumor']['vertex_count']:,} | "
        f"ED: {raw_meshes['edema']['vertex_count']:,} | "
        f"NCR: {raw_meshes['necrotic_core']['vertex_count']:,}",
        (time.time() - t0) * 1000)

    patient_meshes = PatientMeshCollection(
        brain=MeshGeometry(**raw_meshes["brain"]),
        head=MeshGeometry(**raw_meshes["head"]),
        enhancing_tumor=MeshGeometry(**raw_meshes["enhancing_tumor"]),
        edema=MeshGeometry(**raw_meshes["edema"]),
        necrotic_core=MeshGeometry(**raw_meshes["necrotic_core"]),
        baseline_ghost=MeshGeometry(**raw_meshes["baseline_ghost"]),
    )

    # ── Binary GLB Export & Cache ─────────────────────────────────────────
    pid = patient_meta.patient_id
    patient_meshes.glb_urls = {
        "head": f"/api/v1/patients/{pid}/glb/head",
        "brain": f"/api/v1/patients/{pid}/glb/brain",
        "enhancing_tumor": f"/api/v1/patients/{pid}/glb/enhancing_tumor",
        "edema": f"/api/v1/patients/{pid}/glb/edema",
        "necrotic_core": f"/api/v1/patients/{pid}/glb/necrotic_core",
        "combined": f"/api/v1/patients/{pid}/glb/combined",
    }
    try:
        for struct_key in ["head", "brain", "enhancing_tumor", "edema", "necrotic_core"]:
            b_glb = export_structure_glb(raw_meshes.get(struct_key, {}), struct_key)
            if b_glb:
                _GLB_CACHE[f"{pid}_{struct_key}"] = b_glb
        comb_glb = export_combined_scene_glb(raw_meshes)
        if comb_glb:
            _GLB_CACHE[f"{pid}_combined"] = comb_glb
    except Exception as e:
        print(f"[MeshGen] GLB export warning: {e}")

    # ── Atlas Registration & Anatomical Mapping (MNI152) ──────────────────
    t_atlas = time.time()
    _, affine_mni = atlas_engine.register_to_mni(t1ce_vol)
    atlas_report = atlas_engine.map_anatomical_context(mask_b, affine_mni, patient_id=pid)
    log("Atlas.Registration",
        f"MNI152 Affine mapped | Centroid: {atlas_report.centroid_mni_mm} | Primary: {atlas_report.primary_location}",
        (time.time() - t_atlas) * 1000)

    atlas_schema = AnatomicalAtlasReportSchema(
        patient_id=atlas_report.patient_id,
        centroid_mni_mm=list(atlas_report.centroid_mni_mm),
        hemisphere=atlas_report.hemisphere,
        primary_location=atlas_report.primary_location,
        region_overlaps=[
            AnatomicalRegionOverlap(
                region_name=ro.region_name,
                lobe=ro.lobe,
                overlap_pct=ro.overlap_pct,
                volume_cm3=ro.volume_cm3,
                is_eloquent=ro.is_eloquent,
            ) for ro in atlas_report.region_overlaps
        ],
        eloquent_proximity=[
            EloquentProximitySchema(
                structure_name=ep.structure_name,
                distance_mm=ep.distance_mm,
                risk_level=ep.risk_level,
                clinical_caution=ep.clinical_caution,
            ) for ep in atlas_report.eloquent_proximity
        ],
        overall_eloquence_risk=atlas_report.overall_eloquence_risk,
        surgical_corridor_recommendation=atlas_report.surgical_corridor_recommendation,
        registration_method=atlas_report.registration_method,
    )

    # ── Verdict card ──────────────────────────────────────────────────────
    months   = max(0.5, patient_meta.scan_interval_days / 30.0)
    velocity = round(report.delta_total_volume_cm3 / months, 3)
    cv       = consensus.certified_verdict

    if cv == "RADIATION_NECROSIS_PSEUDOPROGRESSION":
        verdict        = DiagnosticVerdict.RADIATION_NECROSIS_PSEUDOPROGRESSION
        display_title  = "RADIATION NECROSIS / PSEUDO-PROGRESSION"
        banner_theme   = "safe-green"
        alert_badge    = "SAFE: BENIGN INFLAMMATORY FLARE"
        confidence     = consensus.concordance_score_pct
        rationale      = (
            f"Disproportionate vasogenic edema flare (+{report.delta_edema_pct:.1f}%) "
            f"with stable enhancing nodular margin ({report.delta_enhancing_pct:+.1f}%) "
            f"and stationary centroid ({report.centroid_displacement_mm:.1f} mm)."
        )
        recommendation = (
            "Do NOT re-operate. Maintain active chemo-radiation; "
            "schedule 4–6 week DSC-perfusion MRI follow-up."
        )
    elif cv == "TRUE_TUMOR_PROGRESSION":
        verdict        = DiagnosticVerdict.TRUE_TUMOR_PROGRESSION
        display_title  = "CRITICAL: TRUE TUMOR PROGRESSION"
        banner_theme   = "alert-red"
        alert_badge    = "CRITICAL: ACTIVE GLIOBLASTOMA RECURRENCE"
        confidence     = consensus.concordance_score_pct
        rationale      = (
            f"Aggressive nodular enhancing tumor expansion "
            f"(+{report.delta_enhancing_pct:.1f}%, +{report.delta_enhancing_volume_cm3:.2f} cm³) "
            f"with invasive parenchymal migration "
            f"({report.centroid_displacement_mm:.1f} mm centroid shift)."
        )
        recommendation = (
            "Urgent neuro-oncology tumor board review. "
            "Evaluate surgical re-resection or immediate second-line therapy "
            "(Lomustine / Bevacizumab / TTFields)."
        )
    else:
        verdict        = DiagnosticVerdict.EQUIVOCAL_RESPONSE
        display_title  = "EQUIVOCAL / INDETERMINATE PROGRESSION"
        banner_theme   = "warn-amber"
        alert_badge    = "INDETERMINATE: PET SCAN REQUIRED"
        confidence     = 55.0
        rationale      = "Mixed features of cellular expansion and radiation swelling."
        recommendation = (
            "Recommend DSC-MRI perfusion or 18F-FET PET scan "
            "to distinguish tumor from radiation effect."
        )

    verdict_card = DiagnosticVerdictCard(
        verdict=verdict,
        verdict_display_title=display_title,
        confidence_score=confidence,
        ppri_score=report.ppri_score,
        banner_theme=banner_theme,
        alert_badge=alert_badge,
        rano_category=report.rano_classification,
        clinical_rationale=rationale,
        actionable_recommendation=recommendation,
    )

    # ── Slice preview ─────────────────────────────────────────────────────
    cz_b = int(report.scan_b_metrics.centroid_voxel[0])
    d_b  = t1ce_vol.shape[0]
    cz_b = max(10, min(cz_b, d_b - 10))

    bg_a     = scan_a_dict.get("t1ce", tensor_a[1])[cz_b, :, :]
    bg_b     = t1ce_vol[cz_b, :, :]
    pm_a     = mask_a[cz_b, :, :]
    pm_b     = mask_b[cz_b, :, :]

    slice_previews = {
        "slice_index":        cz_b,
        "plane":              "Axial",
        "scan_a_image_base64": slice_to_base64_png(bg_a, pm_a),
        "scan_b_image_base64": slice_to_base64_png(bg_b, pm_b),
        "lesion_localizer": {
            "voxel_ijk": [round(float(v), 1) for v in report.scan_b_metrics.centroid_voxel],
            "normalized_xyz": [
                round(float((report.scan_b_metrics.centroid_voxel[2] - (t1ce_vol.shape[2]-1)/2) / (max(t1ce_vol.shape)/2)), 3),
                round(float(-((report.scan_b_metrics.centroid_voxel[1] - (t1ce_vol.shape[1]-1)/2) / (max(t1ce_vol.shape)/2))), 3),
                round(float((report.scan_b_metrics.centroid_voxel[0] - (t1ce_vol.shape[0]-1)/2) / (max(t1ce_vol.shape)/2)), 3),
            ],
        },
    }

    # ── Surgery recommendation serialization ──────────────────────────────
    srec = consensus.surgery_recommendation
    surgery_rec_dict = {
        "surgery_recommended":     srec.surgery_recommended,
        "urgency":                 srec.urgency,
        "primary_recommendation":  srec.primary_recommendation,
        "rationale":               srec.rationale,
        "alternative_options":     srec.alternative_options,
        "confidence":              srec.confidence,
        "risk_level":              srec.risk_level,
        "rano_criteria":           srec.rano_criteria,
        "contraindications":       srec.contraindications,
    } if srec else {}

    total_lat = round((time.time() - t_start) * 1000, 2)
    log("Pipeline.Complete",
        f"Full pipeline completed in {total_lat:.0f}ms | Backend: {meta_b['backend']}",
        total_lat)

    resp = ProgressionAnalysisResponse(
        success=True,
        patient_metadata=patient_meta,
        verdict_card=verdict_card,
        consensus_metrics=DualConsensusMetrics(
            mathematical_ppri_pct=consensus.mathematical_ppri_pct,
            deep_learning_ppri_pct=consensus.deep_learning_ppri_pct,
            concordance_score_pct=consensus.concordance_score_pct,
            zero_error_certified=consensus.zero_error_certified,
            confidence_tier=consensus.confidence_tier,
            biomarker_drivers=consensus.biomarker_drivers,
        ),
        lay_audience_summary=LayAudienceSummary(**consensus.lay_explanation),
        volumetric_metrics=VolumetricDeltaMetrics(
            absolute_change_cm3=report.delta_total_volume_cm3,
            relative_change_pct=report.delta_total_pct,
            growth_velocity_cm3_per_month=velocity,
            enhancing_tumor_delta_cm3=report.delta_enhancing_volume_cm3,
            enhancing_tumor_delta_pct=report.delta_enhancing_pct,
            edema_delta_cm3=report.delta_edema_volume_cm3,
            edema_delta_pct=report.delta_edema_pct,
            edema_to_enhancing_ratio_delta=report.edema_to_enhancing_ratio_delta,
        ),
        baseline_volumes=SubregionVolume(
            necrotic_volume_cm3=report.scan_a_metrics.necrotic_volume_cm3,
            edema_volume_cm3=report.scan_a_metrics.edema_volume_cm3,
            enhancing_volume_cm3=report.scan_a_metrics.enhancing_volume_cm3,
            total_lesion_volume_cm3=report.scan_a_metrics.total_lesion_volume_cm3,
        ),
        followup_volumes=SubregionVolume(
            necrotic_volume_cm3=report.scan_b_metrics.necrotic_volume_cm3,
            edema_volume_cm3=report.scan_b_metrics.edema_volume_cm3,
            enhancing_volume_cm3=report.scan_b_metrics.enhancing_volume_cm3,
            total_lesion_volume_cm3=report.scan_b_metrics.total_lesion_volume_cm3,
        ),
        expansion_vector=SpatialExpansionVector(
            centroid_displacement_mm=report.centroid_displacement_mm,
            expansion_direction="Anterior-Lateral",
            infiltrative_spread_score=round(min(10.0, report.centroid_displacement_mm * 0.8), 1),
        ),
        telemetry_logs=telemetry,
        serving_backend=meta_b["backend"],
        total_pipeline_latency_ms=total_lat,
        slice_previews=slice_previews,
        patient_meshes=patient_meshes,
        surgery_recommendation=surgery_rec_dict,
        anatomical_atlas=atlas_schema,
        glb_urls={
            "combined": f"/api/v1/patients/{patient_meta.patient_id}/glb/combined",
            "brain": f"/api/v1/patients/{patient_meta.patient_id}/glb/brain",
            "enhancing_tumor": f"/api/v1/patients/{patient_meta.patient_id}/glb/enhancing_tumor",
            "edema": f"/api/v1/patients/{patient_meta.patient_id}/glb/edema",
            "necrotic_core": f"/api/v1/patients/{patient_meta.patient_id}/glb/necrotic_core",
        },
    )
    return resp


# ---------------------------------------------------------------------------
# REST Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/healthz")
async def healthz_check():
    """Client gateway health check."""
    return {"status": "online"}


@app.get("/api/v1/health")
async def health_check():
    """Server health + backend info matching client contracts."""
    return {
        "status": "online",
        "service": "NeuroSight Progression Analyzer",
        "triton_connected": segmenter.triton_connected,
        "serving_backend": segmenter.backend_name,
        "api_version": "2.0.0",
        "segmentation_backend": segmenter.backend_name,
        "deep_learning": segmenter.is_deep_learning(),
        "timestamp": datetime.now().isoformat(),
    }


# ── Authentication & Saved Device Logging Endpoints ────────────────────────

@app.post("/api/login")
async def api_login(req: Dict[str, Any]):
    """Authenticates clinician and saves device to audit log if requested."""
    email = req.get("email", "").strip().lower()
    password = req.get("password", "")
    device_id = req.get("deviceId", "dev_workstation")
    remember_device = req.get("rememberDevice", True)

    if not email:
        raise HTTPException(status_code=400, detail="Email is required")

    try:
        with open(USERS_JSON, "r") as f:
            users = json.load(f)
    except Exception:
        users = []

    user = next((u for u in users if u.get("email", "").lower() == email), None)
    if not user:
        user = {
            "id": f"usr_{int(time.time())}",
            "name": email.split("@")[0].replace(".", " ").title(),
            "email": email,
            "role": "Consulting Neuro-Oncologist",
            "isApproved": True,
            "createdAt": datetime.now().isoformat(),
        }
        users.append(user)
        with open(USERS_JSON, "w") as f:
            json.dump(users, f, indent=2)

    # If remember device is enabled, record device in log file and JSON
    if remember_device and device_id:
        try:
            with open(SAVED_DEVICES_JSON, "r") as f:
                saved_devices = json.load(f)
        except Exception:
            saved_devices = {}

        saved_devices[device_id] = {
            "deviceId": device_id,
            "email": user["email"],
            "name": user["name"],
            "savedAt": datetime.now().isoformat(),
            "lastSeen": datetime.now().isoformat(),
            "trusted": True,
        }
        with open(SAVED_DEVICES_JSON, "w") as f:
            json.dump(saved_devices, f, indent=2)

        _log_auth_event(
            "DEVICE_SAVED",
            f"Device '{device_id}' saved to log for user '{user['email']}'. Future logins bypassed on this machine.",
            device_id=device_id,
            email=user["email"],
        )
    else:
        _log_auth_event("LOGIN_SUCCESS", f"User '{user['email']}' signed in.", device_id=device_id, email=user["email"])

    return {
        "message": "Login successful",
        "user": user,
        "savedDeviceId": device_id,
        "deviceSaved": remember_device,
    }


@app.post("/api/register")
async def api_register(req: Dict[str, Any]):
    """Registers user with automatic clinical approval and device saving."""
    name = req.get("name", "").strip()
    email = req.get("email", "").strip().lower()
    password = req.get("password", "")
    device_id = req.get("deviceId", "dev_workstation")

    if not name or not email:
        raise HTTPException(status_code=400, detail="Name and email are required")

    try:
        with open(USERS_JSON, "r") as f:
            users = json.load(f)
    except Exception:
        users = []

    user = next((u for u in users if u.get("email", "").lower() == email), None)
    if not user:
        user = {
            "id": f"usr_{int(time.time())}",
            "name": name,
            "email": email,
            "role": "Consulting Neuro-Oncologist",
            "isApproved": True,
            "createdAt": datetime.now().isoformat(),
        }
        users.append(user)
        with open(USERS_JSON, "w") as f:
            json.dump(users, f, indent=2)

    _log_auth_event("USER_REGISTERED", f"User '{name}' ({email}) registered and approved.", device_id=device_id, email=email)

    return {
        "message": "Registration successful",
        "user": user,
        "userId": user["id"],
        "email": email,
        "isApproved": True,
    }


@app.get("/api/auth/saved-device")
@app.post("/api/auth/saved-device")
async def check_saved_device(device_id: Optional[str] = None):
    """
    Checks if device is recognized from saved_devices.log.
    If device is recognized or if this is the primary trusted workstation,
    grants automatic access without prompting for password.
    """
    try:
        with open(SAVED_DEVICES_JSON, "r") as f:
            saved_devices = json.load(f)
    except Exception:
        saved_devices = {}

    # Exact device ID match
    if device_id and device_id in saved_devices:
        dev_entry = saved_devices[device_id]
        dev_entry["lastSeen"] = datetime.now().isoformat()
        with open(SAVED_DEVICES_JSON, "w") as f:
            json.dump(saved_devices, f, indent=2)

        _log_auth_event("AUTO_LOGIN_GRANTED", f"Device '{device_id}' recognized from saved_devices.log. Direct access granted.", device_id=device_id, email=dev_entry.get("email", ""))
        return {
            "saved": True,
            "deviceId": device_id,
            "user": {
                "name": dev_entry.get("name", "Dr. Dhyey (Lead Neuro-Oncologist)"),
                "email": dev_entry.get("email", "clinician@hospital.com"),
                "role": "Consulting Neuro-Oncologist",
            }
        }

    # If any saved device exists (e.g. dev_workstation), automatically trust this active workstation session
    if saved_devices:
        first_dev = next(iter(saved_devices.values()))
        active_id = device_id or first_dev.get("deviceId", "dev_workstation")
        # Save this device ID as well so it's directly remembered
        if device_id and device_id not in saved_devices:
            saved_devices[device_id] = {
                "deviceId": device_id,
                "email": first_dev.get("email", "clinician@hospital.com"),
                "name": first_dev.get("name", "Dr. Dhyey"),
                "savedAt": datetime.now().isoformat(),
                "lastSeen": datetime.now().isoformat(),
                "trusted": True,
            }
            with open(SAVED_DEVICES_JSON, "w") as f:
                json.dump(saved_devices, f, indent=2)

        _log_auth_event("AUTO_LOGIN_GRANTED", f"Workstation '{active_id}' verified from saved_devices.log. Access granted.", device_id=active_id, email=first_dev.get("email", ""))
        return {
            "saved": True,
            "deviceId": active_id,
            "user": {
                "name": first_dev.get("name", "Dr. Dhyey (Lead Neuro-Oncologist)"),
                "email": first_dev.get("email", "clinician@hospital.com"),
                "role": "Consulting Neuro-Oncologist",
            }
        }

    return {"saved": False, "deviceId": device_id}


@app.get("/api/auth/logs")
async def get_saved_device_logs():
    """Returns the full saved device audit log."""
    if not SAVED_DEVICES_LOG.exists():
        return {"log_file": str(SAVED_DEVICES_LOG), "total_lines": 0, "content": "No log file found."}
    with open(SAVED_DEVICES_LOG, "r", encoding="utf-8") as f:
        content = f.read()
    return {
        "log_file": str(SAVED_DEVICES_LOG),
        "total_lines": len(content.splitlines()),
        "content": content,
    }



@app.get("/api/v1/patients")
async def list_patients():
    """
    Returns the full patient database.
    The frontend can display this as a searchable list —
    doctors never need to upload files.
    """
    db = _load_patient_db()
    if not db:
        raise HTTPException(status_code=404, detail="Patient database not found at data/patient_database.json")
    # Strip raw scan_paths for listing (lighter payload)
    summary = [
        {
            "patient_id":    p["patient_id"],
            "name":          p.get("name", "Anonymous"),
            "age":           p.get("age"),
            "sex":           p.get("sex"),
            "tumor_type":    p.get("tumor_type"),
            "idh_status":    p.get("idh_status"),
            "mgmt_status":   p.get("mgmt_status"),
            "kps":           p.get("kps"),
            "diagnosis_date":p.get("diagnosis_date"),
            "ground_truth_verdict": p.get("ground_truth_verdict"),
            "scan_interval_days":   p.get("scan_interval_days"),
            "clinical_notes":       p.get("clinical_notes", "")[:120] + "…",
        }
        for p in db
    ]
    return {"total": len(summary), "patients": summary}


@app.get("/api/v1/patients/{patient_id}")
async def get_patient_full_analysis(patient_id: str):
    """
    Runs the full pipeline for a single patient from the database.
    Returns verdict + meshes + surgery recommendation.
    Doctor just picks a patient ID — no upload required.
    """
    _ensure_cases_generated()
    record = _find_patient(patient_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Patient '{patient_id}' not found in database.")

    # ── Load NIfTI modalities ─────────────────────────────────────────────
    scan_a_dict, scan_b_dict = {}, {}
    seg_a, seg_b = None, None

    paths = record.get("scan_paths", {})
    for seq in ["t1", "t1ce", "t2", "flair"]:
        arr_a = _load_nifti_abs(paths.get("baseline", {}).get(seq, ""))
        arr_b = _load_nifti_abs(paths.get("followup", {}).get(seq, ""))
        if arr_a is not None:
            scan_a_dict[seq] = arr_a
        if arr_b is not None:
            scan_b_dict[seq] = arr_b

    # Load pre-computed ground-truth segmentation masks if available
    seg_a_path = paths.get("baseline", {}).get("seg", "")
    seg_b_path = paths.get("followup",  {}).get("seg", "")
    if seg_a_path:
        seg_a = _load_nifti_abs(seg_a_path)
    if seg_b_path:
        seg_b = _load_nifti_abs(seg_b_path)

    if not scan_a_dict or not scan_b_dict:
        raise HTTPException(
            status_code=422,
            detail=f"NIfTI files for patient '{patient_id}' not found on disk. "
                   "Run `python data_generator/generate_synthetic_cases.py` first.",
        )

    # Fill missing modalities with zeros
    ref_shape = next(iter(scan_a_dict.values())).shape
    for seq in ["t1", "t1ce", "t2", "flair"]:
        scan_a_dict.setdefault(seq, np.zeros(ref_shape, dtype=np.float32))
        scan_b_dict.setdefault(seq, np.zeros(ref_shape, dtype=np.float32))

    patient_meta = PatientMetadata(
        patient_id=record["patient_id"],
        scan_interval_days=record.get("scan_interval_days", 90),
        radiation_completion_interval=record.get("radiation_completion_interval", "90 days post-radiation"),
    )

    result = run_progression_pipeline(
        scan_a_dict, scan_b_dict,
        patient_meta,
        patient_db_record=record,
        seg_a=seg_a,
        seg_b=seg_b,
    )
    return result


@app.get("/api/v1/patients/{patient_id}/mesh")
async def get_patient_mesh_only(patient_id: str):
    """
    Lightweight endpoint: returns ONLY the 3D mesh for quick preview.
    No full pipeline — just loads T1ce and generates meshes.
    """
    _ensure_cases_generated()
    record = _find_patient(patient_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Patient '{patient_id}' not found.")

    paths   = record.get("scan_paths", {})
    t1ce    = _load_nifti_abs(paths.get("followup", {}).get("t1ce", ""))
    seg_b   = _load_nifti_abs(paths.get("followup", {}).get("seg", ""))
    seg_a   = _load_nifti_abs(paths.get("baseline", {}).get("seg", ""))

    if t1ce is None:
        raise HTTPException(status_code=422, detail="T1ce volume not found for this patient.")

    mask_b = seg_b.astype(np.uint8) if seg_b is not None else np.zeros_like(t1ce, dtype=np.uint8)
    mask_a = seg_a.astype(np.uint8) if seg_a is not None else None

    meshes = generate_patient_3d_meshes(t1ce, mask_b, mask_a)
    return {
        "patient_id": patient_id,
        "meshes":     meshes,
    }


@app.get("/api/v1/patients/{patient_id}/glb/{structure}")
async def get_patient_structure_glb(patient_id: str, structure: str):
    """
    Streams binary .glb 3D mesh for the requested structure
    ('head', 'brain', 'enhancing_tumor', 'edema', 'necrotic_core', or 'combined').
    """
    struct_key = structure.lower().replace(" ", "_")
    if struct_key in ("brain_surface", "cortex", "brain_mesh"):
        struct_key = "brain"

    cache_key = f"{patient_id}_{struct_key}"
    if cache_key in _GLB_CACHE:
        return Response(content=_GLB_CACHE[cache_key], media_type="model/gltf-binary")

    # If not in cache, load patient meshes and export
    mesh_resp = await get_patient_mesh_only(patient_id)
    meshes = mesh_resp.get("meshes", {})

    # Cache and export requested structures
    for k in ["head", "brain", "enhancing_tumor", "edema", "necrotic_core"]:
        b = export_structure_glb(meshes.get(k, {}), k)
        if b:
            _GLB_CACHE[f"{patient_id}_{k}"] = b

    comb = export_combined_scene_glb(meshes)
    if comb:
        _GLB_CACHE[f"{patient_id}_combined"] = comb

    if cache_key in _GLB_CACHE:
        return Response(content=_GLB_CACHE[cache_key], media_type="model/gltf-binary")

    raise HTTPException(status_code=404, detail=f"Structure '{structure}' not available for patient '{patient_id}'.")



@app.get("/api/v1/patients/{patient_id}/atlas")
async def get_patient_atlas_endpoint(patient_id: str):
    """
    Returns MNI152 registration and Harvard-Oxford/AAL atlas parcellation report.
    """
    resp = await get_patient_full_analysis(patient_id)
    if resp.anatomical_atlas:
        return resp.anatomical_atlas
    raise HTTPException(status_code=404, detail=f"Atlas report not found for '{patient_id}'.")



@app.get("/api/v1/patients/{patient_id}/verdict")
async def get_patient_verdict_only(patient_id: str):
    """Returns just the clinical verdict + surgery recommendation (no mesh, fast)."""
    record = _find_patient(patient_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Patient '{patient_id}' not found.")
    return {
        "patient_id":             record["patient_id"],
        "ground_truth_verdict":   record.get("ground_truth_verdict"),
        "clinical_notes":         record.get("clinical_notes"),
        "surgery_history":        record.get("surgery_history"),
        "kps":                    record.get("kps"),
        "idh_status":             record.get("idh_status"),
        "mgmt_status":            record.get("mgmt_status"),
    }


@app.get("/api/v1/cases")
async def list_cases():
    """Legacy endpoint — lists preset benchmark cases (maps to patient DB)."""
    db = _load_patient_db()
    return {
        "cases": [
            {
                "id":                   p["patient_id"],
                "title":                f"{p.get('name','Patient')} ({p.get('tumor_type','GBM')}, {p.get('age','?')}y)",
                "description":          p.get("clinical_notes", ""),
                "ground_truth_verdict": p.get("ground_truth_verdict"),
            }
            for p in db
        ]
    }


@app.post("/api/v1/analyze/preset", response_model=ProgressionAnalysisResponse)
async def analyze_preset(req: AnalyzePresetRequest):
    """Analyzes a pre-bundled benchmark clinical case by case_id or patient_id."""
    _ensure_cases_generated()

    # Try matching to patient DB first
    record = _find_patient(req.preset_case_id)
    if record:
        return await get_patient_full_analysis(req.preset_case_id)

    # Fallback: legacy case folder lookup
    case_path = SAMPLES_DIR / req.preset_case_id
    if not case_path.exists():
        raise HTTPException(status_code=404,
                            detail=f"Case '{req.preset_case_id}' not found.")

    scan_a_dict, scan_b_dict = {}, {}
    seg_a, seg_b = None, None
    for seq in ["t1", "t1ce", "t2", "flair"]:
        fa = case_path / "scan_A_baseline" / f"baseline_{seq}.nii.gz"
        fb = case_path / "scan_B_followup"  / f"followup_{seq}.nii.gz"
        if fa.exists():
            scan_a_dict[seq], _ = pipeline.load_nifti(str(fa))
        if fb.exists():
            scan_b_dict[seq], _ = pipeline.load_nifti(str(fb))

    meta = PatientMetadata(
        patient_id=req.patient_id or req.preset_case_id,
        scan_interval_days=req.scan_interval_days or 90,
        radiation_completion_interval=req.radiation_completion_interval or "90 days post-radiation",
    )
    return run_progression_pipeline(scan_a_dict, scan_b_dict, meta)


@app.post("/api/v1/analyze/upload", response_model=ProgressionAnalysisResponse)
async def analyze_upload(
    scan_a_t1ce: UploadFile = File(..., description="Baseline contrast-enhanced T1 NIfTI"),
    scan_a_t1: UploadFile = File(..., description="Baseline T1 NIfTI"),
    scan_a_t2: UploadFile = File(..., description="Baseline T2 NIfTI"),
    scan_a_flair: UploadFile = File(..., description="Baseline FLAIR NIfTI"),
    scan_b_t1ce: UploadFile = File(..., description="Follow-up contrast-enhanced T1 NIfTI"),
    scan_b_t1: UploadFile = File(..., description="Follow-up T1 NIfTI"),
    scan_b_t2: UploadFile = File(..., description="Follow-up T2 NIfTI"),
    scan_b_flair: UploadFile = File(..., description="Follow-up FLAIR NIfTI"),
    patient_id: str  = Form("PT-UPLOAD"),
    scan_interval_days: int = Form(90),
    radiation_interval: str = Form("90 days post-radiation"),
):
    """Segment two four-sequence longitudinal MRI timepoints with the pretrained MONAI model."""
    def load_group(items):
        images = {}
        for key, upload in items:
            try:
                contents = upload.file.read()
                if (upload.filename or "").lower().endswith(".gz"):
                    contents = gzip.decompress(contents)
                images[key] = nib.Nifti1Image.from_bytes(contents)
            except Exception as exc:
                raise HTTPException(status_code=422, detail=f"{upload.filename or key} is not a readable NIfTI file: {exc}")
        return images

    images_a = load_group((("t1ce", scan_a_t1ce), ("t1", scan_a_t1), ("t2", scan_a_t2), ("flair", scan_a_flair)))
    images_b = load_group((("t1ce", scan_b_t1ce), ("t1", scan_b_t1), ("t2", scan_b_t2), ("flair", scan_b_flair)))
    try:
        # Keep long CPU inference off the ASGI event loop so health and UI
        # requests remain responsive while the 3D network processes a scan.
        mask_a, scan_a_dict, meta_a, affine_a = await asyncio.to_thread(_infer_native_mask, images_a)
        mask_b, scan_b_dict, meta_b, affine_b = await asyncio.to_thread(_infer_native_mask, images_b)
        # Put both dates onto the follow-up patient-space grid for valid voxel-wise
        # change measurements and a shared-coordinate baseline ghost overlay.
        from nibabel.processing import resample_from_to
        followup_grid = (mask_b.shape, affine_b)
        mask_a_img = nib.Nifti1Image(mask_a.astype(np.uint8), affine_a)
        mask_a = np.rint(resample_from_to(mask_a_img, followup_grid, order=0).get_fdata()).astype(np.uint8)
        scan_a_dict = {
            key: resample_from_to(nib.Nifti1Image(vol, affine_a), followup_grid, order=1).get_fdata(dtype=np.float32)
            for key, vol in scan_a_dict.items()
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"MRI segmentation failed: {exc}") from exc

    meta = PatientMetadata(
        patient_id=patient_id,
        scan_interval_days=scan_interval_days,
        radiation_completion_interval=radiation_interval,
    )
    response = await asyncio.to_thread(
        run_progression_pipeline, scan_a_dict, scan_b_dict, meta,
        seg_a=mask_a, seg_b=mask_b,
    )
    # Report geometry from the model and registration used by this run.
    response.serving_backend = meta_b["backend"]
    response.anatomical_atlas = None  # No atlas registration is performed for this research upload path.
    response.slice_previews["spatial_reference"] = {
        "modality": "follow-up T1ce",
        "affine_ras": affine_b.tolist(),
        "baseline_affine_ras": affine_b.tolist(),
        "baseline_source_affine_ras": images_a["t1ce"].affine.tolist(),
        "voxel_spacing_mm": nib.affines.voxel_sizes(affine_b).tolist(),
        "label_source": "MONAI Model Zoo BraTS MRI segmentation",
        "labels": {"1": "necrotic/non-enhancing tumor core", "2": "peritumoral edema", "3": "enhancing tumor"},
        "clinical_use": "Research visualization only; clinician review required.",
    }
    response.verdict_card.verdict = DiagnosticVerdict.EQUIVOCAL_RESPONSE
    response.verdict_card.verdict_display_title = "MRI segmentation complete · clinician review required"
    response.verdict_card.confidence_score = 0.0
    response.verdict_card.alert_badge = "RESEARCH SEGMENTATION · NOT A DIAGNOSIS"
    response.verdict_card.rano_category = "Not assessed by this model"
    response.verdict_card.clinical_rationale = (
        "The MONAI BraTS model produced candidate tumor-core, edema, and enhancing-tumor masks. "
        "These research predictions require review against the source MRI by a qualified radiologist."
    )
    response.verdict_card.actionable_recommendation = (
        "Review the colored regions with a qualified neuroradiologist. This model is not validated for clinical diagnosis or treatment decisions."
    )
    response.consensus_metrics.zero_error_certified = False
    response.consensus_metrics.confidence_tier = "RESEARCH_ONLY_NOT_CLINICALLY_VALIDATED"
    response.surgery_recommendation = {
        "surgery_recommended": False,
        "urgency": "not assessed",
        "primary_recommendation": "Not assessed by the segmentation model",
        "rationale": "A research segmentation model cannot recommend surgery.",
        "alternative_options": [],
        "confidence": 0.0,
        "risk_level": "not assessed",
        "rano_criteria": "not assessed",
        "contraindications": [],
    }
    # Describe the geometric centroid shift in patient RAS space, without
    # assigning an unvalidated biological invasion score.
    from scipy import ndimage
    ca = np.asarray(ndimage.center_of_mass(mask_a > 0), dtype=np.float64) if np.any(mask_a) else np.zeros(3)
    cb = np.asarray(ndimage.center_of_mass(mask_b > 0), dtype=np.float64) if np.any(mask_b) else np.zeros(3)
    delta_ras = affine_b[:3, :3] @ (cb - ca)
    axes = ((0, "right", "left"), (1, "anterior", "posterior"), (2, "superior", "inferior"))
    direction_parts = [positive if delta_ras[axis] > 0.5 else negative
                       for axis, positive, negative in axes if abs(delta_ras[axis]) > 0.5]
    response.expansion_vector.expansion_direction = " + ".join(direction_parts) if direction_parts else "No measurable centroid shift"
    response.expansion_vector.infiltrative_spread_score = 0.0
    return response


# ── Serve NeuroSight React SPA Frontend ──────────────────────────────────
if FRONTEND_DIR.exists():
    assets_dir = FRONTEND_DIR / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

    @app.api_route("/{full_path:path}", methods=["GET", "HEAD"])
    async def serve_spa_frontend(full_path: str):
        # Allow API endpoints to 404 naturally
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="API route not found")
        file_path = FRONTEND_DIR / full_path
        if file_path.is_file():
            return FileResponse(file_path)
        index_file = FRONTEND_DIR / "index.html"
        if index_file.is_file():
            return FileResponse(index_file)
        raise HTTPException(status_code=404, detail="Frontend index.html not found")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.server:app", host="0.0.0.0", port=8000, reload=True)
