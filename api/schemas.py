"""
Data Contracts and Schema Definitions for Brain Tumor Progression Analyzer.

Defines exact input/output Pydantic contracts between the Frontend Diagnostic Terminal
and the NVIDIA MONAI / Triton Inference Server backend.
"""

from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from pydantic import BaseModel, Field


class DiagnosticVerdict(str, Enum):
    """Binary clinical outcome with equivocal fallback."""
    TRUE_TUMOR_PROGRESSION = "TRUE_TUMOR_PROGRESSION"
    RADIATION_NECROSIS_PSEUDOPROGRESSION = "RADIATION_NECROSIS_PSEUDOPROGRESSION"
    EQUIVOCAL_RESPONSE = "EQUIVOCAL_RESPONSE"


class PatientMetadata(BaseModel):
    """Clinical metadata for the sequential analysis."""
    patient_id: str = Field(..., description="Unique clinical record identifier, e.g. 'PT-84920'")
    scan_interval_days: int = Field(90, ge=1, le=1000, description="Elapsed days between Baseline and Follow-up scans")
    radiation_completion_interval: Optional[str] = Field("90 Days post-radiation", description="Post-RT timeline description")
    primary_diagnosis: Optional[str] = Field("Glioblastoma (IDH-wildtype, WHO Grade 4)", description="Initial histopathology")
    adjuvant_protocol: Optional[str] = Field("Stupp Protocol (TMZ + RT)", description="Current active therapy")


class SubregionVolume(BaseModel):
    """Tumor sub-compartment volume measurements in cubic centimeters (cm³)."""
    necrotic_volume_cm3: float = Field(..., description="Necrotic Core / Non-enhancing tumor (NCR/NET)")
    edema_volume_cm3: float = Field(..., description="Peritumoral Edema (ED)")
    enhancing_volume_cm3: float = Field(..., description="GD-enhancing viable tumor (ET)")
    total_lesion_volume_cm3: float = Field(..., description="Whole tumor lesion burden (WT)")


class VolumetricDeltaMetrics(BaseModel):
    """Quantitative change metrics and velocity over time."""
    absolute_change_cm3: float = Field(..., description="Overall volumetric difference (V_B - V_A) in cm³")
    relative_change_pct: float = Field(..., description="Percentage delta in whole tumor volume")
    growth_velocity_cm3_per_month: float = Field(..., description="Volume expansion speed in cm³/month (30-day normalized)")
    enhancing_tumor_delta_cm3: float = Field(..., description="Absolute change in viable enhancing tumor compartment (cm³)")
    enhancing_tumor_delta_pct: float = Field(..., description="Percentage change in enhancing tumor compartment")
    edema_delta_cm3: float = Field(..., description="Absolute change in vasogenic edema (cm³)")
    edema_delta_pct: float = Field(..., description="Percentage change in vasogenic edema")
    edema_to_enhancing_ratio_delta: float = Field(..., description="Shift in Edema/Enhancing ratio; massive surge indicates PsP")


class SpatialExpansionVector(BaseModel):
    """Voxel-level spatial migration and invasive vector."""
    centroid_displacement_mm: float = Field(..., description="Euclidean shift in 3D center of mass (mm)")
    expansion_direction: str = Field(..., description="Principal anatomical vector (e.g. 'Anterior-Lateral')")
    infiltrative_spread_score: float = Field(..., ge=0.0, le=10.0, description="Boundary irregularity score (0-10)")


class TelemetryEvent(BaseModel):
    """Live telemetry progress item emitted during MONAI preprocessing & Triton serving."""
    step: str = Field(..., description="Subsystem or pipeline stage (e.g. 'MONAI.Spacingd')")
    status: str = Field("OK", description="'PENDING' | 'RUNNING' | 'OK' | 'WARNING'")
    message: str = Field(..., description="Human-readable technical log entry")
    latency_ms: float = Field(..., description="Execution time in milliseconds")
    timestamp: str = Field(..., description="ISO 8601 or clock timestamp")


class DiagnosticVerdictCard(BaseModel):
    """High-impact primary verdict card returned to front-end."""
    verdict: DiagnosticVerdict = Field(..., description="Binary outcome: TRUE_TUMOR_PROGRESSION or RADIATION_NECROSIS_PSEUDOPROGRESSION")
    verdict_display_title: str = Field(..., description="Formatted bold title for the UI banner")
    confidence_score: float = Field(..., ge=0.0, le=100.0, description="Confidence percentage index (e.g. 94.2%)")
    ppri_score: float = Field(..., ge=0.0, le=100.0, description="Pseudo-Progression Risk Index (0=Recurrence, 100=Necrosis)")
    banner_theme: str = Field("alert-red", description="'alert-red' for True Progression, 'safe-green' for Radiation Necrosis")
    alert_badge: str = Field(..., description="Visual badge status (e.g. 'CRITICAL: RESECTION REQUIRED')")
    rano_category: str = Field(..., description="RANO 2.0 criteria (e.g. 'Progressive Disease (PD)')")
    clinical_rationale: str = Field(..., description="Concise explanation of biomarker signatures driving this verdict")
    actionable_recommendation: str = Field(..., description="Immediate recommended clinical next steps")


class DualConsensusMetrics(BaseModel):
    """Mathematical Formula vs. Deep Learning Model Consensus Verification."""
    mathematical_ppri_pct: float = Field(..., description="P_Math (0-100%) from bio-volumetric equation")
    deep_learning_ppri_pct: float = Field(..., description="P_DL (0-100%) from neural network classifier")
    concordance_score_pct: float = Field(..., description="Consensus agreement index (0-100%)")
    zero_error_certified: bool = Field(True, description="True if agreement exceeds 85%")
    confidence_tier: str = Field("CERTIFIED_HIGH_CONFIDENCE", description="Certification level")
    biomarker_drivers: Dict[str, str] = Field(default_factory=dict, description="Leading mathematical drivers")


class LayAudienceSummary(BaseModel):
    """Plain-language compassionate explanation for patients and families."""
    headline: str
    plain_meaning: str
    why_this_happens: str
    what_doctors_recommend: str


class MeshGeometry(BaseModel):
    """3D polygon mesh surface buffer."""
    vertices: List[float] = Field(default_factory=list, description="Flattened [x, y, z, ...]")
    faces: List[int] = Field(default_factory=list, description="Flattened index [i, j, k, ...]")
    normals: List[float] = Field(default_factory=list, description="Flattened per-vertex normals [nx, ny, nz, ...]")
    vertex_count: int = 0
    face_count: int = 0


class PatientMeshCollection(BaseModel):
    """Realistic 3D anatomical patient brain and tumor sub-regions."""
    brain: MeshGeometry
    head: MeshGeometry = Field(default_factory=MeshGeometry, description="Approximate outer head and upper-neck surface")
    enhancing_tumor: MeshGeometry
    edema: MeshGeometry
    necrotic_core: MeshGeometry
    baseline_ghost: MeshGeometry
    glb_urls: Optional[Dict[str, str]] = Field(default_factory=dict, description="URLs to binary .glb files per structure")


class AnatomicalRegionOverlap(BaseModel):
    region_name: str
    lobe: str
    overlap_pct: float
    volume_cm3: float
    is_eloquent: bool = False


class EloquentProximitySchema(BaseModel):
    structure_name: str
    distance_mm: float
    risk_level: str
    clinical_caution: str


class AnatomicalAtlasReportSchema(BaseModel):
    patient_id: str
    centroid_mni_mm: List[float]
    hemisphere: str
    primary_location: str
    region_overlaps: List[AnatomicalRegionOverlap] = Field(default_factory=list)
    eloquent_proximity: List[EloquentProximitySchema] = Field(default_factory=list)
    overall_eloquence_risk: str = "MODERATE"
    surgical_corridor_recommendation: str = ""
    registration_method: str = "SimpleITK Affine MNI152 (Harvard-Oxford & AAL)"


class ProgressionAnalysisResponse(BaseModel):
    """Complete structured JSON response returned by the backend engine."""
    success: bool = True
    patient_metadata: PatientMetadata
    verdict_card: DiagnosticVerdictCard
    consensus_metrics: DualConsensusMetrics
    lay_audience_summary: LayAudienceSummary
    volumetric_metrics: VolumetricDeltaMetrics
    baseline_volumes: SubregionVolume
    followup_volumes: SubregionVolume
    expansion_vector: SpatialExpansionVector
    telemetry_logs: List[TelemetryEvent]
    serving_backend: str = Field("NVIDIA Triton Inference Server (Docker)", description="Active serving backend")
    total_pipeline_latency_ms: float = Field(..., description="End-to-end processing duration")
    slice_previews: Optional[Dict[str, Any]] = Field(None, description="Base64 or array preview slices for 2D comparison")
    patient_meshes: Optional[PatientMeshCollection] = Field(None, description="Real 3D anatomical patient meshes")
    surgery_recommendation: Optional[Dict[str, Any]] = Field(None, description="Structured surgery/treatment recommendation with urgency tier")
    anatomical_atlas: Optional[AnatomicalAtlasReportSchema] = Field(None, description="MNI152 Harvard-Oxford anatomical localization report")
    glb_urls: Optional[Dict[str, str]] = Field(None, description="Direct streaming/download URLs for binary .glb 3D models")



class AnalyzePresetRequest(BaseModel):
    """Request model when selecting benchmark clinical cases."""
    preset_case_id: str = Field("case_001_true_progression", description="'case_001_true_progression' or 'case_002_pseudo_progression'")
    patient_id: Optional[str] = "PT-84920"
    scan_interval_days: Optional[int] = 90
    radiation_completion_interval: Optional[str] = "90 Days post-radiation"
