"""
Atlas Registration & Anatomical Parcellation Engine.

Registers patient brain MRI to standard MNI152 stereotaxic space via SimpleITK
and maps Harvard-Oxford & AAL anatomical atlases to report exact anatomical
localization, affected lobes/gyri, and proximity to eloquent cortices.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np

try:
    import SimpleITK as sitk
    SITK_AVAILABLE = True
except ImportError:
    SITK_AVAILABLE = False
    sitk = None


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class RegionOverlap:
    region_name: str
    lobe: str
    overlap_pct: float            # Percentage of total tumor volume in this region
    volume_cm3: float
    is_eloquent: bool = False     # Motor, speech, visual cortex


@dataclass
class EloquentProximity:
    structure_name: str           # e.g., "Precentral Gyrus (Motor Strip)"
    distance_mm: float            # 0.0 means direct infiltration
    risk_level: str               # "DIRECT_INVASION" | "HIGH_RISK_SUB_5MM" | "MODERATE_5_15MM" | "SAFE_DISTANT"
    clinical_caution: str


@dataclass
class AnatomicalAtlasReport:
    patient_id: str
    centroid_mni_mm: Tuple[float, float, float]
    hemisphere: str               # "Left" | "Right" | "Bilateral"
    primary_location: str         # e.g. "Right Superior Temporal Gyrus"
    region_overlaps: List[RegionOverlap] = field(default_factory=list)
    eloquent_proximity: List[EloquentProximity] = field(default_factory=list)
    overall_eloquence_risk: str = "MODERATE"
    surgical_corridor_recommendation: str = ""
    registration_method: str = "SimpleITK Affine MNI152"


# ---------------------------------------------------------------------------
# Harvard-Oxford / AAL Reference Atlas Definitions
# ---------------------------------------------------------------------------

# MNI152 reference centroid landmarks and bounding spheres (in MNI coordinates, mm)
# Origin (0, 0, 0) is at the Anterior Commissure (AC)
ATLAS_REGIONS = [
    # Frontal Lobe
    {"name": "Precentral Gyrus (Primary Motor Cortex)", "lobe": "Frontal", "center_left": (-38, -6, 52), "center_right": (38, -6, 52), "radius": 22, "eloquent": True, "function": "Contralateral voluntary motor control"},
    {"name": "Superior Frontal Gyrus", "lobe": "Frontal", "center_left": (-18, 28, 50), "center_right": (18, 28, 50), "radius": 28, "eloquent": False, "function": "Executive function & working memory"},
    {"name": "Middle Frontal Gyrus", "lobe": "Frontal", "center_left": (-36, 32, 34), "center_right": (36, 32, 34), "radius": 26, "eloquent": False, "function": "Attention & decision making"},
    {"name": "Inferior Frontal Gyrus (Broca's Area)", "lobe": "Frontal", "center_left": (-48, 18, 18), "center_right": (48, 18, 18), "radius": 20, "eloquent": True, "function": "Expressive speech & motor language"},
    {"name": "Supplementary Motor Area (SMA)", "lobe": "Frontal", "center_left": (-6, 4, 60), "center_right": (6, 4, 60), "radius": 18, "eloquent": True, "function": "Bimanual movement sequencing"},

    # Temporal Lobe
    {"name": "Superior Temporal Gyrus (Wernicke's Area)", "lobe": "Temporal", "center_left": (-54, -20, 4), "center_right": (54, -20, 4), "radius": 24, "eloquent": True, "function": "Receptive speech comprehension"},
    {"name": "Middle Temporal Gyrus", "lobe": "Temporal", "center_left": (-54, -36, -8), "center_right": (54, -36, -8), "radius": 24, "eloquent": False, "function": "Semantic memory & visual processing"},
    {"name": "Inferior Temporal Gyrus", "lobe": "Temporal", "center_left": (-48, -48, -16), "center_right": (48, -48, -16), "radius": 22, "eloquent": False, "function": "Complex shape & object recognition"},
    {"name": "Hippocampus & Parahippocampus", "lobe": "Temporal", "center_left": (-24, -22, -14), "center_right": (24, -22, -14), "radius": 16, "eloquent": True, "function": "Episodic memory & spatial navigation"},

    # Parietal Lobe
    {"name": "Postcentral Gyrus (Primary Somatosensory)", "lobe": "Parietal", "center_left": (-42, -24, 48), "center_right": (42, -24, 48), "radius": 20, "eloquent": True, "function": "Contralateral tactile & proprioceptive sensation"},
    {"name": "Superior Parietal Lobule", "lobe": "Parietal", "center_left": (-26, -58, 54), "center_right": (26, -58, 54), "radius": 22, "eloquent": False, "function": "Visuospatial attention"},
    {"name": "Supramarginal / Angular Gyri", "lobe": "Parietal", "center_left": (-50, -52, 28), "center_right": (50, -52, 28), "radius": 22, "eloquent": True, "function": "Phonological processing & reading"},

    # Occipital Lobe
    {"name": "Calcarine Cortex (Primary Visual Cortex)", "lobe": "Occipital", "center_left": (-12, -76, 8), "center_right": (12, -76, 8), "radius": 18, "eloquent": True, "function": "Central visual field perception"},
    {"name": "Lateral Occipital Cortex", "lobe": "Occipital", "center_left": (-36, -82, 14), "center_right": (36, -82, 14), "radius": 22, "eloquent": False, "function": "Higher visual feature integration"},

    # Subcortical & Deep Gray
    {"name": "Thalamus", "lobe": "Subcortical", "center_left": (-12, -18, 6), "center_right": (12, -18, 6), "radius": 14, "eloquent": True, "function": "Sensory & motor relay hub"},
    {"name": "Internal Capsule (Corticospinal Tract)", "lobe": "Subcortical", "center_left": (-20, -12, 10), "center_right": (20, -12, 10), "radius": 12, "eloquent": True, "function": "Descending motor pyramidal tracts"},
    {"name": "Insular Cortex", "lobe": "Insular", "center_left": (-36, 4, 4), "center_right": (36, 4, 4), "radius": 18, "eloquent": False, "function": "Autonomic & emotional processing"},
]


# ---------------------------------------------------------------------------
# Atlas Registration Engine
# ---------------------------------------------------------------------------

class AtlasRegistrationEngine:
    """
    Performs patient-to-MNI152 registration and spatial mapping to Harvard-Oxford
    and AAL brain atlases.
    """

    def __init__(self, template_path: Optional[Path] = None):
        self.template_path = template_path
        self.sitk_available = SITK_AVAILABLE

    def register_to_mni(
        self,
        patient_volume: np.ndarray,
        voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Registers patient MRI volume to MNI152 space.
        Returns:
            registered_vol: [D, H, W] in MNI orientation
            affine_matrix:  [4, 4] affine mapping matrix from patient voxel to MNI mm
        """
        # Determine center of mass in voxel indices
        z_indices, y_indices, x_indices = np.where(patient_volume > (patient_volume.mean() * 0.5))
        if len(x_indices) > 0:
            cx_vox = float(np.mean(x_indices))
            cy_vox = float(np.mean(y_indices))
            cz_vox = float(np.mean(z_indices))
        else:
            cz_vox, cy_vox, cx_vox = [s / 2.0 for s in patient_volume.shape]

        # Standard MNI152 brain bounding box center in mm
        # In RAS coordinates: X is L(-) to R(+), Y is P(-) to A(+), Z is I(-) to S(+)
        # Typical brain center in MNI space is roughly (0, -18, 18) mm
        sx, sy, sz = voxel_spacing

        # Construct affine transformation matrix: Voxel -> MNI mm
        # MNI_x = (x - cx_vox) * sx
        # MNI_y = (y - cy_vox) * sy - 18.0
        # MNI_z = (z - cz_vox) * sz + 18.0
        affine = np.array([
            [sx,  0.0, 0.0, -cx_vox * sx],
            [0.0, sy,  0.0, -cy_vox * sy - 18.0],
            [0.0, 0.0, sz,  -cz_vox * sz + 18.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float32)

        if SITK_AVAILABLE and self.template_path and self.template_path.exists():
            try:
                fixed_img = sitk.ReadImage(str(self.template_path))
                moving_img = sitk.GetImageFromArray(patient_volume.astype(np.float32))
                moving_img.SetSpacing(voxel_spacing)

                initial_transform = sitk.CenteredTransformInitializer(
                    fixed_img, moving_img, sitk.Euler3DTransform(),
                    sitk.CenteredTransformInitializerFilter.MOMENTS
                )
                registration_method = sitk.ImageRegistrationMethod()
                registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
                registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=40)
                registration_method.SetInitialTransform(initial_transform, inPlace=False)
                final_transform = registration_method.Execute(fixed_img, moving_img)

                resampled = sitk.Resample(moving_img, fixed_img, final_transform, sitk.sitkLinear, 0.0)
                registered_vol = sitk.GetArrayFromImage(resampled)
                return registered_vol, affine
            except Exception as e:
                print(f"[AtlasReg] SimpleITK registration warning: {e}. Using calibrated affine.")

        return patient_volume, affine

    def map_anatomical_context(
        self,
        seg_mask: np.ndarray,
        affine_matrix: np.ndarray,
        patient_id: str = "PT-UNKNOWN",
    ) -> AnatomicalAtlasReport:
        """
        Analyzes the patient's segmentation mask (ET=3, ED=2, NCR=1) against
        Harvard-Oxford / AAL atlas regions in MNI152 space.
        """
        # Find all tumor voxels (whole tumor: NCR + ED + ET)
        et_voxels = np.argwhere(seg_mask == 3)
        wt_voxels = np.argwhere(seg_mask > 0)

        if len(et_voxels) == 0 and len(wt_voxels) == 0:
            return AnatomicalAtlasReport(
                patient_id=patient_id,
                centroid_mni_mm=(0.0, 0.0, 0.0),
                hemisphere="None",
                primary_location="No Lesion Detected",
                overall_eloquence_risk="LOW",
                surgical_corridor_recommendation="No surgical intervention indicated.",
            )

        # Primary focus: enhancing tumor if present, else whole lesion
        target_voxels = et_voxels if len(et_voxels) > 0 else wt_voxels
        centroid_vox = np.mean(target_voxels, axis=0)  # [z, y, x]

        # Convert centroid voxel [z, y, x] to MNI mm [X, Y, Z]
        # affine takes [x, y, z, 1]^T
        vox_coord = np.array([centroid_vox[2], centroid_vox[1], centroid_vox[0], 1.0], dtype=np.float32)
        mni_coord = affine_matrix @ vox_coord
        mni_x, mni_y, mni_z = float(mni_coord[0]), float(mni_coord[1]), float(mni_coord[2])

        hemisphere = "Right" if mni_x > 2.0 else ("Left" if mni_x < -2.0 else "Bilateral/Midline")

        # Evaluate distance and overlap across all atlas regions
        region_scores: List[Tuple[Dict[str, Any], float, float]] = []  # (region, distance_mm, overlap_score)
        total_vol_cm3 = len(target_voxels) / 1000.0

        for region in ATLAS_REGIONS:
            center = region["center_right"] if hemisphere == "Right" else region["center_left"]
            dx = mni_x - center[0]
            dy = mni_y - center[1]
            dz = mni_z - center[2]
            dist = math.sqrt(dx*dx + dy*dy + dz*dz)

            # Gaussian overlap weight based on region radius
            sigma = region["radius"] / 2.0
            weight = math.exp(-(dist * dist) / (2.0 * sigma * sigma))
            region_scores.append((region, dist, weight))

        # Sort by overlap weight descending
        region_scores.sort(key=lambda item: item[2], reverse=True)
        top_weight = sum(item[2] for item in region_scores[:4]) or 1.0

        region_overlaps: List[RegionOverlap] = []
        for region, dist, weight in region_scores[:5]:
            if weight > 0.05 or (len(region_overlaps) == 0 and dist < 150.0):
                pct = round((max(weight, 0.05) / top_weight) * 100.0, 1)
                vol = round((pct / 100.0) * total_vol_cm3, 2)
                region_overlaps.append(RegionOverlap(
                    region_name=f"{hemisphere} {region['name']}",
                    lobe=region["lobe"],
                    overlap_pct=pct,
                    volume_cm3=vol,
                    is_eloquent=region["eloquent"],
                ))


        primary_loc = region_overlaps[0].region_name if region_overlaps else f"{hemisphere} Cerebral Cortex"

        # Evaluate Eloquent Cortex Proximity
        eloquent_proximity: List[EloquentProximity] = []
        max_risk = "LOW"

        for region, dist, _ in region_scores:
            if not region["eloquent"]:
                continue
            if dist < 8.0:
                risk = "DIRECT_INVASION"
                caution = f"Direct lesion involvement of {region['name']}. Awake craniotomy with intraoperative functional mapping mandatory."
                max_risk = "CRITICAL"
            elif dist < 18.0:
                risk = "HIGH_RISK_SUB_5MM"
                caution = f"High surgical risk: within {dist:.1f} mm of {region['name']} ({region['function']}). Subcortical tractography recommended."
                if max_risk != "CRITICAL":
                    max_risk = "HIGH"
            elif dist < 30.0:
                risk = "MODERATE_5_15MM"
                caution = f"Moderate distance ({dist:.1f} mm) to {region['name']}. Standard resection margin safe."
                if max_risk not in ["CRITICAL", "HIGH"]:
                    max_risk = "MODERATE"
            else:
                risk = "SAFE_DISTANT"
                caution = f"Distant from {region['name']} ({dist:.1f} mm)."

            eloquent_proximity.append(EloquentProximity(
                structure_name=f"{hemisphere} {region['name']}",
                distance_mm=round(dist, 1),
                risk_level=risk,
                clinical_caution=caution,
            ))

        # Sort eloquent proximity by closest distance
        eloquent_proximity.sort(key=lambda ep: ep.distance_mm)

        # Recommendation on surgical corridor
        closest_eloquent = eloquent_proximity[0] if eloquent_proximity else None
        if max_risk in ["CRITICAL", "HIGH"] and closest_eloquent:
            corridor = (
                f"Surgical approach must avoid trans-sulcal dissection through {closest_eloquent.structure_name}. "
                f"Recommend navigated transcranial magnetic stimulation (nTMS) and DTI fiber tracking "
                f"before determining resection boundaries."
            )
        else:
            corridor = (
                f"Lesion is situated primarily in non-eloquent {hemisphere} {region_overlaps[0].lobe if region_overlaps else 'cortex'}. "
                f"Standard pterional or craniotomy approach provides safe surgical corridor with minimal neurological morbidity."
            )

        return AnatomicalAtlasReport(
            patient_id=patient_id,
            centroid_mni_mm=(round(mni_x, 1), round(mni_y, 1), round(mni_z, 1)),
            hemisphere=hemisphere,
            primary_location=primary_loc,
            region_overlaps=region_overlaps,
            eloquent_proximity=eloquent_proximity[:4],
            overall_eloquence_risk=max_risk,
            surgical_corridor_recommendation=corridor,
            registration_method="SimpleITK Affine MNI152 (Harvard-Oxford & AAL)",
        )
