"""
High-Fidelity 3D Multi-Modal Sequential Brain MRI Phantom Generator.

Generates realistic paired NIfTI (.nii.gz) scans with:
  - Cortical surface folding (gyri/sulci simulation via sinusoidal deformation)
  - Anatomically correct tissue contrasts (WM, GM, CSF, ventricles)
  - GBM sub-region pathology: Necrotic Core, Enhancing Tumor, Vasogenic Edema
  - 10 cases: 5 True Tumor Progression + 5 Pseudo-Progression (Radiation Necrosis)

Volume shape: 128x128x128 at 1mm³ isotropic spacing.
"""

import os
from pathlib import Path
import numpy as np
import nibabel as nib
from scipy.ndimage import gaussian_filter, binary_dilation, label as ndlabel


# ---------------------------------------------------------------------------
# Cortical Surface with Gyri/Sulci Simulation
# ---------------------------------------------------------------------------

def generate_brain_phantom(shape=(128, 128, 128), seed=42):
    """
    Creates an anatomically-inspired brain volume with:
    - Outer cortex surface deformed by sinusoidal waves (gyri/sulci)
    - White matter interior
    - Lateral ventricles (CSF cavities)
    - Realistic per-modality contrast (T1, T1ce, T2, FLAIR)

    Returns a dict with 'brain_mask', 't1', 't1ce', 't2', 'flair' arrays.
    """
    rng = np.random.default_rng(seed)
    d, h, w = shape

    # --- Coordinate grids (normalized -1 to +1) ---
    z_idx, y_idx, x_idx = np.mgrid[0:d, 0:h, 0:w]
    zn = (z_idx / (d - 1)) * 2 - 1   # [-1, 1]
    yn = (y_idx / (h - 1)) * 2 - 1
    xn = (x_idx / (w - 1)) * 2 - 1

    # --- Base ellipsoid (slightly asymmetric for realism) ---
    az, ay, ax = 0.84, 0.88, 0.78   # Semi-axis ratios
    r_base = (zn / az) ** 2 + (yn / ay) ** 2 + (xn / ax) ** 2  # 0=center, 1=surface

    # --- Gyri/Sulci: sinusoidal deformation on the surface ---
    # Frequencies chosen to match real GBM scan atlas (~3-6 major gyri per axis)
    freq_z  = rng.uniform(2.8, 4.2)
    freq_y  = rng.uniform(3.0, 4.5)
    freq_x  = rng.uniform(2.5, 3.8)
    phase_z = rng.uniform(0, np.pi)
    phase_y = rng.uniform(0, np.pi)
    phase_x = rng.uniform(0, np.pi)
    amp     = 0.06   # 6% radial deformation → visible folds

    gyral_deform = (
        amp * np.sin(freq_z * np.pi * zn + phase_z) *
              np.sin(freq_y * np.pi * yn + phase_y) *
              np.cos(freq_x * np.pi * xn + phase_x)
    )

    r_deformed = r_base + gyral_deform

    # --- Tissue masks ---
    brain_mask  = r_deformed <= 1.0
    cortex_mask = (r_deformed >= 0.82) & brain_mask          # GM cortical ribbon
    wm_mask     = (r_deformed < 0.82) & (r_deformed >= 0.30) & brain_mask  # WM
    deep_mask   = (r_deformed < 0.30) & brain_mask            # deep structures

    # Lateral ventricles (butterfly shape, symmetric)
    vent_mask = (
        (np.abs(xn) < 0.18) &
        (yn > -0.10) & (yn < 0.28) &
        (np.abs(zn) < 0.22) &
        brain_mask
    )
    # Small 3rd ventricle (midline)
    vent3_mask = (
        (np.abs(xn) < 0.06) &
        (yn > 0.05) & (yn < 0.20) &
        (np.abs(zn) < 0.12)
    )
    vent_mask = vent_mask | vent3_mask

    # Basal ganglia / thalami (slightly hypo on T2)
    bg_mask = deep_mask & ~vent_mask

    # --- Per-modality intensities (BraTS-like scale 0-1) ---
    t1    = np.zeros(shape, dtype=np.float32)
    t1ce  = np.zeros(shape, dtype=np.float32)
    t2    = np.zeros(shape, dtype=np.float32)
    flair = np.zeros(shape, dtype=np.float32)

    # WM: bright T1, intermediate T2
    t1[wm_mask]    = 0.82 + rng.normal(0, 0.03, wm_mask.sum()).astype(np.float32)
    t1ce[wm_mask]  = 0.80
    t2[wm_mask]    = 0.38
    flair[wm_mask] = 0.42

    # GM cortex: intermediate T1, slightly brighter T2
    t1[cortex_mask]    = 0.62
    t1ce[cortex_mask]  = 0.60
    t2[cortex_mask]    = 0.52
    flair[cortex_mask] = 0.50

    # Basal ganglia/thalami: intermediate
    t1[bg_mask]    = 0.65
    t1ce[bg_mask]  = 0.63
    t2[bg_mask]    = 0.44
    flair[bg_mask] = 0.40

    # Lateral ventricles (CSF): dark T1, bright T2, dark FLAIR
    t1[vent_mask]    = 0.12
    t1ce[vent_mask]  = 0.10
    t2[vent_mask]    = 0.92
    flair[vent_mask] = 0.08   # ← FLAIR suppresses free water

    # Skull / extra-cerebral (thin hyperintense rim for MRI skull effect)
    skull_mask = (r_deformed > 1.0) & (r_deformed < 1.08)
    t1[skull_mask]    = 0.30
    t1ce[skull_mask]  = 0.28
    t2[skull_mask]    = 0.20
    flair[skull_mask] = 0.22

    # --- Acquisition blur (PSF simulation: 0.9mm isotropic Gaussian) ---
    sigma = 0.9
    t1    = gaussian_filter(np.clip(t1,    0, 1), sigma=sigma)
    t1ce  = gaussian_filter(np.clip(t1ce,  0, 1), sigma=sigma)
    t2    = gaussian_filter(np.clip(t2,    0, 1), sigma=sigma)
    flair = gaussian_filter(np.clip(flair, 0, 1), sigma=sigma)

    # --- Rician noise (scanner noise model) ---
    noise_std = 0.018
    def rician_noise(arr):
        real  = arr + rng.normal(0, noise_std, arr.shape).astype(np.float32)
        imag  = rng.normal(0, noise_std, arr.shape).astype(np.float32)
        return np.sqrt(real**2 + imag**2).astype(np.float32)

    t1    = np.clip(rician_noise(t1),    0, 1)
    t1ce  = np.clip(rician_noise(t1ce),  0, 1)
    t2    = np.clip(rician_noise(t2),    0, 1)
    flair = np.clip(rician_noise(flair), 0, 1)

    return {
        "brain_mask": brain_mask,
        "vent_mask":  vent_mask,
        "wm_mask":    wm_mask,
        "t1":    t1,
        "t1ce":  t1ce,
        "t2":    t2,
        "flair": flair,
    }


# ---------------------------------------------------------------------------
# GBM Pathology Injection
# ---------------------------------------------------------------------------

def add_tumor_pathology(
    phantom: dict,
    center: tuple,
    r_core: float,
    r_enh: float,
    r_edema: float,
    nodular_enhancement: float = 1.0,
    rng: np.random.Generator = None,
):
    """
    Injects realistic GBM sub-regions into a phantom:
      Label 1 – Necrotic Core (NCR): T1 dark, T2 bright fluid
      Label 2 – Peritumoral Edema (ED): high T2 & FLAIR
      Label 3 – Enhancing Tumor (ET): ring/nodule enhancement on T1ce

    Returns updated modalities + integer segmentation mask.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    t1    = phantom["t1"].copy()
    t1ce  = phantom["t1ce"].copy()
    t2    = phantom["t2"].copy()
    flair = phantom["flair"].copy()
    shape = t1.shape

    d, h, w = shape
    z_idx, y_idx, x_idx = np.mgrid[0:d, 0:h, 0:w]
    cz, cy, cx = center

    # Ellipsoidal distance (slightly elongated anteriorly, realistic GBM morphology)
    az_t, ay_t, ax_t = 1.0, 1.15, 0.90
    dist_sq = (
        ((z_idx - cz) / az_t) ** 2 +
        ((y_idx - cy) / ay_t) ** 2 +
        ((x_idx - cx) / ax_t) ** 2
    )

    # Add irregular boundary using band-limited noise
    angle_noise = rng.normal(0, 0.8, shape).astype(np.float32)
    angle_noise = gaussian_filter(angle_noise, sigma=3.0)
    dist_noisy  = np.sqrt(np.maximum(dist_sq, 0)) + angle_noise * 0.5

    core_mask  = dist_noisy <= r_core
    enh_mask   = (dist_noisy > r_core) & (dist_noisy <= r_enh)
    edema_mask = (dist_noisy > r_enh)  & (dist_noisy <= r_edema)

    # Only inside brain
    brain_mask = phantom["brain_mask"]
    core_mask  &= brain_mask
    enh_mask   &= brain_mask
    edema_mask &= brain_mask

    # --- Apply modality-specific signal changes ---
    # Vasogenic edema: high T2, high FLAIR
    flair[edema_mask] = np.clip(flair[edema_mask] + 0.50, 0, 1.0)
    t2[edema_mask]    = np.clip(t2[edema_mask]    + 0.45, 0, 1.0)
    t1[edema_mask]    = np.clip(t1[edema_mask]    - 0.10, 0, 1.0)

    # Enhancing tumor ring: gadolinium uptake on T1ce
    enh_signal = 0.55 * nodular_enhancement
    t1ce[enh_mask] = np.clip(t1ce[enh_mask] + enh_signal, 0, 1.0)
    t1[enh_mask]   = np.clip(t1[enh_mask]   - 0.05, 0, 1.0)
    t2[enh_mask]   = np.clip(t2[enh_mask]   + 0.15, 0, 1.0)

    # Necrotic core: T1 dark (fluid), T1ce dark, T2 bright fluid
    t1[core_mask]    = 0.18
    t1ce[core_mask]  = 0.16
    t2[core_mask]    = 0.88
    flair[core_mask] = 0.22   # necrosis is slightly bright on FLAIR

    # Build segmentation mask (BraTS label convention)
    seg_mask = np.zeros(shape, dtype=np.uint8)
    seg_mask[edema_mask] = 2
    seg_mask[enh_mask]   = 3
    seg_mask[core_mask]  = 1

    # Smooth boundaries slightly
    t1    = gaussian_filter(t1,    sigma=0.6)
    t1ce  = gaussian_filter(t1ce,  sigma=0.6)
    t2    = gaussian_filter(t2,    sigma=0.6)
    flair = gaussian_filter(flair, sigma=0.6)

    return {
        "t1":     np.clip(t1,    0, 1),
        "t1ce":   np.clip(t1ce,  0, 1),
        "t2":     np.clip(t2,    0, 1),
        "flair":  np.clip(flair, 0, 1),
        "seg":    seg_mask,
    }


# ---------------------------------------------------------------------------
# NIfTI IO
# ---------------------------------------------------------------------------

def save_nifti_modalities(modal_dict: dict, output_dir: str, prefix: str = "scan",
                           voxel_size_mm: float = 1.0):
    """Saves modality arrays + segmentation mask as compressed NIfTI files."""
    os.makedirs(output_dir, exist_ok=True)
    affine = np.diag([voxel_size_mm, voxel_size_mm, voxel_size_mm, 1.0]).astype(np.float32)

    for seq in ["t1", "t1ce", "t2", "flair"]:
        data = modal_dict[seq]
        img  = nib.Nifti1Image(data, affine)
        path = os.path.join(output_dir, f"{prefix}_{seq}.nii.gz")
        nib.save(img, path)

    if "seg" in modal_dict:
        seg_img  = nib.Nifti1Image(modal_dict["seg"].astype(np.uint8), affine)
        seg_path = os.path.join(output_dir, f"{prefix}_seg.nii.gz")
        nib.save(seg_img, seg_path)
        print(f"  Saved {len(modal_dict)} volumes → {output_dir}/")


# ---------------------------------------------------------------------------
# Case Generator
# ---------------------------------------------------------------------------

def _make_case(base_path: Path, case_id: str,
               center_a: tuple, center_b: tuple,
               r_core_a: float, r_enh_a: float, r_edema_a: float, enh_a: float,
               r_core_b: float, r_enh_b: float, r_edema_b: float, enh_b: float,
               seed: int = 0):
    """Generates one paired (baseline, follow-up) case and saves NIfTI files."""
    case_dir = base_path / case_id
    rng = np.random.default_rng(seed)

    # --- Baseline Scan A ---
    phantom_a = generate_brain_phantom(shape=(128, 128, 128), seed=seed)
    scan_a    = add_tumor_pathology(phantom_a, center_a, r_core_a, r_enh_a, r_edema_a, enh_a, rng=rng)
    save_nifti_modalities(scan_a, str(case_dir / "scan_A_baseline"), prefix="baseline")

    # --- Follow-up Scan B ---
    rng_b     = np.random.default_rng(seed + 1000)
    phantom_b = generate_brain_phantom(shape=(128, 128, 128), seed=seed + 1)
    scan_b    = add_tumor_pathology(phantom_b, center_b, r_core_b, r_enh_b, r_edema_b, enh_b, rng=rng_b)
    save_nifti_modalities(scan_b, str(case_dir / "scan_B_followup"), prefix="followup")

    return case_dir


def generate_all_cases(base_dir: str = "data/samples"):
    """Builds 10 high-fidelity paired sequential cases (5 TP + 5 PP)."""
    print("=" * 70)
    print("Generating High-Fidelity 3D Multi-Modal Sequential Cases")
    print("=" * 70)
    base_path = Path(base_dir)

    # ----------------------------------------------------------------
    # TRUE PROGRESSION cases (expanding ET, centroid shift)
    # ----------------------------------------------------------------
    tp_cases = [
        # (case_id, center_a, center_b, core_a, enh_a, ed_a, nod_a, core_b, enh_b, ed_b, nod_b, seed)
        ("case_001_true_progression", (64, 72, 68), (60, 78, 74),
         4,  8, 14, 0.85,
         8, 18, 22, 1.40, 1),
        ("case_003_true_progression", (50, 60, 55), (44, 65, 62),
         5, 10, 16, 0.90,
         10, 20, 25, 1.50, 3),
        ("case_005_true_progression", (70, 50, 60), (65, 58, 68),
         3,  7, 12, 0.80,
         9, 17, 21, 1.35, 5),
        ("case_007_true_progression", (60, 80, 64), (54, 88, 70),
         6, 12, 18, 1.00,
         12, 22, 28, 1.60, 7),
        ("case_009_true_progression", (68, 45, 72), (62, 52, 80),
         4,  9, 15, 0.88,
         11, 19, 24, 1.45, 9),
    ]

    # ----------------------------------------------------------------
    # PSEUDO-PROGRESSION cases (edema flare, stable ET, stationary centroid)
    # ----------------------------------------------------------------
    pp_cases = [
        ("case_002_pseudo_progression", (64, 50, 55), (64, 50, 55),
         5, 10, 15, 0.75,
         5, 10, 30, 0.78, 2),
        ("case_004_pseudo_progression", (55, 65, 60), (55, 65, 60),
         4,  9, 13, 0.70,
         4,  9, 28, 0.72, 4),
        ("case_006_pseudo_progression", (72, 58, 50), (72, 58, 50),
         6, 11, 16, 0.80,
         6, 11, 32, 0.82, 6),
        ("case_008_pseudo_progression", (60, 42, 68), (60, 42, 68),
         3,  8, 12, 0.68,
         3,  8, 26, 0.70, 8),
        ("case_010_pseudo_progression", (50, 78, 62), (50, 78, 62),
         5, 10, 14, 0.73,
         5, 10, 29, 0.75, 10),
    ]

    all_cases = tp_cases + pp_cases
    for (case_id, ca, cb,
         rca, rea, rda, na,
         rcb, reb, rdb, nb, seed) in all_cases:
        print(f"\n→ Generating: {case_id}")
        _make_case(base_path, case_id, ca, cb,
                   rca, rea, rda, na,
                   rcb, reb, rdb, nb, seed)

    print(f"\n✅ Successfully generated {len(all_cases)} cases in: {base_dir}")
    print("   Each case: 4 modalities × 2 timepoints + segmentation mask = 10 NIfTI files")


if __name__ == "__main__":
    generate_all_cases()
