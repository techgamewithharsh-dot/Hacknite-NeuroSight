"""
Real Volumetric Brain & Tumor Mesh Extractor.

Runs Marching Cubes on actual MRI NIfTI volumes to produce:
  - High-resolution cortical brain surface (gyri/sulci visible)
  - Enhancing Tumor (ET) mesh — label 3
  - Peritumoral Edema (ED) mesh — label 2
  - Necrotic Core (NCR) mesh — label 1
  - Baseline ghost tumor (for temporal comparison)

All meshes include per-vertex normals for realistic Phong/PBR lighting
in WebGL Three.js. Coordinates are centered at origin and scaled to
a ±1 unit bounding box for consistent Three.js scene framing.
"""

from __future__ import annotations

from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from scipy.ndimage import gaussian_filter, binary_closing, binary_fill_holes, binary_dilation


# ── Marching Cubes (skimage) ───────────────────────────────────────────────
from skimage.measure import marching_cubes


# ---------------------------------------------------------------------------
# Per-vertex normal computation
# ---------------------------------------------------------------------------

def compute_vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """
    Computes smooth per-vertex normals by averaging adjacent face normals.

    Args:
        vertices: [N, 3] float32
        faces:    [M, 3] int32

    Returns:
        normals: [N, 3] float32  (unit length)
    """
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    edge1 = v1 - v0
    edge2 = v2 - v0
    face_normals = np.cross(edge1, edge2)

    # Normalize face normals
    norms = np.linalg.norm(face_normals, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    face_normals /= norms

    # Accumulate to vertices
    vertex_normals = np.zeros_like(vertices)
    for i in range(3):
        np.add.at(vertex_normals, faces[:, i], face_normals)

    # Normalize vertex normals
    norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    vertex_normals /= norms

    return vertex_normals.astype(np.float32)


# ---------------------------------------------------------------------------
# Surface extraction
# ---------------------------------------------------------------------------

def extract_surface_mesh(
    volume_3d: np.ndarray,
    threshold: float = 0.5,
    step_size: int = 2,
    smoothing_sigma: float = 0.8,
    closing_iters: int = 2,
    max_faces: int = 120_000,
    reference_shape: Optional[Tuple[int, int, int]] = None,
) -> Dict[str, Any]:
    """
    Extracts an isosurface from a 3D binary/float volume using Marching Cubes.
    Applies Gaussian smoothing and binary morphological closing to eliminate
    holes and reduce voxel stair-stepping before meshing.

    Returns a dict compatible with the MeshGeometry Pydantic schema.
    """
    if volume_3d is None or not np.any(volume_3d > threshold):
        return {"vertices": [], "faces": [], "normals": [],
                "vertex_count": 0, "face_count": 0}

    vol = volume_3d.astype(np.float32)

    # ── Morphological closing: fill small holes before marching cubes ──
    if closing_iters > 0:
        binary_vol = vol > threshold
        binary_vol = binary_closing(binary_vol, iterations=closing_iters)
        binary_vol = binary_fill_holes(binary_vol)
        vol = binary_vol.astype(np.float32)
        threshold  = 0.5

    # ── Gaussian smoothing: reduces stair-step artifacts ──
    if smoothing_sigma > 0:
        vol = gaussian_filter(vol, sigma=smoothing_sigma)

    try:
        verts, faces, _, _ = marching_cubes(vol, level=threshold, step_size=step_size)
    except Exception as e:
        print(f"[MeshGen] Marching Cubes failed: {e}")
        return {"vertices": [], "faces": [], "normals": [],
                "vertex_count": 0, "face_count": 0}

    if len(faces) == 0:
        return {"vertices": [], "faces": [], "normals": [],
                "vertex_count": 0, "face_count": 0}

    # ── Decimate if too large (keep rendering snappy) ──
    if len(faces) > max_faces:
        # Uniform random face subsampling (preserves topology reasonably)
        idx = np.random.choice(len(faces), max_faces, replace=False)
        faces = faces[idx]
        # Reindex to compact vertex set
        unique_verts, inv = np.unique(faces, return_inverse=True)
        faces   = inv.reshape(-1, 3).astype(np.int32)
        verts   = verts[unique_verts]

    # Preserve spatial registration across the brain, lesion and outer head.
    shape = np.asarray(reference_shape or volume_3d.shape, dtype=np.float32)
    center = (shape - 1.0) / 2.0
    scale = float(max(shape)) / 2.0 or 1.0
    verts = ((verts - center) / scale).astype(np.float32)

    # ── Per-vertex normals ──
    normals = compute_vertex_normals(verts, faces.astype(np.int32))

    return {
        "vertices":     [round(float(v), 4) for v in verts.flatten()],
        "faces":        [int(f) for f in faces.flatten()],
        "normals":      [round(float(n), 4) for n in normals.flatten()],
        "vertex_count": int(len(verts)),
        "face_count":   int(len(faces)),
    }


def make_head_neck_shell(brain_mask: np.ndarray) -> np.ndarray:
    """Create a smooth, asymmetric outer head and upper-neck envelope from brain bounds."""
    coords = np.argwhere(brain_mask)
    if not len(coords):
        return brain_mask
    lo, hi = coords.min(axis=0), coords.max(axis=0)
    shape = np.asarray(brain_mask.shape)
    center = (lo + hi) / 2.0
    radii = (hi - lo + 1) / 2.0
    # Expand beyond brain to approximate scalp/skull and extend inferiorly for neck.
    radii = radii * np.array([1.34, 1.28, 1.27])
    radii[0] *= 1.08  # natural posterior/anterior asymmetry
    zz, yy, xx = np.ogrid[:shape[0], :shape[1], :shape[2]]
    ellipsoid = (((zz-center[0])/radii[0])**2 +
                 ((yy-center[1])/radii[1])**2 +
                 ((xx-center[2])/radii[2])**2) <= 1.0
    # Taper a short neck from the inferior brain into the upper cervical region.
    neck_center_z = hi[0] + radii[0] * 0.46
    neck_radius_z = max(radii[0] * 0.57, 1.0)
    neck_radius_y = max(radii[1] * 0.43, 1.0)
    neck_radius_x = max(radii[2] * 0.44, 1.0)
    neck = (((zz-neck_center_z)/neck_radius_z)**2 +
            ((yy-center[1])/neck_radius_y)**2 +
            ((xx-center[2])/neck_radius_x)**2) <= 1.0
    return binary_dilation(ellipsoid | neck, iterations=1)


# ---------------------------------------------------------------------------
# Skull-strip proxy: remove skull from T1ce before brain surface extraction
# ---------------------------------------------------------------------------

def skull_strip(t1ce_volume: np.ndarray) -> np.ndarray:
    """
    Fast skull-stripping proxy for synthetic MRI data:
    1. Intensity threshold + Gaussian to get initial brain mask
    2. Keep largest connected component
    3. Binary fill holes
    4. Erode skull (outer ~3 voxels)

    Returns binary brain parenchyma mask [D, H, W] bool.
    """
    from scipy.ndimage import (
        label as ndlabel, binary_erosion, binary_fill_holes
    )

    vol = t1ce_volume.astype(np.float32)
    # Normalize to [0,1]
    vmin, vmax = vol.min(), vol.max()
    if vmax > vmin:
        vol = (vol - vmin) / (vmax - vmin)

    # Initial threshold: 5th percentile of nonzero voxels, capped at 0.95
    nonzero_vals = vol[vol > 0.02]
    thresh = np.percentile(nonzero_vals, 5) if len(nonzero_vals) > 100 else 0.10
    thresh = min(thresh, 0.95)   # never threshold out real tissue
    brain_bin = vol >= thresh

    # Keep largest connected component
    labeled, n_comp = ndlabel(brain_bin)
    if n_comp == 0:
        return brain_bin

    comp_sizes = np.bincount(labeled.flatten())
    comp_sizes[0] = 0  # ignore background
    largest = comp_sizes.argmax()
    brain_bin = (labeled == largest)

    # Fill internal holes (ventricles etc.)
    brain_bin = binary_fill_holes(brain_bin)

    # Erode 2 voxels to remove thin skull rim
    struct = np.ones((3, 3, 3), dtype=bool)
    brain_bin = binary_erosion(brain_bin, structure=struct, iterations=2)

    return brain_bin


# ---------------------------------------------------------------------------
# Main entry point used by api/server.py
# ---------------------------------------------------------------------------

def generate_patient_3d_meshes(
    t1ce_volume: np.ndarray,
    seg_mask: np.ndarray,
    baseline_seg: Optional[np.ndarray] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Generates all 5 anatomical meshes for one patient:
      • brain        – cortical surface from skull-stripped T1ce
      • enhancing_tumor  – ET (label 3)
      • edema            – ED (label 2)
      • necrotic_core    – NCR (label 1)
      • baseline_ghost   – baseline ET silhouette (label 3 from scan A)

    Args:
        t1ce_volume:  [D, H, W] float32  (contrast-enhanced T1, raw voxel values)
        seg_mask:     [D, H, W] uint8    (BraTS labels 0/1/2/3)
        baseline_seg: [D, H, W] uint8    (optional, for ghost overlay)

    Returns:
        Dict mapping region name → MeshGeometry-compatible dict.
    """
    print("[MeshGen] Starting patient mesh extraction…")

    # ── 1. Brain cortical surface ──────────────────────────────────────────
    brain_bin = skull_strip(t1ce_volume)
    print(f"  Brain voxels: {brain_bin.sum():,}")
    shape = tuple(t1ce_volume.shape)
    head_mesh = extract_surface_mesh(
        make_head_neck_shell(brain_bin).astype(np.float32), threshold=0.5,
        step_size=2, smoothing_sigma=2.2, closing_iters=2,
        max_faces=100_000, reference_shape=shape,
    )
    brain_mesh = extract_surface_mesh(
        brain_bin.astype(np.float32),
        threshold=0.5,
        step_size=2,          # step=2 → ~40k–80k vertices for 128³ volume
        smoothing_sigma=1.2,  # smooth cortex but preserve gyri shapes
        closing_iters=3,
        max_faces=100_000,
        reference_shape=shape,
    )
    print(f"  Brain mesh: {brain_mesh['vertex_count']:,} verts, "
          f"{brain_mesh['face_count']:,} faces")

    # ── 2. Enhancing Tumor ────────────────────────────────────────────────
    et_bin  = (seg_mask == 3).astype(np.float32)
    et_mesh = extract_surface_mesh(
        et_bin, threshold=0.5, step_size=1,
        smoothing_sigma=0.8, closing_iters=2, max_faces=30_000,
        reference_shape=shape,
    )
    print(f"  ET mesh: {et_mesh['vertex_count']:,} verts")

    # ── 3. Peritumoral Edema ──────────────────────────────────────────────
    ed_bin  = (seg_mask == 2).astype(np.float32)
    ed_mesh = extract_surface_mesh(
        ed_bin, threshold=0.5, step_size=2,
        smoothing_sigma=1.0, closing_iters=2, max_faces=40_000,
        reference_shape=shape,
    )
    print(f"  ED mesh: {ed_mesh['vertex_count']:,} verts")

    # ── 4. Necrotic Core ──────────────────────────────────────────────────
    ncr_bin  = (seg_mask == 1).astype(np.float32)
    ncr_mesh = extract_surface_mesh(
        ncr_bin, threshold=0.5, step_size=1,
        smoothing_sigma=0.6, closing_iters=2, max_faces=20_000,
        reference_shape=shape,
    )
    print(f"  NCR mesh: {ncr_mesh['vertex_count']:,} verts")

    # ── 5. Baseline Ghost ─────────────────────────────────────────────────
    ghost_mesh = {"vertices": [], "faces": [], "normals": [],
                  "vertex_count": 0, "face_count": 0}
    if baseline_seg is not None:
        ghost_bin = (baseline_seg == 3).astype(np.float32)
        if ghost_bin.any():
            ghost_mesh = extract_surface_mesh(
                ghost_bin, threshold=0.5, step_size=1,
                smoothing_sigma=0.8, closing_iters=2, max_faces=25_000,
                reference_shape=shape,
            )
    print(f"  Ghost mesh: {ghost_mesh['vertex_count']:,} verts")
    print("[MeshGen] Done.")

    return {
        "brain":            brain_mesh,
        "head":             head_mesh,
        "enhancing_tumor":  et_mesh,
        "edema":            ed_mesh,
        "necrotic_core":    ncr_mesh,
        "baseline_ghost":   ghost_mesh,
    }


# ---------------------------------------------------------------------------
# Trimesh & Binary GLB Exporters
# ---------------------------------------------------------------------------

try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    trimesh = None
    TRIMESH_AVAILABLE = False


STRUCTURE_PALETTE = {
    "brain":            {"name": "Brain_Surface",     "rgba": [140, 210, 235, 75]},    # Translucent pial membrane
    "enhancing_tumor":  {"name": "Enhancing_Tumor",   "rgba": [245, 45, 65, 240]},     # Solid vibrant red/pink
    "edema":            {"name": "Peritumoral_Edema", "rgba": [45, 215, 135, 100]},    # Luminous green
    "necrotic_core":    {"name": "Necrotic_Core",     "rgba": [150, 45, 185, 235]},    # Deep purple
    "baseline_ghost":   {"name": "Baseline_Ghost",    "rgba": [255, 180, 50, 65]},     # Amber silhouette
}


def mesh_dict_to_trimesh(
    mesh_dict: Dict[str, Any],
    structure_key: str = "brain",
    palette_info: Optional[Dict[str, Any]] = None,
) -> Optional["trimesh.Trimesh"]:
    """
    Converts MeshGeometry dict (flat vertices, faces, normals or numpy arrays) into a Trimesh object
    with PBR vertex colors, smoothing, and decimation.
    """
    if not TRIMESH_AVAILABLE or not mesh_dict:
        return None

    # Handle key aliases (e.g. brain_surface -> brain)
    key_norm = structure_key.lower().replace(" ", "_")
    if key_norm in ("brain_surface", "cortex", "brain_mesh"):
        key_norm = "brain"

    # Extract vertices
    if "vertices" in mesh_dict:
        raw_verts = mesh_dict["vertices"]
    elif "verts" in mesh_dict:
        raw_verts = mesh_dict["verts"]
    else:
        return None

    verts = np.asarray(raw_verts, dtype=np.float32).reshape(-1, 3)
    if len(verts) == 0:
        return None

    # Extract faces
    if "faces" in mesh_dict:
        raw_faces = mesh_dict["faces"]
    else:
        return None

    faces = np.asarray(raw_faces, dtype=np.int32).reshape(-1, 3)
    if len(faces) == 0:
        return None

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

    # Assign per-vertex RGBA color
    if palette_info and "rgba" in palette_info:
        info = palette_info
    else:
        info = STRUCTURE_PALETTE.get(key_norm, {"name": structure_key, "rgba": [200, 200, 200, 255]})

    rgba = np.array(info["rgba"], dtype=np.uint8)
    mesh.visual.vertex_colors = np.tile(rgba, (len(verts), 1))

    # Optional mild Laplacian smoothing
    if len(verts) > 500:
        try:
            trimesh.smoothing.filter_laplacian(mesh, iterations=2)
        except Exception:
            pass

    return mesh


def export_structure_glb(
    mesh_dict: Dict[str, Any],
    structure_key: str = "brain",
    palette_info: Optional[Dict[str, Any]] = None,
) -> Optional[bytes]:
    """
    Exports a single anatomical structure as a standalone binary .glb file.
    """
    mesh = mesh_dict_to_trimesh(mesh_dict, structure_key, palette_info=palette_info)
    if mesh is None:
        return None
    return mesh.export(file_type="glb")



def export_combined_scene_glb(meshes_dict: Dict[str, Dict[str, Any]]) -> Optional[bytes]:
    """
    Assembles all anatomical meshes into a single scene with named nodes
    (Brain_Surface, Enhancing_Tumor, Peritumoral_Edema, Necrotic_Core).
    This allows Three.js / React Three Fiber to load one efficient .glb file
    and toggle layers with zero network overhead.
    """
    if not TRIMESH_AVAILABLE:
        return None

    scene = trimesh.Scene()
    added_any = False

    for key, info in STRUCTURE_PALETTE.items():
        if key in meshes_dict:
            m_data = meshes_dict[key]
            v_cnt = m_data.get("vertex_count", len(m_data.get("vertices", m_data.get("verts", []))))
            if v_cnt > 0:
                mesh = mesh_dict_to_trimesh(m_data, key)
                if mesh is not None:
                    scene.add_geometry(mesh, node_name=info["name"])
                    added_any = True


    if not added_any:
        return None

    return scene.export(file_type="glb")
