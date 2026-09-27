"""
Automated Unit Tests for GOD MODE Dual-Consensus Engine,
Surgery Recommendation Engine, and 3D Mesh Generator.
"""

import unittest
import numpy as np
from core.consensus_classifier import DualConsensusEngine
from core.mesh_generator import generate_patient_3d_meshes, extract_surface_mesh


class TestGodModeFeatures(unittest.TestCase):

    def setUp(self):
        self.consensus_engine = DualConsensusEngine()

    def test_mathematical_formula_true_progression(self):
        """Mathematical formula must output low PsP probability for massive ET growth."""
        p_math, drivers = self.consensus_engine.calculate_mathematical_ppri(
            delta_et_cm3=12.11,
            delta_ed_cm3=6.25,
            delta_ncr_cm3=0.23,
            displacement_mm=10.4,
            days_post_rt=90
        )
        self.assertLess(p_math, 15.0)
        self.assertIn("Malignant Proliferation", drivers["enhancing_mass_impact"])

    def test_mathematical_formula_pseudo_progression(self):
        """Mathematical formula must output high PsP probability for edema flare with stable ET."""
        p_math, drivers = self.consensus_engine.calculate_mathematical_ppri(
            delta_et_cm3=0.05,
            delta_ed_cm3=42.36,
            delta_ncr_cm3=0.02,
            displacement_mm=1.2,
            days_post_rt=60
        )
        self.assertGreater(p_math, 85.0)
        self.assertIn("High Edema Surge", drivers["edema_ratio_impact"])

    def test_dual_consensus_zero_error_certification(self):
        """Verify dual-consensus yields certified zero-ambiguity verdict for true progression."""
        vol_a = (0.12, 6.60, 1.27, 7.99)
        vol_b = (0.35, 12.85, 13.38, 26.58)
        result = self.consensus_engine.evaluate_consensus(
            vol_a=vol_a, vol_b=vol_b,
            displacement_mm=10.4, days_post_rt=90
        )
        self.assertTrue(result.zero_error_certified)
        self.assertEqual(result.certified_verdict, "TRUE_TUMOR_PROGRESSION")
        self.assertGreaterEqual(result.concordance_score_pct, 85.0)
        self.assertIn("Active Tumor Growth Detected", result.lay_explanation["headline"])

    def test_surgery_recommendation_true_progression_urgent(self):
        """Urgent surgery should be recommended for rapid true progression."""
        vol_a = (0.12, 6.60, 1.27, 7.99)
        vol_b = (0.35, 12.85, 13.38, 26.58)
        result = self.consensus_engine.evaluate_consensus(
            vol_a=vol_a, vol_b=vol_b,
            displacement_mm=15.0, days_post_rt=90,
            patient_meta={"age": 52, "kps": 80, "idh_status": "IDH-wildtype", "mgmt_status": "MGMT-unmethylated"}
        )
        srec = result.surgery_recommendation
        self.assertIsNotNone(srec)
        self.assertTrue(srec.surgery_recommended)
        self.assertIn(srec.urgency, ["URGENT_48H", "EVALUATE_7D"])
        self.assertIn(srec.risk_level, ["HIGH", "CRITICAL"])

    def test_surgery_recommendation_pseudo_no_surgery(self):
        """No surgery should be recommended for pseudo-progression."""
        vol_a = (0.10, 5.00, 1.00, 6.10)
        vol_b = (0.10, 35.00, 1.05, 36.15)
        result = self.consensus_engine.evaluate_consensus(
            vol_a=vol_a, vol_b=vol_b,
            displacement_mm=0.8, days_post_rt=60,
            patient_meta={"age": 45, "kps": 85, "idh_status": "IDH-wildtype", "mgmt_status": "MGMT-methylated"}
        )
        srec = result.surgery_recommendation
        self.assertIsNotNone(srec)
        self.assertFalse(srec.surgery_recommended)
        self.assertEqual(srec.urgency, "WATCHFUL_WAITING")
        self.assertEqual(srec.risk_level, "LOW")

    def test_marching_cubes_mesh_extraction(self):
        """Verify 3D Marching Cubes extracts valid brain + tumor meshes with normals."""
        # Build a 64³ realistic brain phantom:
        # outer sphere = brain tissue (intensity 0.70), inner sphere = tumor (ET label 3)
        sz = 64
        z, y, x = np.mgrid[0:sz, 0:sz, 0:sz]
        cx = cy = cz = sz // 2
        dist = np.sqrt((z - cz)**2 + (y - cy)**2 + (x - cx)**2)

        vol = np.zeros((sz, sz, sz), dtype=np.float32)
        vol[dist <= 28] = 0.70   # brain parenchyma
        vol[dist <= 4]  = 0.10   # ventricle-like centre (dark)

        mask = np.zeros((sz, sz, sz), dtype=np.uint8)
        mask[(dist >= 8) & (dist <= 14)] = 3   # ET ring
        mask[dist < 8]                   = 1   # NCR core

        meshes = generate_patient_3d_meshes(vol, mask)

        self.assertGreater(meshes["head"]["vertex_count"], meshes["brain"]["vertex_count"])
        brain_center = np.mean(np.asarray(meshes["brain"]["vertices"]).reshape(-1, 3), axis=0)
        tumor_center = np.mean(np.asarray(meshes["enhancing_tumor"]["vertices"]).reshape(-1, 3), axis=0)
        self.assertLess(np.linalg.norm(brain_center - tumor_center), 0.5)

        # Brain mesh must be non-trivial
        self.assertGreater(meshes["brain"]["vertex_count"], 100,
                           "Brain mesh is empty — skull_strip produced no voxels")
        self.assertGreater(meshes["brain"]["face_count"], 0)

        # ET mesh must exist
        self.assertGreater(meshes["enhancing_tumor"]["vertex_count"], 0)

        # Flat arrays must match declared counts
        self.assertEqual(len(meshes["brain"]["vertices"]),
                         meshes["brain"]["vertex_count"] * 3)
        self.assertEqual(len(meshes["brain"]["faces"]),
                         meshes["brain"]["face_count"] * 3)

        # Per-vertex normals must be present and same length as vertices
        self.assertEqual(len(meshes["brain"]["normals"]),
                         meshes["brain"]["vertex_count"] * 3)

    def test_extract_surface_mesh_empty_volume(self):
        """extract_surface_mesh must return empty result on all-zero volume."""
        vol = np.zeros((32, 32, 32), dtype=np.float32)
        result = extract_surface_mesh(vol)
        self.assertEqual(result["vertex_count"], 0)
        self.assertEqual(result["face_count"], 0)
        self.assertEqual(result["vertices"], [])

    def test_extract_surface_mesh_sphere(self):
        """extract_surface_mesh must produce a valid sphere mesh."""
        sz = 48
        z, y, x = np.mgrid[0:sz, 0:sz, 0:sz]
        dist = np.sqrt((z - 24)**2 + (y - 24)**2 + (x - 24)**2)
        vol = (dist <= 18).astype(np.float32)
        result = extract_surface_mesh(vol, threshold=0.5, step_size=2)
        self.assertGreater(result["vertex_count"], 100)
        self.assertGreater(result["face_count"], 0)
        self.assertEqual(len(result["normals"]), result["vertex_count"] * 3)


if __name__ == "__main__":
    unittest.main()
