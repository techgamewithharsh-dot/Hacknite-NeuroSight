"""
Tests for MNI152 Atlas Registration and 3D GLB Mesh Generation endpoints.
"""

import os
import unittest
import numpy as np
from fastapi.testclient import TestClient
from api.server import app
from core.atlas_registration import AtlasRegistrationEngine, AnatomicalAtlasReport
from core.mesh_generator import (
    mesh_dict_to_trimesh,
    export_structure_glb,
    export_combined_scene_glb,
    STRUCTURE_PALETTE,
)


class TestAtlasAndGLB(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)

    def test_mesh_to_glb_export(self):
        """Verify trimesh conversion and GLB export produces valid GLB binary data."""
        # Create a simple tetrahedron mesh dict
        verts = np.array([
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
            [0.0, 0.0, 10.0],
        ], dtype=np.float32)
        faces = np.array([
            [0, 1, 2],
            [0, 1, 3],
            [0, 2, 3],
            [1, 2, 3],
        ], dtype=np.int32)
        mesh_dict = {"verts": verts, "faces": faces}

        tri = mesh_dict_to_trimesh(mesh_dict, "enhancing_tumor", STRUCTURE_PALETTE["enhancing_tumor"])
        self.assertIsNotNone(tri)
        self.assertEqual(len(tri.vertices), 4)

        # Export single structure GLB
        glb_bytes = export_structure_glb(mesh_dict, "enhancing_tumor", STRUCTURE_PALETTE["enhancing_tumor"])
        self.assertIsNotNone(glb_bytes)
        self.assertGreater(len(glb_bytes), 100)
        # GLB files start with magic bytes b'glTF' (0x46546C67)
        self.assertTrue(glb_bytes.startswith(b"glTF"))

        # Export combined scene GLB
        scene_dict = {
            "enhancing_tumor": {"vertices": verts.flatten().tolist(), "faces": faces.flatten().tolist()},
            "edema": {"vertices": verts.flatten().tolist(), "faces": faces.flatten().tolist()},
        }
        scene_glb = export_combined_scene_glb(scene_dict)
        self.assertIsNotNone(scene_glb)
        self.assertGreater(len(scene_glb), 200)
        self.assertTrue(scene_glb.startswith(b"glTF"))

    def test_atlas_registration_engine_synthetic(self):
        """Verify AtlasRegistrationEngine computes coordinates and regional overlap."""
        engine = AtlasRegistrationEngine()
        # Synthetic segmentation volume with some tumor voxels (100x100x100)
        seg = np.zeros((100, 100, 100), dtype=np.uint8)
        # Add enhancing tumor cluster in right hemisphere
        seg[60:75, 65:80, 50:65] = 3  # Enhancing Tumor
        seg[55:80, 60:85, 45:70] = 2  # Edema surrounds it
        seg[60:75, 65:80, 50:65] = 3  # Re-apply tumor core

        report = engine.map_anatomical_context(seg, affine_matrix=np.eye(4))
        self.assertIsInstance(report, AnatomicalAtlasReport)
        self.assertIsNotNone(report.primary_location)
        self.assertIn(report.hemisphere, ["Left", "Right", "Bilateral"])
        self.assertEqual(len(report.centroid_mni_mm), 3)
        self.assertGreater(len(report.region_overlaps), 0)
        self.assertGreater(len(report.eloquent_proximity), 0)
        self.assertTrue(len(report.surgical_corridor_recommendation) > 0)

    def test_patient_atlas_endpoint(self):
        """Verify /api/v1/patients/{patient_id}/atlas returns anatomical context for PT-001."""
        resp = self.client.get("/api/v1/patients/PT-001/atlas")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["patient_id"], "PT-001")
        self.assertIn("primary_location", data)
        self.assertIn("hemisphere", data)
        self.assertIn("centroid_mni_mm", data)
        self.assertEqual(len(data["centroid_mni_mm"]), 3)
        self.assertIn("region_overlaps", data)
        self.assertIn("eloquent_proximity", data)
        self.assertIn("surgical_corridor_recommendation", data)

    def test_patient_glb_combined_endpoint(self):
        """Verify /api/v1/patients/{patient_id}/glb/combined returns a valid GLB binary."""
        resp = self.client.get("/api/v1/patients/PT-001/glb/combined")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.headers.get("content-type"), "model/gltf-binary")
        content = resp.content
        self.assertTrue(content.startswith(b"glTF"), "Expected GLB binary format starting with glTF magic header")

    def test_patient_glb_individual_structures(self):
        """Verify individual GLB endpoints for tumor, edema, brain surface."""
        for structure in ["enhancing_tumor", "edema", "brain_surface"]:
            resp = self.client.get(f"/api/v1/patients/PT-001/glb/{structure}")
            self.assertEqual(resp.status_code, 200, f"Failed for {structure}")
            self.assertEqual(resp.headers.get("content-type"), "model/gltf-binary")
            self.assertTrue(resp.content.startswith(b"glTF"))

    def test_patient_glb_nonexistent_patient(self):
        """Verify 404 for nonexistent patient."""
        resp = self.client.get("/api/v1/patients/PT-NONEXISTENT/glb/combined")
        self.assertEqual(resp.status_code, 404)

    def test_patient_detail_includes_glb_and_atlas(self):
        """Verify patient detail response includes glb_urls and anatomical_atlas fields."""
        resp = self.client.get("/api/v1/patients/PT-001")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("glb_urls", data)
        self.assertIn("combined", data["glb_urls"])
        self.assertIn("enhancing_tumor", data["glb_urls"])
        self.assertIn("anatomical_atlas", data)
        self.assertIsNotNone(data["anatomical_atlas"])
        self.assertIn("primary_location", data["anatomical_atlas"])


if __name__ == "__main__":
    unittest.main()
