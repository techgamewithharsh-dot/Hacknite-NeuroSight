"""
Automated Verification Tests for API Contracts & FastAPI Endpoints.
"""

import unittest
from fastapi.testclient import TestClient
from api.server import app
from api.schemas import (
    DiagnosticVerdict,
    ProgressionAnalysisResponse,
    AnalyzePresetRequest,
)


class TestAPIContracts(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)

    # ── Health ──────────────────────────────────────────────────────────
    def test_health_endpoint(self):
        """Verify health check returns service online status."""
        resp = self.client.get("/api/v1/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "online")
        # New schema uses segmentation_backend instead of serving_backend
        self.assertIn("segmentation_backend", data)
        self.assertIn("api_version", data)

    # ── Patient database ─────────────────────────────────────────────────
    def test_patients_list_endpoint(self):
        """Verify patient database returns at least 1 patient."""
        resp = self.client.get("/api/v1/patients")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("patients", data)
        self.assertGreater(len(data["patients"]), 0)
        # Check required fields exist in first patient
        p = data["patients"][0]
        for field in ["patient_id", "name", "age", "tumor_type", "ground_truth_verdict"]:
            self.assertIn(field, p, f"Missing field '{field}' in patient listing")

    def test_cases_endpoint(self):
        """Verify legacy /cases endpoint lists at least 2 cases."""
        resp = self.client.get("/api/v1/cases")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("cases", data)
        self.assertGreaterEqual(len(data["cases"]), 2)

    # ── Preset analysis (reads from generated NIfTI files) ───────────────
    def test_preset_case_001_true_progression_contract(self):
        """Verify Case 001 returns TRUE_TUMOR_PROGRESSION adhering to schema."""
        payload = {
            "preset_case_id": "case_001_true_progression",
            "patient_id":     "PT-84920",
            "scan_interval_days": 90,
            "radiation_completion_interval": "90 Days post-radiation",
        }
        resp = self.client.post("/api/v1/analyze/preset", json=payload)
        self.assertEqual(resp.status_code, 200, msg=resp.text)
        data = resp.json()

        validated = ProgressionAnalysisResponse(**data)
        self.assertEqual(validated.verdict_card.verdict,
                         DiagnosticVerdict.TRUE_TUMOR_PROGRESSION)
        self.assertLess(validated.verdict_card.ppri_score, 35.0)
        self.assertGreater(validated.verdict_card.confidence_score, 80.0)
        self.assertEqual(validated.verdict_card.banner_theme, "alert-red")
        self.assertGreater(validated.volumetric_metrics.enhancing_tumor_delta_cm3, 0.5)
        self.assertIsNotNone(validated.slice_previews)

        # Surgery recommendation must be present and recommend surgery
        self.assertIsNotNone(validated.surgery_recommendation)
        self.assertIn("surgery_recommended", validated.surgery_recommendation)
        self.assertIn("urgency", validated.surgery_recommendation)

    def test_preset_case_002_pseudo_progression_contract(self):
        """Verify Case 002 returns RADIATION_NECROSIS_PSEUDOPROGRESSION adhering to schema."""
        payload = {
            "preset_case_id": "case_002_pseudo_progression",
            "patient_id":     "PT-31045",
            "scan_interval_days": 60,
            "radiation_completion_interval": "60 Days post-radiation",
        }
        resp = self.client.post("/api/v1/analyze/preset", json=payload)
        self.assertEqual(resp.status_code, 200, msg=resp.text)
        data = resp.json()

        validated = ProgressionAnalysisResponse(**data)
        self.assertEqual(validated.verdict_card.verdict,
                         DiagnosticVerdict.RADIATION_NECROSIS_PSEUDOPROGRESSION)
        self.assertGreater(validated.verdict_card.ppri_score, 65.0)
        self.assertEqual(validated.verdict_card.banner_theme, "safe-green")
        self.assertGreater(validated.volumetric_metrics.edema_delta_cm3, 5.0)

        # Surgery recommendation must say NO surgery for pseudo-progression
        self.assertIsNotNone(validated.surgery_recommendation)
        self.assertFalse(validated.surgery_recommendation["surgery_recommended"])

    # ── Patient-by-ID endpoint ───────────────────────────────────────────
    def test_patient_by_id_endpoint(self):
        """Verify /patients/PT-001 runs full pipeline and returns valid response."""
        resp = self.client.get("/api/v1/patients/PT-001")
        self.assertEqual(resp.status_code, 200, msg=resp.text)
        data = resp.json()
        validated = ProgressionAnalysisResponse(**data)
        self.assertTrue(validated.success)
        self.assertIsNotNone(validated.patient_meshes)
        # Brain mesh must have real geometry
        self.assertGreater(validated.patient_meshes.brain.vertex_count, 0)
        self.assertGreater(validated.patient_meshes.head.vertex_count, 0)

    def test_patient_not_found(self):
        """Verify 404 for unknown patient IDs."""
        resp = self.client.get("/api/v1/patients/PT-NONEXISTENT")
        self.assertEqual(resp.status_code, 404)

    def test_patient_mesh_endpoint(self):
        """Verify the lightweight /mesh endpoint returns mesh data."""
        resp = self.client.get("/api/v1/patients/PT-001/mesh")
        self.assertEqual(resp.status_code, 200, msg=resp.text)
        data = resp.json()
        self.assertIn("meshes", data)
        self.assertIn("brain", data["meshes"])
        self.assertIn("vertex_count", data["meshes"]["brain"])

    # ── Auth & Saved Device Logging ──────────────────────────────────────
    def test_healthz_endpoint(self):
        """Verify /api/healthz returns online status."""
        resp = self.client.get("/api/healthz")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["status"], "online")

    def test_login_and_saved_device_logging(self):
        """Verify clinician login saves device and writes to audit log."""
        payload = {
            "email": "clinician@hospital.com",
            "password": "clinician123",
            "deviceId": "dev_test_suite_99",
            "rememberDevice": True
        }
        resp = self.client.post("/api/login", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["message"], "Login successful")
        self.assertTrue(data["deviceSaved"])
        self.assertEqual(data["savedDeviceId"], "dev_test_suite_99")

    def test_auth_saved_device_verification(self):
        """Verify saved device can bypass login via /api/auth/saved-device."""
        resp = self.client.get("/api/auth/saved-device?device_id=dev_test_suite_99")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["saved"])
        self.assertEqual(data["deviceId"], "dev_test_suite_99")
        self.assertIn("user", data)

    def test_get_saved_device_logs(self):
        """Verify saved device audit log can be retrieved."""
        resp = self.client.get("/api/auth/logs")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("log_file", data)
        self.assertGreater(data["total_lines"], 0)
        self.assertIn("NEUROSIGHT", data["content"])


if __name__ == "__main__":
    unittest.main()
