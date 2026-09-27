"""
Unit tests for the Brain Tumor Progression & Pseudo-Progression Analyzer.
"""

import unittest
import numpy as np
from core.progression_analyzer import ProgressionAnalyzer, VolumetricProfile
from core.monai_pipeline import BrainMRIPipeline


class TestProgressionAnalyzer(unittest.TestCase):

    def setUp(self):
        self.analyzer = ProgressionAnalyzer(voxel_spacing=(1.0, 1.0, 1.0))
        self.pipeline = BrainMRIPipeline()

    def test_volume_calculation(self):
        """Verify volumetric measurements convert accurately from voxel counts."""
        mask = np.zeros((30, 30, 30), dtype=np.uint8)
        # 1000 voxels of enhancing tumor (Label 3)
        mask[10:20, 10:20, 10:20] = 3

        profile = self.analyzer.compute_volume_profile(mask)
        # 1000 voxels at 1mm^3 spacing = 1.0 cm^3
        self.assertAlmostEqual(profile.enhancing_volume_cm3, 1.0, places=2)
        self.assertAlmostEqual(profile.total_lesion_volume_cm3, 1.0, places=2)
        self.assertEqual(profile.edema_volume_cm3, 0.0)

    def test_true_progression_discrimination(self):
        """Verify expanding enhancing mass triggers True Progression verdict and low PPRI."""
        mask_a = np.zeros((50, 50, 50), dtype=np.uint8)
        mask_a[20:25, 20:25, 20:25] = 3 # Small enhancing baseline (125 voxels = 0.125 cm3)
        mask_a[18:27, 18:27, 18:27] = np.where(mask_a[18:27, 18:27, 18:27] == 0, 2, 3)

        mask_b = np.zeros((50, 50, 50), dtype=np.uint8)
        # Marked focal nodular enhancing expansion + shifted centroid
        mask_b[25:35, 25:35, 25:35] = 3 # 1000 voxels = 1.0 cm3 (+700% ET)

        report = self.analyzer.analyze_progression(mask_a, mask_b, days_between_scans=90)
        self.assertLess(report.ppri_score, 35.0)
        self.assertEqual(report.clinical_verdict, "True Tumor Progression (Glioblastoma Recurrence)")
        self.assertEqual(report.rano_classification, "Progressive Disease (PD)")

    def test_pseudo_progression_discrimination(self):
        """Verify massive edema flare with stable nodular ET triggers Pseudo-Progression verdict and high PPRI."""
        mask_a = np.zeros((60, 60, 60), dtype=np.uint8)
        mask_a[28:32, 28:32, 28:32] = 3 # Stable ET
        mask_a[25:35, 25:35, 25:35] = np.where(mask_a[25:35, 25:35, 25:35] == 0, 2, 3) # Edema

        mask_b = np.zeros((60, 60, 60), dtype=np.uint8)
        mask_b[28:32, 28:32, 28:32] = 3 # Stable ET (identical)
        # Massive edema expansion (flare)
        mask_b[15:45, 15:45, 15:45] = np.where(mask_b[15:45, 15:45, 15:45] == 0, 2, 3)

        report = self.analyzer.analyze_progression(mask_a, mask_b, days_between_scans=60)
        self.assertGreaterEqual(report.ppri_score, 65.0)
        self.assertEqual(report.clinical_verdict, "Pseudo-Progression (Radiation Necrosis)")
        self.assertEqual(report.rano_classification, "Suspected Pseudo-Progression (RANO PsP)")

    def test_pipeline_crop_pad(self):
        """Verify crop or pad operations maintain expected target tensor shape."""
        tensor = np.zeros((4, 80, 80, 80), dtype=np.float32)
        target_shape = (96, 96, 96)
        padded = self.pipeline.crop_or_pad_spatial(tensor, target_shape)
        self.assertEqual(padded.shape, (4, 96, 96, 96))


if __name__ == "__main__":
    unittest.main()
