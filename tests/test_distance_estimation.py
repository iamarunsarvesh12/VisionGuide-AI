import sys
import os
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.object_tracking.interface import Track
from modules.hazard_memory.models import HazardMemoryRecord
from modules.distance_estimation.estimator import MonocularDistanceEstimator
from modules.distance_estimation.models import DistanceResult


class TestDistanceEstimation(unittest.TestCase):
    """
    Unit test suite for Module 06 — Monocular Distance Estimation.
    """

    def setUp(self):
        """Initialize estimator instance before each test."""
        self.estimator = MonocularDistanceEstimator(
            focal_length_px=600.0,
            near_threshold_m=1.5,
            medium_threshold_m=3.0
        )
        self.estimator.initialize()

    def tearDown(self):
        """Clean up estimator state."""
        self.estimator.reset()

    def test_01_estimator_initialization(self):
        """Test 1: Validate estimator initialization."""
        self.assertTrue(self.estimator._is_initialized)
        stats = self.estimator.get_statistics()
        self.assertEqual(stats["method"], "monocular_bbox")
        self.assertEqual(stats["focal_length_px"], 600.0)

    def test_02_valid_tracking_input(self):
        """Test 2: Validate estimation on standard Track input."""
        trk = Track(
            track_id=1, class_id=0, class_name="person", confidence=0.9,
            bounding_box=[100.0, 100.0, 300.0, 500.0],
            center_x=200.0, center_y=300.0, width=200.0, height=400.0,
            tracking_state="TRACKED"
        )
        res = self.estimator.estimate_distance(trk)
        self.assertIsInstance(res, DistanceResult)
        self.assertEqual(res.track_id, 1)
        self.assertEqual(res.class_name, "person")
        self.assertEqual(res.distance_status, "MEASURED")

    def test_03_bounding_box_processing(self):
        """Test 3: Validate bbox coordinate processing and height extraction."""
        trk = Track(
            track_id=2, class_id=1, class_name="chair", confidence=0.8,
            bounding_box=[50.0, 100.0, 250.0, 400.0],
            center_x=150.0, center_y=250.0, width=200.0, height=300.0,
            tracking_state="TRACKED"
        )
        res = self.estimator.estimate_distance(trk)
        # Chair ref height = 0.85m, focal = 600, height = 300px -> d = (0.85 * 600) / 300 = 1.70m
        self.assertAlmostEqual(res.estimated_distance_m, 1.70, places=2)
        self.assertEqual(res.distance_category, "MEDIUM")

    def test_04_distance_category_generation(self):
        """Test 4: Validate distance category string outputs."""
        trk = Track(
            track_id=1, class_id=0, class_name="person", confidence=0.9,
            bounding_box=[0.0, 0.0, 100.0, 100.0],
            center_x=50.0, center_y=50.0, width=100.0, height=100.0,
            tracking_state="TRACKED"
        )
        res = self.estimator.estimate_distance(trk)
        self.assertIn(res.distance_category, ["NEAR", "MEDIUM", "FAR", "UNKNOWN"])

    def test_05_near_classification(self):
        """Test 5: Validate NEAR classification (< 1.5m)."""
        # Person (1.7m ref height) with large 800px box -> d = (1.7 * 600) / 800 = 1.275m (< 1.5m)
        trk = Track(
            track_id=1, class_id=0, class_name="person", confidence=0.9,
            bounding_box=[0.0, 0.0, 400.0, 800.0],
            center_x=200.0, center_y=400.0, width=400.0, height=800.0,
            tracking_state="TRACKED"
        )
        res = self.estimator.estimate_distance(trk)
        self.assertEqual(res.distance_category, "NEAR")
        self.assertLessEqual(res.estimated_distance_m, 1.5)

    def test_06_medium_classification(self):
        """Test 6: Validate MEDIUM classification (1.5m - 3.0m)."""
        # Person (1.7m ref height) with 450px box -> d = (1.7 * 600) / 450 = 2.267m (1.5m - 3.0m)
        trk = Track(
            track_id=1, class_id=0, class_name="person", confidence=0.9,
            bounding_box=[0.0, 0.0, 200.0, 450.0],
            center_x=100.0, center_y=225.0, width=200.0, height=450.0,
            tracking_state="TRACKED"
        )
        res = self.estimator.estimate_distance(trk)
        self.assertEqual(res.distance_category, "MEDIUM")
        self.assertGreater(res.estimated_distance_m, 1.5)
        self.assertLessEqual(res.estimated_distance_m, 3.0)

    def test_07_far_classification(self):
        """Test 7: Validate FAR classification (> 3.0m)."""
        # Person (1.7m ref height) with small 150px box -> d = (1.7 * 600) / 150 = 6.80m (> 3.0m)
        trk = Track(
            track_id=1, class_id=0, class_name="person", confidence=0.9,
            bounding_box=[0.0, 0.0, 50.0, 150.0],
            center_x=25.0, center_y=75.0, width=50.0, height=150.0,
            tracking_state="TRACKED"
        )
        res = self.estimator.estimate_distance(trk)
        self.assertEqual(res.distance_category, "FAR")
        self.assertGreater(res.estimated_distance_m, 3.0)

    def test_08_invalid_bounding_box_handling(self):
        """Test 8: Validate handling of zero/invalid bounding box height."""
        trk = Track(
            track_id=1, class_id=0, class_name="person", confidence=0.9,
            bounding_box=[0.0, 0.0, 0.0, 0.0],
            center_x=0.0, center_y=0.0, width=0.0, height=0.0,
            tracking_state="TRACKED"
        )
        res = self.estimator.estimate_distance(trk)
        self.assertEqual(res.distance_category, "UNKNOWN")
        self.assertIsNone(res.estimated_distance_m)

    def test_09_missing_track_handling(self):
        """Test 9: Validate handling of None input."""
        res = self.estimator.estimate_distance(None)
        self.assertEqual(res.distance_category, "UNKNOWN")

    def test_10_multiple_object_distance_estimation(self):
        """Test 10: Validate batch distance estimation for multiple objects."""
        trk1 = Track(
            track_id=1, class_id=0, class_name="person", confidence=0.9,
            bounding_box=[0, 0, 100, 800], center_x=50, center_y=400, width=100, height=800,
            tracking_state="TRACKED"
        )
        trk2 = Track(
            track_id=2, class_id=1, class_name="chair", confidence=0.8,
            bounding_box=[0, 0, 50, 100], center_x=25, center_y=50, width=50, height=100,
            tracking_state="TRACKED"
        )
        results = self.estimator.estimate_batch([trk1, trk2])
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].distance_category, "NEAR")
        self.assertEqual(results[1].distance_category, "FAR")

    def test_11_confidence_bounds(self):
        """Test 11: Validate distance confidence bounds [0.0, 1.0]."""
        trk = Track(
            track_id=1, class_id=0, class_name="person", confidence=0.95,
            bounding_box=[0, 0, 100, 300], center_x=50, center_y=150, width=100, height=300,
            tracking_state="TRACKED"
        )
        res = self.estimator.estimate_distance(trk)
        self.assertGreaterEqual(res.distance_confidence, 0.0)
        self.assertLessEqual(res.distance_confidence, 1.0)

    def test_12_phmu_integration(self):
        """Test 12: Validate PHMU HazardMemoryRecord integration."""
        rec = HazardMemoryRecord(
            track_id=5, object_class="door", class_id=3,
            bounding_box=[100.0, 50.0, 300.0, 450.0],
            center_x=200.0, center_y=250.0, width=200.0, height=400.0,
            memory_confidence=0.88, memory_state="ACTIVE"
        )
        res = self.estimator.estimate_distance(rec)
        self.assertEqual(res.track_id, 5)
        self.assertEqual(res.class_name, "door")
        # Door (2.0m ref height) with 400px box -> d = (2.0 * 600) / 400 = 3.0m
        self.assertEqual(res.distance_category, "MEDIUM")

    def test_13_remembered_object_handling(self):
        """Test 13: Validate remembered object policy (LAST_OBSERVED status)."""
        # Step 1: Active observation
        rec_active = HazardMemoryRecord(
            track_id=7, object_class="chair", class_id=1,
            bounding_box=[100.0, 100.0, 300.0, 400.0],
            center_x=200.0, center_y=250.0, width=200.0, height=300.0,
            memory_confidence=0.9, memory_state="ACTIVE"
        )
        res_act = self.estimator.estimate_distance(rec_active)
        self.assertEqual(res_act.distance_status, "MEASURED")
        self.assertEqual(res_act.distance_category, "MEDIUM")

        # Step 2: Unobserved REMEMBERED state with decayed confidence (e.g. 0.5)
        rec_remembered = HazardMemoryRecord(
            track_id=7, object_class="chair", class_id=1,
            bounding_box=[100.0, 100.0, 300.0, 400.0],
            center_x=200.0, center_y=250.0, width=200.0, height=300.0,
            memory_confidence=0.5, memory_state="REMEMBERED"
        )
        res_rem = self.estimator.estimate_distance(rec_remembered)
        self.assertEqual(res_rem.distance_status, "LAST_OBSERVED")
        self.assertEqual(res_rem.distance_category, "MEDIUM")
        self.assertEqual(res_rem.distance_confidence, 0.5)

    def test_14_reset_and_calibration(self):
        """Test 14: Validate reset and custom class calibration."""
        # Calibrate person reference height using known 2.0m distance with 510px height
        # H_ref = (2.0 * 510) / 600 = 1.70m
        new_ref = self.estimator.calibrate_class("person", 2.0, 510.0)
        self.assertAlmostEqual(new_ref, 1.70, places=2)

        self.estimator.reset()
        stats = self.estimator.get_statistics()
        self.assertEqual(stats["total_estimations"], 0)

    def test_15_core_synthetic_distance_experiment(self):
        """
        Test 15: Core Synthetic Distance Experiment.
        Tests same object class across large, medium, and small apparent sizes.
        Expected: Large -> NEAR, Medium -> MEDIUM, Small -> FAR.
        """
        # Person (1.7m ref height), focal = 600px
        # Large bbox (700px) -> d = 1.45m -> NEAR
        trk_large = Track(
            track_id=1, class_id=0, class_name="person", confidence=0.9,
            bounding_box=[0, 0, 300, 700], center_x=150, center_y=350, width=300, height=700,
            tracking_state="TRACKED"
        )
        # Medium bbox (350px) -> d = 2.91m -> MEDIUM
        trk_medium = Track(
            track_id=1, class_id=0, class_name="person", confidence=0.9,
            bounding_box=[0, 0, 150, 350], center_x=75, center_y=175, width=150, height=350,
            tracking_state="TRACKED"
        )
        # Small bbox (100px) -> d = 10.20m -> FAR
        trk_small = Track(
            track_id=1, class_id=0, class_name="person", confidence=0.9,
            bounding_box=[0, 0, 40, 100], center_x=20, center_y=50, width=40, height=100,
            tracking_state="TRACKED"
        )

        res_near = self.estimator.estimate_distance(trk_large)
        res_med = self.estimator.estimate_distance(trk_medium)
        res_far = self.estimator.estimate_distance(trk_small)

        self.assertEqual(res_near.distance_category, "NEAR")
        self.assertEqual(res_med.distance_category, "MEDIUM")
        self.assertEqual(res_far.distance_category, "FAR")


if __name__ == "__main__":
    unittest.main()
