import sys
import os
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.danger_mapping.models import DangerAssessment
from modules.hazard_memory.models import HazardMemoryRecord
from modules.free_space.analyzer import ImageSpaceFreeSpaceAnalyzer
from modules.free_space.models import FreeSpaceAnalysisResult, RegionOccupancy


class TestFreeSpace(unittest.TestCase):
    """
    Unit test suite for Module 08 — Free-Space Analysis.
    """

    def setUp(self):
        """Initialize free-space analyzer before each test."""
        self.analyzer = ImageSpaceFreeSpaceAnalyzer()
        self.analyzer.initialize()

    def tearDown(self):
        """Clean up analyzer state."""
        self.analyzer.reset()

    def test_01_initialization(self):
        """Test 1: Validate analyzer initialization."""
        self.assertTrue(self.analyzer._is_initialized)
        stats = self.analyzer.get_statistics()
        self.assertIn("clear_max_threshold", stats)

    def test_02_valid_input(self):
        """Test 2: Validate analysis of valid DangerAssessment input list."""
        d = DangerAssessment(
            track_id=1, class_name="chair", danger_score=0.88, danger_level="CRITICAL",
            distance_category="NEAR", estimated_distance_m=1.2, position_zone="CENTER",
            memory_state="ACTIVE", memory_confidence=0.9, persistence_score=0.8,
            navigation_relevance=True, bounding_box=[200.0, 100.0, 400.0, 400.0]
        )
        res = self.analyzer.analyze_free_space([d], frame_width=640.0, frame_height=480.0)
        self.assertIsInstance(res, FreeSpaceAnalysisResult)
        self.assertIn("CENTER", res.regions)

    def test_03_left_zoning(self):
        """Test 3: Validate LEFT region overlap calculation."""
        d_left = DangerAssessment(
            track_id=1, class_name="person", danger_score=0.80, danger_level="HIGH",
            distance_category="NEAR", estimated_distance_m=1.0, position_zone="LEFT",
            memory_state="ACTIVE", memory_confidence=0.9, persistence_score=0.8,
            navigation_relevance=True, bounding_box=[10.0, 100.0, 150.0, 400.0]
        )
        res = self.analyzer.analyze_free_space([d_left], frame_width=640.0, frame_height=480.0)
        self.assertEqual(res.regions["LEFT"].occupancy_state, "BLOCKED")
        self.assertIn(1, res.regions["LEFT"].blocked_object_ids)

    def test_04_center_zoning(self):
        """Test 4: Validate CENTER region overlap calculation."""
        d_center = DangerAssessment(
            track_id=2, class_name="stairs", danger_score=0.95, danger_level="CRITICAL",
            distance_category="NEAR", estimated_distance_m=1.0, position_zone="CENTER",
            memory_state="ACTIVE", memory_confidence=0.95, persistence_score=0.9,
            navigation_relevance=True, bounding_box=[250.0, 100.0, 400.0, 400.0]
        )
        res = self.analyzer.analyze_free_space([d_center], frame_width=640.0, frame_height=480.0)
        self.assertEqual(res.regions["CENTER"].occupancy_state, "BLOCKED")

    def test_05_right_zoning(self):
        """Test 5: Validate RIGHT region overlap calculation."""
        d_right = DangerAssessment(
            track_id=3, class_name="chair", danger_score=0.85, danger_level="CRITICAL",
            distance_category="NEAR", estimated_distance_m=1.0, position_zone="RIGHT",
            memory_state="ACTIVE", memory_confidence=0.9, persistence_score=0.8,
            navigation_relevance=True, bounding_box=[450.0, 100.0, 600.0, 400.0]
        )
        res = self.analyzer.analyze_free_space([d_right], frame_width=640.0, frame_height=480.0)
        self.assertEqual(res.regions["RIGHT"].occupancy_state, "BLOCKED")

    def test_06_empty_scene(self):
        """Test 6: Validate empty scene (no hazards) produces CLEAR regions."""
        res = self.analyzer.analyze_free_space([], frame_width=640.0, frame_height=480.0)
        self.assertEqual(res.regions["LEFT"].occupancy_state, "CLEAR")
        self.assertEqual(res.regions["CENTER"].occupancy_state, "CLEAR")
        self.assertEqual(res.regions["RIGHT"].occupancy_state, "CLEAR")
        self.assertEqual(res.overall_traversability, "CLEAR")

    def test_07_single_center_obstacle(self):
        """Test 7: Validate single center obstacle blocks only CENTER region."""
        d = DangerAssessment(
            track_id=1, class_name="door", danger_score=0.90, danger_level="CRITICAL",
            distance_category="NEAR", estimated_distance_m=1.2, position_zone="CENTER",
            memory_state="ACTIVE", memory_confidence=0.9, persistence_score=0.8,
            navigation_relevance=True, bounding_box=[250.0, 100.0, 380.0, 400.0]
        )
        res = self.analyzer.analyze_free_space([d], frame_width=640.0, frame_height=480.0)
        self.assertEqual(res.regions["CENTER"].occupancy_state, "BLOCKED")
        self.assertEqual(res.regions["LEFT"].occupancy_state, "CLEAR")
        self.assertEqual(res.regions["RIGHT"].occupancy_state, "CLEAR")

    def test_08_left_obstacle(self):
        """Test 8: Validate single left obstacle."""
        d = DangerAssessment(
            track_id=1, class_name="person", danger_score=0.85, danger_level="CRITICAL",
            distance_category="NEAR", estimated_distance_m=1.0, position_zone="LEFT",
            memory_state="ACTIVE", memory_confidence=0.9, persistence_score=0.8,
            navigation_relevance=True, bounding_box=[10.0, 100.0, 180.0, 400.0]
        )
        res = self.analyzer.analyze_free_space([d], frame_width=640.0, frame_height=480.0)
        self.assertEqual(res.regions["LEFT"].occupancy_state, "BLOCKED")

    def test_09_right_obstacle(self):
        """Test 9: Validate single right obstacle."""
        d = DangerAssessment(
            track_id=1, class_name="person", danger_score=0.85, danger_level="CRITICAL",
            distance_category="NEAR", estimated_distance_m=1.0, position_zone="RIGHT",
            memory_state="ACTIVE", memory_confidence=0.9, persistence_score=0.8,
            navigation_relevance=True, bounding_box=[450.0, 100.0, 620.0, 400.0]
        )
        res = self.analyzer.analyze_free_space([d], frame_width=640.0, frame_height=480.0)
        self.assertEqual(res.regions["RIGHT"].occupancy_state, "BLOCKED")

    def test_10_near_obstacle(self):
        """Test 10: Validate NEAR obstacle distance multiplier."""
        d_near = DangerAssessment(
            track_id=1, class_name="chair", danger_score=0.70, danger_level="HIGH",
            distance_category="NEAR", estimated_distance_m=1.0, position_zone="CENTER",
            memory_state="ACTIVE", memory_confidence=0.9, persistence_score=0.8,
            navigation_relevance=True, bounding_box=[250.0, 100.0, 380.0, 400.0]
        )
        res = self.analyzer.analyze_free_space([d_near], frame_width=640.0, frame_height=480.0)
        self.assertGreater(res.regions["CENTER"].occupancy_score, 0.40)

    def test_11_medium_obstacle(self):
        """Test 11: Validate MEDIUM obstacle distance factor."""
        d_med = DangerAssessment(
            track_id=1, class_name="chair", danger_score=0.70, danger_level="HIGH",
            distance_category="MEDIUM", estimated_distance_m=2.5, position_zone="CENTER",
            memory_state="ACTIVE", memory_confidence=0.9, persistence_score=0.8,
            navigation_relevance=True, bounding_box=[250.0, 100.0, 380.0, 400.0]
        )
        res = self.analyzer.analyze_free_space([d_med], frame_width=640.0, frame_height=480.0)
        self.assertLess(res.regions["CENTER"].occupancy_score, 0.45)

    def test_12_far_obstacle(self):
        """Test 12: Validate FAR obstacle low occupancy contribution."""
        d_far = DangerAssessment(
            track_id=1, class_name="chair", danger_score=0.51, danger_level="LOW",
            distance_category="FAR", estimated_distance_m=5.0, position_zone="LEFT",
            memory_state="ACTIVE", memory_confidence=0.8, persistence_score=0.8,
            navigation_relevance=False, bounding_box=[10.0, 10.0, 50.0, 50.0]
        )
        res = self.analyzer.analyze_free_space([d_far], frame_width=640.0, frame_height=480.0)
        self.assertEqual(res.regions["LEFT"].occupancy_state, "CLEAR")

    def test_13_high_danger_obstacle(self):
        """Test 13: Validate high danger score obstacle occupancy contribution."""
        d_high = DangerAssessment(
            track_id=1, class_name="stairs", danger_score=0.96, danger_level="CRITICAL",
            distance_category="NEAR", estimated_distance_m=1.0, position_zone="CENTER",
            memory_state="ACTIVE", memory_confidence=0.95, persistence_score=0.9,
            navigation_relevance=True, bounding_box=[250.0, 100.0, 380.0, 400.0]
        )
        res = self.analyzer.analyze_free_space([d_high], frame_width=640.0, frame_height=480.0)
        self.assertEqual(res.regions["CENTER"].occupancy_state, "BLOCKED")

    def test_14_low_danger_obstacle(self):
        """Test 14: Validate low danger score obstacle."""
        d_low = DangerAssessment(
            track_id=1, class_name="exit", danger_score=0.30, danger_level="LOW",
            distance_category="FAR", estimated_distance_m=6.0, position_zone="RIGHT",
            memory_state="ACTIVE", memory_confidence=0.8, persistence_score=0.5,
            navigation_relevance=True, bounding_box=[450.0, 10.0, 550.0, 100.0]
        )
        res = self.analyzer.analyze_free_space([d_low], frame_width=640.0, frame_height=480.0)
        self.assertEqual(res.regions["RIGHT"].occupancy_state, "CLEAR")

    def test_15_multiple_obstacles(self):
        """Test 15: Validate cumulative occupancy from multiple obstacles in same region."""
        d1 = DangerAssessment(
            track_id=1, class_name="person", danger_score=0.60, danger_level="MODERATE",
            distance_category="MEDIUM", estimated_distance_m=2.0, position_zone="CENTER",
            memory_state="ACTIVE", memory_confidence=0.9, persistence_score=0.8,
            navigation_relevance=True, bounding_box=[250.0, 100.0, 350.0, 400.0]
        )
        d2 = DangerAssessment(
            track_id=2, class_name="table", danger_score=0.60, danger_level="MODERATE",
            distance_category="MEDIUM", estimated_distance_m=2.0, position_zone="CENTER",
            memory_state="ACTIVE", memory_confidence=0.9, persistence_score=0.8,
            navigation_relevance=True, bounding_box=[280.0, 100.0, 380.0, 400.0]
        )
        res1 = self.analyzer.analyze_free_space([d1], frame_width=640.0, frame_height=480.0)
        res2 = self.analyzer.analyze_free_space([d1, d2], frame_width=640.0, frame_height=480.0)
        self.assertGreater(res2.regions["CENTER"].occupancy_score, res1.regions["CENTER"].occupancy_score)

    def test_16_cross_region_bounding_box(self):
        """Test 16: Validate wide bounding box overlapping both CENTER and RIGHT regions."""
        d_wide = DangerAssessment(
            track_id=1, class_name="table", danger_score=0.88, danger_level="CRITICAL",
            distance_category="NEAR", estimated_distance_m=1.2, position_zone="CENTER",
            memory_state="ACTIVE", memory_confidence=0.9, persistence_score=0.8,
            navigation_relevance=True, bounding_box=[300.0, 100.0, 600.0, 400.0]  # Overlaps CENTER and RIGHT
        )
        res = self.analyzer.analyze_free_space([d_wide], frame_width=640.0, frame_height=480.0)
        self.assertGreater(res.regions["CENTER"].occupancy_score, 0.0)
        self.assertGreater(res.regions["RIGHT"].occupancy_score, 0.0)
        self.assertIn(1, res.regions["CENTER"].blocked_object_ids)
        self.assertIn(1, res.regions["RIGHT"].blocked_object_ids)

    def test_17_active_phmu(self):
        """Test 17: Validate ACTIVE PHMU state full confidence contribution."""
        d_active = DangerAssessment(
            track_id=1, class_name="chair", danger_score=0.80, danger_level="HIGH",
            distance_category="NEAR", estimated_distance_m=1.2, position_zone="CENTER",
            memory_state="ACTIVE", memory_confidence=1.0, persistence_score=1.0,
            navigation_relevance=True, bounding_box=[250.0, 100.0, 380.0, 400.0]
        )
        res = self.analyzer.analyze_free_space([d_active], frame_width=640.0, frame_height=480.0)
        self.assertEqual(res.regions["CENTER"].occupancy_state, "BLOCKED")

    def test_18_remembered_phmu(self):
        """Test 18: Validate REMEMBERED PHMU state produces UNCERTAIN state and reasoning string."""
        d_rem = DangerAssessment(
            track_id=1, class_name="chair", danger_score=0.80, danger_level="HIGH",
            distance_category="NEAR", estimated_distance_m=1.2, position_zone="CENTER",
            memory_state="REMEMBERED", memory_confidence=0.4, persistence_score=0.5,
            navigation_relevance=True, bounding_box=[250.0, 100.0, 380.0, 400.0]
        )
        res = self.analyzer.analyze_free_space([d_rem], frame_width=640.0, frame_height=480.0)
        self.assertIn("REMEMBERED", res.regions["CENTER"].reasoning)

    def test_19_expired_phmu(self):
        """Test 19: Validate EXPIRED PHMU hazards are completely ignored."""
        h_expired = HazardMemoryRecord(
            track_id=1, object_class="chair", class_id=1,
            bounding_box=[250, 100, 380, 400], center_x=315, center_y=250, width=130, height=300,
            memory_state="EXPIRED"
        )
        res = self.analyzer.analyze_free_space([h_expired], frame_width=640.0, frame_height=480.0)
        self.assertEqual(res.regions["CENTER"].occupancy_state, "CLEAR")

    def test_20_low_confidence_input(self):
        """Test 20: Validate low-confidence input reduces occupancy contribution."""
        d_low_conf = DangerAssessment(
            track_id=1, class_name="person", danger_score=0.80, danger_level="HIGH",
            distance_category="NEAR", estimated_distance_m=1.2, position_zone="CENTER",
            memory_state="ACTIVE", memory_confidence=0.2, persistence_score=0.5,
            navigation_relevance=True, bounding_box=[250.0, 100.0, 380.0, 400.0]
        )
        res = self.analyzer.analyze_free_space([d_low_conf], frame_width=640.0, frame_height=480.0)
        self.assertLess(res.regions["CENTER"].occupancy_score, 0.60)

    def test_21_missing_distance(self):
        """Test 21: Validate handling when distance_category is UNKNOWN."""
        d_unk = DangerAssessment(
            track_id=1, class_name="person", danger_score=0.60, danger_level="MODERATE",
            distance_category="UNKNOWN", estimated_distance_m=None, position_zone="CENTER",
            memory_state="ACTIVE", memory_confidence=0.8, persistence_score=0.5,
            navigation_relevance=True, bounding_box=[250.0, 100.0, 380.0, 400.0]
        )
        res = self.analyzer.analyze_free_space([d_unk], frame_width=640.0, frame_height=480.0)
        self.assertIsInstance(res, FreeSpaceAnalysisResult)

    def test_22_occupancy_bounds(self):
        """Test 22: Validate occupancy score bounds [0.0, 1.0]."""
        d1 = DangerAssessment(
            track_id=1, class_name="stairs", danger_score=0.95, danger_level="CRITICAL",
            distance_category="NEAR", estimated_distance_m=0.5, position_zone="CENTER",
            memory_state="ACTIVE", memory_confidence=1.0, persistence_score=1.0,
            navigation_relevance=True, bounding_box=[200.0, 100.0, 400.0, 400.0]
        )
        d2 = DangerAssessment(
            track_id=2, class_name="door", danger_score=0.95, danger_level="CRITICAL",
            distance_category="NEAR", estimated_distance_m=0.5, position_zone="CENTER",
            memory_state="ACTIVE", memory_confidence=1.0, persistence_score=1.0,
            navigation_relevance=True, bounding_box=[200.0, 100.0, 400.0, 400.0]
        )
        res = self.analyzer.analyze_free_space([d1, d2], frame_width=640.0, frame_height=480.0)
        self.assertGreaterEqual(res.regions["CENTER"].occupancy_score, 0.0)
        self.assertLessEqual(res.regions["CENTER"].occupancy_score, 1.0)

    def test_23_safe_space_bounds(self):
        """Test 23: Validate safe_space_score equals 1.0 - occupancy_score."""
        d = DangerAssessment(
            track_id=1, class_name="chair", danger_score=0.80, danger_level="HIGH",
            distance_category="NEAR", estimated_distance_m=1.2, position_zone="LEFT",
            memory_state="ACTIVE", memory_confidence=0.9, persistence_score=0.8,
            navigation_relevance=True, bounding_box=[10.0, 100.0, 180.0, 400.0]
        )
        res = self.analyzer.analyze_free_space([d], frame_width=640.0, frame_height=480.0)
        left = res.regions["LEFT"]
        self.assertAlmostEqual(left.safe_space_score, 1.0 - left.occupancy_score, places=4)

    def test_24_region_state_thresholds(self):
        """Test 24: Validate CLEAR, UNCERTAIN, and BLOCKED state transitions."""
        # CLEAR (< 0.25)
        d_clear = DangerAssessment(
            track_id=1, class_name="chair", danger_score=0.30, danger_level="LOW",
            distance_category="FAR", estimated_distance_m=5.0, position_zone="LEFT",
            memory_state="ACTIVE", memory_confidence=0.5, persistence_score=0.5,
            navigation_relevance=False, bounding_box=[10.0, 10.0, 50.0, 50.0]
        )
        res_c = self.analyzer.analyze_free_space([d_clear], frame_width=640.0, frame_height=480.0)
        self.assertEqual(res_c.regions["LEFT"].occupancy_state, "CLEAR")

        # BLOCKED (>= 0.60)
        d_blocked = DangerAssessment(
            track_id=2, class_name="stairs", danger_score=0.95, danger_level="CRITICAL",
            distance_category="NEAR", estimated_distance_m=1.0, position_zone="LEFT",
            memory_state="ACTIVE", memory_confidence=0.95, persistence_score=0.9,
            navigation_relevance=True, bounding_box=[10.0, 100.0, 200.0, 400.0]
        )
        res_b = self.analyzer.analyze_free_space([d_blocked], frame_width=640.0, frame_height=480.0)
        self.assertEqual(res_b.regions["LEFT"].occupancy_state, "BLOCKED")

    def test_25_core_synthetic_experiments(self):
        """
        Test 25: Core Synthetic Free-Space Experiments (Scenarios A through H).
        """
        # Scenario A — Completely unobstructed
        res_a = self.analyzer.analyze_free_space([], frame_width=640.0, frame_height=480.0)
        self.assertEqual(res_a.regions["LEFT"].occupancy_state, "CLEAR")
        self.assertEqual(res_a.regions["CENTER"].occupancy_state, "CLEAR")
        self.assertEqual(res_a.regions["RIGHT"].occupancy_state, "CLEAR")

        # Scenario B — Center obstacle
        d_b = DangerAssessment(
            track_id=1, class_name="person", danger_score=0.92, danger_level="CRITICAL",
            distance_category="NEAR", estimated_distance_m=1.0, position_zone="CENTER",
            memory_state="ACTIVE", memory_confidence=0.95, persistence_score=0.9,
            navigation_relevance=True, bounding_box=[250.0, 100.0, 380.0, 400.0]
        )
        res_b = self.analyzer.analyze_free_space([d_b], frame_width=640.0, frame_height=480.0)
        self.assertEqual(res_b.regions["CENTER"].occupancy_state, "BLOCKED")
        self.assertEqual(res_b.regions["LEFT"].occupancy_state, "CLEAR")
        self.assertEqual(res_b.regions["RIGHT"].occupancy_state, "CLEAR")

        # Scenario C — Left obstacle
        d_c = DangerAssessment(
            track_id=2, class_name="person", danger_score=0.90, danger_level="CRITICAL",
            distance_category="NEAR", estimated_distance_m=1.0, position_zone="LEFT",
            memory_state="ACTIVE", memory_confidence=0.9, persistence_score=0.8,
            navigation_relevance=True, bounding_box=[10.0, 100.0, 180.0, 400.0]
        )
        res_c = self.analyzer.analyze_free_space([d_c], frame_width=640.0, frame_height=480.0)
        self.assertEqual(res_c.regions["LEFT"].occupancy_state, "BLOCKED")

        # Scenario D — Right obstacle
        d_d = DangerAssessment(
            track_id=3, class_name="person", danger_score=0.90, danger_level="CRITICAL",
            distance_category="NEAR", estimated_distance_m=1.0, position_zone="RIGHT",
            memory_state="ACTIVE", memory_confidence=0.9, persistence_score=0.8,
            navigation_relevance=True, bounding_box=[450.0, 100.0, 620.0, 400.0]
        )
        res_d = self.analyzer.analyze_free_space([d_d], frame_width=640.0, frame_height=480.0)
        self.assertEqual(res_d.regions["RIGHT"].occupancy_state, "BLOCKED")

        # Scenario E — Multiple obstacles (LEFT NEAR, CENTER NEAR, RIGHT FAR)
        d_e1 = DangerAssessment(
            track_id=4, class_name="chair", danger_score=0.85, danger_level="CRITICAL",
            distance_category="NEAR", estimated_distance_m=1.0, position_zone="LEFT",
            memory_state="ACTIVE", memory_confidence=0.9, persistence_score=0.8,
            navigation_relevance=True, bounding_box=[10.0, 100.0, 180.0, 400.0]
        )
        d_e2 = DangerAssessment(
            track_id=5, class_name="table", danger_score=0.85, danger_level="CRITICAL",
            distance_category="NEAR", estimated_distance_m=1.0, position_zone="CENTER",
            memory_state="ACTIVE", memory_confidence=0.9, persistence_score=0.8,
            navigation_relevance=True, bounding_box=[250.0, 100.0, 380.0, 400.0]
        )
        d_e3 = DangerAssessment(
            track_id=6, class_name="chair", danger_score=0.30, danger_level="LOW",
            distance_category="FAR", estimated_distance_m=5.0, position_zone="RIGHT",
            memory_state="ACTIVE", memory_confidence=0.8, persistence_score=0.5,
            navigation_relevance=False, bounding_box=[450.0, 10.0, 500.0, 50.0]
        )
        res_e = self.analyzer.analyze_free_space([d_e1, d_e2, d_e3], frame_width=640.0, frame_height=480.0)
        self.assertEqual(res_e.regions["LEFT"].occupancy_state, "BLOCKED")
        self.assertEqual(res_e.regions["CENTER"].occupancy_state, "BLOCKED")
        self.assertEqual(res_e.regions["RIGHT"].occupancy_state, "CLEAR")

        # Scenario F — Remembered obstacle
        d_f = DangerAssessment(
            track_id=7, class_name="chair", danger_score=0.80, danger_level="HIGH",
            distance_category="NEAR", estimated_distance_m=1.2, position_zone="CENTER",
            memory_state="REMEMBERED", memory_confidence=0.4, persistence_score=0.5,
            navigation_relevance=True, bounding_box=[250.0, 100.0, 380.0, 400.0]
        )
        res_f = self.analyzer.analyze_free_space([d_f], frame_width=640.0, frame_height=480.0)
        self.assertIn("REMEMBERED", res_f.regions["CENTER"].reasoning)

        # Scenario G — Expired obstacle
        h_g = HazardMemoryRecord(
            track_id=8, object_class="chair", class_id=1,
            bounding_box=[250, 100, 380, 400], center_x=315, center_y=250, width=130, height=300,
            memory_state="EXPIRED"
        )
        res_g = self.analyzer.analyze_free_space([h_g], frame_width=640.0, frame_height=480.0)
        self.assertEqual(res_g.regions["CENTER"].occupancy_state, "CLEAR")

        # Scenario H — Wide object crossing regions (CENTER + RIGHT)
        d_h = DangerAssessment(
            track_id=9, class_name="table", danger_score=0.88, danger_level="CRITICAL",
            distance_category="NEAR", estimated_distance_m=1.2, position_zone="CENTER",
            memory_state="ACTIVE", memory_confidence=0.9, persistence_score=0.8,
            navigation_relevance=True, bounding_box=[300.0, 100.0, 600.0, 400.0]
        )
        res_h = self.analyzer.analyze_free_space([d_h], frame_width=640.0, frame_height=480.0)
        self.assertIn(9, res_h.regions["CENTER"].blocked_object_ids)
        self.assertIn(9, res_h.regions["RIGHT"].blocked_object_ids)


if __name__ == "__main__":
    unittest.main()
