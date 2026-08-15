import sys
import os
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.distance_estimation.models import DistanceResult
from modules.hazard_memory.models import HazardMemoryRecord
from modules.danger_mapping.mapper import ContextAwareDangerMapper
from modules.danger_mapping.models import DangerAssessment


class TestDangerMapping(unittest.TestCase):
    """
    Unit test suite for Module 07 — Context-Aware Danger Mapping.
    """

    def setUp(self):
        """Initialize danger mapper before each test."""
        self.mapper = ContextAwareDangerMapper()
        self.mapper.initialize()

    def tearDown(self):
        """Clean up mapper state."""
        self.mapper.reset()

    def test_01_initialization(self):
        """Test 1: Validate mapper initialization."""
        self.assertTrue(self.mapper._is_initialized)
        stats = self.mapper.get_statistics()
        self.assertIn("object", stats["weights"])

    def test_02_valid_hazard_input(self):
        """Test 2: Validate assessment of standard DistanceResult input."""
        dist_res = DistanceResult(
            track_id=1, class_name="person", distance_category="NEAR",
            distance_confidence=0.9, estimated_distance_m=1.2,
            bounding_box=[200.0, 100.0, 400.0, 400.0],
            distance_status="MEASURED"
        )
        res = self.mapper.assess_danger(dist_res, frame_width=640.0)
        self.assertIsInstance(res, DangerAssessment)
        self.assertEqual(res.track_id, 1)
        self.assertEqual(res.class_name, "person")

    def test_03_position_zone_left(self):
        """Test 3: Validate LEFT position zone classification (x_norm < 0.33)."""
        zone = self.mapper._determine_position_zone(center_x=100.0, frame_width=640.0)
        self.assertEqual(zone, "LEFT")

    def test_04_position_zone_center(self):
        """Test 4: Validate CENTER position zone classification (0.33 <= x_norm <= 0.67)."""
        zone = self.mapper._determine_position_zone(center_x=320.0, frame_width=640.0)
        self.assertEqual(zone, "CENTER")

    def test_05_position_zone_right(self):
        """Test 5: Validate RIGHT position zone classification (x_norm > 0.67)."""
        zone = self.mapper._determine_position_zone(center_x=550.0, frame_width=640.0)
        self.assertEqual(zone, "RIGHT")

    def test_06_near_distance_scoring(self):
        """Test 6: Validate NEAR distance factor contribution."""
        f_near = self.mapper._get_distance_factor("NEAR")
        self.assertEqual(f_near, 1.00)

    def test_07_medium_distance_scoring(self):
        """Test 7: Validate MEDIUM distance factor contribution."""
        f_med = self.mapper._get_distance_factor("MEDIUM")
        self.assertEqual(f_med, 0.50)

    def test_08_far_distance_scoring(self):
        """Test 8: Validate FAR distance factor contribution."""
        f_far = self.mapper._get_distance_factor("FAR")
        self.assertEqual(f_far, 0.20)

    def test_09_high_risk_object_rule(self):
        """Test 9: Validate high-risk object rule (stairs)."""
        f_stairs = self.mapper._get_object_hazard_factor("stairs")
        self.assertGreaterEqual(f_stairs, 0.90)

    def test_10_low_risk_object_rule(self):
        """Test 10: Validate low-risk object rule (distant chair on LEFT)."""
        dist_far_left = DistanceResult(
            track_id=1, class_name="chair", distance_category="FAR",
            distance_confidence=0.8, estimated_distance_m=5.0,
            bounding_box=[10.0, 10.0, 50.0, 50.0],
            distance_status="MEASURED"
        )
        res = self.mapper.assess_danger(dist_far_left, frame_width=640.0)
        self.assertEqual(res.danger_level, "LOW")

    def test_11_persistence_factor(self):
        """Test 11: Validate persistence factor contribution in scoring."""
        h_high_pers = HazardMemoryRecord(
            track_id=1, object_class="chair", class_id=1,
            bounding_box=[200, 100, 400, 400], center_x=300, center_y=250, width=200, height=300,
            persistence_score=1.0, memory_confidence=1.0, memory_state="ACTIVE"
        )
        h_low_pers = HazardMemoryRecord(
            track_id=2, object_class="chair", class_id=1,
            bounding_box=[200, 100, 400, 400], center_x=300, center_y=250, width=200, height=300,
            persistence_score=0.1, memory_confidence=1.0, memory_state="ACTIVE"
        )
        res1 = self.mapper.assess_danger(h_high_pers, frame_width=640.0)
        res2 = self.mapper.assess_danger(h_low_pers, frame_width=640.0)
        self.assertGreater(res1.danger_score, res2.danger_score)

    def test_12_memory_confidence_factor(self):
        """Test 12: Validate memory confidence factor contribution."""
        h_full_conf = HazardMemoryRecord(
            track_id=1, object_class="person", class_id=0,
            bounding_box=[200, 100, 400, 400], center_x=300, center_y=250, width=200, height=300,
            memory_confidence=1.0, memory_state="ACTIVE"
        )
        h_decay_conf = HazardMemoryRecord(
            track_id=2, object_class="person", class_id=0,
            bounding_box=[200, 100, 400, 400], center_x=300, center_y=250, width=200, height=300,
            memory_confidence=0.2, memory_state="REMEMBERED"
        )
        res1 = self.mapper.assess_danger(h_full_conf, frame_width=640.0)
        res2 = self.mapper.assess_danger(h_decay_conf, frame_width=640.0)
        self.assertGreater(res1.danger_score, res2.danger_score)

    def test_13_remembered_hazard_handling(self):
        """Test 13: Validate REMEMBERED hazard handling and reasoning generation."""
        h_rem = HazardMemoryRecord(
            track_id=7, object_class="chair", class_id=1,
            bounding_box=[200, 100, 400, 400], center_x=300, center_y=250, width=200, height=300,
            memory_confidence=0.5, memory_state="REMEMBERED"
        )
        res = self.mapper.assess_danger(h_rem, frame_width=640.0)
        self.assertIn("REMEMBERED", res.reasoning.upper())
        self.assertTrue(any("REMEMBERED" in f for f in res.danger_factors))

    def test_14_expired_hazard_removal(self):
        """Test 14: Validate exclusion of EXPIRED hazards in batch processing."""
        h_active = HazardMemoryRecord(
            track_id=1, object_class="person", class_id=0,
            bounding_box=[200, 100, 400, 400], center_x=300, center_y=250, width=200, height=300,
            memory_state="ACTIVE"
        )
        h_expired = HazardMemoryRecord(
            track_id=2, object_class="chair", class_id=1,
            bounding_box=[200, 100, 400, 400], center_x=300, center_y=250, width=200, height=300,
            memory_state="EXPIRED"
        )
        batch_res = self.mapper.assess_batch([h_active, h_expired], frame_width=640.0)
        self.assertEqual(len(batch_res), 1)
        self.assertEqual(batch_res[0].track_id, 1)

    def test_15_multiple_hazard_ranking(self):
        """Test 15: Validate hazard ranking by danger_score descending."""
        d_low = DistanceResult(
            track_id=1, class_name="chair", distance_category="FAR",
            distance_confidence=0.8, estimated_distance_m=5.0, bounding_box=[10, 10, 50, 50]
        )
        d_high = DistanceResult(
            track_id=2, class_name="stairs", distance_category="NEAR",
            distance_confidence=0.9, estimated_distance_m=1.0, bounding_box=[200, 100, 400, 400]
        )
        ranked = self.mapper.assess_batch([d_low, d_high], frame_width=640.0)
        self.assertEqual(len(ranked), 2)
        self.assertEqual(ranked[0].track_id, 2, "Highest danger (stairs) must be ranked first")
        self.assertEqual(ranked[1].track_id, 1)

    def test_16_danger_score_bounds(self):
        """Test 16: Validate danger score bounds [0.0, 1.0]."""
        d = DistanceResult(
            track_id=1, class_name="stairs", distance_category="NEAR",
            distance_confidence=1.0, estimated_distance_m=0.5, bounding_box=[200, 100, 400, 400]
        )
        res = self.mapper.assess_danger(d, frame_width=640.0)
        self.assertGreaterEqual(res.danger_score, 0.0)
        self.assertLessEqual(res.danger_score, 1.0)

    def test_17_navigation_relevance(self):
        """Test 17: Validate navigation_relevance boolean mapping."""
        d_stairs = DistanceResult(
            track_id=1, class_name="stairs", distance_category="FAR",
            distance_confidence=0.8, estimated_distance_m=5.0, bounding_box=[10, 10, 50, 50]
        )
        res = self.mapper.assess_danger(d_stairs, frame_width=640.0)
        self.assertTrue(res.navigation_relevance, "Stairs must be marked navigation_relevance=True")

    def test_18_invalid_input_handling(self):
        """Test 18: Validate safe handling of None input."""
        res = self.mapper.assess_danger(None, frame_width=640.0)
        self.assertEqual(res.danger_level, "LOW")
        self.assertFalse(res.navigation_relevance)

    def test_19_missing_distance_handling(self):
        """Test 19: Validate handling when distance category is UNKNOWN."""
        d_unk = DistanceResult(
            track_id=1, class_name="person", distance_category="UNKNOWN",
            distance_confidence=0.5, estimated_distance_m=None, bounding_box=[200, 100, 400, 400]
        )
        res = self.mapper.assess_danger(d_unk, frame_width=640.0)
        self.assertIsInstance(res, DangerAssessment)

    def test_20_missing_motion_handling(self):
        """Test 20: Validate handling when motion factor is unspecified (uses neutral 0.50)."""
        d = DistanceResult(
            track_id=1, class_name="chair", distance_category="MEDIUM",
            distance_confidence=0.8, estimated_distance_m=2.0, bounding_box=[200, 100, 400, 400]
        )
        res = self.mapper.assess_danger(d, frame_width=640.0)
        self.assertIsInstance(res, DangerAssessment)

    def test_21_core_context_aware_experiment(self):
        """
        Test 21: Core Context-Aware Experiment (5 Scenarios).
        Validates relative scoring across Scenarios A, B, C, D, and E.
        """
        # Scenario A: Chair, FAR, LEFT, ACTIVE -> LOW
        scen_a = DistanceResult(
            track_id=1, class_name="chair", distance_category="FAR",
            distance_confidence=0.8, estimated_distance_m=5.0, bounding_box=[10, 10, 50, 50]
        )
        res_a = self.mapper.assess_danger(scen_a, frame_width=640.0)

        # Scenario B: Chair, NEAR, CENTER, ACTIVE -> HIGH/MODERATE (> A)
        scen_b = DistanceResult(
            track_id=2, class_name="chair", distance_category="NEAR",
            distance_confidence=0.9, estimated_distance_m=1.2, bounding_box=[200, 100, 400, 400]
        )
        res_b = self.mapper.assess_danger(scen_b, frame_width=640.0)

        # Scenario C: Stairs, NEAR, CENTER, ACTIVE -> CRITICAL / Very High
        scen_c = DistanceResult(
            track_id=3, class_name="stairs", distance_category="NEAR",
            distance_confidence=0.95, estimated_distance_m=1.0, bounding_box=[200, 100, 400, 400]
        )
        res_c = self.mapper.assess_danger(scen_c, frame_width=640.0)

        # Scenario D: Person, NEAR, CENTER, approaching motion -> Higher than stationary
        scen_d_stat = DistanceResult(
            track_id=4, class_name="person", distance_category="NEAR",
            distance_confidence=0.9, estimated_distance_m=1.2, bounding_box=[200, 100, 400, 400]
        )
        scen_d_move = DistanceResult(
            track_id=5, class_name="person", distance_category="NEAR",
            distance_confidence=0.9, estimated_distance_m=1.2, bounding_box=[200, 100, 400, 400]
        )
        setattr(scen_d_move, "motion_factor", 1.0)
        setattr(scen_d_stat, "motion_factor", 0.0)
        res_d_stat = self.mapper.assess_danger(scen_d_stat, frame_width=640.0)
        res_d_move = self.mapper.assess_danger(scen_d_move, frame_width=640.0)

        # Scenario E: Chair, NEAR, CENTER, REMEMBERED -> Reduced score vs ACTIVE
        scen_e = HazardMemoryRecord(
            track_id=6, object_class="chair", class_id=1,
            bounding_box=[200, 100, 400, 400], center_x=300, center_y=250, width=200, height=300,
            memory_confidence=0.4, memory_state="REMEMBERED"
        )
        setattr(scen_e, "distance_category", "NEAR")
        res_e = self.mapper.assess_danger(scen_e, frame_width=640.0)

        # Confirm expected context relationships
        self.assertEqual(res_a.danger_level, "LOW")
        self.assertGreater(res_b.danger_score, res_a.danger_score, "Scenario B (NEAR, CENTER) > Scenario A (FAR, LEFT)")
        self.assertEqual(res_c.danger_level, "CRITICAL", "Scenario C (Stairs NEAR CENTER) must be CRITICAL")
        self.assertGreater(res_d_move.danger_score, res_d_stat.danger_score, "Scenario D (Approaching motion) > Stationary")
        self.assertLess(res_e.danger_score, res_b.danger_score, "Scenario E (REMEMBERED) < Scenario B (ACTIVE)")


if __name__ == "__main__":
    unittest.main()
