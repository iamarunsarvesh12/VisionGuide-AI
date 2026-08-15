import sys
import os
import time
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.danger_mapping.models import DangerAssessment
from modules.free_space.models import RegionOccupancy, FreeSpaceAnalysisResult
from modules.decision_engine.models import (
    NavigationCommand,
    DecisionInput,
    DecisionResult,
    RegionDecisionScore,
)
from modules.decision_engine.engine import ContextAwareDecisionEngine


class TestDecisionEngine(unittest.TestCase):
    """
    Unit test suite and Core Decision Experiment for Module 09 — Context-Aware Decision Engine.
    """

    def setUp(self):
        """Initialize decision engine before each test."""
        self.engine = ContextAwareDecisionEngine()
        self.engine.initialize()

    def tearDown(self):
        """Reset decision engine state."""
        self.engine.reset()

    def test_01_initialization(self):
        """Test 1: Validate engine initialization."""
        self.assertTrue(self.engine._is_initialized)
        stats = self.engine.get_statistics()
        self.assertIn("command_counts", stats)
        self.assertEqual(stats["total_decisions_made"], 0)

    def test_02_valid_decision_input(self):
        """Test 2: Validate handling of structured DecisionInput."""
        inp = DecisionInput(
            timestamp=time.time(),
            frame_id=1,
            regions={
                "LEFT": RegionOccupancy("LEFT", "CLEAR", 0.0, 1.0),
                "CENTER": RegionOccupancy("CENTER", "CLEAR", 0.0, 1.0),
                "RIGHT": RegionOccupancy("RIGHT", "CLEAR", 0.0, 1.0),
            },
            hazards=[],
        )
        res = self.engine.decide(inp)
        self.assertIsInstance(res, DecisionResult)
        self.assertEqual(res.command, NavigationCommand.FORWARD.value)

    def test_03_clear_center_forward(self):
        """Test 3: Clear CENTER region yields FORWARD command."""
        inp = DecisionInput(
            timestamp=1.0,
            regions={
                "LEFT": {"occupancy_state": "CLEAR", "safe_space_score": 0.90},
                "CENTER": {"occupancy_state": "CLEAR", "safe_space_score": 0.95},
                "RIGHT": {"occupancy_state": "CLEAR", "safe_space_score": 0.90},
            },
        )
        res = self.engine.decide(inp)
        self.assertEqual(res.command, NavigationCommand.FORWARD.value)
        self.assertEqual(res.selected_region, "CENTER")

    def test_04_left_clear_left(self):
        """Test 4: CENTER blocked and LEFT clear yields LEFT command."""
        inp = DecisionInput(
            timestamp=1.0,
            regions={
                "LEFT": {"occupancy_state": "CLEAR", "safe_space_score": 0.85},
                "CENTER": {"occupancy_state": "BLOCKED", "safe_space_score": 0.10, "blocked_object_ids": [1]},
                "RIGHT": {"occupancy_state": "BLOCKED", "safe_space_score": 0.15, "blocked_object_ids": [2]},
            },
        )
        res = self.engine.decide(inp)
        self.assertEqual(res.command, NavigationCommand.LEFT.value)
        self.assertEqual(res.selected_region, "LEFT")

    def test_05_right_clear_right(self):
        """Test 5: CENTER blocked and RIGHT clear yields RIGHT command."""
        inp = DecisionInput(
            timestamp=1.0,
            regions={
                "LEFT": {"occupancy_state": "BLOCKED", "safe_space_score": 0.10, "blocked_object_ids": [1]},
                "CENTER": {"occupancy_state": "BLOCKED", "safe_space_score": 0.10, "blocked_object_ids": [2]},
                "RIGHT": {"occupancy_state": "CLEAR", "safe_space_score": 0.88},
            },
        )
        res = self.engine.decide(inp)
        self.assertEqual(res.command, NavigationCommand.RIGHT.value)
        self.assertEqual(res.selected_region, "RIGHT")

    def test_06_all_blocked_stop(self):
        """Test 6: All regions blocked yields STOP command."""
        inp = DecisionInput(
            timestamp=1.0,
            regions={
                "LEFT": {"occupancy_state": "BLOCKED", "safe_space_score": 0.05, "blocked_object_ids": [1]},
                "CENTER": {"occupancy_state": "BLOCKED", "safe_space_score": 0.05, "blocked_object_ids": [2]},
                "RIGHT": {"occupancy_state": "BLOCKED", "safe_space_score": 0.05, "blocked_object_ids": [3]},
            },
        )
        res = self.engine.decide(inp)
        self.assertEqual(res.command, NavigationCommand.STOP.value)
        self.assertIsNone(res.selected_region)
        self.assertEqual(sorted(res.blocking_hazards), [1, 2, 3])

    def test_07_critical_center_stop(self):
        """Test 7: Critical hazard in CENTER with no safe alternatives yields STOP command."""
        crit_hazard = DangerAssessment(
            track_id=7, class_name="glass_wall", danger_score=0.95, danger_level="CRITICAL",
            distance_category="NEAR", estimated_distance_m=1.0, position_zone="CENTER",
            memory_state="ACTIVE", memory_confidence=1.0, persistence_score=1.0,
            navigation_relevance=True, bounding_box=[250.0, 100.0, 400.0, 400.0]
        )
        inp = DecisionInput(
            timestamp=1.0,
            regions={
                "LEFT": {"occupancy_state": "UNCERTAIN", "safe_space_score": 0.40},
                "CENTER": {"occupancy_state": "BLOCKED", "safe_space_score": 0.05, "blocked_object_ids": [7]},
                "RIGHT": {"occupancy_state": "UNCERTAIN", "safe_space_score": 0.42},
            },
            hazards=[crit_hazard],
        )
        res = self.engine.decide(inp)
        self.assertEqual(res.command, NavigationCommand.STOP.value)
        self.assertIn(7, res.blocking_hazards)

    def test_08_critical_all_regions_stop(self):
        """Test 8: Critical hazards across all regions yields STOP command."""
        h1 = DangerAssessment(1, "stairs", 0.95, "CRITICAL", "NEAR", 1.0, "LEFT", "ACTIVE", 1.0, 1.0, True, bounding_box=[10, 100, 150, 400])
        h2 = DangerAssessment(2, "glass_wall", 0.95, "CRITICAL", "NEAR", 1.0, "CENTER", "ACTIVE", 1.0, 1.0, True, bounding_box=[250, 100, 400, 400])
        h3 = DangerAssessment(3, "table", 0.90, "CRITICAL", "NEAR", 1.0, "RIGHT", "ACTIVE", 1.0, 1.0, True, bounding_box=[450, 100, 600, 400])
        inp = DecisionInput(timestamp=1.0, hazards=[h1, h2, h3])
        res = self.engine.decide(inp)
        self.assertEqual(res.command, NavigationCommand.STOP.value)

    def test_09_high_center_danger(self):
        """Test 9: High danger in center suppresses FORWARD and prefers safe direction."""
        h_center = DangerAssessment(1, "person", 0.75, "HIGH", "NEAR", 1.2, "CENTER", "ACTIVE", 0.9, 0.8, True, bounding_box=[250, 100, 400, 400])
        inp = DecisionInput(
            timestamp=1.0,
            regions={
                "LEFT": {"occupancy_state": "CLEAR", "safe_space_score": 0.85},
                "CENTER": {"occupancy_state": "CLEAR", "safe_space_score": 0.60},
                "RIGHT": {"occupancy_state": "CLEAR", "safe_space_score": 0.50},
            },
            hazards=[h_center],
        )
        res = self.engine.decide(inp)
        self.assertEqual(res.command, NavigationCommand.LEFT.value)

    def test_10_low_center_danger(self):
        """Test 10: Low center danger allows FORWARD execution."""
        h_center = DangerAssessment(1, "chair", 0.25, "LOW", "FAR", 3.5, "CENTER", "ACTIVE", 0.9, 0.8, True, bounding_box=[250, 100, 300, 200])
        inp = DecisionInput(
            timestamp=1.0,
            regions={
                "LEFT": {"occupancy_state": "CLEAR", "safe_space_score": 0.80},
                "CENTER": {"occupancy_state": "CLEAR", "safe_space_score": 0.85},
                "RIGHT": {"occupancy_state": "CLEAR", "safe_space_score": 0.70},
            },
            hazards=[h_center],
        )
        res = self.engine.decide(inp)
        self.assertEqual(res.command, NavigationCommand.FORWARD.value)

    def test_11_safe_space_comparison(self):
        """Test 11: System selects region with higher safe space score."""
        inp = DecisionInput(
            timestamp=1.0,
            regions={
                "LEFT": {"occupancy_state": "CLEAR", "safe_space_score": 0.85},
                "CENTER": {"occupancy_state": "BLOCKED", "safe_space_score": 0.10},
                "RIGHT": {"occupancy_state": "CLEAR", "safe_space_score": 0.60},
            },
        )
        res = self.engine.decide(inp)
        self.assertEqual(res.command, NavigationCommand.LEFT.value)

    def test_12_danger_comparison(self):
        """Test 12: System selects region with lower danger score when safe spaces are similar."""
        h_left = DangerAssessment(1, "person", 0.70, "HIGH", "NEAR", 1.5, "LEFT", "ACTIVE", 0.9, 0.8, True, bounding_box=[50, 100, 150, 400])
        h_right = DangerAssessment(2, "chair", 0.20, "LOW", "FAR", 3.5, "RIGHT", "ACTIVE", 0.9, 0.8, True, bounding_box=[450, 100, 550, 300])
        inp = DecisionInput(
            timestamp=1.0,
            regions={
                "LEFT": {"occupancy_state": "CLEAR", "safe_space_score": 0.80},
                "CENTER": {"occupancy_state": "BLOCKED", "safe_space_score": 0.10},
                "RIGHT": {"occupancy_state": "CLEAR", "safe_space_score": 0.80},
            },
            hazards=[h_left, h_right],
        )
        res = self.engine.decide(inp)
        self.assertEqual(res.command, NavigationCommand.RIGHT.value)

    def test_13_confidence_comparison(self):
        """Test 13: Region with higher confidence scores better."""
        inp = DecisionInput(
            timestamp=1.0,
            regions={
                "LEFT": {"occupancy_state": "CLEAR", "safe_space_score": 0.80, "confidence": 0.95},
                "CENTER": {"occupancy_state": "BLOCKED", "safe_space_score": 0.10},
                "RIGHT": {"occupancy_state": "CLEAR", "safe_space_score": 0.80, "confidence": 0.40},
            },
        )
        res = self.engine.decide(inp)
        self.assertEqual(res.command, NavigationCommand.LEFT.value)

    def test_14_multiple_hazards(self):
        """Test 14: Handles multiple hazards across regions."""
        h1 = DangerAssessment(1, "chair", 0.60, "MODERATE", "MEDIUM", 2.0, "LEFT", "ACTIVE", 0.9, 0.8, True, bounding_box=[50, 100, 150, 400])
        h2 = DangerAssessment(2, "table", 0.85, "CRITICAL", "NEAR", 1.1, "CENTER", "ACTIVE", 0.9, 0.8, True, bounding_box=[250, 100, 400, 400])
        inp = DecisionInput(timestamp=1.0, hazards=[h1, h2])
        res = self.engine.decide(inp)
        self.assertEqual(res.command, NavigationCommand.RIGHT.value)

    def test_15_remembered_hazard(self):
        """Test 15: PHMU REMEMBERED hazard degrades region confidence and reduces candidate score."""
        h_rem = DangerAssessment(
            track_id=5, class_name="chair", danger_score=0.75, danger_level="HIGH",
            distance_category="NEAR", estimated_distance_m=1.2, position_zone="CENTER",
            memory_state="REMEMBERED", memory_confidence=0.8, persistence_score=0.7,
            navigation_relevance=True, bounding_box=[250.0, 100.0, 400.0, 400.0]
        )
        inp = DecisionInput(
            timestamp=1.0,
            regions={
                "LEFT": {"occupancy_state": "CLEAR", "safe_space_score": 0.85},
                "CENTER": {"occupancy_state": "CLEAR", "safe_space_score": 0.65},
                "RIGHT": {"occupancy_state": "CLEAR", "safe_space_score": 0.70},
            },
            hazards=[h_rem],
        )
        res = self.engine.decide(inp)
        # Should prefer LEFT due to remembered hazard in CENTER
        self.assertEqual(res.command, NavigationCommand.LEFT.value)

    def test_16_active_hazard(self):
        """Test 16: Active hazard directly impacts scoring."""
        h_act = DangerAssessment(
            track_id=1, class_name="stairs", danger_score=0.90, danger_level="CRITICAL",
            distance_category="NEAR", estimated_distance_m=1.0, position_zone="LEFT",
            memory_state="ACTIVE", memory_confidence=1.0, persistence_score=1.0,
            navigation_relevance=True, bounding_box=[50, 100, 150, 400]
        )
        inp = DecisionInput(
            timestamp=1.0,
            regions={
                "LEFT": {"occupancy_state": "BLOCKED", "safe_space_score": 0.10},
                "CENTER": {"occupancy_state": "BLOCKED", "safe_space_score": 0.10},
                "RIGHT": {"occupancy_state": "CLEAR", "safe_space_score": 0.85},
            },
            hazards=[h_act],
        )
        res = self.engine.decide(inp)
        self.assertEqual(res.command, NavigationCommand.RIGHT.value)

    def test_17_expired_hazard(self):
        """Test 17: Expired hazard is ignored completely."""
        h_exp = DangerAssessment(
            track_id=9, class_name="chair", danger_score=0.95, danger_level="CRITICAL",
            distance_category="NEAR", estimated_distance_m=1.0, position_zone="CENTER",
            memory_state="EXPIRED", memory_confidence=0.0, persistence_score=0.0,
            navigation_relevance=False, bounding_box=[250.0, 100.0, 400.0, 400.0]
        )
        inp = DecisionInput(
            timestamp=1.0,
            regions={
                "LEFT": {"occupancy_state": "CLEAR", "safe_space_score": 0.80},
                "CENTER": {"occupancy_state": "CLEAR", "safe_space_score": 0.90},
                "RIGHT": {"occupancy_state": "CLEAR", "safe_space_score": 0.80},
            },
            hazards=[h_exp],
        )
        res = self.engine.decide(inp)
        self.assertEqual(res.command, NavigationCommand.FORWARD.value)

    def test_18_uncertain_region(self):
        """Test 18: UNCERTAIN region is penalized appropriately."""
        inp = DecisionInput(
            timestamp=1.0,
            regions={
                "LEFT": {"occupancy_state": "CLEAR", "safe_space_score": 0.70},
                "CENTER": {"occupancy_state": "BLOCKED", "safe_space_score": 0.10},
                "RIGHT": {"occupancy_state": "UNCERTAIN", "safe_space_score": 0.70},
            },
        )
        res = self.engine.decide(inp)
        self.assertEqual(res.command, NavigationCommand.LEFT.value)

    def test_19_all_uncertain(self):
        """Test 19: All regions UNCERTAIN yields STOP decision for safety."""
        inp = DecisionInput(
            timestamp=1.0,
            regions={
                "LEFT": {"occupancy_state": "UNCERTAIN", "safe_space_score": 0.45},
                "CENTER": {"occupancy_state": "UNCERTAIN", "safe_space_score": 0.45},
                "RIGHT": {"occupancy_state": "UNCERTAIN", "safe_space_score": 0.45},
            },
        )
        res = self.engine.decide(inp)
        self.assertEqual(res.command, NavigationCommand.STOP.value)

    def test_20_previous_command_retention(self):
        """Test 20: Retains previous command when candidate score diff is below switching margin."""
        # Frame 1: LEFT selected
        inp1 = DecisionInput(
            timestamp=1.0,
            regions={
                "LEFT": {"occupancy_state": "CLEAR", "safe_space_score": 0.75},
                "CENTER": {"occupancy_state": "BLOCKED", "safe_space_score": 0.10},
                "RIGHT": {"occupancy_state": "CLEAR", "safe_space_score": 0.70},
            },
        )
        res1 = self.engine.decide(inp1)
        self.assertEqual(res1.command, NavigationCommand.LEFT.value)

        # Frame 2: RIGHT score slightly higher (0.73 vs 0.72), diff < switching_margin (0.10)
        inp2 = DecisionInput(
            timestamp=1.1,
            regions={
                "LEFT": {"occupancy_state": "CLEAR", "safe_space_score": 0.72},
                "CENTER": {"occupancy_state": "BLOCKED", "safe_space_score": 0.10},
                "RIGHT": {"occupancy_state": "CLEAR", "safe_space_score": 0.73},
            },
            previous_command="LEFT",
        )
        res2 = self.engine.decide(inp2)
        self.assertEqual(res2.command, NavigationCommand.LEFT.value)

    def test_21_switching_margin(self):
        """Test 21: Validates exact switching margin behavior."""
        margin = self.engine.config.get("switching_margin", 0.10)
        self.assertEqual(margin, 0.10)

    def test_22_strong_command_switch(self):
        """Test 22: Strong improvement exceeding switching margin triggers command switch."""
        # Frame 1: LEFT selected
        inp1 = DecisionInput(
            timestamp=1.0,
            regions={
                "LEFT": {"occupancy_state": "CLEAR", "safe_space_score": 0.70},
                "CENTER": {"occupancy_state": "BLOCKED", "safe_space_score": 0.10},
                "RIGHT": {"occupancy_state": "CLEAR", "safe_space_score": 0.60},
            },
        )
        self.engine.decide(inp1)

        # Frame 2: RIGHT score strongly improves (0.95 vs 0.55), diff >> margin
        inp2 = DecisionInput(
            timestamp=2.0,  # Elapsed time > hold duration
            regions={
                "LEFT": {"occupancy_state": "CLEAR", "safe_space_score": 0.55},
                "CENTER": {"occupancy_state": "BLOCKED", "safe_space_score": 0.10},
                "RIGHT": {"occupancy_state": "CLEAR", "safe_space_score": 0.95},
            },
        )
        res2 = self.engine.decide(inp2)
        self.assertEqual(res2.command, NavigationCommand.RIGHT.value)

    def test_23_decision_confidence_bounds(self):
        """Test 23: Ensures confidence score is bounded within [0.0, 1.0]."""
        inp = DecisionInput(
            timestamp=1.0,
            regions={
                "LEFT": {"occupancy_state": "CLEAR", "safe_space_score": 0.80, "confidence": 1.5},
                "CENTER": {"occupancy_state": "CLEAR", "safe_space_score": 0.80, "confidence": -0.5},
                "RIGHT": {"occupancy_state": "CLEAR", "safe_space_score": 0.80, "confidence": 0.9},
            },
        )
        res = self.engine.decide(inp)
        self.assertGreaterEqual(res.confidence, 0.0)
        self.assertLessEqual(res.confidence, 1.0)

    def test_24_invalid_input_handling(self):
        """Test 24: Gracefully handles empty or incomplete input."""
        inp = DecisionInput(timestamp=1.0, regions={}, hazards=[])
        res = self.engine.decide(inp)
        self.assertIn(res.command, [c.value for c in NavigationCommand])

    def test_25_deterministic_repeated_decision(self):
        """Test 25: Identical input yields identical decision result."""
        inp = DecisionInput(
            timestamp=1.0,
            regions={
                "LEFT": {"occupancy_state": "CLEAR", "safe_space_score": 0.80},
                "CENTER": {"occupancy_state": "BLOCKED", "safe_space_score": 0.20},
                "RIGHT": {"occupancy_state": "CLEAR", "safe_space_score": 0.65},
            },
        )
        res1 = self.engine.decide(inp)
        self.engine.reset()
        res2 = self.engine.decide(inp)
        self.assertEqual(res1.command, res2.command)
        self.assertEqual(res1.decision_score, res2.decision_score)
        self.assertEqual(res1.reason, res2.reason)

    def test_26_core_decision_experiments(self):
        """
        Deterministic Core Decision Experiment across 12 Scenarios specified in Phase 8 Prompt.
        """
        print("\n=== RUNNING CORE DECISION EXPERIMENT (12 SCENARIOS) ===")

        scenarios = [
            {
                "id": 1,
                "name": "Scenario 1 — Completely Clear",
                "input": DecisionInput(timestamp=1.0, regions={"LEFT": {"occupancy_state": "CLEAR", "safe_space_score": 0.90}, "CENTER": {"occupancy_state": "CLEAR", "safe_space_score": 0.90}, "RIGHT": {"occupancy_state": "CLEAR", "safe_space_score": 0.90}}),
                "expected": NavigationCommand.FORWARD.value,
            },
            {
                "id": 2,
                "name": "Scenario 2 — Center Blocked, Left Clear",
                "input": DecisionInput(timestamp=1.0, regions={"LEFT": {"occupancy_state": "CLEAR", "safe_space_score": 0.85}, "CENTER": {"occupancy_state": "BLOCKED", "safe_space_score": 0.10}, "RIGHT": {"occupancy_state": "BLOCKED", "safe_space_score": 0.10}}),
                "expected": NavigationCommand.LEFT.value,
            },
            {
                "id": 3,
                "name": "Scenario 3 — Center Blocked, Right Clear",
                "input": DecisionInput(timestamp=1.0, regions={"LEFT": {"occupancy_state": "BLOCKED", "safe_space_score": 0.10}, "CENTER": {"occupancy_state": "BLOCKED", "safe_space_score": 0.10}, "RIGHT": {"occupancy_state": "CLEAR", "safe_space_score": 0.85}}),
                "expected": NavigationCommand.RIGHT.value,
            },
            {
                "id": 4,
                "name": "Scenario 4 — Everything Blocked",
                "input": DecisionInput(timestamp=1.0, regions={"LEFT": {"occupancy_state": "BLOCKED", "safe_space_score": 0.05}, "CENTER": {"occupancy_state": "BLOCKED", "safe_space_score": 0.05}, "RIGHT": {"occupancy_state": "BLOCKED", "safe_space_score": 0.05}}),
                "expected": NavigationCommand.STOP.value,
            },
            {
                "id": 5,
                "name": "Scenario 5 — Critical Center Hazard",
                "input": DecisionInput(
                    timestamp=1.0,
                    regions={"LEFT": {"occupancy_state": "BLOCKED", "safe_space_score": 0.10}, "CENTER": {"occupancy_state": "BLOCKED", "safe_space_score": 0.05}, "RIGHT": {"occupancy_state": "BLOCKED", "safe_space_score": 0.10}},
                    hazards=[DangerAssessment(1, "glass_wall", 0.95, "CRITICAL", "NEAR", 1.0, "CENTER", "ACTIVE", 1.0, 1.0, True, bounding_box=[250, 100, 400, 400])],
                ),
                "expected": NavigationCommand.STOP.value,
            },
            {
                "id": 6,
                "name": "Scenario 6 — Center Clear",
                "input": DecisionInput(timestamp=1.0, regions={"LEFT": {"occupancy_state": "CLEAR", "safe_space_score": 0.70}, "CENTER": {"occupancy_state": "CLEAR", "safe_space_score": 0.90}, "RIGHT": {"occupancy_state": "CLEAR", "safe_space_score": 0.70}}),
                "expected": NavigationCommand.FORWARD.value,
            },
            {
                "id": 7,
                "name": "Scenario 7 — Left Better Than Right",
                "input": DecisionInput(timestamp=1.0, regions={"LEFT": {"occupancy_state": "CLEAR", "safe_space_score": 0.85}, "CENTER": {"occupancy_state": "BLOCKED", "safe_space_score": 0.10}, "RIGHT": {"occupancy_state": "CLEAR", "safe_space_score": 0.60}}),
                "expected": NavigationCommand.LEFT.value,
            },
            {
                "id": 8,
                "name": "Scenario 8 — Right Better Than Left",
                "input": DecisionInput(timestamp=1.0, regions={"LEFT": {"occupancy_state": "CLEAR", "safe_space_score": 0.55}, "CENTER": {"occupancy_state": "BLOCKED", "safe_space_score": 0.10}, "RIGHT": {"occupancy_state": "CLEAR", "safe_space_score": 0.82}}),
                "expected": NavigationCommand.RIGHT.value,
            },
            {
                "id": 9,
                "name": "Scenario 9 — Remembered Hazard",
                "input": DecisionInput(
                    timestamp=1.0,
                    regions={"LEFT": {"occupancy_state": "CLEAR", "safe_space_score": 0.85}, "CENTER": {"occupancy_state": "CLEAR", "safe_space_score": 0.65}, "RIGHT": {"occupancy_state": "CLEAR", "safe_space_score": 0.70}},
                    hazards=[DangerAssessment(1, "chair", 0.75, "HIGH", "NEAR", 1.2, "CENTER", "REMEMBERED", 0.8, 0.7, True, bounding_box=[250, 100, 400, 400])],
                ),
                "expected": NavigationCommand.LEFT.value,
            },
            {
                "id": 10,
                "name": "Scenario 10 — Expired Hazard",
                "input": DecisionInput(
                    timestamp=1.0,
                    regions={"LEFT": {"occupancy_state": "CLEAR", "safe_space_score": 0.80}, "CENTER": {"occupancy_state": "CLEAR", "safe_space_score": 0.90}, "RIGHT": {"occupancy_state": "CLEAR", "safe_space_score": 0.80}},
                    hazards=[DangerAssessment(1, "chair", 0.95, "CRITICAL", "NEAR", 1.0, "CENTER", "EXPIRED", 0.0, 0.0, False, bounding_box=[250, 100, 400, 400])],
                ),
                "expected": NavigationCommand.FORWARD.value,
            },
            {
                "id": 11,
                "name": "Scenario 11 — Command Stability",
                "setup": lambda eng: eng.decide(DecisionInput(timestamp=1.0, regions={"LEFT": {"occupancy_state": "CLEAR", "safe_space_score": 0.75}, "CENTER": {"occupancy_state": "BLOCKED", "safe_space_score": 0.10}, "RIGHT": {"occupancy_state": "CLEAR", "safe_space_score": 0.70}})),
                "input": DecisionInput(timestamp=1.1, regions={"LEFT": {"occupancy_state": "CLEAR", "safe_space_score": 0.72}, "CENTER": {"occupancy_state": "BLOCKED", "safe_space_score": 0.10}, "RIGHT": {"occupancy_state": "CLEAR", "safe_space_score": 0.73}}, previous_command="LEFT"),
                "expected": NavigationCommand.LEFT.value,
            },
            {
                "id": 12,
                "name": "Scenario 12 — Strong Right Improvement",
                "setup": lambda eng: eng.decide(DecisionInput(timestamp=1.0, regions={"LEFT": {"occupancy_state": "CLEAR", "safe_space_score": 0.75}, "CENTER": {"occupancy_state": "BLOCKED", "safe_space_score": 0.10}, "RIGHT": {"occupancy_state": "CLEAR", "safe_space_score": 0.60}})),
                "input": DecisionInput(timestamp=2.5, regions={"LEFT": {"occupancy_state": "CLEAR", "safe_space_score": 0.55}, "CENTER": {"occupancy_state": "BLOCKED", "safe_space_score": 0.10}, "RIGHT": {"occupancy_state": "CLEAR", "safe_space_score": 0.90}}),
                "expected": NavigationCommand.RIGHT.value,
            },
        ]

        all_passed = True
        for sc in scenarios:
            self.engine.reset()
            if "setup" in sc:
                sc["setup"](self.engine)
            res = self.engine.decide(sc["input"])
            status = "PASS" if res.command == sc["expected"] else "FAIL"
            if status == "FAIL":
                all_passed = False
            print(f"Scenario {sc['id']:02d}: {sc['name']:<40} Expected: {sc['expected']:<8} Actual: {res.command:<8} -> {status}")
            self.assertEqual(res.command, sc["expected"], f"Failed scenario {sc['id']}: {sc['name']}")

        self.assertTrue(all_passed, "All 12 Core Decision Scenarios must pass.")


if __name__ == "__main__":
    unittest.main()
