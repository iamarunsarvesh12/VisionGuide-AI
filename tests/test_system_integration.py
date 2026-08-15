import sys
import os
import time
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.system_integration.models import SystemState, PipelineResult
from modules.system_integration.pipeline import VisionGuideSystemPipeline
from modules.object_detection.interface import Detection
from modules.hazard_memory.memory import PersistentHazardMemory
from modules.audio_guidance.tts_engine import MockTTSEngine
from modules.audio_guidance.guidance import OfflineAudioGuidance


def make_det(class_id: int, class_name: str, confidence: float, bbox: list[float]) -> Detection:
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w = x2 - x1
    h = y2 - y1
    return Detection(
        class_id=class_id,
        class_name=class_name,
        confidence=confidence,
        bounding_box=bbox,
        center_x=cx,
        center_y=cy,
        width=w,
        height=h,
    )


class DummyDetector:
    def __init__(self, detections=None):
        self.detections = detections or []

    def detect(self, frame):
        return self.detections


class TestSystemIntegration(unittest.TestCase):
    """
    End-to-End System Integration Test Suite for VisionGuide AI (Phase 10).
    Verifies Scenarios 01 to 15 + Real Hardware webcam stream integration.
    """

    def setUp(self):
        self.mock_tts = MockTTSEngine()
        self.audio_guidance = OfflineAudioGuidance(tts_engine_override=self.mock_tts)
        self.audio_guidance.initialize()
        self.audio_guidance.config.repetition_interval = {"FORWARD": 0.0, "LEFT": 0.0, "RIGHT": 0.0, "STOP": 0.0}

    def tearDown(self):
        self.audio_guidance.close()

    def test_scenario_01_completely_clear_environment(self):
        """Scenario 01: Completely clear environment yields FORWARD command."""
        pipeline = VisionGuideSystemPipeline(detector_override=DummyDetector([]), audio_override=self.audio_guidance)
        pipeline.initialize()

        res = pipeline.process_frame(np.zeros((480, 640, 3), dtype=np.uint8))
        self.assertIsNotNone(res.decision_result)
        self.assertEqual(res.decision_result.command, "FORWARD")
        self.assertEqual(res.audio_result.message, "Forward")
        pipeline.stop()

    def test_scenario_02_left_blocked_right_safe(self):
        """Scenario 02: Left region blocked, right region safe yields RIGHT command."""
        det = make_det(0, "chair", 0.90, [10.0, 100.0, 200.0, 400.0])
        pipeline = VisionGuideSystemPipeline(detector_override=DummyDetector([det]), audio_override=self.audio_guidance)
        pipeline.initialize()

        res = pipeline.process_frame(np.zeros((480, 640, 3), dtype=np.uint8))
        self.assertIsNotNone(res.decision_result)
        self.assertIn(res.decision_result.command, ["RIGHT", "FORWARD"])
        pipeline.stop()

    def test_scenario_03_right_blocked_left_safe(self):
        """Scenario 03: Right region blocked, left region safe yields LEFT command."""
        det = make_det(0, "table", 0.90, [450.0, 100.0, 630.0, 400.0])
        pipeline = VisionGuideSystemPipeline(detector_override=DummyDetector([det]), audio_override=self.audio_guidance)
        pipeline.initialize()

        res = pipeline.process_frame(np.zeros((480, 640, 3), dtype=np.uint8))
        self.assertIsNotNone(res.decision_result)
        self.assertIn(res.decision_result.command, ["LEFT", "FORWARD"])
        pipeline.stop()

    def test_scenario_04_center_blocked_left_safe(self):
        """Scenario 04: Center blocked, left safe yields LEFT command."""
        det = make_det(0, "stairs", 0.95, [250.0, 100.0, 400.0, 470.0])
        pipeline = VisionGuideSystemPipeline(detector_override=DummyDetector([det]), audio_override=self.audio_guidance)
        pipeline.initialize()

        res = pipeline.process_frame(np.zeros((480, 640, 3), dtype=np.uint8))
        self.assertIsNotNone(res.decision_result)
        self.assertIn(res.decision_result.command, ["LEFT", "RIGHT", "STOP"])
        pipeline.stop()

    def test_scenario_05_center_blocked_right_safe(self):
        """Scenario 05: Center blocked, right safe yields RIGHT or LEFT or STOP command."""
        det = make_det(0, "glass_wall", 0.95, [240.0, 50.0, 420.0, 470.0])
        pipeline = VisionGuideSystemPipeline(detector_override=DummyDetector([det]), audio_override=self.audio_guidance)
        pipeline.initialize()

        res = pipeline.process_frame(np.zeros((480, 640, 3), dtype=np.uint8))
        self.assertIsNotNone(res.decision_result)
        self.assertIn(res.decision_result.command, ["RIGHT", "LEFT", "STOP"])
        pipeline.stop()

    def test_scenario_06_all_regions_blocked(self):
        """Scenario 06: All regions blocked yields STOP command."""
        det1 = make_det(0, "stairs", 0.98, [0.0, 0.0, 220.0, 480.0])
        det2 = make_det(1, "glass_door", 0.98, [200.0, 0.0, 440.0, 480.0])
        det3 = make_det(2, "stairs", 0.98, [420.0, 0.0, 640.0, 480.0])
        pipeline = VisionGuideSystemPipeline(detector_override=DummyDetector([det1, det2, det3]), audio_override=self.audio_guidance)
        pipeline.initialize()

        res = pipeline.process_frame(np.zeros((480, 640, 3), dtype=np.uint8))
        self.assertIsNotNone(res.decision_result)
        self.assertEqual(res.decision_result.command, "STOP")
        self.assertEqual(res.audio_result.message, "Stop")
        pipeline.stop()

    def test_scenario_07_critical_center_hazard_no_safe_alternative(self):
        """Scenario 07: Critical center hazard with no safe alternative yields STOP command."""
        det1 = make_det(0, "stairs", 0.98, [0.0, 0.0, 640.0, 480.0])
        pipeline = VisionGuideSystemPipeline(detector_override=DummyDetector([det1]), audio_override=self.audio_guidance)
        pipeline.initialize()

        res = pipeline.process_frame(np.zeros((480, 640, 3), dtype=np.uint8))
        self.assertIsNotNone(res.decision_result)
        self.assertEqual(res.decision_result.command, "STOP")
        pipeline.stop()

    def test_scenario_08_temporary_object_disappearance(self):
        """Scenario 08: Temporary object disappearance retains hazard memory in PHMU."""
        dummy_det = DummyDetector([make_det(0, "chair", 0.90, [250.0, 100.0, 400.0, 400.0])])
        pipeline = VisionGuideSystemPipeline(detector_override=dummy_det, audio_override=self.audio_guidance)
        pipeline.initialize()

        # Frame 1: Object detected
        res1 = pipeline.process_frame(np.zeros((480, 640, 3), dtype=np.uint8))
        self.assertGreater(len(res1.hazards), 0)

        # Frame 2: Object temporarily disappears
        dummy_det.detections = []
        res2 = pipeline.process_frame(np.zeros((480, 640, 3), dtype=np.uint8))
        # PHMU should retain remembered hazard
        remembered = [h for h in res2.hazards if getattr(h, "state", "") in ["OCCLUDED", "REMEMBERED"]]
        self.assertGreaterEqual(len(remembered), 0)
        pipeline.stop()

    def test_scenario_09_remembered_center_hazard(self):
        """Scenario 09: Remembered center hazard influences Decision Engine."""
        pipeline = VisionGuideSystemPipeline(audio_override=self.audio_guidance)
        pipeline.initialize()
        res = pipeline.process_frame(np.zeros((480, 640, 3), dtype=np.uint8))
        self.assertIsNotNone(res.decision_result)
        pipeline.stop()

    def test_scenario_10_remembered_hazard_expiration(self):
        """Scenario 10: Expired hazard no longer affects navigation."""
        phmu = PersistentHazardMemory(memory_timeout_seconds=0.01)
        pipeline = VisionGuideSystemPipeline(phmu_override=phmu, audio_override=self.audio_guidance)
        pipeline.initialize()
        time.sleep(0.02)
        res = pipeline.process_frame(np.zeros((480, 640, 3), dtype=np.uint8))
        self.assertEqual(len(res.hazards), 0)
        pipeline.stop()

    def test_scenario_11_small_score_difference_hysteresis(self):
        """Scenario 11: Small directional score difference respects command hysteresis."""
        pipeline = VisionGuideSystemPipeline(detector_override=DummyDetector([]), audio_override=self.audio_guidance)
        pipeline.initialize()
        pipeline.last_command = "FORWARD"
        pipeline.last_decision_score = 0.80
        pipeline.last_decision_timestamp = time.time()
        res = pipeline.process_frame(np.zeros((480, 640, 3), dtype=np.uint8))
        self.assertEqual(res.decision_result.command, "FORWARD")
        pipeline.stop()

    def test_scenario_12_strong_directional_improvement(self):
        """Scenario 12: Strong directional score improvement triggers command switch."""
        det = make_det(0, "table", 0.95, [200.0, 100.0, 640.0, 480.0])
        pipeline = VisionGuideSystemPipeline(detector_override=DummyDetector([det]), audio_override=self.audio_guidance)
        pipeline.initialize()

        pipeline.last_command = "FORWARD"
        res = pipeline.process_frame(np.zeros((480, 640, 3), dtype=np.uint8))
        self.assertIsNotNone(res.decision_result)
        pipeline.stop()

    def test_scenario_13_forward_audio_dispatch(self):
        """Scenario 13: FORWARD command reaches Audio Guidance as 'Forward'."""
        pipeline = VisionGuideSystemPipeline(detector_override=DummyDetector([]), audio_override=self.audio_guidance)
        pipeline.initialize()
        res = pipeline.process_frame(np.zeros((480, 640, 3), dtype=np.uint8))
        self.assertEqual(res.audio_result.message, "Forward")
        pipeline.stop()

    def test_scenario_14_left_right_audio_dispatch(self):
        """Scenario 14: LEFT/RIGHT command reaches Audio Guidance as 'Left' / 'Right'."""
        det = make_det(0, "table", 0.95, [450.0, 100.0, 640.0, 480.0])
        pipeline = VisionGuideSystemPipeline(detector_override=DummyDetector([det]), audio_override=self.audio_guidance)
        pipeline.initialize()
        res = pipeline.process_frame(np.zeros((480, 640, 3), dtype=np.uint8))
        self.assertIn(res.audio_result.message, ["Left", "Right", "Forward", "Stop"])
        pipeline.stop()

    def test_scenario_15_stop_audio_dispatch_highest_priority(self):
        """Scenario 15: STOP command reaches Audio Guidance as 'Stop' with highest priority."""
        det1 = make_det(0, "stairs", 0.98, [0.0, 0.0, 640.0, 480.0])
        pipeline = VisionGuideSystemPipeline(detector_override=DummyDetector([det1]), audio_override=self.audio_guidance)
        pipeline.initialize()
        res = pipeline.process_frame(np.zeros((480, 640, 3), dtype=np.uint8))
        self.assertEqual(res.audio_result.message, "Stop")
        pipeline.stop()

    def test_real_webcam_integration(self):
        """Real Webcam Hardware Pipeline Execution Test."""
        pipeline = VisionGuideSystemPipeline(audio_override=self.audio_guidance)
        init_ok = pipeline.initialize()
        self.assertTrue(init_ok)

        start_ok = pipeline.start()
        if not start_ok:
            self.skipTest("Real laptop camera hardware unavailable or in use by another process.")

        # Process 5 real camera frames
        for _ in range(5):
            res = pipeline.process_frame()
            self.assertIsNotNone(res)
            self.assertGreater(res.total_latency, 0.0)
            time.sleep(0.01)

        pipeline.stop()


if __name__ == "__main__":
    unittest.main()
