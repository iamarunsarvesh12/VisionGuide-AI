import sys
import os
import time
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.decision_engine.models import DecisionInput, DecisionResult, NavigationCommand
from modules.decision_engine.engine import ContextAwareDecisionEngine
from modules.audio_guidance.guidance import OfflineAudioGuidance
from modules.audio_guidance.tts_engine import MockTTSEngine
from modules.audio_guidance.models import AudioResult


class TestAudioIntegration(unittest.TestCase):
    """
    Integrated Pipeline Test connecting Perception & Navigation Decision Engine (Module 09)
    to Offline Audio Guidance (Module 10).
    
    Verifies that end-to-end perception output decisions map deterministically to spoken audio messages:
      FORWARD -> "Forward"
      LEFT    -> "Left"
      RIGHT   -> "Right"
      STOP    -> "Stop"
    """

    def setUp(self):
        """Initialize Decision Engine and Audio Guidance modules."""
        self.decision_engine = ContextAwareDecisionEngine()
        self.decision_engine.initialize()

        self.mock_tts = MockTTSEngine()
        self.audio_guidance = OfflineAudioGuidance(tts_engine_override=self.mock_tts)
        self.audio_guidance.initialize()
        # Set zero repetition interval for predictable sequential testing
        self.audio_guidance.config.repetition_interval = {
            "FORWARD": 0.0,
            "LEFT": 0.0,
            "RIGHT": 0.0,
            "STOP": 0.0,
        }

    def tearDown(self):
        """Reset and shutdown modules."""
        self.decision_engine.reset()
        self.audio_guidance.close()

    def test_01_decision_to_audio_forward(self):
        """Test FORWARD decision converted to 'Forward' audio message."""
        dec = DecisionResult(
            command="FORWARD",
            selected_region="CENTER",
            confidence=0.90,
            decision_score=0.88,
            reason="Center region is clear",
            timestamp=time.time(),
        )
        audio_res = self.audio_guidance.speak_command(dec)
        self.assertTrue(audio_res.success)
        self.assertEqual(audio_res.command, "FORWARD")
        self.assertEqual(audio_res.message, "Forward")

    def test_02_decision_to_audio_left(self):
        """Test LEFT decision converted to 'Left' audio message."""
        dec = DecisionResult(
            command="LEFT",
            selected_region="LEFT",
            confidence=0.85,
            decision_score=0.80,
            reason="Left region has higher safe-space score",
            timestamp=time.time(),
        )
        audio_res = self.audio_guidance.speak_command(dec)
        self.assertTrue(audio_res.success)
        self.assertEqual(audio_res.command, "LEFT")
        self.assertEqual(audio_res.message, "Left")

    def test_03_decision_to_audio_right(self):
        """Test RIGHT decision converted to 'Right' audio message."""
        dec = DecisionResult(
            command="RIGHT",
            selected_region="RIGHT",
            confidence=0.86,
            decision_score=0.81,
            reason="Right region has higher safe-space score",
            timestamp=time.time(),
        )
        audio_res = self.audio_guidance.speak_command(dec)
        self.assertTrue(audio_res.success)
        self.assertEqual(audio_res.command, "RIGHT")
        self.assertEqual(audio_res.message, "Right")

    def test_04_decision_to_audio_stop(self):
        """Test STOP decision converted to 'Stop' audio message."""
        dec = DecisionResult(
            command="STOP",
            selected_region=None,
            confidence=0.95,
            decision_score=0.0,
            reason="Emergency stop triggered by critical hazard",
            timestamp=time.time(),
        )
        audio_res = self.audio_guidance.speak_command(dec)
        self.assertTrue(audio_res.success)
        self.assertEqual(audio_res.command, "STOP")
        self.assertEqual(audio_res.message, "Stop")

    def test_05_sequential_pipeline_integration(self):
        """Test sequence of decision engine outputs passed through audio guidance."""
        sequence = ["FORWARD", "LEFT", "RIGHT", "STOP"]
        expected_texts = ["Forward", "Left", "Right", "Stop"]

        for cmd, exp_text in zip(sequence, expected_texts):
            dec = DecisionResult(command=cmd, selected_region=cmd, confidence=0.88, decision_score=0.85, reason=f"Test {cmd}")
            audio_res = self.audio_guidance.speak_command(dec)
            self.assertTrue(audio_res.success)
            self.assertEqual(audio_res.message, exp_text)
            time.sleep(0.01)

        # Verify MockTTSEngine recorded exact spoken sequence
        self.assertEqual(self.mock_tts.spoken_messages, expected_texts)


if __name__ == "__main__":
    unittest.main()
