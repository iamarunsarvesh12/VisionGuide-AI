import sys
import os
import time
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.audio_guidance.models import (
    NavigationAudioMessage,
    AudioResult,
    AudioConfiguration,
    AudioStatus,
)
from modules.audio_guidance.tts_engine import MockTTSEngine, Pyttsx3TTSEngine
from modules.audio_guidance.audio_output import AudioOutputDevice
from modules.audio_guidance.guidance import OfflineAudioGuidance
from modules.decision_engine.models import DecisionResult, NavigationCommand


class TestOfflineAudioGuidance(unittest.TestCase):
    """
    Deterministic Unit Test Suite for Module 10 — Offline Audio Guidance.
    Uses MockTTSEngine for 100% silent, fast, hardware-independent execution.
    """

    def setUp(self):
        """Set up OfflineAudioGuidance with MockTTSEngine before each test."""
        self.mock_tts = MockTTSEngine()
        self.guidance = OfflineAudioGuidance(tts_engine_override=self.mock_tts)
        self.guidance.initialize()

    def tearDown(self):
        """Shutdown OfflineAudioGuidance after each test."""
        self.guidance.close()

    def test_01_module_initialization(self):
        """1. Test Module Initialization."""
        self.assertTrue(self.guidance._is_initialized)
        self.assertTrue(self.guidance.is_available())

    def test_02_tts_availability(self):
        """2. Test TTS Availability."""
        self.assertTrue(self.mock_tts._is_initialized)
        self.assertEqual(self.mock_tts.volume, 1.0)
        self.assertEqual(self.mock_tts.speech_rate, 170)

    def test_03_voice_discovery(self):
        """3. Test Voice Discovery."""
        voices = self.mock_tts.get_voices()
        self.assertGreater(len(voices), 0)
        self.assertIn("name", voices[0])

    def test_04_left_command(self):
        """4. Test LEFT Command."""
        res = self.guidance.speak_command("LEFT")
        self.assertTrue(res.success)
        self.assertEqual(res.command, "LEFT")
        self.assertEqual(res.message, "Left")

    def test_05_right_command(self):
        """5. Test RIGHT Command."""
        res = self.guidance.speak_command("RIGHT")
        self.assertTrue(res.success)
        self.assertEqual(res.command, "RIGHT")
        self.assertEqual(res.message, "Right")

    def test_06_forward_command(self):
        """6. Test FORWARD Command."""
        res = self.guidance.speak_command("FORWARD")
        self.assertTrue(res.success)
        self.assertEqual(res.command, "FORWARD")
        self.assertEqual(res.message, "Forward")

    def test_07_stop_command(self):
        """7. Test STOP Command."""
        res = self.guidance.speak_command("STOP")
        self.assertTrue(res.success)
        self.assertEqual(res.command, "STOP")
        self.assertEqual(res.message, "Stop")

    def test_08_invalid_command(self):
        """8. Test Invalid Command Handling."""
        res = self.guidance.speak_command("INVALID_CMD")
        self.assertFalse(res.success)
        self.assertIn("Invalid command", str(res.error))

    def test_09_command_to_message_mapping(self):
        """9. Test Command-to-Message Mapping."""
        for cmd, text in [("LEFT", "Left"), ("RIGHT", "Right"), ("FORWARD", "Forward"), ("STOP", "Stop")]:
            res = self.guidance.speak_command(cmd)
            self.assertEqual(res.message, text)
            time.sleep(0.01)

    def test_10_priority_ordering(self):
        """10. Test Priority Ordering."""
        m_stop = NavigationAudioMessage("STOP", "Stop", 100, time.time())
        m_fwd = NavigationAudioMessage("FORWARD", "Forward", 50, time.time())
        self.assertGreater(m_stop.priority, m_fwd.priority)

    def test_11_stop_priority(self):
        """11. Test STOP Priority Override."""
        self.guidance.speak_command("FORWARD")
        res_stop = self.guidance.speak_command("STOP")
        self.assertTrue(res_stop.success)
        self.assertEqual(res_stop.command, "STOP")
        stats = self.guidance.get_statistics()
        self.assertGreater(stats["total_stop_overrides"], 0)

    def test_12_repetition_suppression(self):
        """12. Test Repetition Suppression."""
        res1 = self.guidance.speak_command("LEFT")
        self.assertTrue(res1.success)
        res2 = self.guidance.speak_command("LEFT")
        self.assertTrue(res2.success)
        self.assertIn("Suppressed", str(res2.error))

    def test_13_forward_repetition_interval(self):
        """13. Test Forward Repetition Interval."""
        self.guidance.config.repetition_interval["FORWARD"] = 0.1
        res1 = self.guidance.speak_command("FORWARD")
        self.assertTrue(res1.success)
        time.sleep(0.15)
        res2 = self.guidance.speak_command("FORWARD")
        self.assertIsNone(res2.error)

    def test_14_left_repetition_interval(self):
        """14. Test Left Repetition Interval."""
        self.guidance.config.repetition_interval["LEFT"] = 0.1
        res1 = self.guidance.speak_command("LEFT")
        self.assertTrue(res1.success)
        time.sleep(0.15)
        res2 = self.guidance.speak_command("LEFT")
        self.assertIsNone(res2.error)

    def test_15_right_repetition_interval(self):
        """15. Test Right Repetition Interval."""
        self.guidance.config.repetition_interval["RIGHT"] = 0.1
        res1 = self.guidance.speak_command("RIGHT")
        self.assertTrue(res1.success)
        time.sleep(0.15)
        res2 = self.guidance.speak_command("RIGHT")
        self.assertIsNone(res2.error)

    def test_16_stop_repetition_interval(self):
        """16. Test Stop Repetition Interval."""
        self.guidance.config.repetition_interval["STOP"] = 0.1
        res1 = self.guidance.speak_command("STOP")
        self.assertTrue(res1.success)
        time.sleep(0.15)
        res2 = self.guidance.speak_command("STOP")
        self.assertIsNone(res2.error)

    def test_17_immediate_command_change(self):
        """17. Test Immediate Command Change."""
        res1 = self.guidance.speak_command("FORWARD")
        self.assertTrue(res1.success)
        res2 = self.guidance.speak_command("LEFT")
        self.assertTrue(res2.success)
        self.assertIsNone(res2.error)

    def test_18_forward_to_left_transition(self):
        """18. Test FORWARD → LEFT Transition."""
        self.guidance.speak_command("FORWARD")
        res = self.guidance.speak_command("LEFT")
        self.assertEqual(res.command, "LEFT")
        self.assertEqual(res.message, "Left")

    def test_19_left_to_right_transition(self):
        """19. Test LEFT → RIGHT Transition."""
        self.guidance.speak_command("LEFT")
        res = self.guidance.speak_command("RIGHT")
        self.assertEqual(res.command, "RIGHT")
        self.assertEqual(res.message, "Right")

    def test_20_forward_to_stop_transition(self):
        """20. Test FORWARD → STOP Transition."""
        self.guidance.speak_command("FORWARD")
        res = self.guidance.speak_command("STOP")
        self.assertEqual(res.command, "STOP")
        self.assertEqual(res.message, "Stop")

    def test_21_stop_to_forward_transition(self):
        """21. Test STOP → FORWARD Transition."""
        self.guidance.speak_command("STOP")
        res = self.guidance.speak_command("FORWARD")
        self.assertEqual(res.command, "FORWARD")
        self.assertEqual(res.message, "Forward")

    def test_22_tts_failure_handling(self):
        """22. Test TTS Failure Handling."""
        class FailingTTS(MockTTSEngine):
            def speak(self, text: str) -> bool:
                raise RuntimeError("Hardware audio error")

        fail_g = OfflineAudioGuidance(tts_engine_override=FailingTTS())
        fail_g.initialize()
        res = fail_g.speak_command("STOP")
        self.assertTrue(res.success)
        fail_g.close()

    def test_23_audio_output_failure_handling(self):
        """23. Test Audio Output Failure Handling."""
        out = AudioOutputDevice()
        out.is_available = False
        out.error_message = "Bluetooth output disconnected"
        self.assertFalse(out.is_available)

    def test_24_statistics(self):
        """24. Test Telemetry & Statistics."""
        self.guidance.speak_command("FORWARD")
        stats = self.guidance.get_statistics()
        self.assertIn("total_commands_received", stats)
        self.assertIn("total_commands_spoken", stats)

    def test_25_reset(self):
        """25. Test Reset behavior."""
        self.guidance.speak_command("LEFT")
        self.guidance.reset()
        stats = self.guidance.get_statistics()
        self.assertEqual(stats["total_commands_received"], 0)

    def test_26_last_audio_result(self):
        """26. Test Last Audio Result."""
        res1 = self.guidance.speak_command("LEFT")
        res2 = self.guidance.get_last_audio()
        self.assertIsNotNone(res2)
        self.assertEqual(res1.command, res2.command)

    def test_27_disabled_audio_mode(self):
        """27. Test Disabled Audio Mode."""
        self.guidance.config.enabled = False
        res = self.guidance.speak_command("FORWARD")
        self.assertIsNotNone(res)

    def test_28_configuration_loading(self):
        """28. Test Configuration Loading."""
        cfg = AudioConfiguration(speech_rate=180, volume=0.8)
        self.assertEqual(cfg.speech_rate, 180)
        self.assertEqual(cfg.volume, 0.8)

    def test_29_bluetooth_default_output(self):
        """29. Test Bluetooth/Default Output Handling."""
        status = self.guidance.audio_output.get_status()
        self.assertIn("selected_output_device", status)
        self.assertIn("is_bluetooth", status)

    def test_30_safety_fallback(self):
        """30. Test Safety Fallback with DecisionResult input."""
        dec = DecisionResult("STOP", None, 0.95, 0.0, "Critical danger ahead")
        res = self.guidance.speak_command(dec)
        self.assertTrue(res.success)
        self.assertEqual(res.command, "STOP")
        self.assertEqual(res.message, "Stop")


if __name__ == "__main__":
    unittest.main()
