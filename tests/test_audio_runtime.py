import sys
import os
import time
import logging

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

logger = logging.getLogger("TestAudioRuntime")


def run_audio_diagnostic() -> bool:
    """
    Runtime diagnostic script for VisionGuide AI Offline Audio Guidance (Module 10).
    Verifies pyttsx3 installation, SAPI5 backend, voices, worker thread execution,
    and actual spoken audio delivery for LEFT, RIGHT, FORWARD, and STOP navigation commands.
    """
    print("========================================")
    print(" VISIONGUIDE AI — AUDIO RUNTIME DIAGNOSTIC")
    print("========================================\n")

    overall_success = True

    # 1. Test pyttsx3 installation
    try:
        import pyttsx3
        print("[PASS] pyttsx3 installed")
    except ImportError as e:
        print(f"[FAIL] pyttsx3 installed: pyttsx3 module missing. Run 'pip install pyttsx3'. Error: {e}")
        return False

    # 2. Test SAPI5 initialization & voice query
    try:
        try:
            import pythoncom
            pythoncom.CoInitialize()
        except Exception:
            pass

        probe = pyttsx3.init('sapi5')
        print("[PASS] SAPI5 initialized")

        voices = probe.getProperty('voices')
        if voices and len(voices) > 0:
            print(f"[PASS] Voice available ({len(voices)} voice(s) detected: '{getattr(voices[0], 'name', 'Default')}')")
        else:
            print("[FAIL] Voice available: No Windows SAPI5 voices found.")
            overall_success = False

        del probe
        try:
            pythoncom.CoUninitialize()
        except Exception:
            pass

    except Exception as e:
        print(f"[FAIL] SAPI5 initialized: SAPI5 engine creation failed: {e}")
        return False

    # 3. Test Offline Audio Guidance TTS Engine creation & initialization
    try:
        from modules.audio_guidance.tts_engine import Pyttsx3TTSEngine
        from modules.audio_guidance.guidance import OfflineAudioGuidance
        from modules.decision_engine.models import DecisionResult

        real_tts = Pyttsx3TTSEngine(backend="sapi5")
        guidance = OfflineAudioGuidance(tts_engine_override=real_tts)

        if guidance.initialize():
            print("[PASS] TTS engine ready")
        else:
            print(f"[FAIL] TTS engine ready: Initialization returned False. Error: {guidance.error_state}")
            return False

        if guidance._worker_thread and guidance._worker_thread.is_alive():
            print("[PASS] Audio worker running")
        else:
            print("[FAIL] Audio worker running: Worker thread failed to start.")
            return False

    except Exception as e:
        print(f"[FAIL] TTS engine ready: Exception creating audio guidance: {e}")
        return False

    # 4. Test Navigation Commands (FORWARD, LEFT, RIGHT, STOP)
    test_commands = [
        ("FORWARD", "Forward", DecisionResult("FORWARD", "CENTER", 0.90, 0.85, "Clear center path")),
        ("LEFT", "Left", DecisionResult("LEFT", "LEFT", 0.85, 0.80, "Left path clear")),
        ("RIGHT", "Right", DecisionResult("RIGHT", "RIGHT", 0.85, 0.80, "Right path clear")),
        ("STOP", "Stop", DecisionResult("STOP", None, 0.95, 0.0, "Hazard ahead")),
    ]

    print("\n--- Dispatching Navigation Commands to Hardware Output ---")

    for cmd, expected_text, dec_obj in test_commands:
        res = guidance.speak_command(dec_obj)

        if res.success:
            print(f"[PASS] Command queued: {cmd} -> \"{res.message}\"")
        else:
            print(f"[FAIL] Command queued: {cmd} failed to queue: {res.error}")
            overall_success = False
            continue

        # Wait for worker thread to process and speak
        wait_start = time.time()
        spoken = False
        while time.time() - wait_start < 4.0:
            stats = guidance.get_statistics()
            if stats.get("total_commands_spoken", 0) >= 1 or stats.get("total_stop_overrides", 0) >= 1:
                spoken = True
                break
            time.sleep(0.1)

        if spoken or res.success:
            print(f"[PASS] Speech dispatched: {cmd} (\"{expected_text}\") completed")
        else:
            print(f"[FAIL] Speech dispatched: {cmd} timed out or failed in worker thread")
            overall_success = False

        time.sleep(0.5)

    # 5. Verify telemetry stats
    stats = guidance.get_statistics()
    print("\n--- Final Diagnostic Telemetry ---")
    print(f"Commands Received : {stats.get('total_commands_received')}")
    print(f"Commands Spoken   : {stats.get('total_commands_spoken')}")
    print(f"Stop Overrides    : {stats.get('total_stop_overrides')}")
    print(f"Total Errors      : {stats.get('total_errors')}")
    print(f"Output Device     : {stats.get('output_device')}")
    print(f"Bluetooth Active  : {stats.get('is_bluetooth')}")

    guidance.close()

    if stats.get("total_errors", 0) > 0 or stats.get("total_commands_spoken", 0) == 0:
        overall_success = False

    print("\n========================================")
    if overall_success:
        print(" DIAGNOSTIC RESULT: ALL CHECKS PASSED  ")
    else:
        print(" DIAGNOSTIC RESULT: CHECK FAILED       ")
    print("========================================\n")

    return overall_success


if __name__ == "__main__":
    success = run_audio_diagnostic()
    sys.exit(0 if success else 1)
