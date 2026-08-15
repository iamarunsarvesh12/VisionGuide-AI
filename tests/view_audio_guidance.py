import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.audio_guidance.guidance import OfflineAudioGuidance
from modules.audio_guidance.tts_engine import Pyttsx3TTSEngine
from modules.decision_engine.models import DecisionResult


def main():
    """
    Live Visual Interactive Audio Guidance Test Script for VisionGuide AI.
    Terminal-based control interface allowing interactive verification of offline SAPI5 speech delivery.
    """
    print("Initializing Offline Audio Guidance with Real SAPI5 Pyttsx3 Engine...")
    real_tts = Pyttsx3TTSEngine(backend="sapi5")
    guidance = OfflineAudioGuidance(tts_engine_override=real_tts)

    if not guidance.initialize():
        print("[ERROR] Failed to initialize Audio Guidance with SAPI5 Pyttsx3.")
        return

    curr_cmd = "FORWARD"
    curr_conf = 0.90
    curr_msg = "Forward"

    try:
        while True:
            os.system('cls' if os.name == 'nt' else 'clear')
            stats = guidance.get_statistics()
            out_device = stats.get("output_device", "Default Audio Output")
            audio_status = "ACTIVE / SPEAKING" if stats.get("is_speaking") else "READY"

            print("========================================")
            print("VISIONGUIDE AI — AUDIO GUIDANCE TEST")
            print("========================================")
            print(f"Current Command : {curr_cmd}")
            print(f"Confidence      : {curr_conf:.2f}")
            print(f"Audio Message   : \"{curr_msg}\"")
            print(f"TTS Engine      : SAPI5")
            print(f"Output Device   : {out_device}")
            print(f"Audio Status    : {audio_status}")
            print("\nPress:\n")
            print("L = LEFT")
            print("R = RIGHT")
            print("F = FORWARD")
            print("S = STOP")
            print("Q = QUIT")
            print("========================================\n")

            key = input("Enter choice (L/R/F/S/Q): ").strip().upper()

            if key == 'Q':
                print("Exiting Audio Guidance Visual Test...")
                break
            elif key == 'L':
                curr_cmd = "LEFT"
                curr_conf = 0.86
                dec = DecisionResult("LEFT", "LEFT", curr_conf, 0.82, "Left region has higher safe space")
                res = guidance.speak_command(dec)
                curr_msg = res.message if res.message else "Left"
            elif key == 'R':
                curr_cmd = "RIGHT"
                curr_conf = 0.84
                dec = DecisionResult("RIGHT", "RIGHT", curr_conf, 0.80, "Right region clear")
                res = guidance.speak_command(dec)
                curr_msg = res.message if res.message else "Right"
            elif key == 'F':
                curr_cmd = "FORWARD"
                curr_conf = 0.92
                dec = DecisionResult("FORWARD", "CENTER", curr_conf, 0.88, "Center region clear")
                res = guidance.speak_command(dec)
                curr_msg = res.message if res.message else "Forward"
            elif key == 'S':
                curr_cmd = "STOP"
                curr_conf = 0.98
                dec = DecisionResult("STOP", None, curr_conf, 0.0, "Hazard blocking path")
                res = guidance.speak_command(dec)
                curr_msg = res.message if res.message else "Stop"
            else:
                print("Invalid input. Press L, R, F, S, or Q.")
                time.sleep(0.5)

    finally:
        guidance.close()
        print("Audio Guidance Visual Test terminated cleanly.")


if __name__ == "__main__":
    main()
