import time
import sys
import os
import psutil

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.audio_guidance.models import AudioConfiguration, NavigationAudioMessage
from modules.audio_guidance.tts_engine import MockTTSEngine, Pyttsx3TTSEngine
from modules.audio_guidance.guidance import OfflineAudioGuidance
from modules.decision_engine.models import DecisionResult, NavigationCommand


def run_benchmark(num_commands: int = 100):
    """
    Empirical benchmark script for Module 10 — Offline Audio Guidance.
    Measures TTS initialization, command processing, message generation,
    mock TTS, actual SAPI5 TTS (if available), memory, and suppression performance.
    """
    print("================================================================")
    print("      VISIONGUIDE AI — MODULE 10 AUDIO GUIDANCE BENCHMARK       ")
    print("================================================================")
    process = psutil.Process(os.getpid())
    ram_start_mb = process.memory_info().rss / (1024 * 1024)

    # 1. TTS Initialization Latency
    t_init_0 = time.perf_counter()
    mock_tts = MockTTSEngine(simulate_latency_ms=1.0)
    guidance = OfflineAudioGuidance(tts_engine_override=mock_tts)
    guidance.initialize()
    t_init_1 = time.perf_counter()
    tts_init_latency_ms = (t_init_1 - t_init_0) * 1000.0
    print(f"OfflineAudioGuidance & Mock TTS Initialization Latency: {tts_init_latency_ms:.3f} ms")

    # 2. Command Processing & Message Generation Latency
    print(f"\nBenchmarking Command Processing across {num_commands} decisions...")
    cmds = ["LEFT", "RIGHT", "FORWARD", "STOP"]
    proc_lats = []
    for i in range(num_commands):
        c = cmds[i % len(cmds)]
        t_p0 = time.perf_counter()
        res = guidance.speak_command(c)
        t_p1 = time.perf_counter()
        proc_lats.append((t_p1 - t_p0) * 1000.0)
        time.sleep(0.005)

    avg_proc_lat_ms = sum(proc_lats) / len(proc_lats)
    print(f"Average Command Processing Latency: {avg_proc_lat_ms:.5f} ms")

    # 3. Message Generation Micro-Benchmark
    print(f"\nBenchmarking Micro NavigationAudioMessage Generation...")
    msg_lats = []
    for i in range(num_commands):
        t_m0 = time.perf_counter()
        msg = NavigationAudioMessage("LEFT", "Left", 80, time.time(), False, "Safe left corridor")
        t_m1 = time.perf_counter()
        msg_lats.append((t_m1 - t_m0) * 1000.0)

    avg_msg_lat_ms = sum(msg_lats) / len(msg_lats)
    print(f"Average NavigationAudioMessage Generation Latency: {avg_msg_lat_ms:.5f} ms")

    # Wait for queue drain
    time.sleep(0.5)
    stats = guidance.get_statistics()
    suppression_rate = (stats["total_suppressed"] / stats["total_commands_received"]) * 100.0 if stats["total_commands_received"] else 0.0
    print(f"\nCommands Received   : {stats['total_commands_received']}")
    print(f"Commands Spoken     : {stats['total_commands_spoken']}")
    print(f"Commands Suppressed : {stats['total_suppressed']} ({suppression_rate:.1f}%)")
    print(f"STOP Overrides      : {stats['total_stop_overrides']}")

    # 4. Actual SAPI5 Pyttsx3 TTS Initialization Latency
    print("\nMeasuring Actual Windows SAPI5 pyttsx3 Initialization Latency...")
    t_sapi_0 = time.perf_counter()
    real_tts = Pyttsx3TTSEngine(backend="sapi5")
    sapi_ok = real_tts.initialize(volume=1.0, speech_rate=170)
    t_sapi_1 = time.perf_counter()
    sapi_init_ms = (t_sapi_1 - t_sapi_0) * 1000.0
    print(f"Actual SAPI5 pyttsx3 Init Latency: {sapi_init_ms:.2f} ms (Success: {sapi_ok})")
    real_tts.close()

    # 5. Decision -> Audio Guidance Integrated Dispatch Overhead
    print("\nMeasuring Integrated Decision -> Audio Guidance Dispatch Overhead...")
    fast_mock = MockTTSEngine(simulate_latency_ms=0.0)
    integ_guidance = OfflineAudioGuidance(tts_engine_override=fast_mock)
    integ_guidance.initialize()

    dummy_decisions = [
        DecisionResult("FORWARD", "CENTER", 0.90, 0.85, "Center clear"),
        DecisionResult("LEFT", "LEFT", 0.85, 0.75, "Center blocked; left clear"),
        DecisionResult("RIGHT", "RIGHT", 0.88, 0.78, "Center blocked; right clear"),
        DecisionResult("STOP", None, 0.95, 0.00, "Emergency stop"),
    ]

    integ_lats = []
    for d in dummy_decisions:
        t_d0 = time.perf_counter()
        audio_res = integ_guidance.speak_command(d)
        t_d1 = time.perf_counter()
        integ_lats.append((t_d1 - t_d0) * 1000.0)

    avg_integ_lat_ms = sum(integ_lats) / len(integ_lats)
    print(f"Average Integrated DecisionResult-to-Audio Dispatch Latency: {avg_integ_lat_ms:.5f} ms")

    guidance.close()
    integ_guidance.close()

    ram_end_mb = process.memory_info().rss / (1024 * 1024)
    print(f"\nTotal System Memory (RAM) Consumption: {ram_end_mb:.2f} MB")
    print("========================================================\n")

    return {
        "init_latency_ms": round(tts_init_latency_ms, 3),
        "command_processing_latency_ms": round(avg_proc_lat_ms, 5),
        "message_gen_latency_ms": round(avg_msg_lat_ms, 5),
        "sapi5_init_latency_ms": round(sapi_init_ms, 2),
        "integrated_dispatch_ms": round(avg_integ_lat_ms, 5),
        "suppression_rate_pct": round(suppression_rate, 1),
        "ram_mb": round(ram_end_mb, 2),
    }


if __name__ == "__main__":
    run_benchmark(num_commands=100)
