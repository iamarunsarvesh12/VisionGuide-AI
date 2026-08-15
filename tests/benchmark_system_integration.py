import time
import sys
import os
import psutil
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.system_integration.pipeline import VisionGuideSystemPipeline
from modules.audio_guidance.tts_engine import MockTTSEngine
from modules.audio_guidance.guidance import OfflineAudioGuidance


def run_benchmark(num_frames: int = 50):
    """
    Empirical End-to-End System Benchmark for VisionGuide AI (Phase 10).
    Measures per-module latencies, total frame processing latency, end-to-end FPS,
    RAM memory consumption, CPU utilization, and identifies the primary system bottleneck.
    """
    print("================================================================")
    print("       VISIONGUIDE AI — PHASE 10 SYSTEM INTEGRATION BENCHMARK   ")
    print("================================================================")

    process = psutil.Process(os.getpid())
    ram_start_mb = process.memory_info().rss / (1024 * 1024)

    mock_tts = MockTTSEngine(simulate_latency_ms=0.0)
    audio_guidance = OfflineAudioGuidance(tts_engine_override=mock_tts)

    pipeline = VisionGuideSystemPipeline(audio_override=audio_guidance)

    print("\nInitializing complete 10-module pipeline...")
    t0_init = time.perf_counter()
    if not pipeline.initialize():
        print(f"[ERROR] Pipeline initialization failed: {pipeline.error_message}")
        return
    t1_init = time.perf_counter()
    init_latency_ms = (t1_init - t0_init) * 1000.0
    print(f"Pipeline Initialization Latency: {init_latency_ms:.2f} ms")

    print(f"\nProcessing {num_frames} frames through full end-to-end pipeline...")
    synthetic_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    per_module_latencies = {
        "camera": [],
        "yolo": [],
        "tracking": [],
        "phmu": [],
        "distance": [],
        "danger": [],
        "free_space": [],
        "decision": [],
        "audio": [],
    }
    total_latencies = []

    t_start_bench = time.perf_counter()

    for i in range(num_frames):
        res = pipeline.process_frame(synthetic_frame)
        total_latencies.append(res.total_latency)
        for mod, lat in res.module_latencies.items():
            if mod in per_module_latencies:
                per_module_latencies[mod].append(lat)

    t_end_bench = time.perf_counter()

    pipeline.stop()
    ram_end_mb = process.memory_info().rss / (1024 * 1024)
    cpu_percent = psutil.cpu_percent(interval=0.1)

    avg_total_lat_ms = sum(total_latencies) / len(total_latencies) if total_latencies else 0.0
    fps = 1000.0 / avg_total_lat_ms if avg_total_lat_ms > 0 else 0.0

    print("\n================================================================")
    print("                    PER-MODULE LATENCY BREAKDOWN                ")
    print("================================================================")
    avg_mod_lats = {}
    for mod, lats in per_module_latencies.items():
        avg_l = sum(lats) / len(lats) if lats else 0.0
        avg_mod_lats[mod] = avg_l
        pct = (avg_l / avg_total_lat_ms) * 100.0 if avg_total_lat_ms > 0 else 0.0
        print(f"  - {mod.upper():<12}: {avg_l:7.2f} ms  ({pct:5.1f}%)")

    print("----------------------------------------------------------------")
    print(f"  TOTAL END-TO-END LATENCY : {avg_total_lat_ms:.2f} ms")
    print(f"  SYSTEM PIPELINE FPS       : {fps:.2f} FPS")
    print(f"  SYSTEM RAM CONSUMPTION    : {ram_end_mb:.2f} MB")
    print(f"  SYSTEM CPU UTILIZATION    : {cpu_percent:.1f}%")
    print("----------------------------------------------------------------")

    # Bottleneck identification
    dominant_mod = max(avg_mod_lats.items(), key=lambda x: x[1]) if avg_mod_lats else ("NONE", 0.0)
    print(f"\nPRIMARY SYSTEM BOTTLENECK : {dominant_mod[0].upper()} ({dominant_mod[1]:.2f} ms per frame)")
    print("========================================================\n")

    return {
        "init_latency_ms": round(init_latency_ms, 2),
        "avg_total_latency_ms": round(avg_total_lat_ms, 2),
        "fps": round(fps, 2),
        "ram_mb": round(ram_end_mb, 2),
        "cpu_percent": round(cpu_percent, 1),
        "dominant_bottleneck": dominant_mod[0].upper(),
        "module_latencies_ms": {k: round(v, 2) for k, v in avg_mod_lats.items()},
    }


if __name__ == "__main__":
    run_benchmark(num_frames=50)
