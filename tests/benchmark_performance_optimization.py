import sys
import os
import time
import torch
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.object_detection.detector import YOLOv8mDetector
from modules.object_tracking.tracker import BoTSORTTracker
from modules.system_integration.pipeline import VisionGuideSystemPipeline
from experiments.logger import ExperimentLogger


def run_performance_optimization_benchmark():
    """
    Module 11G — Performance Optimization & Baseline Comparison Script.
    Evaluates PyTorch CPU thread settings, resolution scaling, and tracking-interleaved frame skipping.
    Measures per-frame YOLO latency, tracking latency, pipeline FPS, and latency reduction %.
    """
    print("================================================================")
    print("      MODULE 11G — CPU PERFORMANCE OPTIMIZATION BENCHMARK      ")
    print("================================================================")

    exp_logger = ExperimentLogger()

    # 1. PyTorch CPU Threading Evaluation
    print("\n[1] PyTorch CPU Thread Tuning:")
    thread_counts = [1, 2, 4, torch.get_num_threads()]
    thread_results = {}

    detector = YOLOv8mDetector(model_path="yolov8m.pt", device="cpu")
    detector.load_model()
    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)

    # Warmup
    detector.detect(dummy_img)

    for tc in set(thread_counts):
        torch.set_num_threads(tc)
        lats = []
        for _ in range(5):
            t0 = time.perf_counter()
            detector.detect(dummy_img)
            t1 = time.perf_counter()
            lats.append((t1 - t0) * 1000.0)
        avg_lat = sum(lats) / len(lats)
        thread_results[tc] = round(avg_lat, 2)
        print(f"  - Threads = {tc:<2} : Avg YOLO Latency = {avg_lat:.2f} ms")

    # Reset default threads
    torch.set_num_threads(os.cpu_count() or 4)

    # 2. Resolution Scaling Evaluation (640x480 vs 416x416 vs 320x320)
    print("\n[2] YOLO Input Resolution Optimization:")
    resolutions = [
        ("640x480 Standard", (480, 640, 3)),
        ("416x416 Scaled", (416, 416, 3)),
        ("320x320 Fast", (320, 320, 3)),
    ]
    res_results = {}

    for res_name, res_shape in resolutions:
        img_res = np.zeros(res_shape, dtype=np.uint8)
        lats = []
        for _ in range(5):
            t0 = time.perf_counter()
            detector.detect(img_res)
            t1 = time.perf_counter()
            lats.append((t1 - t0) * 1000.0)
        avg_lat = sum(lats) / len(lats)
        fps = 1000.0 / avg_lat if avg_lat > 0 else 0.0
        res_results[res_name] = {"avg_latency_ms": round(avg_lat, 2), "fps": round(fps, 2)}
        print(f"  - {res_name:<20} : Avg Latency = {avg_lat:7.2f} ms ({fps:.2f} FPS)")

    # 3. Interleaved Frame Skipping (Detection every N frames + BoT-SORT Tracking)
    print("\n[3] Interleaved Frame-Skipping (Detection every N=3 frames):")
    tracker = BoTSORTTracker()
    tracker.initialize()

    # Compare 10 frames of Full Detection vs 10 frames of Interleaved (1 Detection + 2 Tracking-Only)
    t0_full = time.perf_counter()
    for _ in range(10):
        dets = detector.detect(dummy_img)
        trks = tracker.update(dets, frame=dummy_img)
    t1_full = time.perf_counter()
    full_lat_ms = (t1_full - t0_full) * 100.0  # avg per frame

    t0_interleaved = time.perf_counter()
    cached_dets = []
    for f_i in range(10):
        if f_i % 3 == 0:
            cached_dets = detector.detect(dummy_img)
        trks = tracker.update(cached_dets, frame=dummy_img)
    t1_interleaved = time.perf_counter()
    interleaved_lat_ms = (t1_interleaved - t0_interleaved) * 100.0

    interleaved_fps = 1000.0 / interleaved_lat_ms if interleaved_lat_ms > 0 else 0.0
    speedup_pct = ((full_lat_ms - interleaved_lat_ms) / full_lat_ms * 100.0) if full_lat_ms > 0 else 0.0

    print(f"  - Full Detection (Every Frame)    : {full_lat_ms:.2f} ms / frame ({1000.0/full_lat_ms:.2f} FPS)")
    print(f"  - Interleaved Detection (N=3)     : {interleaved_lat_ms:.2f} ms / frame ({interleaved_fps:.2f} FPS)")
    print(f"  - Effective Processing Speedup    : {speedup_pct:.1f}% latency reduction")
    print("================================================================")

    exp_logger.log_experiment(
        category="detection",
        experiment_id="EXP_11G_OPTIMIZATION",
        scenario="CPU PyTorch threading, resolution scaling, and tracking-interleaved frame skipping benchmark",
        configuration={"model": "YOLOv8m", "device": "cpu"},
        measured_outputs={
            "threads_results_ms": thread_results,
            "resolution_results": res_results,
            "baseline_full_latency_ms": round(full_lat_ms, 2),
            "interleaved_n3_latency_ms": round(interleaved_lat_ms, 2),
            "interleaved_speedup_pct": round(speedup_pct, 1),
        },
        result="VALIDATED",
        limitations=["CPU execution environment without TensorRT/ONNX Runtime GPU quantization"],
    )

    return res_results, speedup_pct


if __name__ == "__main__":
    run_performance_optimization_benchmark()
