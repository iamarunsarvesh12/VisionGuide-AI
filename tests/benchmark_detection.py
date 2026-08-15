import time
import sys
import os
import psutil
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.camera_input.camera import WebcamInput
from modules.object_detection.detector import YOLOv8mDetector


def run_benchmark(num_frames: int = 30):
    """
    Empirical benchmark script for YOLOv8m object detection on CPU.
    Measures model load latency, per-frame inference latency, inference FPS,
    RAM usage, and min/max bounds.
    """
    print("=== STARTING YOLOV8M OBJECT DETECTION BENCHMARK ===")
    process = psutil.Process(os.getpid())
    ram_start_mb = process.memory_info().rss / (1024 * 1024)

    # 1. Measure Model Load Time
    t_load_start = time.perf_counter()
    detector = YOLOv8mDetector(model_path="yolov8m.pt", confidence_threshold=0.35, device="cpu")
    loaded = detector.load_model()
    t_load_end = time.perf_counter()
    model_load_ms = (t_load_end - t_load_start) * 1000.0

    if not loaded:
        print("[ERROR] Failed to load YOLOv8m model.")
        sys.exit(1)

    print(f"YOLOv8m Model Load Latency: {model_load_ms:.2f} ms")

    # 2. Acquire camera stream or synthetic frames
    camera = WebcamInput(camera_index=0, width=640, height=480)
    cam_active = camera.start()

    frames = []
    if cam_active:
        print("Capturing live camera frames for benchmark...")
        for _ in range(num_frames):
            ret, frame = camera.read()
            if ret and frame is not None:
                frames.append(frame)
        camera.stop()

    if len(frames) < num_frames:
        print("Supplementing with synthetic 640x480 frames...")
        needed = num_frames - len(frames)
        for _ in range(needed):
            syn_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            frames.append(syn_frame)

    # 3. Perform Inference Benchmark
    print(f"Benchmarking inference across {len(frames)} frames...")
    latencies_ms = []
    total_detections_count = 0

    # Warmup inference
    detector.detect(frames[0])

    t_bench_start = time.perf_counter()
    for frame in frames:
        t_f0 = time.perf_counter()
        dets = detector.detect(frame)
        t_f1 = time.perf_counter()
        latencies_ms.append((t_f1 - t_f0) * 1000.0)
        total_detections_count += len(dets)
    t_bench_end = time.perf_counter()

    total_bench_duration_s = t_bench_end - t_bench_start
    avg_inference_fps = len(frames) / total_bench_duration_s
    avg_latency_ms = sum(latencies_ms) / len(latencies_ms) if latencies_ms else 0.0
    min_latency_ms = min(latencies_ms) if latencies_ms else 0.0
    max_latency_ms = max(latencies_ms) if latencies_ms else 0.0

    ram_end_mb = process.memory_info().rss / (1024 * 1024)
    ram_diff_mb = ram_end_mb - ram_start_mb

    detector.release()

    print("\n=== YOLOV8M CPU INFERENCE PERFORMANCE RESULTS ===")
    print("Model Architecture: YOLOv8m (Medium)")
    print("Execution Backend: CPU (PyTorch 2.10.0+cpu)")
    print(f"Input Frame Resolution: 640x480")
    print(f"Model Loading Latency: {model_load_ms:.2f} ms")
    print(f"Total Frames Processed: {len(frames)}")
    print(f"Total Benchmark Duration: {total_bench_duration_s:.2f} s")
    print(f"Average Inference Latency: {avg_latency_ms:.2f} ms")
    print(f"Minimum Inference Latency: {min_latency_ms:.2f} ms")
    print(f"Maximum Inference Latency: {max_latency_ms:.2f} ms")
    print(f"Average Inference FPS: {avg_inference_fps:.2f} FPS")
    print(f"Total Objects Detected: {total_detections_count}")
    print(f"RAM Memory Consumption: {ram_end_mb:.2f} MB (Delta: +{ram_diff_mb:.2f} MB)")

    return {
        "model": "YOLOv8m",
        "device": "cpu",
        "model_load_ms": round(model_load_ms, 2),
        "avg_latency_ms": round(avg_latency_ms, 2),
        "min_latency_ms": round(min_latency_ms, 2),
        "max_latency_ms": round(max_latency_ms, 2),
        "fps": round(avg_inference_fps, 2),
        "ram_mb": round(ram_end_mb, 2),
    }


if __name__ == "__main__":
    run_benchmark(num_frames=20)
