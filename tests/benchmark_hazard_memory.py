import time
import sys
import os
import psutil
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.camera_input.camera import WebcamInput
from modules.object_detection.detector import YOLOv8mDetector
from modules.object_tracking.tracker import BoTSORTTracker
from modules.hazard_memory.memory import PersistentHazardMemory
from modules.object_tracking.interface import Track


def run_benchmark(num_hazards: int = 100, num_frames: int = 20):
    """
    Empirical benchmark script for Module 05 — Persistent Hazard Memory Unit (PHMU).
    Measures update latency, memory creation latency, lookup latency, expiration latency,
    RAM usage, and pipeline integration latency.
    """
    print("=== STARTING PHMU HAZARD MEMORY BENCHMARK ===")
    process = psutil.Process(os.getpid())
    ram_start_mb = process.memory_info().rss / (1024 * 1024)

    phmu = PersistentHazardMemory(memory_timeout_seconds=3.0, decay_rate=0.2)
    phmu.initialize()

    # 1. Micro-Benchmark: Memory Creation Latency
    print(f"\nBenchmarking Memory Creation for {num_hazards} synthetic hazards...")
    synthetic_tracks = []
    for i in range(1, num_hazards + 1):
        trk = Track(
            track_id=i, class_id=i % 10, class_name=f"class_{i%10}", confidence=0.85,
            bounding_box=[float(i), float(i), float(i+50), float(i+50)],
            center_x=float(i+25), center_y=float(i+25), width=50.0, height=50.0,
            tracking_state="NEW"
        )
        synthetic_tracks.append(trk)

    t_create_0 = time.perf_counter()
    phmu.update(synthetic_tracks, current_time=1000.0)
    t_create_1 = time.perf_counter()

    creation_latency_ms = (t_create_1 - t_create_0) * 1000.0
    avg_creation_per_hazard_ms = creation_latency_ms / num_hazards

    print(f"Total Creation Latency ({num_hazards} hazards): {creation_latency_ms:.3f} ms")
    print(f"Average Creation Latency Per Hazard: {avg_creation_per_hazard_ms:.4f} ms")

    # 2. Micro-Benchmark: Lookup & Update Latency
    print("Benchmarking Memory Lookup & State Update Latency over 1000 iterations...")
    t_lookup_0 = time.perf_counter()
    for _ in range(1000):
        _ = phmu.get_hazard(50)
    t_lookup_1 = time.perf_counter()
    avg_lookup_ms = ((t_lookup_1 - t_lookup_0) * 1000.0) / 1000.0

    t_up_0 = time.perf_counter()
    for step in range(100):
        # Update with subset of tracks (simulating partial occlusion)
        subset = synthetic_tracks[:50]
        phmu.update(subset, current_time=1000.0 + (step * 0.1))
    t_up_1 = time.perf_counter()

    avg_update_ms = ((t_up_1 - t_up_0) * 1000.0) / 100.0

    # 3. Micro-Benchmark: Expiration Processing Latency
    print("Benchmarking Memory Expiration Latency...")
    t_exp_0 = time.perf_counter()
    expired = phmu.expire_memories(current_time=1005.0)  # Jump past 3.0s timeout
    t_exp_1 = time.perf_counter()

    expiration_latency_ms = (t_exp_1 - t_exp_0) * 1000.0
    print(f"Expiration Latency (Purged {len(expired)} hazards): {expiration_latency_ms:.3f} ms")

    phmu.clear()

    # 4. Pipeline Integration Benchmark (Camera + YOLO + BoT-SORT + PHMU)
    print(f"\nExecuting Full Pipeline (Camera + YOLO + BoT-SORT + PHMU) across {num_frames} frames...")
    camera = WebcamInput(camera_index=0, width=640, height=480)
    detector = YOLOv8mDetector(model_path="yolov8m.pt", confidence_threshold=0.35, device="cpu")
    tracker = BoTSORTTracker(iou_threshold=0.3, max_age=15)

    cam_ok = camera.start()
    det_ok = detector.load_model()
    trk_ok = tracker.initialize()
    phmu_ok = phmu.initialize()

    if not (cam_ok and det_ok and trk_ok and phmu_ok):
        print("[ERROR] Pipeline initialization failed.")
        if cam_ok: camera.stop()
        if det_ok: detector.release()
        sys.exit(1)

    cam_lats, det_lats, trk_lats, phmu_lats, e2e_lats = [], [], [], [], []
    t_pipeline_start = time.perf_counter()

    for i in range(num_frames):
        t_e2e_0 = time.perf_counter()

        # Step 1: Camera
        t_c0 = time.perf_counter()
        ret, frame = camera.read()
        t_c1 = time.perf_counter()

        if not ret or frame is None:
            continue

        # Step 2: YOLO Detection
        t_d0 = time.perf_counter()
        detections = detector.detect(frame)
        t_d1 = time.perf_counter()

        # Step 3: BoT-SORT Tracking
        t_t0 = time.perf_counter()
        tracks = tracker.update(detections, frame)
        t_t1 = time.perf_counter()

        # Step 4: PHMU Memory Update
        t_p0 = time.perf_counter()
        active_hazards = phmu.update(tracks, current_time=time.time(), frame_index=i+1)
        t_p1 = time.perf_counter()

        t_e2e_1 = time.perf_counter()

        cam_lats.append((t_c1 - t_c0) * 1000.0)
        det_lats.append((t_d1 - t_d0) * 1000.0)
        trk_lats.append((t_t1 - t_t0) * 1000.0)
        phmu_lats.append((t_p1 - t_p0) * 1000.0)
        e2e_lats.append((t_e2e_1 - t_e2e_0) * 1000.0)

    t_pipeline_end = time.perf_counter()

    camera.stop()
    detector.release()
    tracker.reset()
    phmu.clear()

    total_pipeline_time_s = t_pipeline_end - t_pipeline_start
    frames_processed = len(e2e_lats)

    avg_cam_ms = sum(cam_lats) / frames_processed if frames_processed else 0
    avg_det_ms = sum(det_lats) / frames_processed if frames_processed else 0
    avg_trk_ms = sum(trk_lats) / frames_processed if frames_processed else 0
    avg_phmu_ms = sum(phmu_lats) / frames_processed if frames_processed else 0
    avg_e2e_ms = sum(e2e_lats) / frames_processed if frames_processed else 0

    e2e_fps = frames_processed / total_pipeline_time_s if total_pipeline_time_s > 0 else 0

    ram_end_mb = process.memory_info().rss / (1024 * 1024)

    print("\n=== PHMU & PIPELINE BENCHMARK RESULTS ===")
    print(f"Max Active Hazards Tested: {num_hazards}")
    print(f"Average PHMU Update Latency: {avg_update_ms:.4f} ms")
    print(f"Average Memory Creation Latency: {avg_creation_per_hazard_ms:.4f} ms")
    print(f"Average Memory Lookup Latency: {avg_lookup_ms:.5f} ms")
    print(f"Expiration Processing Latency: {expiration_latency_ms:.3f} ms")

    print("\n--- Pipeline Latency Breakdown (Per Frame Average) ---")
    print(f"  1. Camera Input Latency : {avg_cam_ms:.2f} ms")
    print(f"  2. YOLOv8m CPU Inference: {avg_det_ms:.2f} ms")
    print(f"  3. BoT-SORT Tracking    : {avg_trk_ms:.2f} ms")
    print(f"  4. PHMU Memory Update   : {avg_phmu_ms:.3f} ms")
    print(f"  Total End-to-End Latency: {avg_e2e_ms:.2f} ms")

    print(f"\nEnd-to-End Pipeline FPS: {e2e_fps:.2f} FPS")
    print(f"RAM Memory Consumption: {ram_end_mb:.2f} MB")

    return {
        "phmu_update_ms": round(avg_update_ms, 4),
        "creation_ms": round(avg_creation_per_hazard_ms, 4),
        "lookup_ms": round(avg_lookup_ms, 5),
        "expiration_ms": round(expiration_latency_ms, 3),
        "phmu_pipeline_ms": round(avg_phmu_ms, 3),
        "e2e_pipeline_ms": round(avg_e2e_ms, 2),
        "e2e_fps": round(e2e_fps, 2),
        "ram_mb": round(ram_end_mb, 2),
    }


if __name__ == "__main__":
    run_benchmark(num_hazards=100, num_frames=15)
