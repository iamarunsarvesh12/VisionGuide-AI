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
from modules.distance_estimation.estimator import MonocularDistanceEstimator
from modules.object_tracking.interface import Track


def run_benchmark(num_objects: int = 100, num_frames: int = 15):
    """
    Empirical benchmark script for Module 06 — Monocular Distance Estimation.
    Measures micro-benchmark latencies and full 5-stage integrated pipeline performance.
    """
    print("=== STARTING DISTANCE ESTIMATION BENCHMARK ===")
    process = psutil.Process(os.getpid())
    ram_start_mb = process.memory_info().rss / (1024 * 1024)

    estimator = MonocularDistanceEstimator(focal_length_px=600.0)
    estimator.initialize()

    # 1. Micro-Benchmark: Synthetic Batch Distance Estimation
    print(f"\nBenchmarking Distance Estimation across {num_objects} synthetic hazards...")
    synthetic_tracks = []
    for i in range(1, num_objects + 1):
        trk = Track(
            track_id=i, class_id=i % 10, class_name="person" if i % 2 == 0 else "chair",
            confidence=0.88,
            bounding_box=[float(i), float(i), float(i + 150), float(i + 300)],
            center_x=float(i + 75), center_y=float(i + 150), width=150.0, height=300.0,
            tracking_state="TRACKED"
        )
        synthetic_tracks.append(trk)

    t_est_0 = time.perf_counter()
    results = estimator.estimate_batch(synthetic_tracks)
    t_est_1 = time.perf_counter()

    batch_latency_ms = (t_est_1 - t_est_0) * 1000.0
    per_object_latency_ms = batch_latency_ms / num_objects if num_objects else 0

    print(f"Total Batch Estimation Latency ({num_objects} objects): {batch_latency_ms:.3f} ms")
    print(f"Average Per-Object Estimation Latency: {per_object_latency_ms:.5f} ms")

    estimator.reset()

    # 2. Integrated 5-Stage Pipeline Execution (Camera + YOLO + BoT-SORT + PHMU + Distance)
    print(f"\nExecuting 5-Stage Pipeline across {num_frames} live camera frames...")
    camera = WebcamInput(camera_index=0, width=640, height=480)
    detector = YOLOv8mDetector(model_path="yolov8m.pt", confidence_threshold=0.35, device="cpu")
    tracker = BoTSORTTracker(iou_threshold=0.3, max_age=15)
    phmu = PersistentHazardMemory(memory_timeout_seconds=3.0, decay_rate=0.2)

    cam_ok = camera.start()
    det_ok = detector.load_model()
    trk_ok = tracker.initialize()
    phmu_ok = phmu.initialize()
    est_ok = estimator.initialize()

    if not (cam_ok and det_ok and trk_ok and phmu_ok and est_ok):
        print("[ERROR] Pipeline initialization failed.")
        if cam_ok: camera.stop()
        if det_ok: detector.release()
        sys.exit(1)

    cam_lats, det_lats, trk_lats, phmu_lats, dist_lats, e2e_lats = [], [], [], [], [], []
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
        hazards = phmu.update(tracks, current_time=time.time(), frame_index=i+1)
        t_p1 = time.perf_counter()

        # Step 5: Distance Estimation
        t_dist0 = time.perf_counter()
        dist_results = estimator.estimate_batch(hazards)
        t_dist1 = time.perf_counter()

        t_e2e_1 = time.perf_counter()

        cam_lats.append((t_c1 - t_c0) * 1000.0)
        det_lats.append((t_d1 - t_d0) * 1000.0)
        trk_lats.append((t_t1 - t_t0) * 1000.0)
        phmu_lats.append((t_p1 - t_p0) * 1000.0)
        dist_lats.append((t_dist1 - t_dist0) * 1000.0)
        e2e_lats.append((t_e2e_1 - t_e2e_0) * 1000.0)

    t_pipeline_end = time.perf_counter()

    camera.stop()
    detector.release()
    tracker.reset()
    phmu.clear()
    estimator.reset()

    total_pipeline_time_s = t_pipeline_end - t_pipeline_start
    frames_processed = len(e2e_lats)

    avg_cam_ms = sum(cam_lats) / frames_processed if frames_processed else 0
    avg_det_ms = sum(det_lats) / frames_processed if frames_processed else 0
    avg_trk_ms = sum(trk_lats) / frames_processed if frames_processed else 0
    avg_phmu_ms = sum(phmu_lats) / frames_processed if frames_processed else 0
    avg_dist_ms = sum(dist_lats) / frames_processed if frames_processed else 0
    avg_e2e_ms = sum(e2e_lats) / frames_processed if frames_processed else 0

    e2e_fps = frames_processed / total_pipeline_time_s if total_pipeline_time_s > 0 else 0
    ram_end_mb = process.memory_info().rss / (1024 * 1024)

    print("\n=== PIPELINE LATENCY BREAKDOWN (5 STAGES) ===")
    print(f"Total Frames Processed: {frames_processed}")
    print(f"Total Benchmark Duration: {total_pipeline_time_s:.2f} s")
    print(f"  1. Camera Input Latency   : {avg_cam_ms:.2f} ms")
    print(f"  2. YOLOv8m CPU Inference  : {avg_det_ms:.2f} ms")
    print(f"  3. BoT-SORT Tracking      : {avg_trk_ms:.2f} ms")
    print(f"  4. PHMU Hazard Memory     : {avg_phmu_ms:.3f} ms")
    print(f"  5. Distance Estimation    : {avg_dist_ms:.3f} ms")
    print(f"  Total End-to-End Latency  : {avg_e2e_ms:.2f} ms")

    print(f"\nEnd-to-End Pipeline FPS: {e2e_fps:.2f} FPS")
    print(f"RAM Memory Consumption: {ram_end_mb:.2f} MB")

    return {
        "batch_latency_ms": round(batch_latency_ms, 3),
        "per_object_latency_ms": round(per_object_latency_ms, 5),
        "distance_pipeline_ms": round(avg_dist_ms, 3),
        "e2e_pipeline_ms": round(avg_e2e_ms, 2),
        "e2e_fps": round(e2e_fps, 2),
        "ram_mb": round(ram_end_mb, 2),
    }


if __name__ == "__main__":
    run_benchmark(num_objects=100, num_frames=15)
