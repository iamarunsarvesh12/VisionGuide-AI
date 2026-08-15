import time
import sys
import os
import psutil
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.camera_input.camera import WebcamInput
from modules.object_detection.detector import YOLOv8mDetector
from modules.object_tracking.tracker import BoTSORTTracker


def run_benchmark(num_frames: int = 20):
    """
    Empirical benchmark script for Module 04 — BoT-SORT Multi-Object Tracking
    integrated with Module 01 (Camera Input) and Module 03 (YOLOv8m Detection).
    """
    print("=== STARTING BOT-SORT MULTI-OBJECT TRACKING BENCHMARK ===")
    process = psutil.Process(os.getpid())
    ram_start_mb = process.memory_info().rss / (1024 * 1024)

    # 1. Initialize Modules
    t_init_start = time.perf_counter()
    camera = WebcamInput(camera_index=0, width=640, height=480)
    detector = YOLOv8mDetector(model_path="yolov8m.pt", confidence_threshold=0.35, device="cpu")
    tracker = BoTSORTTracker(iou_threshold=0.3, max_age=15)
    
    cam_ok = camera.start()
    det_ok = detector.load_model()
    trk_ok = tracker.initialize()
    t_init_end = time.perf_counter()

    pipeline_init_ms = (t_init_end - t_init_start) * 1000.0

    if not (cam_ok and det_ok and trk_ok):
        print("[ERROR] Pipeline initialization failed.")
        if cam_ok: camera.stop()
        if det_ok: detector.release()
        sys.exit(1)

    print(f"Pipeline Initialization Latency: {pipeline_init_ms:.2f} ms")

    # 2. End-to-End Pipeline Execution
    print(f"Executing End-to-End Pipeline over {num_frames} live frames...")
    cam_latencies_ms = []
    det_latencies_ms = []
    trk_latencies_ms = []
    e2e_latencies_ms = []
    
    unique_track_ids = set()
    track_persistence_log = {}

    t_pipeline_start = time.perf_counter()

    for i in range(num_frames):
        t_e2e_0 = time.perf_counter()

        # Step 1: Camera Read
        t_c0 = time.perf_counter()
        ret, frame = camera.read()
        t_c1 = time.perf_counter()
        cam_lat = (t_c1 - t_c0) * 1000.0

        if not ret or frame is None:
            continue

        # Step 2: YOLOv8m Inference
        t_d0 = time.perf_counter()
        detections = detector.detect(frame)
        t_d1 = time.perf_counter()
        det_lat = (t_d1 - t_d0) * 1000.0

        # Step 3: BoT-SORT Tracking
        t_t0 = time.perf_counter()
        tracks = tracker.update(detections, frame)
        t_t1 = time.perf_counter()
        trk_lat = (t_t1 - t_t0) * 1000.0

        t_e2e_1 = time.perf_counter()
        e2e_lat = (t_e2e_1 - t_e2e_0) * 1000.0

        cam_latencies_ms.append(cam_lat)
        det_latencies_ms.append(det_lat)
        trk_latencies_ms.append(trk_lat)
        e2e_latencies_ms.append(e2e_lat)

        # Track persistence logging
        for trk in tracks:
            unique_track_ids.add(trk.track_id)
            if trk.track_id not in track_persistence_log:
                track_persistence_log[trk.track_id] = {"class": trk.class_name, "frames_seen": 1}
            else:
                track_persistence_log[trk.track_id]["frames_seen"] += 1

    t_pipeline_end = time.perf_counter()

    camera.stop()
    detector.release()
    tracker.reset()

    total_pipeline_time_s = t_pipeline_end - t_pipeline_start
    frames_processed = len(e2e_latencies_ms)
    
    avg_cam_ms = sum(cam_latencies_ms) / frames_processed if frames_processed else 0
    avg_det_ms = sum(det_latencies_ms) / frames_processed if frames_processed else 0
    avg_trk_ms = sum(trk_latencies_ms) / frames_processed if frames_processed else 0
    avg_e2e_ms = sum(e2e_latencies_ms) / frames_processed if frames_processed else 0

    camera_fps = 1000.0 / avg_cam_ms if avg_cam_ms > 0 else 0
    yolo_fps = 1000.0 / avg_det_ms if avg_det_ms > 0 else 0
    tracking_throughput_fps = 1000.0 / avg_trk_ms if avg_trk_ms > 0 else 0
    e2e_fps = frames_processed / total_pipeline_time_s if total_pipeline_time_s > 0 else 0

    ram_end_mb = process.memory_info().rss / (1024 * 1024)

    print("\n=== PIPELINE PERFORMANCE COMPARISON & RESULTS ===")
    print(f"Total Frames Processed: {frames_processed}")
    print(f"Total Pipeline Duration: {total_pipeline_time_s:.2f} s")
    print("\n--- Latency Breakdown (Per Frame Average) ---")
    print(f"  1. Camera Input Latency : {avg_cam_ms:.2f} ms")
    print(f"  2. YOLOv8m Inference     : {avg_det_ms:.2f} ms")
    print(f"  3. BoT-SORT Tracking     : {avg_trk_ms:.2f} ms")
    print(f"  Total End-to-End Latency : {avg_e2e_ms:.2f} ms")

    print("\n--- FPS & Throughput Comparison ---")
    print(f"  Camera Native FPS        : {camera_fps:.2f} FPS")
    print(f"  YOLOv8m CPU Inference    : {yolo_fps:.2f} FPS")
    print(f"  BoT-SORT Standalone Rate : {tracking_throughput_fps:.2f} FPS")
    print(f"  End-to-End Pipeline FPS  : {e2e_fps:.2f} FPS")

    print("\n--- Track Persistence Summary ---")
    print(f"  Unique Track IDs Created : {len(unique_track_ids)}")
    for trk_id, meta in track_persistence_log.items():
        print(f"   - Track ID {trk_id} ({meta['class']}): Persisted across {meta['frames_seen']} frames")

    print(f"\nRAM Memory Consumption: {ram_end_mb:.2f} MB")

    return {
        "tracker": "BoT-SORT",
        "cam_ms": round(avg_cam_ms, 2),
        "det_ms": round(avg_det_ms, 2),
        "trk_ms": round(avg_trk_ms, 2),
        "e2e_ms": round(avg_e2e_ms, 2),
        "yolo_fps": round(yolo_fps, 2),
        "tracking_fps": round(tracking_throughput_fps, 2),
        "e2e_fps": round(e2e_fps, 2),
        "track_count": len(unique_track_ids),
        "ram_mb": round(ram_end_mb, 2),
    }


if __name__ == "__main__":
    run_benchmark(num_frames=15)
