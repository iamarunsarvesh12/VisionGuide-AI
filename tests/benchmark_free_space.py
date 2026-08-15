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
from modules.danger_mapping.mapper import ContextAwareDangerMapper
from modules.free_space.analyzer import ImageSpaceFreeSpaceAnalyzer
from modules.danger_mapping.models import DangerAssessment


def run_benchmark(num_hazards: int = 100, num_frames: int = 15):
    """
    Empirical benchmark script for Module 08 — Free-Space Analysis.
    Measures micro-benchmark latencies and full 7-stage integrated pipeline performance.
    """
    print("=== STARTING FREE-SPACE ANALYSIS BENCHMARK ===")
    process = psutil.Process(os.getpid())
    ram_start_mb = process.memory_info().rss / (1024 * 1024)

    analyzer = ImageSpaceFreeSpaceAnalyzer()
    analyzer.initialize()

    # 1. Micro-Benchmark: Synthetic Free-Space Analysis across 100 hazards
    print(f"\nBenchmarking Free-Space Analysis across {num_hazards} synthetic hazards...")
    synthetic_assessments = []
    classes = ["person", "chair", "table", "door", "stairs", "cabinet"]
    categories = ["NEAR", "MEDIUM", "FAR"]
    levels = ["CRITICAL", "HIGH", "MODERATE", "LOW"]

    for i in range(1, num_hazards + 1):
        cls_name = classes[i % len(classes)]
        cat = categories[i % len(categories)]
        lvl = levels[i % len(levels)]
        center_x = float((i * 13) % 640)
        d = DangerAssessment(
            track_id=i, class_name=cls_name, danger_score=0.85 if lvl == "CRITICAL" else 0.40,
            danger_level=lvl, distance_category=cat, estimated_distance_m=1.2 if cat == "NEAR" else 3.0,
            position_zone="CENTER", memory_state="ACTIVE", memory_confidence=0.9, persistence_score=0.8,
            navigation_relevance=True, bounding_box=[center_x - 40, 100.0, center_x + 40, 400.0]
        )
        synthetic_assessments.append(d)

    t_fs_0 = time.perf_counter()
    fs_result = analyzer.analyze_free_space(synthetic_assessments, frame_width=640.0, frame_height=480.0)
    t_fs_1 = time.perf_counter()

    micro_latency_ms = (t_fs_1 - t_fs_0) * 1000.0
    per_object_latency_ms = micro_latency_ms / num_hazards if num_hazards else 0

    print(f"Total Batch Free-Space Analysis Latency ({num_hazards} hazards): {micro_latency_ms:.3f} ms")
    print(f"Average Per-Hazard Free-Space Analysis Latency: {per_object_latency_ms:.5f} ms")

    analyzer.reset()

    # 2. Integrated 7-Stage Pipeline Execution
    print(f"\nExecuting 7-Stage Pipeline across {num_frames} live camera frames...")
    camera = WebcamInput(camera_index=0, width=640, height=480)
    detector = YOLOv8mDetector(model_path="yolov8m.pt", confidence_threshold=0.35, device="cpu")
    tracker = BoTSORTTracker(iou_threshold=0.3, max_age=15)
    phmu = PersistentHazardMemory(memory_timeout_seconds=3.0, decay_rate=0.2)
    estimator = MonocularDistanceEstimator(focal_length_px=600.0)
    mapper = ContextAwareDangerMapper()

    cam_ok = camera.start()
    det_ok = detector.load_model()
    trk_ok = tracker.initialize()
    phmu_ok = phmu.initialize()
    est_ok = estimator.initialize()
    map_ok = mapper.initialize()
    fs_ok = analyzer.initialize()

    if not (cam_ok and det_ok and trk_ok and phmu_ok and est_ok and map_ok and fs_ok):
        print("[ERROR] Pipeline initialization failed.")
        if cam_ok: camera.stop()
        if det_ok: detector.release()
        sys.exit(1)

    cam_lats, det_lats, trk_lats, phmu_lats, dist_lats, danger_lats, fs_lats, e2e_lats = [], [], [], [], [], [], [], []
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

        # Step 6: Danger Mapping
        t_dang0 = time.perf_counter()
        ranked_assessments = mapper.assess_batch(dist_results, frame_width=640.0)
        t_dang1 = time.perf_counter()

        # Step 7: Free-Space Analysis
        t_fs0 = time.perf_counter()
        free_space_res = analyzer.analyze_free_space(ranked_assessments, frame_width=640.0, frame_height=480.0)
        t_fs1 = time.perf_counter()

        t_e2e_1 = time.perf_counter()

        cam_lats.append((t_c1 - t_c0) * 1000.0)
        det_lats.append((t_d1 - t_d0) * 1000.0)
        trk_lats.append((t_t1 - t_t0) * 1000.0)
        phmu_lats.append((t_p1 - t_p0) * 1000.0)
        dist_lats.append((t_dist1 - t_dist0) * 1000.0)
        danger_lats.append((t_dang1 - t_dang0) * 1000.0)
        fs_lats.append((t_fs1 - t_fs0) * 1000.0)
        e2e_lats.append((t_e2e_1 - t_e2e_0) * 1000.0)

    t_pipeline_end = time.perf_counter()

    camera.stop()
    detector.release()
    tracker.reset()
    phmu.clear()
    estimator.reset()
    mapper.reset()
    analyzer.reset()

    total_pipeline_time_s = t_pipeline_end - t_pipeline_start
    frames_processed = len(e2e_lats)

    avg_cam_ms = sum(cam_lats) / frames_processed if frames_processed else 0
    avg_det_ms = sum(det_lats) / frames_processed if frames_processed else 0
    avg_trk_ms = sum(trk_lats) / frames_processed if frames_processed else 0
    avg_phmu_ms = sum(phmu_lats) / frames_processed if frames_processed else 0
    avg_dist_ms = sum(dist_lats) / frames_processed if frames_processed else 0
    avg_danger_ms = sum(danger_lats) / frames_processed if frames_processed else 0
    avg_fs_ms = sum(fs_lats) / frames_processed if frames_processed else 0
    avg_e2e_ms = sum(e2e_lats) / frames_processed if frames_processed else 0

    e2e_fps = frames_processed / total_pipeline_time_s if total_pipeline_time_s > 0 else 0
    ram_end_mb = process.memory_info().rss / (1024 * 1024)

    print("\n=== PIPELINE LATENCY BREAKDOWN (7 STAGES) ===")
    print(f"Total Frames Processed: {frames_processed}")
    print(f"Total Benchmark Duration: {total_pipeline_time_s:.2f} s")
    print(f"  1. Camera Input Latency   : {avg_cam_ms:.2f} ms")
    print(f"  2. YOLOv8m CPU Inference  : {avg_det_ms:.2f} ms")
    print(f"  3. BoT-SORT Tracking      : {avg_trk_ms:.2f} ms")
    print(f"  4. PHMU Hazard Memory     : {avg_phmu_ms:.3f} ms")
    print(f"  5. Distance Estimation    : {avg_dist_ms:.3f} ms")
    print(f"  6. Context Danger Mapping : {avg_danger_ms:.3f} ms")
    print(f"  7. Free-Space Analysis    : {avg_fs_ms:.3f} ms")
    print(f"  Total End-to-End Latency  : {avg_e2e_ms:.2f} ms")

    print(f"\nEnd-to-End Pipeline FPS: {e2e_fps:.2f} FPS")
    print(f"RAM Memory Consumption: {ram_end_mb:.2f} MB")

    return {
        "batch_latency_ms": round(micro_latency_ms, 3),
        "per_object_latency_ms": round(per_object_latency_ms, 5),
        "free_space_pipeline_ms": round(avg_fs_ms, 3),
        "e2e_pipeline_ms": round(avg_e2e_ms, 2),
        "e2e_fps": round(e2e_fps, 2),
        "ram_mb": round(ram_end_mb, 2),
    }


if __name__ == "__main__":
    run_benchmark(num_hazards=100, num_frames=15)
