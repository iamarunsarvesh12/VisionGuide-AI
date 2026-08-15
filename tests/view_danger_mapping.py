import sys
import os
import time
import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.camera_input.camera import WebcamInput
from modules.object_detection.detector import YOLOv8mDetector
from modules.object_tracking.tracker import BoTSORTTracker
from modules.hazard_memory.memory import PersistentHazardMemory
from modules.distance_estimation.estimator import MonocularDistanceEstimator
from modules.danger_mapping.mapper import ContextAwareDangerMapper


def main():
    """
    Interactive Live Danger Mapping Viewer.
    Connects Camera Input, YOLOv8m Detection, BoT-SORT Tracking, PHMU Hazard Memory,
    Monocular Distance Estimation, and Context-Aware Danger Mapping.
    Displays risk levels (CRITICAL, HIGH, MODERATE, LOW), scores, position zones,
    memory states, and 6-stage latency telemetry.
    Press 'q' or ESC to exit.
    """
    print("Initializing VisionGuide AI Live Danger Mapping Viewer...")
    camera = WebcamInput(camera_index=0, width=640, height=480, target_fps=30)
    detector = YOLOv8mDetector(model_path="yolov8m.pt", confidence_threshold=0.35, device="cpu")
    tracker = BoTSORTTracker(iou_threshold=0.3, max_age=15)
    phmu = PersistentHazardMemory(memory_timeout_seconds=3.0, decay_rate=0.2)
    estimator = MonocularDistanceEstimator(focal_length_px=600.0)
    mapper = ContextAwareDangerMapper()

    if not camera.start():
        print("[ERROR] Failed to start camera.")
        return

    print("Loading YOLOv8m model on CPU & initializing tracking, memory, distance, and danger modules...")
    if not detector.load_model():
        print("[ERROR] Failed to load YOLOv8m model.")
        camera.stop()
        return

    tracker.initialize()
    phmu.initialize()
    estimator.initialize()
    mapper.initialize()

    print("\nLive Context-Aware Danger Mapping demonstration started. Press 'q' or ESC to quit.")

    frame_count = 0

    try:
        while True:
            t_pipe_0 = time.perf_counter()
            ret, frame = camera.read()
            if not ret or frame is None:
                print("[WARNING] Frame read failed.")
                break

            frame_count += 1
            now = time.time()

            # Step 1: Detection
            t_d0 = time.perf_counter()
            detections = detector.detect(frame)
            t_d1 = time.perf_counter()

            # Step 2: Tracking
            t_t0 = time.perf_counter()
            tracks = tracker.update(detections, frame)
            t_t1 = time.perf_counter()

            # Step 3: PHMU Memory Update
            t_p0 = time.perf_counter()
            hazards = phmu.update(tracks, current_time=now, frame_index=frame_count)
            t_p1 = time.perf_counter()

            # Step 4: Distance Estimation
            t_dist0 = time.perf_counter()
            dist_results = estimator.estimate_batch(hazards)
            t_dist1 = time.perf_counter()

            # Step 5: Danger Mapping
            t_dang0 = time.perf_counter()
            assessments = mapper.assess_batch(dist_results, frame_width=640.0)
            t_dang1 = time.perf_counter()

            t_pipe_1 = time.perf_counter()

            det_ms = (t_d1 - t_d0) * 1000.0
            trk_ms = (t_t1 - t_t0) * 1000.0
            phmu_ms = (t_p1 - t_p0) * 1000.0
            dist_ms = (t_dist1 - t_dist0) * 1000.0
            dang_ms = (t_dang1 - t_dang0) * 1000.0
            pipe_ms = (t_pipe_1 - t_pipe_0) * 1000.0
            pipe_fps = (1.0 / (t_pipe_1 - t_pipe_0)) if (t_pipe_1 - t_pipe_0) > 0 else 0.0

            # Render Danger Mapping overlays
            for a in assessments:
                x1, y1, x2, y2 = [int(v) for v in a.bounding_box]
                lvl = a.danger_level
                
                # Color code by danger level
                if lvl == "CRITICAL":
                    color = (0, 0, 255)      # Red
                elif lvl == "HIGH":
                    color = (0, 140, 255)    # Deep Orange
                elif lvl == "MODERATE":
                    color = (0, 255, 255)    # Yellow
                else:
                    color = (0, 255, 0)      # Green (Low Danger)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                label = f"ID:{a.track_id} {a.class_name} [{lvl}: {a.danger_score:.2f}] {a.position_zone} ({a.memory_state})"

                (w, h_text), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), color, -1)
                cv2.putText(frame, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0) if lvl in ("MODERATE", "LOW") else (255, 255, 255), 1)

            # Telemetry Overlay
            top_hazard = assessments[0] if assessments else None
            top_str = f"Top Risk: ID:{top_hazard.track_id} {top_hazard.class_name} ({top_hazard.danger_level})" if top_hazard else "Top Risk: NONE"
            
            info_text = f"Context Danger Mapper | Hazards: {len(assessments)} | {top_str}"
            perf_text = f"YOLO:{det_ms:.0f}ms|Trk:{trk_ms:.1f}ms|PHMU:{phmu_ms:.2f}ms|Dist:{dist_ms:.2f}ms|Danger:{dang_ms:.2f}ms|Pipe:{pipe_ms:.0f}ms ({pipe_fps:.2f}FPS)"

            cv2.putText(frame, info_text, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 0), 2)
            cv2.putText(frame, perf_text, (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)

            cv2.imshow("VisionGuide AI - Module 07 Danger Mapping", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break

    finally:
        mapper.reset()
        estimator.reset()
        phmu.clear()
        tracker.reset()
        detector.release()
        camera.stop()
        cv2.destroyAllWindows()
        print("Live Danger Mapping viewer shut down cleanly.")


if __name__ == "__main__":
    main()
