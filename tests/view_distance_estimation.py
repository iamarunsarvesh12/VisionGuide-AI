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


def main():
    """
    Interactive Live Distance Estimation Viewer.
    Connects Camera Input, YOLOv8m Detection, BoT-SORT Tracking, PHMU Hazard Memory,
    and Monocular Distance Estimation.
    Displays bounding boxes, proximity categories (NEAR, MEDIUM, FAR), metric estimates,
    distance status (MEASURED vs LAST_OBSERVED), and 5-stage latency telemetry.
    Press 'q' or ESC to exit.
    """
    print("Initializing VisionGuide AI Live Distance Estimation Viewer...")
    camera = WebcamInput(camera_index=0, width=640, height=480, target_fps=30)
    detector = YOLOv8mDetector(model_path="yolov8m.pt", confidence_threshold=0.35, device="cpu")
    tracker = BoTSORTTracker(iou_threshold=0.3, max_age=15)
    phmu = PersistentHazardMemory(memory_timeout_seconds=3.0, decay_rate=0.2)
    estimator = MonocularDistanceEstimator(focal_length_px=600.0)

    if not camera.start():
        print("[ERROR] Failed to start camera.")
        return

    print("Loading YOLOv8m model on CPU & initializing tracking, memory, and distance modules...")
    if not detector.load_model():
        print("[ERROR] Failed to load YOLOv8m model.")
        camera.stop()
        return

    tracker.initialize()
    phmu.initialize()
    estimator.initialize()

    print("\nLive Distance Estimation demonstration started. Press 'q' or ESC to quit.")

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

            t_pipe_1 = time.perf_counter()

            det_ms = (t_d1 - t_d0) * 1000.0
            trk_ms = (t_t1 - t_t0) * 1000.0
            phmu_ms = (t_p1 - t_p0) * 1000.0
            dist_ms = (t_dist1 - t_dist0) * 1000.0
            pipe_ms = (t_pipe_1 - t_pipe_0) * 1000.0
            pipe_fps = (1.0 / (t_pipe_1 - t_pipe_0)) if (t_pipe_1 - t_pipe_0) > 0 else 0.0

            # Render Distance Estimation overlays
            for d in dist_results:
                x1, y1, x2, y2 = [int(v) for v in d.bounding_box]
                cat = d.distance_category
                
                # Color code by distance category
                if cat == "NEAR":
                    color = (0, 0, 255)      # Red (Critical Proximity)
                elif cat == "MEDIUM":
                    color = (0, 165, 255)    # Orange
                elif cat == "FAR":
                    color = (0, 255, 0)      # Green
                else:
                    color = (128, 128, 128)  # Grey

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                d_str = f"{d.estimated_distance_m:.1f}m" if d.estimated_distance_m else "?"
                label = f"ID:{d.track_id} {d.class_name} [{cat}: {d_str}] ({d.distance_status})"

                (w, h_text), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), color, -1)
                cv2.putText(frame, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

            # Telemetry Overlay
            info_text = f"Monocular Distance Estimator | Hazards: {len(dist_results)}"
            perf_text = f"YOLO:{det_ms:.0f}ms | Trk:{trk_ms:.1f}ms | PHMU:{phmu_ms:.2f}ms | Dist:{dist_ms:.2f}ms | Pipe:{pipe_ms:.0f}ms ({pipe_fps:.2f} FPS)"

            cv2.putText(frame, info_text, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
            cv2.putText(frame, perf_text, (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)

            cv2.imshow("VisionGuide AI - Module 06 Distance Estimation", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break

    finally:
        estimator.reset()
        phmu.clear()
        tracker.reset()
        detector.release()
        camera.stop()
        cv2.destroyAllWindows()
        print("Live Distance Estimation viewer shut down cleanly.")


if __name__ == "__main__":
    main()
