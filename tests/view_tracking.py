import sys
import os
import time
import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.camera_input.camera import WebcamInput
from modules.object_detection.detector import YOLOv8mDetector
from modules.object_tracking.tracker import BoTSORTTracker


def main():
    """
    Interactive Live Tracking Viewer.
    Connects Module 01 (Camera Input), Module 03 (YOLOv8m Detection), and Module 04 (BoT-SORT Tracking).
    Displays persistent track IDs, bounding boxes, labels, confidence scores, and latency telemetry.
    Press 'q' or ESC to exit.
    """
    print("Initializing VisionGuide AI Live Tracking Viewer...")
    camera = WebcamInput(camera_index=0, width=640, height=480, target_fps=30)
    detector = YOLOv8mDetector(model_path="yolov8m.pt", confidence_threshold=0.35, device="cpu")
    tracker = BoTSORTTracker(iou_threshold=0.3, max_age=15)

    if not camera.start():
        print("[ERROR] Failed to start camera.")
        return

    print("Loading YOLOv8m model on CPU and initializing BoT-SORT tracker...")
    if not detector.load_model():
        print("[ERROR] Failed to load YOLOv8m model.")
        camera.stop()
        return

    tracker.initialize()

    print("Live multi-object tracking started. Press 'q' or ESC to quit.")

    try:
        while True:
            t_pipe_0 = time.perf_counter()
            ret, frame = camera.read()
            if not ret or frame is None:
                print("[WARNING] Frame read failed.")
                break

            # 1. Detection
            t_det_0 = time.perf_counter()
            detections = detector.detect(frame)
            t_det_1 = time.perf_counter()
            det_ms = (t_det_1 - t_det_0) * 1000.0

            # 2. Tracking
            t_trk_0 = time.perf_counter()
            tracks = tracker.update(detections, frame)
            t_trk_1 = time.perf_counter()
            trk_ms = (t_trk_1 - t_trk_0) * 1000.0

            t_pipe_1 = time.perf_counter()
            pipe_ms = (t_pipe_1 - t_pipe_0) * 1000.0
            pipe_fps = (1.0 / (t_pipe_1 - t_pipe_0)) if (t_pipe_1 - t_pipe_0) > 0 else 0.0

            # Draw tracks
            for trk in tracks:
                x1, y1, x2, y2 = [int(v) for v in trk.bounding_box]
                label = f"ID:{trk.track_id} | {trk.class_name}: {trk.confidence:.2f} [{trk.tracking_state}]"

                # Color based on state
                color = (0, 255, 0) if trk.tracking_state == "TRACKED" else (0, 165, 255)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x1, y1 - 22), (x1 + w, y1), color, -1)
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            # Telemetry overlay
            info_text = f"YOLOv8m + BoT-SORT | Active Tracks: {len(tracks)}"
            perf_text = f"YOLO: {det_ms:.1f}ms | Track: {trk_ms:.2f}ms | Pipe: {pipe_ms:.1f}ms ({pipe_fps:.2f} FPS)"

            cv2.putText(frame, info_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(frame, perf_text, (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            cv2.imshow("VisionGuide AI - Module 04 BoT-SORT Live Tracking", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break

    finally:
        tracker.reset()
        detector.release()
        camera.stop()
        cv2.destroyAllWindows()
        print("Live tracking viewer shut down cleanly.")


if __name__ == "__main__":
    main()
