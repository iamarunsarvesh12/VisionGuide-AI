import sys
import os
import time
import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.camera_input.camera import WebcamInput
from modules.object_detection.detector import YOLOv8mDetector


def main():
    """
    Interactive Live Detection Viewer.
    Connects Module 01 (Camera Input) and Module 03 (YOLOv8m Object Detection).
    Displays bounding boxes, labels, confidence scores, FPS, and inference latency.
    Press 'q' or ESC to exit.
    """
    print("Initializing VisionGuide AI Live Detection Viewer...")
    camera = WebcamInput(camera_index=0, width=640, height=480, target_fps=30)
    detector = YOLOv8mDetector(model_path="yolov8m.pt", confidence_threshold=0.35)

    if not camera.start():
        print("[ERROR] Failed to start camera.")
        return

    print("Loading YOLOv8m model on CPU...")
    if not detector.load_model():
        print("[ERROR] Failed to load YOLOv8m model.")
        camera.stop()
        return

    print("Live detection started. Press 'q' or ESC to quit.")

    try:
        while True:
            ret, frame = camera.read()
            if not ret or frame is None:
                print("[WARNING] Frame read failed.")
                break

            t0 = time.perf_counter()
            detections = detector.detect(frame)
            t1 = time.perf_counter()

            inf_latency_ms = (t1 - t0) * 1000.0
            inf_fps = (1.0 / (t1 - t0)) if (t1 - t0) > 0 else 0.0

            # Draw detections
            for det in detections:
                x1, y1, x2, y2 = [int(v) for v in det.bounding_box]
                label = f"{det.class_name}: {det.confidence:.2f}"

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw label background box
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), (0, 255, 0), -1)
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            # Draw telemetry overlay
            overlay_text = f"Model: YOLOv8m (CPU) | Detections: {len(detections)}"
            perf_text = f"Inference Latency: {inf_latency_ms:.1f} ms | FPS: {inf_fps:.2f}"

            cv2.putText(frame, overlay_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, perf_text, (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            cv2.imshow("VisionGuide AI - Module 03 YOLOv8m Live Detection", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break

    finally:
        detector.release()
        camera.stop()
        cv2.destroyAllWindows()
        print("Live detection viewer shut down cleanly.")


if __name__ == "__main__":
    main()
