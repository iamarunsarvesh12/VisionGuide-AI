import sys
import os
import time
import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.camera_input.camera import WebcamInput
from modules.object_detection.detector import YOLOv8mDetector
from modules.object_tracking.tracker import BoTSORTTracker
from modules.hazard_memory.memory import PersistentHazardMemory


def main():
    """
    Interactive Live Hazard Memory Viewer (PHMU Demo).
    Connects Camera Input, YOLOv8m Detection, BoT-SORT Tracking, and PHMU Hazard Memory.
    Displays persistent hazard memory states (ACTIVE, OCCLUDED, REMEMBERED, RECOVERED),
    confidence decay, persistence scores, and real-time telemetry.
    Press 'q' or ESC to exit.
    """
    print("Initializing VisionGuide AI Live Hazard Memory Viewer...")
    camera = WebcamInput(camera_index=0, width=640, height=480, target_fps=30)
    detector = YOLOv8mDetector(model_path="yolov8m.pt", confidence_threshold=0.35, device="cpu")
    tracker = BoTSORTTracker(iou_threshold=0.3, max_age=15)
    phmu = PersistentHazardMemory(memory_timeout_seconds=3.0, decay_rate=0.2)

    if not camera.start():
        print("[ERROR] Failed to start camera.")
        return

    print("Loading YOLOv8m model on CPU, initializing BoT-SORT tracker & PHMU memory...")
    if not detector.load_model():
        print("[ERROR] Failed to load YOLOv8m model.")
        camera.stop()
        return

    tracker.initialize()
    phmu.initialize()

    print("\nLive PHMU hazard memory demonstration started. Press 'q' or ESC to quit.")

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

            # 1. Detection
            t_det_0 = time.perf_counter()
            detections = detector.detect(frame)
            t_det_1 = time.perf_counter()

            # 2. Tracking
            t_trk_0 = time.perf_counter()
            tracks = tracker.update(detections, frame)
            t_trk_1 = time.perf_counter()

            # 3. PHMU Memory Update
            t_phmu_0 = time.perf_counter()
            hazards = phmu.update(tracks, current_time=now, frame_index=frame_count)
            t_phmu_1 = time.perf_counter()

            t_pipe_1 = time.perf_counter()

            det_ms = (t_det_1 - t_det_0) * 1000.0
            trk_ms = (t_trk_1 - t_trk_0) * 1000.0
            phmu_ms = (t_phmu_1 - t_phmu_0) * 1000.0
            pipe_ms = (t_pipe_1 - t_pipe_0) * 1000.0
            pipe_fps = (1.0 / (t_pipe_1 - t_pipe_0)) if (t_pipe_1 - t_pipe_0) > 0 else 0.0

            # Render PHMU Hazard Memory Records
            for h in hazards:
                x1, y1, x2, y2 = [int(v) for v in h.bounding_box]
                state_str = h.memory_state
                
                # Color code by memory state
                if state_str == "ACTIVE":
                    color = (0, 255, 0)  # Green
                elif state_str == "RECOVERED":
                    color = (255, 255, 0)  # Cyan
                elif state_str in ("OCCLUDED", "REMEMBERED"):
                    color = (0, 165, 255)  # Orange/Yellow (Remembered Hazard)
                else:
                    color = (0, 0, 255)  # Red (Expired)

                # Dotted line style for remembered/occluded hazards
                line_style = cv2.LINE_AA if state_str == "ACTIVE" else cv2.LINE_4
                thickness = 2 if state_str in ("ACTIVE", "RECOVERED") else 1

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness, line_style)

                label = f"ID:{h.track_id} {h.object_class} [{state_str}] P:{h.persistence_score:.2f} C:{h.memory_confidence:.2f}"
                if state_str in ("OCCLUDED", "REMEMBERED"):
                    label += f" ({h.time_since_last_seen:.1f}s ago)"

                (w, h_text), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), color, -1)
                cv2.putText(frame, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

            # Dashboard Telemetry Overlay
            stats = phmu.get_statistics()
            info_text = f"PHMU Hazard Memory | Hazards in Store: {len(hazards)} (Active:{stats['state_breakdown'].get('ACTIVE',0)} Rem:{stats['state_breakdown'].get('REMEMBERED',0)+stats['state_breakdown'].get('OCCLUDED',0)})"
            perf_text = f"YOLO:{det_ms:.0f}ms | Track:{trk_ms:.2f}ms | PHMU:{phmu_ms:.3f}ms | Pipe:{pipe_ms:.0f}ms ({pipe_fps:.2f} FPS)"

            cv2.putText(frame, info_text, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
            cv2.putText(frame, perf_text, (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

            cv2.imshow("VisionGuide AI - Module 05 PHMU Live Hazard Memory", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break

    finally:
        phmu.clear()
        tracker.reset()
        detector.release()
        camera.stop()
        cv2.destroyAllWindows()
        print("Live PHMU viewer shut down cleanly.")


if __name__ == "__main__":
    main()
