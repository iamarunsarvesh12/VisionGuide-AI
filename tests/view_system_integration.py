import sys
import os
import time
import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.system_integration.pipeline import VisionGuideSystemPipeline
from modules.system_integration.models import SystemState


def main():
    """
    Live Visual Integration Inspection Script for VisionGuide AI (Phase 10).
    Displays real-time OpenCV window rendering bounding boxes, track IDs, PHMU states,
    danger levels, regional free-space boundaries, final command, and per-module latencies.
    """
    print("================================================================")
    print("      VISIONGUIDE AI — PHASE 10 SYSTEM VISUAL INSPECTION        ")
    print("================================================================")

    pipeline = VisionGuideSystemPipeline(config_path="config/config.yaml")

    if not pipeline.initialize():
        print(f"[ERROR] Failed to initialize VisionGuide Pipeline: {pipeline.error_message}")
        return

    print("Pipeline initialized successfully. Starting webcam hardware stream...")
    if not pipeline.start():
        print("[WARNING] Could not start live webcam hardware. Running on synthetic frames.")

    try:
        while True:
            res = pipeline.process_frame()

            # Create visual display canvas
            frame = res.camera_status.get("raw_frame", None)
            if frame is None or not isinstance(frame, np.ndarray):
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
            else:
                frame = frame.copy()

            # 1. Overlay Regional Free Space Dividers
            h, w = frame.shape[:2]
            w_left = int(w * 0.33)
            w_right = int(w * 0.67)

            cv2.line(frame, (w_left, 0), (w_left, h), (100, 100, 100), 1)
            cv2.line(frame, (w_right, 0), (w_right, h), (100, 100, 100), 1)

            cv2.putText(frame, "LEFT", (w_left // 2 - 20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(frame, "CENTER", (w_left + 40, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(frame, "RIGHT", (w_right + 40, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            # 2. Overlay Bounding Boxes & Danger Assessments
            if res.danger_assessments:
                for da in res.danger_assessments:
                    bbox = getattr(da, "bbox", [0, 0, 50, 50])
                    x1, y1, x2, y2 = [int(v) for v in bbox]
                    tid = getattr(da, "track_id", 0)
                    lvl = getattr(da, "danger_level", "LOW")
                    cls_n = getattr(da, "class_name", "object")

                    color = (0, 255, 0) if lvl == "LOW" else (0, 255, 255) if lvl in ["MODERATE", "HIGH"] else (0, 0, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"ID:{tid} {cls_n} ({lvl})", (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

            # 3. Overlay Navigation Decision & Spoken Guidance
            cmd = res.decision_result.command if res.decision_result else "STOP"
            msg = res.audio_result.message if res.audio_result else ""

            cmd_color = (0, 255, 0) if cmd == "FORWARD" else (0, 255, 255) if cmd in ["LEFT", "RIGHT"] else (0, 0, 255)

            # Banner at bottom
            cv2.rectangle(frame, (0, h - 70), (w, h), (20, 20, 20), -1)
            cv2.putText(frame, f"COMMAND: {cmd}", (15, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, cmd_color, 2)
            cv2.putText(frame, f"AUDIO: \"{msg}\"", (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            cv2.putText(frame, f"FPS: {res.pipeline_fps:.1f} | Latency: {res.total_latency:.1f}ms", (w - 240, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            cv2.imshow("VisionGuide AI — End-to-End System Visualizer", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in [ord('q'), 27]:  # 'q' or ESC
                print("Closing Visualizer...")
                break

            time.sleep(0.01)

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("Visual Inspection closed cleanly.")


if __name__ == "__main__":
    main()
