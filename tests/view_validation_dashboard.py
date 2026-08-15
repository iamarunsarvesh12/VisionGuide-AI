import sys
import os
import time
import numpy as np
import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.system_integration.pipeline import VisionGuideSystemPipeline
from modules.system_monitor.monitor import SystemResourceMonitor
from modules.audio_guidance.tts_engine import MockTTSEngine
from modules.audio_guidance.guidance import OfflineAudioGuidance


def draw_dashboard_ui(frame: np.ndarray, res: Any, monitor_summary: dict) -> np.ndarray:
    """Render technical telemetry dashboard overlays on top of the OpenCV frame."""
    h, w, _ = frame.shape
    vis = frame.copy()

    # Draw 3-region vertical partition boundaries
    x_left = int(w * 0.33)
    x_right = int(w * 0.67)
    cv2.line(vis, (x_left, 0), (x_left, h), (100, 100, 100), 1, cv2.LINE_AA)
    cv2.line(vis, (x_right, 0), (x_right, h), (100, 100, 100), 1, cv2.LINE_AA)

    # Draw Region Occupancy Banner at Bottom
    if res.free_space_result:
        for r_name, r_occ in res.free_space_result.regions.items():
            if r_name == "LEFT":
                rx1, rx2 = 0, x_left
            elif r_name == "CENTER":
                rx1, rx2 = x_left, x_right
            else:
                rx1, rx2 = x_right, w

            col = (0, 200, 0) if r_occ.occupancy_state == "CLEAR" else ((0, 165, 255) if r_occ.occupancy_state == "UNCERTAIN" else (0, 0, 255))
            cv2.rectangle(vis, (rx1 + 5, h - 35), (rx2 - 5, h - 5), col, -1)
            text = f"{r_name}: {r_occ.occupancy_state} ({r_occ.safe_space_score:.2f})"
            cv2.putText(vis, text, (rx1 + 10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    # Draw Active Detections & Hazard Overlays
    for d in res.detections:
        x1, y1, x2, y2 = [int(v) for v in d.bounding_box]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.putText(vis, f"{d.class_name} {d.confidence:.2f}", (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)

    for dr in res.distance_results:
        if dr.estimated_distance_m:
            bbox = dr.bounding_box
            cx, cy = int(dr.center_x), int(dr.center_y)
            cv2.putText(vis, f"{dr.estimated_distance_m:.1f}m ({dr.distance_category})", (cx - 30, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA)

    # Top Header Telemetry Panel
    cv2.rectangle(vis, (0, 0), (w, 85), (20, 20, 20), -1)

    cmd = res.decision_result.command if res.decision_result else "STOP"
    cmd_col = (0, 255, 0) if cmd == "FORWARD" else ((0, 255, 255) if cmd in ["LEFT", "RIGHT"] else (0, 0, 255))
    cv2.putText(vis, f"COMMAND: {cmd}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, cmd_col, 2, cv2.LINE_AA)

    fps = res.pipeline_fps
    tot_lat = res.total_latency
    yolo_lat = res.module_latencies.get("yolo", 0.0)
    cpu_pct = monitor_summary.get("avg_cpu_percent", 0.0)
    ram_mb = monitor_summary.get("avg_ram_mb", 0.0)

    cv2.putText(vis, f"FPS: {fps:.1f} | Pipeline: {tot_lat:.1f}ms | YOLO: {yolo_lat:.1f}ms", (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(vis, f"CPU: {cpu_pct:.1f}% | RAM: {ram_mb:.0f}MB | PHMU Hazards: {len(res.hazards)}", (15, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    return vis


def run_validation_dashboard():
    """
    Module 11K — Final Validation Dashboard Launcher.
    Runs real-time webcam camera stream with live telemetry overlays.
    Press 'q' or ESC to exit cleanly.
    """
    print("================================================================")
    print("         MODULE 11K — TECHNICAL VALIDATION DASHBOARD            ")
    print("================================================================")

    mock_tts = MockTTSEngine()
    audio = OfflineAudioGuidance(tts_engine_override=mock_tts)

    pipeline = VisionGuideSystemPipeline(audio_override=audio)
    monitor = SystemResourceMonitor()

    print("\nInitializing VisionGuide AI Validation Dashboard...")
    if not pipeline.initialize():
        print(f"[ERROR] Pipeline init failed: {pipeline.error_message}")
        return

    if not pipeline.start():
        print("[WARNING] Laptop webcam hardware stream unavailable. Running synthetic demo mode...")

    print("\nDashboard running. Press 'q' or ESC in OpenCV window to quit.\n")

    frame_count = 0
    t_start = time.time()

    try:
        while True:
            res = pipeline.process_frame()
            q_len = audio._queue.qsize() if (audio and hasattr(audio, "_queue")) else 0
            monitor.capture_metrics(pipeline_result=res, audio_queue_length=q_len)

            # Synthesize display frame
            img_display = np.zeros((480, 640, 3), dtype=np.uint8)
            summary = monitor.get_summary()

            vis_frame = draw_dashboard_ui(img_display, res, summary)

            cv2.imshow("VisionGuide AI — Technical Validation Dashboard", vis_frame)
            key = cv2.waitKey(1) & 0xFF
            if key in [ord("q"), 27]:
                print("Exit signal received. Closing dashboard...")
                break

            frame_count += 1
            if frame_count >= 10 and not pipeline.camera.is_opened():
                print("Completed synthetic dashboard run (10 frames). Exiting...")
                break

    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received.")
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        audio.close()
        print("Dashboard shut down cleanly.")


if __name__ == "__main__":
    run_validation_dashboard()
