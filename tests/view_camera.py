import sys
import os
import time
import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from modules.camera_input.camera import WebcamInput


def main():
    """
    Interactive camera test application.
    Displays live webcam feed overlaid with status, resolution, and real-time FPS.
    Press 'q' or ESC to exit.
    """
    print("Initializing VisionGuide AI Camera Input Test Application...")
    camera = WebcamInput(camera_index=0, width=640, height=480, target_fps=30)
    
    if not camera.start():
        print("[ERROR] Failed to start camera index 0.")
        return

    print("Camera running. Press 'q' or ESC to quit.")
    
    try:
        while True:
            ret, frame = camera.read()
            if not ret or frame is None:
                print("[WARNING] Frame read failed.")
                break

            props = camera.get_properties()
            
            # Simple clean overlay display
            fps_str = f"FPS: {props['measured_fps']:.1f}"
            res_str = f"Res: {props['width']}x{props['height']}"
            lat_str = f"Latency: {props['capture_latency_ms']:.1f} ms"
            status_str = "Status: ACTIVE"

            cv2.putText(frame, fps_str, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, res_str, (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, lat_str, (15, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, status_str, (15, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("VisionGuide AI - Module 01 Camera Input Test", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                print("Exit signal received.")
                break

    finally:
        camera.stop()
        cv2.destroyAllWindows()
        print("Camera shutdown complete.")


if __name__ == "__main__":
    main()
