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
from modules.free_space.analyzer import ImageSpaceFreeSpaceAnalyzer
from modules.decision_engine.engine import ContextAwareDecisionEngine
from modules.decision_engine.models import DecisionInput


def main():
    """
    Interactive Live Decision Engine Viewer for VisionGuide AI.
    Connects Camera Input, YOLOv8m Detection, BoT-SORT Tracking, PHMU Hazard Memory,
    Monocular Distance Estimation, Context-Aware Danger Mapping, Free-Space Analysis,
    and Context-Aware Decision Engine into an integrated 8-stage visualization.
    
    Displays region boundaries (LEFT, CENTER, RIGHT), regional occupancy & safe-space scores,
    hazard bounding boxes & danger levels, active navigation command (FORWARD, LEFT, RIGHT, STOP),
    decision confidence, machine-readable reasoning, and 8-stage pipeline telemetry.
    Press 'q' or ESC to exit.
    """
    print("Initializing VisionGuide AI Live Decision Engine Viewer...")
    camera = WebcamInput(camera_index=0, width=640, height=480, target_fps=30)
    detector = YOLOv8mDetector(model_path="yolov8m.pt", confidence_threshold=0.35, device="cpu")
    tracker = BoTSORTTracker(iou_threshold=0.3, max_age=15)
    phmu = PersistentHazardMemory(memory_timeout_seconds=3.0, decay_rate=0.2)
    estimator = MonocularDistanceEstimator(focal_length_px=600.0)
    mapper = ContextAwareDangerMapper()
    analyzer = ImageSpaceFreeSpaceAnalyzer()
    engine = ContextAwareDecisionEngine()

    if not camera.start():
        print("[ERROR] Failed to start camera.")
        return

    print("Loading YOLOv8m model on CPU & initializing tracking, memory, distance, danger, free-space, and decision modules...")
    if not detector.load_model():
        print("[ERROR] Failed to load YOLOv8m model.")
        camera.stop()
        return

    tracker.initialize()
    phmu.initialize()
    estimator.initialize()
    mapper.initialize()
    analyzer.initialize()
    engine.initialize()

    print("\nLive Decision Engine demonstration started. Press 'q' or ESC to quit.")

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
            h, w, _ = frame.shape

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
            assessments = mapper.assess_batch(dist_results, frame_width=float(w))
            t_dang1 = time.perf_counter()

            # Step 6: Free-Space Analysis
            t_fs0 = time.perf_counter()
            fs_result = analyzer.analyze_free_space(assessments, frame_width=float(w), frame_height=float(h))
            t_fs1 = time.perf_counter()

            # Step 7: Decision Engine Synthesis
            t_dec0 = time.perf_counter()
            d_input = DecisionInput(
                timestamp=now,
                frame_id=frame_count,
                regions=fs_result.regions,
                hazards=assessments,
                number_of_active_hazards=len([hz for hz in hazards if hz.memory_state == "ACTIVE"]),
                number_of_remembered_hazards=len([hz for hz in hazards if hz.memory_state == "REMEMBERED"]),
            )
            decision_result = engine.decide(d_input)
            t_dec1 = time.perf_counter()

            t_pipe_1 = time.perf_counter()

            det_ms = (t_d1 - t_d0) * 1000.0
            trk_ms = (t_t1 - t_t0) * 1000.0
            phmu_ms = (t_p1 - t_p0) * 1000.0
            dist_ms = (t_dist1 - t_dist0) * 1000.0
            dang_ms = (t_dang1 - t_dang0) * 1000.0
            fs_ms = (t_fs1 - t_fs0) * 1000.0
            dec_ms = (t_dec1 - t_dec0) * 1000.0
            pipe_ms = (t_pipe_1 - t_pipe_0) * 1000.0
            pipe_fps = (1.0 / (t_pipe_1 - t_pipe_0)) if (t_pipe_1 - t_pipe_0) > 0 else 0.0

            # Render vertical region dividers
            x_l = int(w * 0.33)
            x_r = int(w * 0.67)
            cv2.line(frame, (x_l, 0), (x_l, h), (255, 255, 255), 1, cv2.LINE_AA)
            cv2.line(frame, (x_r, 0), (x_r, h), (255, 255, 255), 1, cv2.LINE_AA)

            # Render hazard bounding boxes & danger tags
            for a in assessments:
                bx1, by1, bx2, by2 = [int(v) for v in a.bounding_box]
                lvl = a.danger_level
                box_color = (0, 0, 255) if lvl == "CRITICAL" else (0, 140, 255) if lvl == "HIGH" else (0, 255, 255) if lvl == "MODERATE" else (0, 255, 0)
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), box_color, 2)
                tag = f"ID:{a.track_id} {a.class_name} [{lvl}] {a.estimated_distance_m:.1f}m"
                cv2.putText(frame, tag, (bx1, by1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, box_color, 1)

            # Render Bottom Regional & Decision Telemetry Panels
            panel_height = 80
            cv2.rectangle(frame, (0, h - panel_height), (w, h), (25, 25, 25), -1)

            state_colors = {"CLEAR": (0, 255, 0), "UNCERTAIN": (0, 255, 255), "BLOCKED": (0, 0, 255)}

            # Left Region Info
            left_reg = fs_result.regions["LEFT"]
            l_color = state_colors.get(left_reg.occupancy_state, (255, 255, 255))
            cv2.putText(frame, f"LEFT: {left_reg.occupancy_state}", (15, h - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.48, l_color, 2)
            cv2.putText(frame, f"Safe: {left_reg.safe_space_score:.2f}", (15, h - 38), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1)

            # Center Region Info
            center_reg = fs_result.regions["CENTER"]
            c_color = state_colors.get(center_reg.occupancy_state, (255, 255, 255))
            cv2.putText(frame, f"CENTER: {center_reg.occupancy_state}", (x_l + 15, h - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.48, c_color, 2)
            cv2.putText(frame, f"Safe: {center_reg.safe_space_score:.2f}", (x_l + 15, h - 38), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1)

            # Right Region Info
            right_reg = fs_result.regions["RIGHT"]
            r_color = state_colors.get(right_reg.occupancy_state, (255, 255, 255))
            cv2.putText(frame, f"RIGHT: {right_reg.occupancy_state}", (x_r + 15, h - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.48, r_color, 2)
            cv2.putText(frame, f"Safe: {right_reg.safe_space_score:.2f}", (x_r + 15, h - 38), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1)

            # Highlight Decision Header Overlay
            cmd = decision_result.command
            cmd_colors = {
                "FORWARD": (0, 255, 0),
                "LEFT": (255, 200, 0),
                "RIGHT": (255, 200, 0),
                "STOP": (0, 0, 255),
            }
            c_col = cmd_colors.get(cmd, (255, 255, 255))

            # Banner box at top center
            cv2.rectangle(frame, (180, 10), (460, 50), (40, 40, 40), -1)
            cv2.rectangle(frame, (180, 10), (460, 50), c_col, 2)
            cv2.putText(frame, f"DECISION: {cmd}", (195, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.75, c_col, 2)

            # Reason & Confidence Panel (Lower Status Bar)
            cv2.putText(
                frame,
                f"Conf: {decision_result.confidence:.2f} | Score: {decision_result.decision_score:.2f}",
                (15, h - 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                (255, 255, 0),
                1,
            )
            cv2.putText(
                frame,
                f"Reason: {decision_result.reason[:75]}...",
                (15, h - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.36,
                (220, 220, 220),
                1,
            )

            # Top Telemetry Bar
            perf_text = f"YOLO:{det_ms:.0f}ms|Trk:{trk_ms:.1f}ms|PHMU:{phmu_ms:.2f}ms|Dist:{dist_ms:.1f}ms|Dang:{dang_ms:.1f}ms|FS:{fs_ms:.1f}ms|Dec:{dec_ms:.2f}ms|FPS:{pipe_fps:.1f}"
            cv2.putText(frame, perf_text, (10, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1)

            cv2.imshow("VisionGuide AI - Module 09 Context-Aware Decision Engine", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break

    finally:
        engine.reset()
        analyzer.reset()
        mapper.reset()
        estimator.reset()
        phmu.clear()
        tracker.reset()
        detector.release()
        camera.stop()
        cv2.destroyAllWindows()
        print("Live Decision Engine viewer shut down cleanly.")


if __name__ == "__main__":
    main()
