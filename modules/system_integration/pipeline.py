import time
import os
import sys
import logging
import yaml
from typing import Dict, Any, Optional, List, Tuple
import numpy as np

from modules.system_integration.models import SystemState, ModuleStatusMap, PipelineResult
from modules.system_integration.interface import SystemPipelineInterface

from modules.camera_input.camera import WebcamInput
from modules.object_detection.detector import YOLOv8mDetector
from modules.object_tracking.tracker import BoTSORTTracker
from modules.hazard_memory.memory import PersistentHazardMemory
from modules.distance_estimation.estimator import MonocularDistanceEstimator
from modules.danger_mapping.mapper import ContextAwareDangerMapper
from modules.free_space.analyzer import ImageSpaceFreeSpaceAnalyzer
from modules.decision_engine.engine import ContextAwareDecisionEngine
from modules.decision_engine.models import DecisionInput, DecisionResult
from modules.audio_guidance.guidance import OfflineAudioGuidance

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

# Logger setup
logger = logging.getLogger("SystemPipeline")
logger.setLevel(logging.INFO)
if not logger.handlers:
    c_handler = logging.StreamHandler(sys.stdout)
    c_handler.setFormatter(logging.Formatter("[%(name)s][%(levelname)s] %(message)s"))
    logger.addHandler(c_handler)

    f_handler = logging.FileHandler("logs/system_integration.log", mode="a", encoding="utf-8")
    f_handler.setFormatter(logging.Formatter("%(asctime)s [%(name)s][%(levelname)s] %(message)s"))
    logger.addHandler(f_handler)


class VisionGuideSystemPipeline(SystemPipelineInterface):
    """
    Module 11 / System Integration — End-to-End Orchestration Pipeline for VisionGuide AI.
    
    Integrates Modules 01 through 10 in strict sequential order:
    Camera -> YOLOv8m -> BoT-SORT -> PHMU -> Distance -> Danger -> Free Space -> Decision Engine -> Audio Guidance.
    
    Provides unified status tracking, fail-safe error handling, per-module latency telemetry,
    and deterministic safety arbitration.
    """

    def __init__(
        self,
        config_path: str = "config/config.yaml",
        camera_override: Optional[Any] = None,
        detector_override: Optional[Any] = None,
        tracker_override: Optional[Any] = None,
        phmu_override: Optional[Any] = None,
        distance_override: Optional[Any] = None,
        danger_override: Optional[Any] = None,
        free_space_override: Optional[Any] = None,
        decision_override: Optional[Any] = None,
        audio_override: Optional[Any] = None,
    ):
        self.config_path = config_path
        self.config: Dict[str, Any] = {}
        self.system_status = SystemState.INITIALIZING
        self.module_status = ModuleStatusMap()

        # Component instances / overrides
        self.camera = camera_override
        self.detector = detector_override
        self.tracker = tracker_override
        self.phmu = phmu_override
        self.distance_estimator = distance_override
        self.danger_mapper = danger_override
        self.free_space_analyzer = free_space_override
        self.decision_engine = decision_override
        self.audio_guidance = audio_override

        # Processing state
        self.frame_counter = 0
        self.last_pipeline_result: Optional[PipelineResult] = None
        self.last_decision_score: Optional[float] = None
        self.last_decision_timestamp: Optional[float] = None
        self.last_command: Optional[str] = None
        self.error_message: Optional[str] = None

        # Statistics & Latency records
        self.total_frames_processed = 0
        self.latency_history: List[float] = []

    def initialize(self, config_dict: Optional[Dict[str, Any]] = None) -> bool:
        """Initialize configuration and boot up all 10 subsystems safely."""
        logger.info("Initializing VisionGuide AI Unified System Pipeline...")
        self.system_status = SystemState.INITIALIZING

        try:
            if config_dict is not None:
                self.config = config_dict
            elif os.path.exists(self.config_path):
                with open(self.config_path, "r", encoding="utf-8") as f:
                    self.config = yaml.safe_load(f) or {}

            # 1. Camera Input
            if self.camera is None:
                cam_cfg = self.config.get("camera", {})
                self.camera = WebcamInput(
                    camera_index=cam_cfg.get("index", 0),
                    width=cam_cfg.get("width", 640),
                    height=cam_cfg.get("height", 480),
                    target_fps=cam_cfg.get("target_fps", 30),
                    backend_preference=cam_cfg.get("cap_api_preference", "CAP_ANY"),
                )
            if hasattr(self.camera, "initialize"):
                self.camera.initialize()
            self.module_status.camera = "READY"

            # 2. YOLOv8m Object Detector
            if self.detector is None:
                yolo_cfg = self.config.get("yolo", {})
                self.detector = YOLOv8mDetector(
                    model_path=yolo_cfg.get("model_path", "yolov8m.pt"),
                    confidence_threshold=yolo_cfg.get("confidence_threshold", 0.35),
                    iou_threshold=yolo_cfg.get("iou_threshold", 0.45),
                    device=yolo_cfg.get("device", "cpu"),
                )
            if hasattr(self.detector, "load_model") and not getattr(self.detector, "_is_loaded", False):
                self.detector.load_model()
            elif hasattr(self.detector, "initialize"):
                self.detector.initialize()
            self.module_status.yolo = "READY"

            # 3. BoT-SORT Tracker
            if self.tracker is None:
                self.tracker = BoTSORTTracker()
            if hasattr(self.tracker, "initialize"):
                self.tracker.initialize()
            self.module_status.tracking = "READY"

            # 4. PHMU Hazard Memory
            if self.phmu is None:
                phmu_cfg = self.config.get("phmu", {})
                self.phmu = PersistentHazardMemory(
                    memory_timeout_seconds=phmu_cfg.get("memory_timeout_seconds", 3.0),
                    decay_rate=phmu_cfg.get("decay_rate", 0.2),
                    minimum_memory_confidence=phmu_cfg.get("minimum_memory_confidence", 0.1),
                    persistence_threshold=phmu_cfg.get("persistence_threshold", 0.2),
                )
            if hasattr(self.phmu, "initialize"):
                self.phmu.initialize()
            self.module_status.phmu = "READY"

            # 5. Distance Estimator
            if self.distance_estimator is None:
                dist_cfg = self.config.get("distance_estimation", {})
                profiles = self.config.get("distance_profiles", None)
                self.distance_estimator = MonocularDistanceEstimator(
                    focal_length_px=dist_cfg.get("focal_length_px", 600.0),
                    near_threshold_m=dist_cfg.get("near_threshold_m", 1.5),
                    medium_threshold_m=dist_cfg.get("medium_threshold_m", 3.0),
                    distance_profiles=profiles,
                )
            if hasattr(self.distance_estimator, "initialize"):
                self.distance_estimator.initialize()
            self.module_status.distance = "READY"

            # 6. Danger Mapper
            if self.danger_mapper is None:
                danger_cfg = self.config.get("danger_mapping", {})
                factors = self.config.get("object_hazard_factors", None)
                self.danger_mapper = ContextAwareDangerMapper(
                    weights=danger_cfg.get("weights"),
                    thresholds=danger_cfg.get("thresholds"),
                    object_factors=factors,
                )
            if hasattr(self.danger_mapper, "initialize"):
                self.danger_mapper.initialize()
            self.module_status.danger = "READY"

            # 7. Free Space Analyzer
            if self.free_space_analyzer is None:
                fs_cfg = self.config.get("free_space", {})
                self.free_space_analyzer = ImageSpaceFreeSpaceAnalyzer(
                    clear_max_threshold=fs_cfg.get("thresholds", {}).get("clear_max", 0.20),
                    uncertain_max_threshold=fs_cfg.get("thresholds", {}).get("uncertain_max", 0.45),
                )
            if hasattr(self.free_space_analyzer, "initialize"):
                self.free_space_analyzer.initialize()
            self.module_status.free_space = "READY"

            # 8. Decision Engine
            if self.decision_engine is None:
                dec_cfg = self.config.get("decision_engine", {})
                self.decision_engine = ContextAwareDecisionEngine()
                self.decision_engine.initialize(config_dict=dec_cfg)
            else:
                if hasattr(self.decision_engine, "initialize"):
                    self.decision_engine.initialize()
            self.module_status.decision = "READY"

            # 9. Audio Guidance
            if self.audio_guidance is None:
                aud_cfg = self.config.get("audio_guidance", {})
                self.audio_guidance = OfflineAudioGuidance()
                self.audio_guidance.initialize(config_dict=aud_cfg)
            else:
                if hasattr(self.audio_guidance, "initialize"):
                    self.audio_guidance.initialize()
            self.module_status.audio = "READY"

            self.system_status = SystemState.READY
            logger.info("VisionGuide AI Pipeline initialization completed successfully.")
            return True

        except Exception as e:
            logger.error(f"Pipeline initialization error: {e}")
            self.system_status = SystemState.ERROR
            self.error_message = str(e)
            return False

    def start(self) -> bool:
        """Start camera hardware stream."""
        if not self.camera:
            return False
        ok = self.camera.start()
        if ok:
            self.system_status = SystemState.RUNNING
            self.module_status.camera = "RUNNING"
            logger.info("VisionGuide AI Camera Stream started.")
        else:
            self.system_status = SystemState.ERROR
            self.module_status.camera = "ERROR"
            self.error_message = "Failed to start camera hardware"
            logger.error("Camera start failed.")
        return ok

    def process_frame(self, frame: Optional[np.ndarray] = None) -> PipelineResult:
        """
        Process a single visual frame through the complete 10-step pipeline:
        Camera -> YOLO -> BoT-SORT -> PHMU -> Distance -> Danger -> Free Space -> Decision -> Audio Guidance.
        """
        t_start = time.perf_counter()
        now = time.time()
        self.frame_counter += 1
        fid = self.frame_counter

        latencies: Dict[str, float] = {}

        # 1. Camera Input
        t0 = time.perf_counter()
        img = frame
        cam_props = {}
        if img is None:
            if self.camera and self.camera.is_opened():
                success, img = self.camera.read()
                cam_props = self.camera.get_properties()
                cam_props["raw_frame"] = img
                if not success or img is None:
                    latencies["camera"] = (time.perf_counter() - t0) * 1000.0
                    self.system_status = SystemState.ERROR
                    self.error_message = "Camera read failed"
                    logger.error("Camera frame read error; issuing emergency STOP audio.")
                    stop_audio = self.audio_guidance.speak_command("STOP") if self.audio_guidance else None
                    return PipelineResult(
                        frame_id=fid,
                        timestamp=now,
                        camera_status={"is_opened": False},
                        audio_result=stop_audio,
                        module_latencies=latencies,
                        total_latency=(time.perf_counter() - t_start) * 1000.0,
                        pipeline_fps=0.0,
                        system_status=SystemState.ERROR,
                        error_status="Camera frame read error",
                    )
            else:
                img = np.zeros((480, 640, 3), dtype=np.uint8)
                cam_props = {"synthetic": True}

        latencies["camera"] = (time.perf_counter() - t0) * 1000.0
        frame_shape = img.shape[:2]  # (height, width)

        # 2. YOLOv8m Object Detection
        t0 = time.perf_counter()
        detections = []
        try:
            if self.detector:
                raw_dets = self.detector.detect(img)
                if isinstance(raw_dets, list):
                    detections = raw_dets
                else:
                    logger.warning("Detector returned non-list type; defaulting to empty list.")
                    detections = []
        except Exception as e:
            logger.error(f"YOLOv8m detection exception on frame {fid}: {e}")
            detections = []
        latencies["yolo"] = (time.perf_counter() - t0) * 1000.0

        # 3. BoT-SORT Object Tracking
        t0 = time.perf_counter()
        tracks = []
        try:
            if self.tracker:
                tracks = self.tracker.update(detections, frame=img)
        except Exception as e:
            logger.error(f"BoT-SORT tracking exception on frame {fid}: {e}")
            tracks = []
        latencies["tracking"] = (time.perf_counter() - t0) * 1000.0

        # 4. PHMU Hazard Memory Unit
        t0 = time.perf_counter()
        hazards = []
        try:
            if self.phmu:
                hazards = self.phmu.update(tracks, current_time=now)
        except Exception as e:
            logger.error(f"PHMU hazard memory exception on frame {fid}: {e}")
            hazards = []
        latencies["phmu"] = (time.perf_counter() - t0) * 1000.0

        # 5. Monocular Distance Estimation
        t0 = time.perf_counter()
        distance_results = []
        try:
            if self.distance_estimator and hazards:
                for h in hazards:
                    dr = self.distance_estimator.estimate_distance(h)
                    if dr:
                        distance_results.append(dr)
        except Exception as e:
            logger.error(f"Distance estimation exception on frame {fid}: {e}")
            distance_results = []
        latencies["distance"] = (time.perf_counter() - t0) * 1000.0

        # 6. Context-Aware Danger Mapping
        t0 = time.perf_counter()
        danger_assessments = []
        try:
            if self.danger_mapper:
                items_to_assess = distance_results if distance_results else hazards
                for item in items_to_assess:
                    da = self.danger_mapper.assess_danger(item, frame_width=float(frame_shape[1]))
                    if da:
                        danger_assessments.append(da)
        except Exception as e:
            logger.error(f"Danger mapping exception on frame {fid}: {e}")
            danger_assessments = []
        latencies["danger"] = (time.perf_counter() - t0) * 1000.0

        # 7. Image-Space Free-Space Analysis
        t0 = time.perf_counter()
        free_space_result = None
        try:
            if self.free_space_analyzer:
                free_space_result = self.free_space_analyzer.analyze_free_space(
                    danger_assessments_or_hazards=danger_assessments,
                    frame_width=float(frame_shape[1]),
                    frame_height=float(frame_shape[0]),
                )
        except Exception as e:
            logger.error(f"Free-space analysis exception on frame {fid}: {e}")
        latencies["free_space"] = (time.perf_counter() - t0) * 1000.0

        # 8. Context-Aware Decision Engine
        t0 = time.perf_counter()
        decision_result = None
        try:
            if self.decision_engine:
                regs = free_space_result.regions if (free_space_result and hasattr(free_space_result, "regions")) else {}
                dec_input = DecisionInput(
                    timestamp=now,
                    frame_id=fid,
                    regions=regs,
                    hazards=danger_assessments,
                    previous_command=self.last_command,
                    previous_decision_score=self.last_decision_score,
                    previous_decision_timestamp=self.last_decision_timestamp,
                    number_of_active_hazards=len([h for h in hazards if getattr(h, "state", "") == "ACTIVE"]),
                    number_of_remembered_hazards=len([h for h in hazards if getattr(h, "state", "") in ["OCCLUDED", "REMEMBERED"]]),
                    available_regions=["LEFT", "CENTER", "RIGHT"],
                    uncertainty_state=(getattr(free_space_result, "overall_state", "") == "UNCERTAIN" if free_space_result else False),
                )
                decision_result = self.decision_engine.decide(dec_input)
                if decision_result:
                    self.last_command = decision_result.command
                    self.last_decision_score = decision_result.decision_score
                    self.last_decision_timestamp = now
        except Exception as e:
            logger.error(f"Decision Engine exception on frame {fid}: {e}")
            decision_result = DecisionResult(
                command="STOP",
                selected_region=None,
                confidence=1.0,
                decision_score=0.0,
                reason=f"Pipeline decision exception fallback: {e}",
                timestamp=now,
            )
        latencies["decision"] = (time.perf_counter() - t0) * 1000.0

        # 9. Offline Audio Guidance & Speech Dispatch
        t0 = time.perf_counter()
        audio_result = None
        try:
            if self.audio_guidance and decision_result:
                audio_result = self.audio_guidance.speak_command(decision_result)
        except Exception as e:
            logger.error(f"Audio Guidance dispatch exception on frame {fid}: {e}")
        latencies["audio"] = (time.perf_counter() - t0) * 1000.0

        # Pipeline Totals & Telemetry
        t_end = time.perf_counter()
        total_lat_ms = (t_end - t_start) * 1000.0
        self.latency_history.append(total_lat_ms)
        if len(self.latency_history) > 100:
            self.latency_history.pop(0)

        fps = 1000.0 / total_lat_ms if total_lat_ms > 0 else 0.0
        self.total_frames_processed += 1

        result = PipelineResult(
            frame_id=fid,
            timestamp=now,
            camera_status=cam_props,
            detections=detections,
            tracks=tracks,
            hazards=hazards,
            distance_results=distance_results,
            danger_assessments=danger_assessments,
            free_space_result=free_space_result,
            decision_result=decision_result,
            audio_result=audio_result,
            module_latencies=latencies,
            total_latency=total_lat_ms,
            pipeline_fps=fps,
            system_status=self.system_status,
            error_status=self.error_message,
        )

        self.last_pipeline_result = result
        return result

    def stop(self) -> None:
        """Safely release all hardware resources across all 10 modules."""
        logger.info("Shutting down VisionGuide AI System Pipeline...")
        if self.camera:
            try:
                self.camera.stop()
            except Exception:
                pass
        if self.phmu:
            try:
                self.phmu.clear()
            except Exception:
                pass
        if self.decision_engine:
            try:
                self.decision_engine.reset()
            except Exception:
                pass
        if self.audio_guidance:
            try:
                self.audio_guidance.close()
            except Exception:
                pass

        self.system_status = SystemState.STOPPED
        self.module_status.camera = "STOPPED"
        self.module_status.audio = "STOPPED"
        logger.info("VisionGuide AI System Pipeline shut down cleanly.")

    def reset(self) -> None:
        """Reset temporal state across tracking, hazard memory, decision hysteresis, and audio guidance."""
        if self.tracker and hasattr(self.tracker, "reset"):
            self.tracker.reset()
        if self.phmu and hasattr(self.phmu, "clear"):
            self.phmu.clear()
        if self.decision_engine and hasattr(self.decision_engine, "reset"):
            self.decision_engine.reset()
        if self.audio_guidance and hasattr(self.audio_guidance, "reset"):
            self.audio_guidance.reset()

        self.last_command = None
        self.last_decision_score = None
        self.last_decision_timestamp = None
        self.latency_history.clear()
        self.error_message = None
        self.system_status = SystemState.READY
        logger.info("VisionGuide AI Pipeline state reset.")

    def get_status(self) -> SystemState:
        return self.system_status

    def get_last_result(self) -> Optional[PipelineResult]:
        return self.last_pipeline_result

    def get_statistics(self) -> Dict[str, Any]:
        """Return aggregated benchmark telemetry and per-module latencies."""
        avg_lat = sum(self.latency_history) / len(self.latency_history) if self.latency_history else 0.0
        avg_fps = 1000.0 / avg_lat if avg_lat > 0 else 0.0
        return {
            "system_status": self.system_status.value,
            "module_statuses": self.module_status.to_dict(),
            "total_frames_processed": self.total_frames_processed,
            "avg_latency_ms": round(avg_lat, 2),
            "avg_fps": round(avg_fps, 2),
            "last_command": self.last_command,
            "error_status": self.error_message,
        }
