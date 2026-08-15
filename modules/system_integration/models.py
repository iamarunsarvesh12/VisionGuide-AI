from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from enum import Enum


class SystemState(str, Enum):
    """
    Operational status states for the unified VisionGuide AI system pipeline.
    """
    INITIALIZING = "INITIALIZING"
    READY = "READY"
    RUNNING = "RUNNING"
    WARNING = "WARNING"
    ERROR = "ERROR"
    STOPPED = "STOPPED"


@dataclass
class ModuleStatusMap:
    """
    Status snapshot tracking individual operational health across all 9 subsystem modules.
    """
    camera: str = "INITIALIZING"
    yolo: str = "INITIALIZING"
    tracking: str = "INITIALIZING"
    phmu: str = "INITIALIZING"
    distance: str = "INITIALIZING"
    danger: str = "INITIALIZING"
    free_space: str = "INITIALIZING"
    decision: str = "INITIALIZING"
    audio: str = "INITIALIZING"

    def to_dict(self) -> Dict[str, str]:
        """Convert module status map to a serializable dictionary."""
        return {
            "camera": self.camera,
            "yolo": self.yolo,
            "tracking": self.tracking,
            "phmu": self.phmu,
            "distance": self.distance,
            "danger": self.danger,
            "free_space": self.free_space,
            "decision": self.decision,
            "audio": self.audio,
        }


@dataclass
class PipelineResult:
    """
    Unified end-to-end Pipeline Execution Result produced per processed video frame.
    Aggregates perception outputs, spatial/danger maps, decision results, spoken audio outputs,
    and micro-benchmark telemetry.
    """
    frame_id: int
    timestamp: float
    camera_status: Dict[str, Any] = field(default_factory=dict)
    detections: List[Any] = field(default_factory=list)
    tracks: List[Any] = field(default_factory=list)
    hazards: List[Any] = field(default_factory=list)
    distance_results: List[Any] = field(default_factory=list)
    danger_assessments: List[Any] = field(default_factory=list)
    free_space_result: Optional[Any] = None
    decision_result: Optional[Any] = None
    audio_result: Optional[Any] = None
    module_latencies: Dict[str, float] = field(default_factory=dict)
    total_latency: float = 0.0
    pipeline_fps: float = 0.0
    system_status: SystemState = SystemState.READY
    error_status: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert pipeline result instance to a serializable dictionary."""
        return {
            "frame_id": self.frame_id,
            "timestamp": round(float(self.timestamp), 3),
            "camera_status": self.camera_status,
            "num_detections": len(self.detections),
            "num_tracks": len(self.tracks),
            "num_hazards": len(self.hazards),
            "num_distances": len(self.distance_results),
            "num_danger_assessments": len(self.danger_assessments),
            "free_space": self.free_space_result.to_dict() if hasattr(self.free_space_result, "to_dict") else str(self.free_space_result),
            "decision": self.decision_result.to_dict() if hasattr(self.decision_result, "to_dict") else str(self.decision_result),
            "audio": self.audio_result.to_dict() if hasattr(self.audio_result, "to_dict") else str(self.audio_result),
            "module_latencies_ms": {k: round(v, 2) for k, v in self.module_latencies.items()},
            "total_latency_ms": round(float(self.total_latency), 2),
            "pipeline_fps": round(float(self.pipeline_fps), 2),
            "system_status": self.system_status.value,
            "error_status": self.error_status,
        }
