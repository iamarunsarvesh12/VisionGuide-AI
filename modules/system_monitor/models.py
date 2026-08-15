from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional


@dataclass
class ResourceMetrics:
    """System resource utilization and performance telemetry snapshot."""
    timestamp: float = 0.0
    cpu_utilization_pct: float = 0.0
    ram_usage_mb: float = 0.0
    ram_utilization_pct: float = 0.0
    pipeline_fps: float = 0.0
    total_pipeline_latency_ms: float = 0.0
    module_latencies_ms: Dict[str, float] = field(default_factory=dict)
    dropped_frames_count: int = 0
    audio_queue_length: int = 0
    active_phmu_hazards_count: int = 0
    remembered_phmu_hazards_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": round(self.timestamp, 2),
            "cpu_utilization_pct": round(self.cpu_utilization_pct, 1),
            "ram_usage_mb": round(self.ram_usage_mb, 2),
            "ram_utilization_pct": round(self.ram_utilization_pct, 1),
            "pipeline_fps": round(self.pipeline_fps, 2),
            "total_pipeline_latency_ms": round(self.total_pipeline_latency_ms, 2),
            "module_latencies_ms": {k: round(v, 2) for k, v in self.module_latencies_ms.items()},
            "dropped_frames_count": self.dropped_frames_count,
            "audio_queue_length": self.audio_queue_length,
            "active_phmu_hazards_count": self.active_phmu_hazards_count,
            "remembered_phmu_hazards_count": self.remembered_phmu_hazards_count,
        }
