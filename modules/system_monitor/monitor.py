import os
import sys
import time
import psutil
import logging
from typing import Dict, Any, List, Optional

from modules.system_monitor.models import ResourceMetrics

os.makedirs("logs", exist_ok=True)

logger = logging.getLogger("SystemMonitor")
logger.setLevel(logging.INFO)
if not logger.handlers:
    c_handler = logging.StreamHandler(sys.stdout)
    c_handler.setFormatter(logging.Formatter("[%(name)s][%(levelname)s] %(message)s"))
    logger.addHandler(c_handler)


class SystemResourceMonitor:
    """
    Real-Time Telemetry & Resource Monitor for VisionGuide AI.
    Tracks CPU load, RAM footprint, pipeline frame latencies, dropped frames,
    audio queue depth, and PHMU memory counts.
    """

    def __init__(self):
        self._process = psutil.Process(os.getpid())
        self.history: List[ResourceMetrics] = []
        self._dropped_frames: int = 0

    def capture_metrics(
        self,
        pipeline_result: Optional[Any] = None,
        audio_queue_length: int = 0
    ) -> ResourceMetrics:
        """Capture current resource utilization and pipeline performance metrics."""
        now = time.time()
        cpu_pct = psutil.cpu_percent(interval=None)
        mem_info = self._process.memory_info()
        ram_mb = mem_info.rss / (1024.0 * 1024.0)
        sys_mem = psutil.virtual_memory()

        tot_lat = getattr(pipeline_result, "total_latency", 0.0) if pipeline_result else 0.0
        fps = getattr(pipeline_result, "pipeline_fps", 0.0) if pipeline_result else 0.0
        mod_lats = getattr(pipeline_result, "module_latencies", {}) if pipeline_result else {}
        hazards = getattr(pipeline_result, "hazards", []) if pipeline_result else []

        active_cnt = len([h for h in hazards if getattr(h, "memory_state", "") == "ACTIVE"])
        remembered_cnt = len([h for h in hazards if getattr(h, "memory_state", "") in ["OCCLUDED", "REMEMBERED"]])

        if tot_lat > 1000.0:  # Dropped frame threshold (> 1s processing latency)
            self._dropped_frames += 1

        metrics = ResourceMetrics(
            timestamp=now,
            cpu_utilization_pct=cpu_pct,
            ram_usage_mb=ram_mb,
            ram_utilization_pct=sys_mem.percent,
            pipeline_fps=fps,
            total_pipeline_latency_ms=tot_lat,
            module_latencies_ms=mod_lats,
            dropped_frames_count=self._dropped_frames,
            audio_queue_length=audio_queue_length,
            active_phmu_hazards_count=active_cnt,
            remembered_phmu_hazards_count=remembered_cnt,
        )

        self.history.append(metrics)
        return metrics

    def get_summary(self) -> Dict[str, Any]:
        """Compute aggregated resource telemetry metrics."""
        if not self.history:
            return {}

        avg_cpu = sum(m.cpu_utilization_pct for m in self.history) / len(self.history)
        avg_ram = sum(m.ram_usage_mb for m in self.history) / len(self.history)
        peak_ram = max(m.ram_usage_mb for m in self.history)
        avg_fps = sum(m.pipeline_fps for m in self.history) / len(self.history)
        avg_lat = sum(m.total_pipeline_latency_ms for m in self.history) / len(self.history)

        return {
            "total_samples": len(self.history),
            "avg_cpu_percent": round(avg_cpu, 1),
            "avg_ram_mb": round(avg_ram, 2),
            "peak_ram_mb": round(peak_ram, 2),
            "avg_fps": round(avg_fps, 2),
            "avg_pipeline_latency_ms": round(avg_lat, 2),
            "total_dropped_frames": self._dropped_frames,
        }
