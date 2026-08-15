from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np

from modules.system_integration.models import SystemState, PipelineResult


class SystemPipelineInterface(ABC):
    """
    Abstract interface for the VisionGuide AI System Pipeline.
    
    Orchestrates the sequential end-to-end execution of all 10 subsystems:
    Camera -> YOLOv8m -> BoT-SORT -> PHMU -> Distance -> Danger -> Free Space -> Decision Engine -> Audio Guidance.
    """

    @abstractmethod
    def initialize(self, config_dict: Optional[Dict[str, Any]] = None) -> bool:
        """
        Initialize configuration and instantiate/boot all 10 underlying subsystems.
        Returns True if successful, False otherwise.
        """
        pass

    @abstractmethod
    def process_frame(self, frame: Optional[np.ndarray] = None) -> PipelineResult:
        """
        Process a single visual frame through the complete 10-module perception,
        spatial memory, risk assessment, decision, and acoustic output pipeline.
        If frame is None, captures a new frame from the active Camera Input module.
        """
        pass

    @abstractmethod
    def start(self) -> bool:
        """
        Start camera acquisition and background processing streams.
        """
        pass

    @abstractmethod
    def stop(self) -> None:
        """
        Safely stop video capture, interrupt active speech, release hardware resources,
        and shut down worker threads across all modules.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """
        Reset internal pipeline state, PHMU temporal memory, decision hysteresis,
        and audio guidance queues.
        """
        pass

    @abstractmethod
    def get_status(self) -> SystemState:
        """
        Return the current system-wide operational state.
        """
        pass

    @abstractmethod
    def get_last_result(self) -> Optional[PipelineResult]:
        """
        Return the most recent processed PipelineResult.
        """
        pass

    @abstractmethod
    def get_statistics(self) -> Dict[str, Any]:
        """
        Return aggregated performance metrics, per-module latencies, and FPS telemetry.
        """
        pass
