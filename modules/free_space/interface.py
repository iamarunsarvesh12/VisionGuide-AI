from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

from modules.free_space.models import FreeSpaceAnalysisResult


class FreeSpaceAnalyzerInterface(ABC):
    """
    Abstract interface for image-space free-space analysis in VisionGuide AI.
    """

    @abstractmethod
    def initialize(self, config_dict: Optional[Dict[str, Any]] = None) -> bool:
        """
        Initialize region boundaries, occupancy thresholds, and lower corridor parameters.
        Returns True if successful, False otherwise.
        """
        pass

    @abstractmethod
    def analyze_free_space(
        self,
        danger_assessments_or_hazards: List[Any],
        frame_width: float = 640.0,
        frame_height: float = 480.0
    ) -> FreeSpaceAnalysisResult:
        """
        Analyze image-space traversability across LEFT, CENTER, and RIGHT navigation regions.
        
        Args:
            danger_assessments_or_hazards (List[Any]): Danger assessments from Module 07 or PHMU records.
            frame_width (float): Horizontal camera resolution in pixels.
            frame_height (float): Vertical camera resolution in pixels.
            
        Returns:
            FreeSpaceAnalysisResult: Structured regional occupancy result.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """
        Reset internal statistics and state containers.
        """
        pass

    @abstractmethod
    def get_statistics(self) -> Dict[str, Any]:
        """
        Return operational free-space analysis statistics.
        """
        pass
