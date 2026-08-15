from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

from modules.danger_mapping.models import DangerAssessment


class DangerMapperInterface(ABC):
    """
    Abstract interface for context-aware danger mapping in VisionGuide AI.
    """

    @abstractmethod
    def initialize(self, config_dict: Optional[Dict[str, Any]] = None) -> bool:
        """
        Initialize scoring weights, danger thresholds, and position zone parameters.
        Returns True if successful, False otherwise.
        """
        pass

    @abstractmethod
    def assess_danger(
        self,
        distance_result_or_hazard: Any,
        frame_width: Optional[float] = None
    ) -> DangerAssessment:
        """
        Compute danger score, level, position zone, and reasoning for a single hazard entry.
        """
        pass

    @abstractmethod
    def assess_batch(
        self,
        distance_results_or_hazards: List[Any],
        frame_width: Optional[float] = None
    ) -> List[DangerAssessment]:
        """
        Assess danger for a list of hazards and return sorted list by danger score (highest first).
        """
        pass

    @abstractmethod
    def rank_hazards(self, assessments: List[DangerAssessment]) -> List[DangerAssessment]:
        """
        Sort list of DangerAssessment objects by danger_score in descending order.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """
        Reset internal statistics counters.
        """
        pass

    @abstractmethod
    def get_statistics(self) -> Dict[str, Any]:
        """
        Return operational danger mapping statistics.
        """
        pass
