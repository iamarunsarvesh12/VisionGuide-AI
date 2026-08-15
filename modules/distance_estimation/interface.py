from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

from modules.distance_estimation.models import DistanceResult


class DistanceEstimatorInterface(ABC):
    """
    Abstract interface for distance estimation subsystems in VisionGuide AI.
    """

    @abstractmethod
    def initialize(self, config_dict: Optional[Dict[str, Any]] = None) -> bool:
        """
        Initialize estimator parameters, thresholds, and class reference profiles.
        Returns True if successful, False otherwise.
        """
        pass

    @abstractmethod
    def estimate_distance(self, track_or_hazard: Any) -> DistanceResult:
        """
        Estimate relative distance and category for a single track or hazard record.
        """
        pass

    @abstractmethod
    def estimate_batch(self, tracks_or_hazards: List[Any]) -> List[DistanceResult]:
        """
        Estimate relative distance and category for a list of tracks or hazard records.
        """
        pass

    @abstractmethod
    def calibrate_class(
        self,
        class_name: str,
        known_distance_m: float,
        observed_height_px: float
    ) -> float:
        """
        Calibrate reference height profile for a specific object class using a known distance observation.
        Returns the calibrated reference height in metres.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """
        Reset internal statistics and reset custom calibration overrides.
        """
        pass

    @abstractmethod
    def get_statistics(self) -> Dict[str, Any]:
        """
        Return estimation statistics.
        """
        pass
