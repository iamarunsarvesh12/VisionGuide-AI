from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

from modules.object_tracking.interface import Track
from modules.hazard_memory.models import HazardMemoryRecord


class HazardMemoryInterface(ABC):
    """
    Abstract interface for the Persistent Hazard Memory Unit (PHMU) in VisionGuide AI.
    """

    @abstractmethod
    def initialize(self, config_dict: Optional[Dict[str, Any]] = None) -> bool:
        """
        Initialize PHMU configuration, decay rates, and memory storage.
        Returns True if successful, False otherwise.
        """
        pass

    @abstractmethod
    def update(
        self,
        tracks: List[Track],
        current_time: Optional[float] = None,
        frame_index: Optional[int] = None
    ) -> List[HazardMemoryRecord]:
        """
        Process current frame BoT-SORT tracks, update memory states, perform decay,
        and manage memory lifecycle transitions.
        
        Args:
            tracks (List[Track]): Current frame tracks from Module 04.
            current_time (Optional[float]): Timestamp override (seconds).
            frame_index (Optional[int]): Sequential frame index counter.
            
        Returns:
            List[HazardMemoryRecord]: Active & retained remembered hazard records.
        """
        pass

    @abstractmethod
    def get_active_hazards(self) -> List[HazardMemoryRecord]:
        """
        Retrieve all currently non-expired hazards (ACTIVE, OCCLUDED, REMEMBERED, RECOVERED).
        """
        pass

    @abstractmethod
    def get_hazard(self, track_id: int) -> Optional[HazardMemoryRecord]:
        """
        Retrieve hazard record for a specific track_id if present.
        """
        pass

    @abstractmethod
    def mark_missing(self, track_id: int, current_time: Optional[float] = None) -> None:
        """
        Explicitly transition a track memory into OCCLUDED / REMEMBERED state.
        """
        pass

    @abstractmethod
    def expire_memories(self, current_time: Optional[float] = None) -> List[int]:
        """
        Purge expired hazard memories exceeding timeout threshold.
        Returns list of expired track_ids.
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """
        Reset memory storage and state statistics.
        """
        pass

    @abstractmethod
    def get_statistics(self) -> Dict[str, Any]:
        """
        Retrieve memory lifecycle execution statistics.
        """
        pass
