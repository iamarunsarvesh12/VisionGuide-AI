from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np

from modules.object_detection.interface import Detection


@dataclass
class Track:
    """
    Standardized multi-object tracking record for VisionGuide AI.
    
    Associates spatial detection bounding boxes with persistent temporal identities (track_id).
    """
    track_id: int
    class_id: int
    class_name: str
    confidence: float
    bounding_box: List[float]  # [x1, y1, x2, y2]
    center_x: float
    center_y: float
    width: float
    height: float
    tracking_state: str  # "NEW", "TRACKED", "LOST", "REMOVED"
    age: int = 1
    hits: int = 1
    time_since_update: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert track instance to a serializable dictionary."""
        return {
            "track_id": self.track_id,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": round(float(self.confidence), 4),
            "bounding_box": [round(float(v), 2) for v in self.bounding_box],
            "center_x": round(float(self.center_x), 2),
            "center_y": round(float(self.center_y), 2),
            "width": round(float(self.width), 2),
            "height": round(float(self.height), 2),
            "tracking_state": self.tracking_state,
            "age": self.age,
            "hits": self.hits,
            "time_since_update": self.time_since_update,
        }


class ObjectTrackerInterface(ABC):
    """
    Abstract interface for object tracking algorithms in VisionGuide AI.
    """

    @abstractmethod
    def initialize(self, tracker_type: str = "botsort", max_age: int = 30) -> bool:
        """
        Initialize tracking engine parameters and state containers.
        Returns True if successful, False otherwise.
        """
        pass

    @abstractmethod
    def update(
        self,
        detections: List[Detection],
        frame: Optional[np.ndarray] = None
    ) -> List[Track]:
        """
        Update tracker state with new frame detections.
        
        Args:
            detections (List[Detection]): Detections produced by ObjectDetectorInterface.
            frame (Optional[np.ndarray]): Current video frame matrix for visual features / ReID.
            
        Returns:
            List[Track]: Active persistent tracked objects.
        """
        pass

    @abstractmethod
    def get_active_tracks(self) -> List[Track]:
        """
        Retrieve all currently active tracks.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """
        Reset tracker state and clear active track memory.
        """
        pass
