from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class HazardMemoryRecord:
    """
    Persistent Hazard Memory Record for VisionGuide AI.
    
    Represents a navigation-relevant hazard maintained over time by the PHMU subsystem.
    Preserves temporal state across temporary occlusions, camera movements, or detector failures.
    """
    track_id: int
    object_class: str
    class_id: int
    bounding_box: List[float]  # [x1, y1, x2, y2]
    center_x: float
    center_y: float
    width: float
    height: float
    
    # Distance and Danger fields (Placeholder None until Phase 5 & Phase 6)
    estimated_distance: Optional[float] = None
    danger_score: Optional[float] = None

    # Temporal Memory & Confidence Metrics
    persistence_score: float = 1.0
    memory_confidence: float = 1.0
    detection_confidence: float = 1.0
    
    # Frame and Time Metadata
    last_seen_timestamp: float = 0.0
    last_seen_frame: int = 0
    observation_count: int = 1
    track_age_frames: int = 1
    time_since_last_seen: float = 0.0

    # Lifecycle State Tracking
    tracking_state: str = "TRACKED"
    memory_state: str = "ACTIVE"  # "ACTIVE", "OCCLUDED", "REMEMBERED", "RECOVERED", "EXPIRED"

    def to_dict(self) -> Dict[str, Any]:
        """Convert hazard memory record to a serializable dictionary."""
        return {
            "track_id": self.track_id,
            "object_class": self.object_class,
            "class_id": self.class_id,
            "bounding_box": [round(float(v), 2) for v in self.bounding_box],
            "center_x": round(float(self.center_x), 2),
            "center_y": round(float(self.center_y), 2),
            "width": round(float(self.width), 2),
            "height": round(float(self.height), 2),
            "estimated_distance": self.estimated_distance,
            "danger_score": self.danger_score,
            "persistence_score": round(float(self.persistence_score), 4),
            "memory_confidence": round(float(self.memory_confidence), 4),
            "detection_confidence": round(float(self.detection_confidence), 4),
            "last_seen_timestamp": round(float(self.last_seen_timestamp), 3),
            "last_seen_frame": self.last_seen_frame,
            "observation_count": self.observation_count,
            "track_age_frames": self.track_age_frames,
            "time_since_last_seen": round(float(self.time_since_last_seen), 3),
            "tracking_state": self.tracking_state,
            "memory_state": self.memory_state,
        }
