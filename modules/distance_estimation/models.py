from dataclasses import dataclass
from typing import List, Dict, Any, Optional


@dataclass
class DistanceResult:
    """
    Standardized Distance Estimation Output for VisionGuide AI.
    
    Represents approximate monocular distance and proximity categorization for a tracked hazard.
    """
    track_id: int
    class_name: str
    distance_category: str  # "NEAR", "MEDIUM", "FAR", "UNKNOWN"
    distance_confidence: float
    estimated_distance_m: Optional[float]
    bounding_box: List[float]  # [x1, y1, x2, y2]
    estimation_method: str = "monocular_bbox"
    distance_status: str = "MEASURED"  # "MEASURED" or "LAST_OBSERVED"

    def to_dict(self) -> Dict[str, Any]:
        """Convert distance result instance to a serializable dictionary."""
        return {
            "track_id": self.track_id,
            "class_name": self.class_name,
            "distance_category": self.distance_category,
            "distance_confidence": round(float(self.distance_confidence), 4),
            "estimated_distance_m": round(float(self.estimated_distance_m), 2) if self.estimated_distance_m is not None else None,
            "bounding_box": [round(float(v), 2) for v in self.bounding_box],
            "estimation_method": self.estimation_method,
            "distance_status": self.distance_status,
        }
