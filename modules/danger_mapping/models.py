from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class DangerAssessment:
    """
    Standardized Context-Aware Danger Assessment for VisionGuide AI.
    
    Encapsulates risk scoring, danger level, spatial position zone, memory state context,
    navigation relevance, and deterministic rule-based reasoning for a tracked hazard.
    """
    track_id: int
    class_name: str
    danger_score: float  # Normalized 0.0 to 1.0
    danger_level: str    # "LOW", "MODERATE", "HIGH", "CRITICAL"
    distance_category: str  # "NEAR", "MEDIUM", "FAR", "UNKNOWN"
    estimated_distance_m: Optional[float]
    position_zone: str   # "LEFT", "CENTER", "RIGHT"
    memory_state: str    # "ACTIVE", "OCCLUDED", "REMEMBERED", "RECOVERED", "EXPIRED"
    memory_confidence: float
    persistence_score: float
    navigation_relevance: bool
    danger_factors: List[str] = field(default_factory=list)
    reasoning: str = ""
    bounding_box: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])

    def to_dict(self) -> Dict[str, Any]:
        """Convert danger assessment instance to a serializable dictionary."""
        return {
            "track_id": self.track_id,
            "class_name": self.class_name,
            "danger_score": round(float(self.danger_score), 4),
            "danger_level": self.danger_level,
            "distance_category": self.distance_category,
            "estimated_distance_m": round(float(self.estimated_distance_m), 2) if self.estimated_distance_m is not None else None,
            "position_zone": self.position_zone,
            "memory_state": self.memory_state,
            "memory_confidence": round(float(self.memory_confidence), 4),
            "persistence_score": round(float(self.persistence_score), 4),
            "navigation_relevance": self.navigation_relevance,
            "danger_factors": self.danger_factors,
            "reasoning": self.reasoning,
            "bounding_box": [round(float(v), 2) for v in self.bounding_box],
        }
