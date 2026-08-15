from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class RegionOccupancy:
    """
    Standardized Image-Space Region Occupancy Assessment for VisionGuide AI.
    
    Represents traversability, occupancy score, safe space score, blocked object IDs,
    dominant danger level, confidence, and reasoning for a specific navigation zone.
    """
    region_name: str  # "LEFT", "CENTER", "RIGHT"
    occupancy_state: str  # "CLEAR", "BLOCKED", "UNCERTAIN"
    occupancy_score: float  # Normalized 0.0 to 1.0
    safe_space_score: float  # Normalized 1.0 - occupancy_score
    blocked_object_ids: List[int] = field(default_factory=list)
    dominant_danger_level: str = "NONE"  # "NONE", "LOW", "MODERATE", "HIGH", "CRITICAL"
    confidence: float = 1.0
    reasoning: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert region occupancy instance to a serializable dictionary."""
        return {
            "region_name": self.region_name,
            "occupancy_state": self.occupancy_state,
            "occupancy_score": round(float(self.occupancy_score), 4),
            "safe_space_score": round(float(self.safe_space_score), 4),
            "blocked_object_ids": self.blocked_object_ids,
            "dominant_danger_level": self.dominant_danger_level,
            "confidence": round(float(self.confidence), 4),
            "reasoning": self.reasoning,
        }


@dataclass
class FreeSpaceAnalysisResult:
    """
    Aggregated Free-Space Scene Traversability Result for VisionGuide AI.
    """
    regions: Dict[str, RegionOccupancy]  # Map of "LEFT", "CENTER", "RIGHT" to RegionOccupancy
    total_hazards_assessed: int = 0
    overall_traversability: str = "CLEAR"  # "CLEAR", "PARTIALLY_BLOCKED", "BLOCKED"

    def to_dict(self) -> Dict[str, Any]:
        """Convert free space analysis result to a serializable dictionary."""
        return {
            "regions": {k: v.to_dict() for k, v in self.regions.items()},
            "total_hazards_assessed": self.total_hazards_assessed,
            "overall_traversability": self.overall_traversability,
        }
