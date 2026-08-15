from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum


class NavigationCommand(str, Enum):
    """
    Fixed command enumeration for Phase 8 Context-Aware Decision Engine.
    Strictly limited to immediate deterministic directional and stopping commands.
    """
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    FORWARD = "FORWARD"
    STOP = "STOP"


@dataclass
class RegionDecisionScore:
    """
    Detailed decision evaluation breakdown for a specific navigation region.
    """
    region_name: str  # "LEFT", "CENTER", "RIGHT"
    safe_space_score: float
    danger_score: float
    confidence: float
    uncertainty: float
    stability_score: float
    final_score: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert region decision score instance to a serializable dictionary."""
        return {
            "region_name": self.region_name,
            "safe_space_score": round(float(self.safe_space_score), 4),
            "danger_score": round(float(self.danger_score), 4),
            "confidence": round(float(self.confidence), 4),
            "uncertainty": round(float(self.uncertainty), 4),
            "stability_score": round(float(self.stability_score), 4),
            "final_score": round(float(self.final_score), 4),
        }


@dataclass
class DecisionInput:
    """
    Structured Input Model for Module 09 — Context-Aware Decision Engine.
    
    Synthesizes regional free-space occupancy assessment, danger assessments,
    PHMU temporal memory state, spatial positioning, and temporal history.
    """
    timestamp: float
    frame_id: Optional[int] = None
    regions: Dict[str, Any] = field(default_factory=dict)  # "LEFT", "CENTER", "RIGHT" -> RegionOccupancy or dict
    hazards: List[Any] = field(default_factory=list)        # List of DangerAssessment or dict
    previous_command: Optional[str] = None
    previous_decision_score: Optional[float] = None
    previous_decision_timestamp: Optional[float] = None
    number_of_active_hazards: int = 0
    number_of_remembered_hazards: int = 0
    available_regions: List[str] = field(default_factory=list)
    uncertainty_state: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert decision input to a serializable dictionary."""
        return {
            "timestamp": round(float(self.timestamp), 3),
            "frame_id": self.frame_id,
            "regions": {
                k: (v.to_dict() if hasattr(v, "to_dict") else v)
                for k, v in self.regions.items()
            },
            "hazards": [
                (h.to_dict() if hasattr(h, "to_dict") else h)
                for h in self.hazards
            ],
            "previous_command": self.previous_command,
            "previous_decision_score": round(float(self.previous_decision_score), 4) if self.previous_decision_score is not None else None,
            "previous_decision_timestamp": round(float(self.previous_decision_timestamp), 3) if self.previous_decision_timestamp is not None else None,
            "number_of_active_hazards": self.number_of_active_hazards,
            "number_of_remembered_hazards": self.number_of_remembered_hazards,
            "available_regions": self.available_regions,
            "uncertainty_state": self.uncertainty_state,
        }


@dataclass
class DecisionResult:
    """
    Structured Output Model for Module 09 — Context-Aware Decision Engine.
    
    Contains the final deterministic navigation directive, target region, confidence,
    scoring metric, human/machine-readable reasoning, blocking hazard track IDs, and alternatives.
    """
    command: str  # "LEFT", "RIGHT", "FORWARD", "STOP"
    selected_region: Optional[str]  # "LEFT", "CENTER", "RIGHT", or None for STOP
    confidence: float  # 0.0 to 1.0
    decision_score: float  # 0.0 to 1.0 (or normalized candidate score)
    reason: str
    blocking_hazards: List[int] = field(default_factory=list)  # Track IDs of blocking hazards
    alternative_regions: List[str] = field(default_factory=list)
    timestamp: float = 0.0
    regional_scores: Dict[str, RegionDecisionScore] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert decision result instance to a serializable dictionary."""
        return {
            "command": self.command,
            "selected_region": self.selected_region,
            "confidence": round(float(self.confidence), 4),
            "decision_score": round(float(self.decision_score), 4),
            "reason": self.reason,
            "blocking_hazards": self.blocking_hazards,
            "alternative_regions": self.alternative_regions,
            "timestamp": round(float(self.timestamp), 3),
            "regional_scores": {
                k: (v.to_dict() if hasattr(v, "to_dict") else v)
                for k, v in self.regional_scores.items()
            },
        }
