from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from modules.decision_engine.models import DecisionInput, DecisionResult


class DecisionEngineInterface(ABC):
    """
    Abstract interface for context-aware decision engine in VisionGuide AI.
    """

    @abstractmethod
    def initialize(self, config_dict: Optional[Dict[str, Any]] = None) -> bool:
        """
        Initialize scoring weights, thresholds, and hysteresis state.
        Returns True if successful, False otherwise.
        """
        pass

    @abstractmethod
    def decide(self, decision_input: DecisionInput) -> DecisionResult:
        """
        Synthesize perception context, free-space scores, danger levels, and temporal history
        into a deterministic navigation command (LEFT, RIGHT, FORWARD, STOP).
        
        Args:
            decision_input (DecisionInput): Complete environmental and temporal state context.
            
        Returns:
            DecisionResult: Structured decision result with command, region, confidence, score, and reasoning.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """
        Reset internal hysteresis tracking, previous commands, and statistics.
        """
        pass

    @abstractmethod
    def get_statistics(self) -> Dict[str, Any]:
        """
        Return operational decision engine statistics.
        """
        pass

    @abstractmethod
    def get_last_decision(self) -> Optional[DecisionResult]:
        """
        Return the most recently generated DecisionResult.
        """
        pass
