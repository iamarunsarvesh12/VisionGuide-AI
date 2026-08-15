import time
import os
import sys
import logging
from typing import List, Dict, Any, Optional

from modules.danger_mapping.models import DangerAssessment
from modules.danger_mapping.interface import DangerMapperInterface

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

# Module logger setup
logger = logging.getLogger("DangerMapping")
logger.setLevel(logging.INFO)
if not logger.handlers:
    c_handler = logging.StreamHandler(sys.stdout)
    c_handler.setFormatter(logging.Formatter("[%(name)s][%(levelname)s] %(message)s"))
    logger.addHandler(c_handler)
    
    f_handler = logging.FileHandler("logs/danger_mapping.log", mode="a", encoding="utf-8")
    f_handler.setFormatter(logging.Formatter("%(asctime)s [%(name)s][%(levelname)s] %(message)s"))
    logger.addHandler(f_handler)

DEFAULT_OBJECT_HAZARD_FACTORS = {
    "stairs": 0.95,
    "glass_door": 0.90,
    "glass_wall": 0.85,
    "person": 0.70,
    "chair": 0.65,
    "table": 0.60,
    "door": 0.50,
    "cabinet": 0.50,
    "corridor": 0.40,
    "exit": 0.30,
    "default": 0.50,
}

DEFAULT_WEIGHTS = {
    "object": 0.25,
    "distance": 0.30,
    "position": 0.25,
    "motion": 0.05,
    "persistence": 0.10,
    "memory": 0.05,
}


class ContextAwareDangerMapper(DangerMapperInterface):
    """
    Context-Aware Danger Mapper implementation for VisionGuide AI.
    
    Evaluates multi-factor risk scores (object type, distance, spatial position zone,
    motion vector, persistence score, and memory confidence) to generate normalized
    danger scores, levels (LOW, MODERATE, HIGH, CRITICAL), and deterministic reasoning strings.
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        thresholds: Optional[Dict[str, float]] = None,
        position_zones: Optional[Dict[str, float]] = None,
        object_factors: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize mapper configuration and weight definitions.
        """
        self.weights = dict(DEFAULT_WEIGHTS)
        if weights:
            self.weights.update(weights)

        self.thresholds = {"critical": 0.85, "high": 0.70, "moderate": 0.55}
        if thresholds:
            self.thresholds.update(thresholds)

        self.position_zones = {"left_max": 0.33, "center_max": 0.67, "frame_width": 640.0}
        if position_zones:
            self.position_zones.update(position_zones)

        self.object_factors = dict(DEFAULT_OBJECT_HAZARD_FACTORS)
        if object_factors:
            self.object_factors.update(object_factors)

        self._is_initialized: bool = False
        self.total_assessments: int = 0
        self.last_mapping_latency_ms: float = 0.0
        self._total_mapping_time_s: float = 0.0

    def initialize(self, config_dict: Optional[Dict[str, Any]] = None) -> bool:
        """Initialize parameters from configuration dictionary."""
        t0 = time.perf_counter()
        if config_dict:
            cfg = config_dict.get("danger_mapping", {})
            if "weights" in cfg:
                self.weights.update(cfg["weights"])
            if "thresholds" in cfg:
                self.thresholds.update(cfg["thresholds"])
            if "position_zones" in cfg:
                self.position_zones.update(cfg["position_zones"])

            if "object_hazard_factors" in config_dict:
                self.object_factors.update(config_dict["object_hazard_factors"])

        self.reset()
        self._is_initialized = True
        t1 = time.perf_counter()

        logger.info(
            f"ContextAwareDangerMapper Initialized (weights={self.weights}, "
            f"thresholds={self.thresholds})"
        )
        return True

    def _determine_position_zone(self, center_x: float, frame_width: Optional[float] = None) -> str:
        """Determine horizontal position zone (LEFT, CENTER, RIGHT)."""
        width = frame_width if (frame_width and frame_width > 0) else self.position_zones.get("frame_width", 640.0)
        x_norm = center_x / float(width) if width > 0 else 0.5

        left_max = self.position_zones.get("left_max", 0.33)
        center_max = self.position_zones.get("center_max", 0.67)

        if x_norm < left_max:
            return "LEFT"
        elif x_norm <= center_max:
            return "CENTER"
        else:
            return "RIGHT"

    def _get_object_hazard_factor(self, class_name: str) -> float:
        """Retrieve object class hazard factor."""
        return self.object_factors.get(class_name.lower(), self.object_factors.get("default", 0.50))

    def _get_distance_factor(self, category: str) -> float:
        """Map distance category to numeric hazard factor."""
        mapping = {"NEAR": 1.00, "MEDIUM": 0.50, "FAR": 0.20, "UNKNOWN": 0.10}
        return mapping.get(category.upper(), 0.10)

    def _get_position_factor(self, zone: str) -> float:
        """Map position zone to corridor obstruction factor."""
        return 1.00 if zone.upper() == "CENTER" else 0.50

    def assess_danger(
        self,
        distance_result_or_hazard: Any,
        frame_width: Optional[float] = None
    ) -> DangerAssessment:
        """
        Calculate context-aware danger score, level, navigation relevance, and deterministic reasoning.
        """
        if not self._is_initialized:
            self.initialize()

        t_start = time.perf_counter()

        if distance_result_or_hazard is None:
            return DangerAssessment(
                track_id=-1, class_name="unknown", danger_score=0.0, danger_level="LOW",
                distance_category="UNKNOWN", estimated_distance_m=None, position_zone="CENTER",
                memory_state="EXPIRED", memory_confidence=0.0, persistence_score=0.0,
                navigation_relevance=False, danger_factors=["Invalid input"], reasoning="No valid hazard input"
            )

        # Extract attributes from DistanceResult, HazardMemoryRecord, or Track
        track_id = getattr(distance_result_or_hazard, "track_id", -1)
        class_name = getattr(distance_result_or_hazard, "object_class", getattr(distance_result_or_hazard, "class_name", "unknown"))
        dist_cat = getattr(distance_result_or_hazard, "distance_category", "UNKNOWN")
        est_dist_m = getattr(distance_result_or_hazard, "estimated_distance_m", None)
        bbox = getattr(distance_result_or_hazard, "bounding_box", [0, 0, 0, 0])
        mem_state = getattr(distance_result_or_hazard, "memory_state", "ACTIVE")
        mem_conf = getattr(distance_result_or_hazard, "distance_confidence", getattr(distance_result_or_hazard, "memory_confidence", getattr(distance_result_or_hazard, "confidence", 1.0)))
        pers_score = getattr(distance_result_or_hazard, "persistence_score", 1.0)
        motion_val = getattr(distance_result_or_hazard, "motion_factor", 0.50)

        # Calculate centroid center_x
        if hasattr(distance_result_or_hazard, "center_x"):
            center_x = getattr(distance_result_or_hazard, "center_x")
        elif bbox and len(bbox) == 4:
            center_x = (bbox[0] + bbox[2]) / 2.0
        else:
            center_x = 320.0

        pos_zone = self._determine_position_zone(center_x, frame_width)

        # Compute individual factors
        f_obj = self._get_object_hazard_factor(class_name)
        f_dist = self._get_distance_factor(dist_cat)
        f_pos = self._get_position_factor(pos_zone)
        f_mot = max(0.0, min(1.0, float(motion_val)))
        f_pers = max(0.0, min(1.0, float(pers_score)))
        f_mem = max(0.0, min(1.0, float(mem_conf)))

        # Weighted Linear Combination
        raw_score = (
            self.weights["object"] * f_obj +
            self.weights["distance"] * f_dist +
            self.weights["position"] * f_pos +
            self.weights["motion"] * f_mot +
            self.weights["persistence"] * f_pers +
            self.weights["memory"] * f_mem
        )
        danger_score = max(0.0, min(1.0, float(raw_score)))

        # Determine Danger Level
        if danger_score >= self.thresholds["critical"]:
            level = "CRITICAL"
        elif danger_score >= self.thresholds["high"]:
            level = "HIGH"
        elif danger_score >= self.thresholds["moderate"]:
            level = "MODERATE"
        else:
            level = "LOW"

        # Determine Navigation Relevance
        nav_relevant = (level in ("CRITICAL", "HIGH", "MODERATE")) or (class_name.lower() in ("door", "exit", "stairs", "corridor"))

        # Build Factors List & Deterministic Reasoning
        factors = []
        reason_parts = []

        if f_obj >= 0.8:
            factors.append(f"High-risk object class: {class_name}")
            reason_parts.append(f"High-risk {class_name}")
        else:
            reason_parts.append(f"{class_name.capitalize()}")

        factors.append(f"Distance category: {dist_cat}")
        reason_parts.append(f"{dist_cat} distance")

        factors.append(f"Position zone: {pos_zone}")
        reason_parts.append(f"in {pos_zone} zone")

        if mem_state in ("OCCLUDED", "REMEMBERED"):
            factors.append(f"REMEMBERED hazard with decayed confidence ({mem_conf:.2f})")
            reason_parts.append("(Remembered hazard)")

        reasoning = ", ".join(reason_parts)

        assessment = DangerAssessment(
            track_id=track_id,
            class_name=class_name,
            danger_score=danger_score,
            danger_level=level,
            distance_category=dist_cat,
            estimated_distance_m=est_dist_m,
            position_zone=pos_zone,
            memory_state=mem_state,
            memory_confidence=mem_conf,
            persistence_score=pers_score,
            navigation_relevance=nav_relevant,
            danger_factors=factors,
            reasoning=reasoning,
            bounding_box=list(bbox)
        )

        t_end = time.perf_counter()
        self.last_mapping_latency_ms = (t_end - t_start) * 1000.0
        self.total_assessments += 1
        self._total_mapping_time_s += (t_end - t_start)

        logger.info(
            f"[Danger] ID={track_id} class={class_name} distance={dist_cat} "
            f"position={pos_zone} score={danger_score:.2f} level={level} state={mem_state}"
        )
        return assessment

    def assess_batch(
        self,
        distance_results_or_hazards: List[Any],
        frame_width: Optional[float] = None
    ) -> List[DangerAssessment]:
        """Process batch list of hazards and return sorted list by danger_score descending."""
        if not distance_results_or_hazards:
            return []
        
        # Exclude EXPIRED hazards
        valid_items = [
            item for item in distance_results_or_hazards
            if getattr(item, "memory_state", "ACTIVE") != "EXPIRED"
        ]

        assessments = [self.assess_danger(item, frame_width) for item in valid_items]
        return self.rank_hazards(assessments)

    def rank_hazards(self, assessments: List[DangerAssessment]) -> List[DangerAssessment]:
        """Sort DangerAssessment records by danger_score in descending order (highest danger first)."""
        if not assessments:
            return []
        return sorted(assessments, key=lambda a: a.danger_score, reverse=True)

    def reset(self) -> None:
        """Reset internal statistics."""
        self.total_assessments = 0
        self._total_mapping_time_s = 0.0
        self.last_mapping_latency_ms = 0.0
        logger.info("ContextAwareDangerMapper reset.")

    def get_statistics(self) -> Dict[str, Any]:
        """Return operational danger mapping statistics."""
        avg_fps = (self.total_assessments / self._total_mapping_time_s) if self._total_mapping_time_s > 0 else 0.0
        avg_lat = (self._total_mapping_time_s * 1000.0 / self.total_assessments) if self.total_assessments > 0 else 0.0

        return {
            "is_initialized": self._is_initialized,
            "weights": self.weights,
            "thresholds": self.thresholds,
            "total_assessments": self.total_assessments,
            "last_mapping_latency_ms": round(self.last_mapping_latency_ms, 3),
            "average_mapping_latency_ms": round(avg_lat, 3),
            "average_mapping_throughput_fps": round(avg_fps, 2),
        }
