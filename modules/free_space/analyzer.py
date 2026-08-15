import time
import os
import sys
import logging
from typing import List, Dict, Any, Optional, Tuple

from modules.free_space.models import RegionOccupancy, FreeSpaceAnalysisResult
from modules.free_space.interface import FreeSpaceAnalyzerInterface

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

# Module logger setup
logger = logging.getLogger("FreeSpace")
logger.setLevel(logging.INFO)
if not logger.handlers:
    c_handler = logging.StreamHandler(sys.stdout)
    c_handler.setFormatter(logging.Formatter("[%(name)s][%(levelname)s] %(message)s"))
    logger.addHandler(c_handler)
    
    f_handler = logging.FileHandler("logs/free_space.log", mode="a", encoding="utf-8")
    f_handler.setFormatter(logging.Formatter("%(asctime)s [%(name)s][%(levelname)s] %(message)s"))
    logger.addHandler(f_handler)

DEFAULT_REGIONS = {
    "LEFT": (0.00, 0.33),
    "CENTER": (0.33, 0.67),
    "RIGHT": (0.67, 1.00),
}


class ImageSpaceFreeSpaceAnalyzer(FreeSpaceAnalyzerInterface):
    """
    Image-Space Free-Space Analyzer implementation for VisionGuide AI.
    
    Evaluates visual scene traversability across horizontal regions (LEFT, CENTER, RIGHT)
    by computing bounding-box interval overlaps, distance factors, danger scores, PHMU memory
    states, and walking corridor lower-mask boosts to produce regional occupancy states
    (CLEAR, BLOCKED, UNCERTAIN) and safe-space scores.
    """

    def __init__(
        self,
        regions: Optional[Dict[str, Tuple[float, float]]] = None,
        clear_max_threshold: float = 0.20,
        uncertain_max_threshold: float = 0.45,
        lower_mask_min_y_norm: float = 0.40,
    ):
        """
        Initialize free-space analyzer parameters.
        """
        self.regions = dict(DEFAULT_REGIONS)
        if regions:
            self.regions.update(regions)

        self.clear_max_threshold = clear_max_threshold
        self.uncertain_max_threshold = uncertain_max_threshold
        self.lower_mask_min_y_norm = lower_mask_min_y_norm

        self._is_initialized: bool = False
        self.total_analyses: int = 0
        self.last_analysis_latency_ms: float = 0.0
        self._total_analysis_time_s: float = 0.0

    def initialize(self, config_dict: Optional[Dict[str, Any]] = None) -> bool:
        """Initialize parameters from configuration dictionary."""
        t0 = time.perf_counter()
        if config_dict:
            cfg = config_dict.get("free_space", {})
            thresh = cfg.get("thresholds", {})
            self.clear_max_threshold = thresh.get("clear_max", self.clear_max_threshold)
            self.uncertain_max_threshold = thresh.get("uncertain_max", self.uncertain_max_threshold)
            self.lower_mask_min_y_norm = cfg.get("lower_mask_min_y_norm", self.lower_mask_min_y_norm)

            regs = cfg.get("regions", {})
            if "left_end" in regs and "center_end" in regs:
                l_end = float(regs["left_end"])
                c_end = float(regs["center_end"])
                self.regions = {
                    "LEFT": (0.00, l_end),
                    "CENTER": (l_end, c_end),
                    "RIGHT": (c_end, 1.00)
                }

        self.reset()
        self._is_initialized = True
        t1 = time.perf_counter()

        logger.info(
            f"ImageSpaceFreeSpaceAnalyzer Initialized (regions={self.regions}, "
            f"clear<={self.clear_max_threshold}, uncertain<={self.uncertain_max_threshold})"
        )
        return True

    def _compute_overlap_fraction(
        self,
        x1_norm: float,
        x2_norm: float,
        r_start: float,
        r_end: float
    ) -> float:
        """
        Calculate horizontal overlap fraction between bounding box [x1_norm, x2_norm] and region [r_start, r_end].
        """
        inter_start = max(x1_norm, r_start)
        inter_end = min(x2_norm, r_end)

        inter_width = max(0.0, inter_end - inter_start)
        region_width = max(0.001, r_end - r_start)

        if inter_width <= 0.0:
            return 0.0

        return max(0.0, min(1.0, inter_width / region_width))

    def _get_distance_factor(self, category: str) -> float:
        """Map distance category to occupancy contribution factor."""
        mapping = {"NEAR": 1.00, "MEDIUM": 0.50, "FAR": 0.20, "UNKNOWN": 0.10}
        return mapping.get(category.upper(), 0.10)

    def _get_memory_state_factor(self, state: str) -> float:
        """Map PHMU memory state to occupancy contribution factor."""
        mapping = {
            "ACTIVE": 1.00,
            "RECOVERED": 1.00,
            "OCCLUDED": 0.80,
            "REMEMBERED": 0.60,
            "EXPIRED": 0.00,
        }
        return mapping.get(state.upper(), 1.00)

    def analyze_free_space(
        self,
        danger_assessments_or_hazards: List[Any],
        frame_width: float = 640.0,
        frame_height: float = 480.0
    ) -> FreeSpaceAnalysisResult:
        """
        Analyze image-space traversability across LEFT, CENTER, and RIGHT navigation regions.
        """
        if not self._is_initialized:
            self.initialize()

        t_start = time.perf_counter()
        w = max(1.0, float(frame_width))
        h = max(1.0, float(frame_height))

        if danger_assessments_or_hazards is None:
            danger_assessments_or_hazards = []

        # Filter out EXPIRED hazards
        valid_items = [
            item for item in danger_assessments_or_hazards
            if getattr(item, "memory_state", "ACTIVE") != "EXPIRED"
        ]

        # Accumulator containers per region
        region_accumulators: Dict[str, Dict[str, Any]] = {
            r_name: {
                "occupancy_score": 0.0,
                "blocked_object_ids": [],
                "danger_levels": [],
                "confidences": [],
                "reason_parts": [],
            }
            for r_name in self.regions.keys()
        }

        # Process contributing hazards
        for item in valid_items:
            track_id = getattr(item, "track_id", -1)
            class_name = getattr(item, "object_class", getattr(item, "class_name", "obstacle"))
            dist_cat = getattr(item, "distance_category", "UNKNOWN")
            danger_lvl = getattr(item, "danger_level", "MODERATE")
            danger_score = getattr(item, "danger_score", 0.50)
            mem_state = getattr(item, "memory_state", "ACTIVE")
            mem_conf = getattr(item, "memory_confidence", getattr(item, "distance_confidence", getattr(item, "confidence", 1.0)))
            bbox = getattr(item, "bounding_box", [0, 0, 0, 0])

            if len(bbox) != 4 or bbox == [0, 0, 0, 0]:
                continue

            x1_norm = max(0.0, min(1.0, bbox[0] / w))
            x2_norm = max(0.0, min(1.0, bbox[2] / w))
            y2_norm = max(0.0, min(1.0, bbox[3] / h))

            if x1_norm >= x2_norm:
                continue

            # Lower mask / walking corridor boost
            v_boost = 1.25 if y2_norm >= self.lower_mask_min_y_norm else 1.00

            # Compute individual factors
            f_dist = self._get_distance_factor(dist_cat)
            f_mem_state = self._get_memory_state_factor(mem_state)

            for r_name, (r_start, r_end) in self.regions.items():
                overlap_frac = self._compute_overlap_fraction(x1_norm, x2_norm, r_start, r_end)

                if overlap_frac > 0.05:
                    # Contribution formula
                    contribution = overlap_frac * f_dist * danger_score * mem_conf * f_mem_state * v_boost
                    
                    acc = region_accumulators[r_name]
                    acc["occupancy_score"] += contribution
                    if track_id not in acc["blocked_object_ids"]:
                        acc["blocked_object_ids"].append(track_id)
                    acc["danger_levels"].append(danger_lvl)
                    acc["confidences"].append(mem_conf)

                    part_str = f"{class_name} ID:{track_id} ({dist_cat})"
                    if mem_state in ("OCCLUDED", "REMEMBERED"):
                        part_str += " [REMEMBERED]"
                    if part_str not in acc["reason_parts"]:
                        acc["reason_parts"].append(part_str)

        # Build final RegionOccupancy objects
        danger_priority = {"CRITICAL": 4, "HIGH": 3, "MODERATE": 2, "LOW": 1, "NONE": 0}
        region_results: Dict[str, RegionOccupancy] = {}

        for r_name in self.regions.keys():
            acc = region_accumulators[r_name]
            occ_score = max(0.0, min(1.0, float(acc["occupancy_score"])))
            safe_score = max(0.0, min(1.0, 1.0 - occ_score))

            # Determine occupancy state
            if occ_score <= self.clear_max_threshold:
                state = "CLEAR"
            elif occ_score <= self.uncertain_max_threshold:
                state = "UNCERTAIN"
            else:
                state = "BLOCKED"

            # Dominant danger level
            if acc["danger_levels"]:
                dom_danger = max(acc["danger_levels"], key=lambda lvl: danger_priority.get(lvl, 0))
            else:
                dom_danger = "NONE"

            # Confidence calculation
            reg_conf = (sum(acc["confidences"]) / len(acc["confidences"])) if acc["confidences"] else 1.0

            # Reasoning string
            if acc["reason_parts"]:
                reasoning = f"{state} region due to {', '.join(acc['reason_parts'])}"
            else:
                reasoning = "CLEAR region; no detected obstacles (unverified open space)"

            reg_occ = RegionOccupancy(
                region_name=r_name,
                occupancy_state=state,
                occupancy_score=occ_score,
                safe_space_score=safe_score,
                blocked_object_ids=list(acc["blocked_object_ids"]),
                dominant_danger_level=dom_danger,
                confidence=reg_conf,
                reasoning=reasoning
            )
            region_results[r_name] = reg_occ

            logger.info(
                f"[FreeSpace] {r_name} occupancy={occ_score:.2f} safe={safe_score:.2f} "
                f"state={state} objects={acc['blocked_object_ids']}"
            )

        # Scene Overall Traversability
        blocked_count = sum(1 for r in region_results.values() if r.occupancy_state == "BLOCKED")
        if blocked_count >= 3:
            overall_trav = "BLOCKED"
        elif blocked_count >= 1 or any(r.occupancy_state == "UNCERTAIN" for r in region_results.values()):
            overall_trav = "PARTIALLY_BLOCKED"
        else:
            overall_trav = "CLEAR"

        final_result = FreeSpaceAnalysisResult(
            regions=region_results,
            total_hazards_assessed=len(valid_items),
            overall_traversability=overall_trav
        )

        t_end = time.perf_counter()
        self.last_analysis_latency_ms = (t_end - t_start) * 1000.0
        self.total_analyses += 1
        self._total_analysis_time_s += (t_end - t_start)

        return final_result

    def reset(self) -> None:
        """Reset internal statistics."""
        self.total_analyses = 0
        self._total_analysis_time_s = 0.0
        self.last_analysis_latency_ms = 0.0
        logger.info("ImageSpaceFreeSpaceAnalyzer reset.")

    def get_statistics(self) -> Dict[str, Any]:
        """Return operational free-space analysis statistics."""
        avg_fps = (self.total_analyses / self._total_analysis_time_s) if self._total_analysis_time_s > 0 else 0.0
        avg_lat = (self._total_analysis_time_s * 1000.0 / self.total_analyses) if self.total_analyses > 0 else 0.0

        return {
            "is_initialized": self._is_initialized,
            "clear_max_threshold": self.clear_max_threshold,
            "uncertain_max_threshold": self.uncertain_max_threshold,
            "lower_mask_min_y_norm": self.lower_mask_min_y_norm,
            "total_analyses": self.total_analyses,
            "last_analysis_latency_ms": round(self.last_analysis_latency_ms, 3),
            "average_analysis_latency_ms": round(avg_lat, 3),
            "average_analysis_throughput_fps": round(avg_fps, 2),
        }
