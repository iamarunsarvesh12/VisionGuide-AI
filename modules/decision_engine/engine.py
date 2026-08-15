import time
import os
import sys
import logging
import yaml
from typing import List, Dict, Any, Optional, Tuple

from modules.decision_engine.models import (
    NavigationCommand,
    RegionDecisionScore,
    DecisionInput,
    DecisionResult,
)
from modules.decision_engine.interface import DecisionEngineInterface

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

# Module logger setup
logger = logging.getLogger("DecisionEngine")
logger.setLevel(logging.INFO)
if not logger.handlers:
    c_handler = logging.StreamHandler(sys.stdout)
    c_handler.setFormatter(logging.Formatter("[%(name)s][%(levelname)s] %(message)s"))
    logger.addHandler(c_handler)

    f_handler = logging.FileHandler("logs/decision_engine.log", mode="a", encoding="utf-8")
    f_handler.setFormatter(logging.Formatter("%(asctime)s [%(name)s][%(levelname)s] %(message)s"))
    logger.addHandler(f_handler)


DEFAULT_CONFIG = {
    "enabled": True,
    "forward_safe_space_threshold": 0.70,
    "min_directional_confidence": 0.50,
    "stop_threshold": 0.30,
    "critical_danger_threshold": 0.85,
    "switching_margin": 0.10,
    "min_command_hold_duration_sec": 0.5,
    "uncertainty_penalty": 0.25,
    "weights": {
        "safe_space": 0.40,
        "danger": 0.30,
        "confidence": 0.15,
        "stability": 0.15,
        "uncertainty": 0.25,
    },
}


class ContextAwareDecisionEngine(DecisionEngineInterface):
    """
    Context-Aware Decision Engine implementation for VisionGuide AI.
    
    Synthesizes regional free-space analysis, context-aware danger assessments,
    PHMU hazard memory states, monocular distances, spatial positioning, and temporal hysteresis
    to produce deterministic, safety-first navigation commands (LEFT, RIGHT, FORWARD, STOP).
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = config_path
        self.config = DEFAULT_CONFIG.copy()
        self._is_initialized = False

        # Hysteresis & State Tracking
        self.previous_command: Optional[str] = None
        self.previous_decision_score: Optional[float] = None
        self.last_command_time: float = 0.0
        self.last_decision_result: Optional[DecisionResult] = None

        # Statistical Metrics
        self.total_decisions_made: int = 0
        self.command_counts: Dict[str, int] = {
            NavigationCommand.LEFT.value: 0,
            NavigationCommand.RIGHT.value: 0,
            NavigationCommand.FORWARD.value: 0,
            NavigationCommand.STOP.value: 0,
        }
        self.total_switches: int = 0

    def initialize(self, config_dict: Optional[Dict[str, Any]] = None) -> bool:
        """Initialize engine parameters, weights, and thresholds."""
        try:
            if config_dict is not None:
                self._apply_config(config_dict)
            elif os.path.exists(self.config_path):
                with open(self.config_path, "r", encoding="utf-8") as f:
                    full_cfg = yaml.safe_load(f)
                    if full_cfg and "decision_engine" in full_cfg:
                        self._apply_config(full_cfg["decision_engine"])

            self._is_initialized = True
            logger.info("ContextAwareDecisionEngine initialized successfully.")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize ContextAwareDecisionEngine: {e}")
            self._is_initialized = False
            return False

    def _apply_config(self, cfg: Dict[str, Any]) -> None:
        """Merge configuration settings."""
        for k, v in cfg.items():
            if k == "weights" and isinstance(v, dict):
                self.config["weights"].update(v)
            else:
                self.config[k] = v

    def reset(self) -> None:
        """Reset temporal state, previous commands, and statistics."""
        self.previous_command = None
        self.previous_decision_score = None
        self.last_command_time = 0.0
        self.last_decision_result = None
        self.total_decisions_made = 0
        self.command_counts = {k: 0 for k in self.command_counts}
        self.total_switches = 0
        logger.info("ContextAwareDecisionEngine state reset.")

    def get_statistics(self) -> Dict[str, Any]:
        """Return operational engine statistics."""
        return {
            "is_initialized": self._is_initialized,
            "total_decisions_made": self.total_decisions_made,
            "command_counts": self.command_counts.copy(),
            "total_switches": self.total_switches,
            "previous_command": self.previous_command,
            "last_decision_score": self.previous_decision_score,
            "weights": self.config.get("weights", {}),
            "switching_margin": self.config.get("switching_margin", 0.10),
        }

    def get_last_decision(self) -> Optional[DecisionResult]:
        """Return the most recently generated DecisionResult."""
        return self.last_decision_result

    def decide(self, decision_input: DecisionInput) -> DecisionResult:
        """
        Main decision-making entry point.
        Evaluates safety overrides, region scores, and stability hysteresis.
        """
        if not self._is_initialized:
            self.initialize()

        current_time = decision_input.timestamp if decision_input.timestamp > 0 else time.time()
        
        # Normalize input regions and hazards
        regions_map = self._normalize_regions(decision_input.regions)
        hazards_list = self._normalize_hazards(decision_input.hazards)

        # Update previous state if passed in input
        if decision_input.previous_command and self.previous_command is None:
            self.previous_command = decision_input.previous_command
        if decision_input.previous_decision_score is not None and self.previous_decision_score is None:
            self.previous_decision_score = decision_input.previous_decision_score

        # Step 1: Calculate Regional Decision Scores
        regional_scores, blocking_hazards_map = self._evaluate_regional_scores(regions_map, hazards_list)

        # Step 2: Evaluate Safety Override Conditions
        override_result = self._evaluate_safety_overrides(
            regions_map, hazards_list, regional_scores, blocking_hazards_map, current_time
        )
        if override_result is not None:
            return self._finalize_decision(override_result, current_time)

        # Step 3: Directional Command Candidate Selection
        candidate_command, candidate_region, candidate_score, reason, blocking_ids, alt_regions = (
            self._select_best_directional_command(regions_map, regional_scores, blocking_hazards_map)
        )

        # Step 4: Apply Command Stability / Hysteresis Policy
        final_command, final_region, final_score, final_reason, switched = self._apply_command_stability(
            candidate_command, candidate_region, candidate_score, reason, regional_scores, current_time
        )

        # Prepare DecisionResult
        res = DecisionResult(
            command=final_command,
            selected_region=final_region if final_command != NavigationCommand.STOP.value else None,
            confidence=round(float(regional_scores[final_region].confidence if final_region in regional_scores else 0.5), 4),
            decision_score=round(float(final_score), 4),
            reason=final_reason,
            blocking_hazards=blocking_ids,
            alternative_regions=alt_regions,
            timestamp=current_time,
            regional_scores=regional_scores,
        )

        return self._finalize_decision(res, current_time, switched=switched)

    def _normalize_regions(self, raw_regions: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Ensure region entries are uniform dictionaries."""
        norm = {}
        for r_name in ["LEFT", "CENTER", "RIGHT"]:
            if r_name in raw_regions:
                item = raw_regions[r_name]
                if hasattr(item, "to_dict"):
                    norm[r_name] = item.to_dict()
                elif isinstance(item, dict):
                    norm[r_name] = item
                else:
                    norm[r_name] = {
                        "region_name": r_name,
                        "occupancy_state": "UNCERTAIN",
                        "occupancy_score": 0.5,
                        "safe_space_score": 0.5,
                        "blocked_object_ids": [],
                        "dominant_danger_level": "NONE",
                        "confidence": 0.5,
                        "reasoning": "",
                    }
            else:
                norm[r_name] = {
                    "region_name": r_name,
                    "occupancy_state": "CLEAR",
                    "occupancy_score": 0.0,
                    "safe_space_score": 1.0,
                    "blocked_object_ids": [],
                    "dominant_danger_level": "NONE",
                    "confidence": 1.0,
                    "reasoning": "",
                }
        return norm

    def _normalize_hazards(self, raw_hazards: List[Any]) -> List[Dict[str, Any]]:
        """Ensure hazard entries are uniform dictionaries."""
        norm = []
        for h in raw_hazards:
            if hasattr(h, "to_dict"):
                norm.append(h.to_dict())
            elif isinstance(item := h, dict):
                norm.append(item)
        return norm

    def _evaluate_regional_scores(
        self,
        regions_map: Dict[str, Dict[str, Any]],
        hazards_list: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, RegionDecisionScore], Dict[str, List[int]]]:
        """
        Compute normalized multi-factor decision score for LEFT, CENTER, and RIGHT regions.
        Formula:
            Score = W_safe * SafeSpace + W_confidence * Confidence + W_stability * Stability
                    - W_danger * Danger - W_uncertainty * Uncertainty
        """
        weights = self.config["weights"]
        w_safe = weights.get("safe_space", 0.40)
        w_danger = weights.get("danger", 0.30)
        w_conf = weights.get("confidence", 0.15)
        w_stab = weights.get("stability", 0.15)
        w_uncert = weights.get("uncertainty", 0.25)
        uncert_penalty_cfg = self.config.get("uncertainty_penalty", 0.25)

        regional_scores: Dict[str, RegionDecisionScore] = {}
        blocking_hazards_map: Dict[str, List[int]] = {"LEFT": [], "CENTER": [], "RIGHT": []}

        for r_name in ["LEFT", "CENTER", "RIGHT"]:
            r_data = regions_map[r_name]
            safe_space = float(r_data.get("safe_space_score", 0.5))
            occ_state = r_data.get("occupancy_state", "CLEAR")
            base_confidence = max(0.0, min(1.0, float(r_data.get("confidence", 1.0))))
            blocked_ids = list(r_data.get("blocked_object_ids", []))

            # Populate blocking hazards from region metadata
            for b_id in blocked_ids:
                if b_id is not None and b_id not in blocking_hazards_map[r_name]:
                    blocking_hazards_map[r_name].append(b_id)

            # Filter hazards belonging to or affecting this region
            r_hazards = []
            for hz in hazards_list:
                pos = hz.get("position_zone", "")
                mem_st = hz.get("memory_state", "ACTIVE")
                
                # Ignore expired hazards completely
                if mem_st == "EXPIRED":
                    continue

                t_id = hz.get("track_id")
                
                # Spatial association check
                bbox = hz.get("bounding_box", [0, 0, 0, 0])
                if pos == r_name or (t_id in blocked_ids) or self._bbox_overlaps_region(bbox, r_name):
                    r_hazards.append(hz)
                    if t_id is not None and t_id not in blocking_hazards_map[r_name]:
                        blocking_hazards_map[r_name].append(t_id)

            # Determine maximum danger score in this region
            max_danger = 0.0
            has_remembered_hazard = False
            for hz in r_hazards:
                d_score = float(hz.get("danger_score", 0.0))
                mem_st = hz.get("memory_state", "ACTIVE")
                mem_conf = float(hz.get("memory_confidence", 1.0))
                
                if mem_st == "REMEMBERED":
                    has_remembered_hazard = True
                    # Remembered hazard danger scaled by memory confidence
                    d_score = d_score * mem_conf
                
                if d_score > max_danger:
                    max_danger = d_score

            # Adjust region confidence if remembered hazards present
            if has_remembered_hazard:
                base_confidence = max(0.2, base_confidence * 0.75)

            # Determine uncertainty penalty
            uncert_val = 0.0
            if occ_state == "UNCERTAIN" or base_confidence < 0.4:
                uncert_val = uncert_penalty_cfg
            elif has_remembered_hazard:
                uncert_val = uncert_penalty_cfg * 0.5

            # Stability bonus if region matches previous command direction (subtle tie-breaker)
            stab_val = 0.0
            if self.previous_command:
                if (self.previous_command == NavigationCommand.FORWARD.value and r_name == "CENTER") or \
                   (self.previous_command == NavigationCommand.LEFT.value and r_name == "LEFT") or \
                   (self.previous_command == NavigationCommand.RIGHT.value and r_name == "RIGHT"):
                    stab_val = 1.0

            # Compute raw score (stability serves as a subtle tie-breaker, main hysteresis handled in switching margin)
            stab_weight = min(0.05, w_stab)
            raw_score = (
                (w_safe * safe_space)
                + (w_conf * base_confidence)
                + (stab_weight * stab_val)
                - (w_danger * max_danger)
                - (w_uncert * uncert_val)
            )

            final_score = max(0.0, min(1.0, raw_score))

            regional_scores[r_name] = RegionDecisionScore(
                region_name=r_name,
                safe_space_score=safe_space,
                danger_score=max_danger,
                confidence=base_confidence,
                uncertainty=uncert_val,
                stability_score=stab_val,
                final_score=final_score,
            )

        return regional_scores, blocking_hazards_map

    def _bbox_overlaps_region(self, bbox: List[float], region_name: str, frame_width: float = 640.0) -> bool:
        """Check if bounding box horizontal interval overlaps region."""
        if not bbox or len(bbox) < 4 or (bbox[0] == 0 and bbox[2] == 0):
            return False
        x1, _, x2, _ = bbox
        if region_name == "LEFT":
            reg_x1, reg_x2 = 0.0, frame_width * 0.33
        elif region_name == "CENTER":
            reg_x1, reg_x2 = frame_width * 0.33, frame_width * 0.67
        else:  # RIGHT
            reg_x1, reg_x2 = frame_width * 0.67, frame_width
        return max(x1, reg_x1) < min(x2, reg_x2)

    def _evaluate_safety_overrides(
        self,
        regions_map: Dict[str, Dict[str, Any]],
        hazards_list: List[Dict[str, Any]],
        regional_scores: Dict[str, RegionDecisionScore],
        blocking_hazards_map: Dict[str, List[int]],
        current_time: float,
    ) -> Optional[DecisionResult]:
        """
        Evaluate highest priority STOP safety conditions.
        """
        stop_thresh = self.config.get("stop_threshold", 0.30)
        crit_danger_thresh = self.config.get("critical_danger_threshold", 0.85)

        # 1. Condition: ALL regions blocked
        all_blocked = all(
            regions_map[r]["occupancy_state"] == "BLOCKED" or regional_scores[r].danger_score >= crit_danger_thresh
            for r in ["LEFT", "CENTER", "RIGHT"]
        )
        if all_blocked:
            all_blocking_ids = sorted(list(set(
                blocking_hazards_map["LEFT"] + blocking_hazards_map["CENTER"] + blocking_hazards_map["RIGHT"]
            )))
            return DecisionResult(
                command=NavigationCommand.STOP.value,
                selected_region=None,
                confidence=0.95,
                decision_score=0.0,
                reason="All navigation regions are blocked or contain critical hazards; emergency stop required.",
                blocking_hazards=all_blocking_ids,
                alternative_regions=[],
                timestamp=current_time,
                regional_scores=regional_scores,
            )

        # 2. Condition: Critical hazard in CENTER and no safe alternative
        center_danger = regional_scores["CENTER"].danger_score
        center_has_critical = center_danger >= crit_danger_thresh or regions_map["CENTER"].get("dominant_danger_level") == "CRITICAL"
        
        left_safe = regional_scores["LEFT"].safe_space_score >= self.config.get("min_directional_confidence", 0.50) and regional_scores["LEFT"].danger_score < crit_danger_thresh and regions_map["LEFT"]["occupancy_state"] != "BLOCKED"
        right_safe = regional_scores["RIGHT"].safe_space_score >= self.config.get("min_directional_confidence", 0.50) and regional_scores["RIGHT"].danger_score < crit_danger_thresh and regions_map["RIGHT"]["occupancy_state"] != "BLOCKED"

        if center_has_critical and not left_safe and not right_safe:
            # Find class of critical hazard if present
            crit_cls = "hazard"
            for hz in hazards_list:
                if hz.get("position_zone") == "CENTER" and float(hz.get("danger_score", 0.0)) >= crit_danger_thresh:
                    crit_cls = hz.get("class_name", "hazard")
                    break

            return DecisionResult(
                command=NavigationCommand.STOP.value,
                selected_region=None,
                confidence=0.90,
                decision_score=0.0,
                reason=f"Center contains a critical {crit_cls} hazard while left and right regions lack sufficient safe space.",
                blocking_hazards=blocking_hazards_map["CENTER"],
                alternative_regions=[],
                timestamp=current_time,
                regional_scores=regional_scores,
            )

        # 3. Condition: Insufficient safe space evidence across all regions (e.g. all UNCERTAIN or low safe space)
        all_low_safe_space = all(regional_scores[r].safe_space_score < stop_thresh for r in ["LEFT", "CENTER", "RIGHT"])
        all_uncertain = all(regions_map[r]["occupancy_state"] == "UNCERTAIN" for r in ["LEFT", "CENTER", "RIGHT"])
        
        if all_low_safe_space or all_uncertain:
            return DecisionResult(
                command=NavigationCommand.STOP.value,
                selected_region=None,
                confidence=0.85,
                decision_score=0.0,
                reason="Insufficient safe space evidence or total environmental uncertainty across all regions.",
                blocking_hazards=sorted(list(set(blocking_hazards_map["LEFT"] + blocking_hazards_map["CENTER"] + blocking_hazards_map["RIGHT"]))),
                alternative_regions=[],
                timestamp=current_time,
                regional_scores=regional_scores,
            )

        return None

    def _select_best_directional_command(
        self,
        regions_map: Dict[str, Dict[str, Any]],
        regional_scores: Dict[str, RegionDecisionScore],
        blocking_hazards_map: Dict[str, List[int]],
    ) -> Tuple[str, str, float, str, List[int], List[str]]:
        """
        Evaluate FORWARD, LEFT, and RIGHT directional commands.
        Returns (command, selected_region, score, reason, blocking_ids, alternative_regions).
        """
        fwd_thresh = self.config.get("forward_safe_space_threshold", 0.70)
        min_dir_conf = self.config.get("min_directional_confidence", 0.50)
        crit_danger_thresh = self.config.get("critical_danger_threshold", 0.85)

        s_left = regional_scores["LEFT"].final_score
        s_center = regional_scores["CENTER"].final_score
        s_right = regional_scores["RIGHT"].final_score

        center_safe_space = regional_scores["CENTER"].safe_space_score
        center_danger = regional_scores["CENTER"].danger_score
        center_occ = regions_map["CENTER"]["occupancy_state"]

        # Check FORWARD eligibility
        forward_eligible = (
            center_safe_space >= fwd_thresh
            and center_danger < 0.70
            and center_occ != "BLOCKED"
        )

        if forward_eligible and s_center >= max(s_left, s_right) - 0.15:
            alts = []
            if regional_scores["LEFT"].safe_space_score >= min_dir_conf and regions_map["LEFT"]["occupancy_state"] != "BLOCKED":
                alts.append("LEFT")
            if regional_scores["RIGHT"].safe_space_score >= min_dir_conf and regions_map["RIGHT"]["occupancy_state"] != "BLOCKED":
                alts.append("RIGHT")

            reason = f"Center region clear with high safe-space confidence ({center_safe_space:.2f}); continue forward."
            return (
                NavigationCommand.FORWARD.value,
                "CENTER",
                s_center,
                reason,
                blocking_hazards_map["CENTER"],
                alts,
            )

        # Compare LEFT vs RIGHT when CENTER is blocked or suboptimal
        left_valid = (
            regional_scores["LEFT"].safe_space_score >= min_dir_conf
            and regional_scores["LEFT"].danger_score < crit_danger_thresh
            and regions_map["LEFT"]["occupancy_state"] != "BLOCKED"
        )
        right_valid = (
            regional_scores["RIGHT"].safe_space_score >= min_dir_conf
            and regional_scores["RIGHT"].danger_score < crit_danger_thresh
            and regions_map["RIGHT"]["occupancy_state"] != "BLOCKED"
        )

        if left_valid and right_valid:
            if s_left >= s_right:
                reason = f"Center region restricted; left region provides higher safe space ({regional_scores['LEFT'].safe_space_score:.2f}) and lower danger."
                return NavigationCommand.LEFT.value, "LEFT", s_left, reason, blocking_hazards_map["CENTER"], ["RIGHT"]
            else:
                reason = f"Center region restricted; right region provides higher safe space ({regional_scores['RIGHT'].safe_space_score:.2f}) and lower danger."
                return NavigationCommand.RIGHT.value, "RIGHT", s_right, reason, blocking_hazards_map["CENTER"], ["LEFT"]

        elif left_valid:
            reason = f"Center region restricted; left region clear ({regional_scores['LEFT'].safe_space_score:.2f})."
            return NavigationCommand.LEFT.value, "LEFT", s_left, reason, blocking_hazards_map["CENTER"], []

        elif right_valid:
            reason = f"Center region restricted; right region clear ({regional_scores['RIGHT'].safe_space_score:.2f})."
            return NavigationCommand.RIGHT.value, "RIGHT", s_right, reason, blocking_hazards_map["CENTER"], []

        # If forward is somewhat valid even if below optimal threshold
        if center_occ != "BLOCKED" and center_danger < crit_danger_thresh and center_safe_space >= min_dir_conf:
            reason = f"Alternative regions unavailable; proceeding forward cautious with safe-space ({center_safe_space:.2f})."
            return NavigationCommand.FORWARD.value, "CENTER", s_center, reason, blocking_hazards_map["CENTER"], []

        # Fallback to STOP if no directional option is sufficiently safe
        all_blocking = sorted(list(set(blocking_hazards_map["LEFT"] + blocking_hazards_map["CENTER"] + blocking_hazards_map["RIGHT"])))
        reason = "Center region blocked and no directional alternatives provide sufficient safe space."
        return NavigationCommand.STOP.value, "CENTER", 0.0, reason, all_blocking, []

    def _apply_command_stability(
        self,
        candidate_command: str,
        candidate_region: str,
        candidate_score: float,
        candidate_reason: str,
        regional_scores: Dict[str, RegionDecisionScore],
        current_time: float,
    ) -> Tuple[str, str, float, str, bool]:
        """
        Apply hysteresis / command-switching margin and hold duration.
        """
        margin = self.config.get("switching_margin", 0.10)
        hold_sec = self.config.get("min_command_hold_duration_sec", 0.5)

        if not self.previous_command:
            return candidate_command, candidate_region, candidate_score, candidate_reason, False

        # STOP command overrides previous command immediately if safety requires it!
        if candidate_command == NavigationCommand.STOP.value:
            return candidate_command, candidate_region, candidate_score, candidate_reason, (self.previous_command != NavigationCommand.STOP.value)

        # If previous command was STOP and now candidate is directional
        if self.previous_command == NavigationCommand.STOP.value:
            if candidate_score >= self.config.get("min_directional_confidence", 0.50):
                return candidate_command, candidate_region, candidate_score, candidate_reason, True
            else:
                return NavigationCommand.STOP.value, candidate_region, 0.0, "Maintaining STOP until candidate region safe-space confidence increases.", False

        # If candidate command matches previous command
        if candidate_command == self.previous_command:
            return candidate_command, candidate_region, candidate_score, candidate_reason, False

        # Candidate is different from previous command (e.g. Previous=LEFT, Candidate=RIGHT)
        # Determine environmental suitability score of previous command's region without double-counted stability bonus
        prev_region = "CENTER"
        if self.previous_command == NavigationCommand.LEFT.value:
            prev_region = "LEFT"
        elif self.previous_command == NavigationCommand.RIGHT.value:
            prev_region = "RIGHT"

        prev_region_score = regional_scores[prev_region].final_score
        w_stab = min(0.05, self.config.get("weights", {}).get("stability", 0.15))
        prev_env_score = prev_region_score - (w_stab * regional_scores[prev_region].stability_score)
        cand_env_score = candidate_score - (w_stab * regional_scores[candidate_region].stability_score) if candidate_region in regional_scores else candidate_score

        score_diff = cand_env_score - prev_env_score

        time_elapsed = current_time - self.last_command_time

        # If improvement is below switching margin OR hold time not met (and prev region is not critical)
        if (score_diff < margin or time_elapsed < hold_sec) and regional_scores[prev_region].danger_score < 0.80:
            retained_reason = (
                f"Retaining previous command {self.previous_command}; "
                f"candidate {candidate_command} score improvement ({score_diff:.2f}) is below switching margin ({margin:.2f})."
            )
            return self.previous_command, prev_region, prev_region_score, retained_reason, False

        # Significant improvement: Approve switch!
        switch_reason = candidate_reason + f" (Switched from {self.previous_command} due to score improvement +{score_diff:.2f})."
        return candidate_command, candidate_region, candidate_score, switch_reason, True

    def _finalize_decision(self, res: DecisionResult, current_time: float, switched: bool = False) -> DecisionResult:
        """Update internal statistics and log decision outcome."""
        self.total_decisions_made += 1
        if res.command in self.command_counts:
            self.command_counts[res.command] += 1

        prev_cmd_str = str(self.previous_command)
        if switched:
            self.total_switches += 1
            self.last_command_time = current_time

        self.previous_command = res.command
        self.previous_decision_score = res.decision_score
        self.last_decision_result = res

        # Log decision
        logger.info(
            f"[Decision] Previous={prev_cmd_str} Current={res.command} "
            f"Switch={switched} Score={res.decision_score:.2f} "
            f"Reason=\"{res.reason}\""
        )

        return res
