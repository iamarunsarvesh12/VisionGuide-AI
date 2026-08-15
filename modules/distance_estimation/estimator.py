import time
import os
import sys
import logging
from typing import List, Dict, Any, Optional

from modules.distance_estimation.models import DistanceResult
from modules.distance_estimation.interface import DistanceEstimatorInterface

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

# Module logger setup
logger = logging.getLogger("DistanceEstimation")
logger.setLevel(logging.INFO)
if not logger.handlers:
    c_handler = logging.StreamHandler(sys.stdout)
    c_handler.setFormatter(logging.Formatter("[%(name)s][%(levelname)s] %(message)s"))
    logger.addHandler(c_handler)
    
    f_handler = logging.FileHandler("logs/distance_estimation.log", mode="a", encoding="utf-8")
    f_handler.setFormatter(logging.Formatter("%(asctime)s [%(name)s][%(levelname)s] %(message)s"))
    logger.addHandler(f_handler)

DEFAULT_PROFILES = {
    "person": 1.70,
    "door": 2.00,
    "chair": 0.85,
    "table": 0.75,
    "stairs": 1.20,
    "glass_door": 2.00,
    "glass_wall": 2.00,
    "cabinet": 1.50,
    "corridor": 2.20,
    "exit": 2.00,
    "default": 1.00,
}


class MonocularDistanceEstimator(DistanceEstimatorInterface):
    """
    Monocular Approximate Distance Estimator for VisionGuide AI.
    
    Estimates relative object distance categories (NEAR, MEDIUM, FAR) and monocular metric
    approximations using bounding box height, class-specific reference profiles, and pinhole optics geometry.
    """

    def __init__(
        self,
        focal_length_px: float = 600.0,
        near_threshold_m: float = 1.5,
        medium_threshold_m: float = 3.0,
        distance_profiles: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize monocular distance estimator parameters.
        
        Args:
            focal_length_px: Prototype camera focal length constant in pixels.
            near_threshold_m: Maximum distance threshold for NEAR category (metres).
            medium_threshold_m: Maximum distance threshold for MEDIUM category (metres).
            distance_profiles: Custom mapping of class_name to reference height (metres).
        """
        self.focal_length_px = focal_length_px
        self.near_threshold_m = near_threshold_m
        self.medium_threshold_m = medium_threshold_m
        self.profiles = dict(DEFAULT_PROFILES)
        if distance_profiles:
            self.profiles.update(distance_profiles)

        self._is_initialized: bool = False
        self._last_known_distances: Dict[int, DistanceResult] = {}

        # Performance metrics
        self.total_estimations: int = 0
        self.last_estimation_latency_ms: float = 0.0
        self._total_estimation_time_s: float = 0.0

    def initialize(self, config_dict: Optional[Dict[str, Any]] = None) -> bool:
        """Initialize parameters from configuration dictionary."""
        t0 = time.perf_counter()
        if config_dict:
            cfg = config_dict.get("distance_estimation", {})
            self.focal_length_px = cfg.get("focal_length_px", self.focal_length_px)
            self.near_threshold_m = cfg.get("near_threshold_m", self.near_threshold_m)
            self.medium_threshold_m = cfg.get("medium_threshold_m", self.medium_threshold_m)

            custom_profs = config_dict.get("distance_profiles", {})
            for cls_name, p_data in custom_profs.items():
                if isinstance(p_data, dict) and "reference_height_m" in p_data:
                    self.profiles[cls_name] = p_data["reference_height_m"]

        self.reset()
        self._is_initialized = True
        t1 = time.perf_counter()

        logger.info(
            f"MonocularDistanceEstimator Initialized (focal_length={self.focal_length_px}px, "
            f"near<={self.near_threshold_m}m, medium<={self.medium_threshold_m}m)"
        )
        return True

    def _get_reference_height(self, class_name: str) -> float:
        """Retrieve class-specific reference height in metres."""
        val = self.profiles.get(class_name.lower(), self.profiles.get("default", 1.00))
        if isinstance(val, dict):
            return float(val.get("reference_height_m", 1.00))
        try:
            return float(val)
        except Exception:
            return 1.00

    def estimate_distance(self, track_or_hazard: Any) -> DistanceResult:
        """
        Estimate relative distance category and approximate metric distance for a single track or hazard record.
        """
        if not self._is_initialized:
            self.initialize()

        t_start = time.perf_counter()

        if track_or_hazard is None:
            return DistanceResult(
                track_id=-1, class_name="unknown", distance_category="UNKNOWN",
                distance_confidence=0.0, estimated_distance_m=None, bounding_box=[0, 0, 0, 0],
                distance_status="UNKNOWN"
            )

        # Extract attributes from Track or HazardMemoryRecord
        track_id = getattr(track_or_hazard, "track_id", -1)
        class_name = getattr(track_or_hazard, "object_class", getattr(track_or_hazard, "class_name", "unknown"))
        bbox = getattr(track_or_hazard, "bounding_box", [0, 0, 0, 0])
        bbox_height = getattr(track_or_hazard, "height", 0.0)
        confidence = getattr(track_or_hazard, "memory_confidence", getattr(track_or_hazard, "confidence", 0.5))
        memory_state = getattr(track_or_hazard, "memory_state", "ACTIVE")

        # Remembered hazard policy: If unobserved in current frame, preserve last known measurement
        if memory_state in ("OCCLUDED", "REMEMBERED") and track_id in self._last_known_distances:
            prev_result = self._last_known_distances[track_id]
            res = DistanceResult(
                track_id=track_id,
                class_name=class_name,
                distance_category=prev_result.distance_category,
                distance_confidence=confidence,  # Use decayed memory confidence
                estimated_distance_m=prev_result.estimated_distance_m,
                bounding_box=list(bbox),
                estimation_method="monocular_bbox",
                distance_status="LAST_OBSERVED"
            )
            t_end = time.perf_counter()
            self.last_estimation_latency_ms = (t_end - t_start) * 1000.0
            return res

        # Validate bounding box height
        if bbox_height <= 0 or bbox == [0, 0, 0, 0]:
            res = DistanceResult(
                track_id=track_id, class_name=class_name, distance_category="UNKNOWN",
                distance_confidence=0.0, estimated_distance_m=None, bounding_box=list(bbox),
                distance_status="UNKNOWN"
            )
            t_end = time.perf_counter()
            self.last_estimation_latency_ms = (t_end - t_start) * 1000.0
            return res

        # Pinhole Camera Geometry Calculation
        ref_height = self._get_reference_height(class_name)
        estimated_d_m = (ref_height * self.focal_length_px) / float(bbox_height)

        # Proximity Categorization
        if estimated_d_m <= self.near_threshold_m:
            category = "NEAR"
        elif estimated_d_m <= self.medium_threshold_m:
            category = "MEDIUM"
        else:
            category = "FAR"

        res = DistanceResult(
            track_id=track_id,
            class_name=class_name,
            distance_category=category,
            distance_confidence=float(confidence),
            estimated_distance_m=float(estimated_d_m),
            bounding_box=list(bbox),
            estimation_method="monocular_bbox",
            distance_status="MEASURED"
        )

        # Cache last known valid measurement
        self._last_known_distances[track_id] = res

        t_end = time.perf_counter()
        self.last_estimation_latency_ms = (t_end - t_start) * 1000.0
        self.total_estimations += 1
        self._total_estimation_time_s += (t_end - t_start)

        logger.info(f"[Distance] ID={track_id} class={class_name} distance={category} ({estimated_d_m:.2f}m) conf={confidence:.2f}")
        return res

    def estimate_batch(self, tracks_or_hazards: List[Any]) -> List[DistanceResult]:
        """Process batch list of tracks or hazard memory records."""
        if not tracks_or_hazards:
            return []
        return [self.estimate_distance(item) for item in tracks_or_hazards]

    def calibrate_class(
        self,
        class_name: str,
        known_distance_m: float,
        observed_height_px: float
    ) -> float:
        """
        Calibrate reference height profile using a known distance measurement.
        Formula: H_ref = (known_distance_m * observed_height_px) / focal_length_px
        """
        if observed_height_px <= 0 or known_distance_m <= 0:
            logger.warning(f"Invalid calibration inputs for class '{class_name}'")
            return self._get_reference_height(class_name)

        calibrated_ref_height = (known_distance_m * observed_height_px) / self.focal_length_px
        self.profiles[class_name.lower()] = float(calibrated_ref_height)
        logger.info(f"Calibrated class '{class_name}': Reference height set to {calibrated_ref_height:.3f} m")
        return calibrated_ref_height

    def reset(self) -> None:
        """Reset internal statistics and cached distances."""
        self._last_known_distances.clear()
        self.total_estimations = 0
        self._total_estimation_time_s = 0.0
        self.last_estimation_latency_ms = 0.0
        logger.info("MonocularDistanceEstimator reset.")

    def get_statistics(self) -> Dict[str, Any]:
        """Return operational estimation statistics."""
        avg_fps = (self.total_estimations / self._total_estimation_time_s) if self._total_estimation_time_s > 0 else 0.0
        avg_lat = (self._total_estimation_time_s * 1000.0 / self.total_estimations) if self.total_estimations > 0 else 0.0

        return {
            "method": "monocular_bbox",
            "is_initialized": self._is_initialized,
            "focal_length_px": self.focal_length_px,
            "near_threshold_m": self.near_threshold_m,
            "medium_threshold_m": self.medium_threshold_m,
            "total_estimations": self.total_estimations,
            "last_estimation_latency_ms": round(self.last_estimation_latency_ms, 3),
            "average_estimation_latency_ms": round(avg_lat, 3),
            "average_estimation_throughput_fps": round(avg_fps, 2),
            "configured_profiles": len(self.profiles),
        }
