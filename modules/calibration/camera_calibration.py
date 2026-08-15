import os
import sys
import time
import yaml
import logging
from typing import Dict, Any, Optional

from modules.calibration.calibration_data import CameraIntrinsics, CalibrationRecord, MountingConfig
from modules.calibration.interface import CameraCalibrationInterface

os.makedirs("logs", exist_ok=True)

logger = logging.getLogger("CameraCalibration")
logger.setLevel(logging.INFO)
if not logger.handlers:
    c_handler = logging.StreamHandler(sys.stdout)
    c_handler.setFormatter(logging.Formatter("[%(name)s][%(levelname)s] %(message)s"))
    logger.addHandler(c_handler)
    
    f_handler = logging.FileHandler("logs/camera_calibration.log", mode="a", encoding="utf-8")
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


class WebcamCalibrator(CameraCalibrationInterface):
    """
    Webcam intrinsic parameter and optical reference calibrator for VisionGuide AI.
    Estimates focal length, manages object reference height profiles,
    handles camera mounting setup, and persists calibration settings.
    """

    def __init__(self, config_path: str = "config/calibration.yaml"):
        self.config_path = config_path
        self.record = CalibrationRecord()
        self.record.reference_profiles = dict(DEFAULT_PROFILES)
        self._is_initialized = False

    def initialize(self, config_path: Optional[str] = None) -> bool:
        """Initialize calibration module and load persisted calibration data."""
        t0 = time.perf_counter()
        target_path = config_path if config_path else self.config_path

        if os.path.exists(target_path):
            try:
                with open(target_path, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}

                int_cfg = cfg.get("intrinsics", {})
                self.record.intrinsics.focal_length_px = float(int_cfg.get("focal_length_px", 600.0))
                self.record.intrinsics.frame_width = float(int_cfg.get("frame_width", 640.0))
                self.record.intrinsics.frame_height = float(int_cfg.get("frame_height", 480.0))
                self.record.intrinsics.fov_horizontal_deg = float(int_cfg.get("fov_horizontal_deg", 60.0))
                self.record.intrinsics.fov_vertical_deg = float(int_cfg.get("fov_vertical_deg", 45.0))

                mnt_cfg = cfg.get("mounting", {})
                self.record.mounting.mounting_height_m = float(mnt_cfg.get("mounting_height_m", 1.20))
                self.record.mounting.pitch_angle_deg = float(mnt_cfg.get("pitch_angle_deg", 0.0))
                self.record.mounting.yaw_angle_deg = float(mnt_cfg.get("yaw_angle_deg", 0.0))
                self.record.mounting.mounting_position = str(mnt_cfg.get("mounting_position", "chest"))

                profs = cfg.get("reference_profiles", {})
                if profs:
                    for k, v in profs.items():
                        if isinstance(v, dict):
                            self.record.reference_profiles[k] = float(v.get("reference_height_m", 1.0))
                        else:
                            self.record.reference_profiles[k] = float(v)

                self.record.is_calibrated = cfg.get("is_calibrated", True)
                self.record.calibration_timestamp = cfg.get("calibration_timestamp", time.time())
                logger.info(f"Loaded existing calibration from '{target_path}' (focal_length={self.record.intrinsics.focal_length_px:.1f}px)")
            except Exception as e:
                logger.warning(f"Could not load calibration file '{target_path}': {e}. Using defaults.")

        self._is_initialized = True
        return True

    def calibrate_focal_length(
        self,
        object_height_px: float,
        known_distance_m: float,
        known_height_m: float
    ) -> float:
        """
        Calculate focal length in pixels using pinhole geometry:
        Formula: f_px = (height_px * distance_m) / height_m
        """
        if object_height_px <= 0 or known_distance_m <= 0 or known_height_m <= 0:
            logger.error("Invalid input parameters for focal length calibration.")
            return self.record.intrinsics.focal_length_px

        f_px = (float(object_height_px) * float(known_distance_m)) / float(known_height_m)
        self.record.intrinsics.focal_length_px = float(f_px)
        self.record.is_calibrated = True
        self.record.calibration_timestamp = time.time()
        logger.info(f"Focal length calibrated: f_px = {f_px:.2f} px (height_px={object_height_px}, dist={known_distance_m}m, height={known_height_m}m)")
        return float(f_px)

    def calibrate_reference_height(self, class_name: str, reference_height_m: float) -> float:
        """Update or calibrate reference height for an object class."""
        if reference_height_m > 0:
            self.record.reference_profiles[class_name.lower()] = float(reference_height_m)
            logger.info(f"Updated reference height for '{class_name}': {reference_height_m:.2f}m")
        return self.record.reference_profiles.get(class_name.lower(), 1.00)

    def get_intrinsics(self) -> CameraIntrinsics:
        """Retrieve current camera intrinsic parameters."""
        return self.record.intrinsics

    def get_mounting_config(self) -> MountingConfig:
        """Retrieve camera mounting configuration."""
        return self.record.mounting

    def save_calibration(self, output_path: str = "config/calibration.yaml") -> bool:
        """Save calibration data to YAML file."""
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            data = self.record.to_dict()
            with open(output_path, "w", encoding="utf-8") as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)
            logger.info(f"Calibration persisted to '{output_path}'.")
            return True
        except Exception as e:
            logger.error(f"Failed to save calibration data to '{output_path}': {e}")
            return False
