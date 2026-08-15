from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from modules.calibration.calibration_data import CameraIntrinsics, CalibrationRecord, MountingConfig


class CameraCalibrationInterface(ABC):
    """
    Abstract interface for webcam calibration in VisionGuide AI.
    Handles intrinsic focal length calibration, reference profile adjustments,
    camera mounting setup, and persistence.
    """

    @abstractmethod
    def initialize(self, config_path: Optional[str] = None) -> bool:
        """Initialize calibration module and load persisted calibration data."""
        pass

    @abstractmethod
    def calibrate_focal_length(
        self,
        object_height_px: float,
        known_distance_m: float,
        known_height_m: float
    ) -> float:
        """Calculate and update focal length in pixels: f_px = (height_px * distance_m) / height_m."""
        pass

    @abstractmethod
    def get_intrinsics(self) -> CameraIntrinsics:
        """Retrieve current camera intrinsic parameters."""
        pass

    @abstractmethod
    def get_mounting_config(self) -> MountingConfig:
        """Retrieve camera mounting and field of view configuration."""
        pass

    @abstractmethod
    def save_calibration(self, output_path: str = "config/calibration.yaml") -> bool:
        """Persist calibration parameters to YAML configuration file."""
        pass
