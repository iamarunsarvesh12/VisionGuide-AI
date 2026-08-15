from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class CameraIntrinsics:
    """Camera intrinsic parameters for monocular optical geometry."""
    focal_length_px: float = 600.0
    frame_width: float = 640.0
    frame_height: float = 480.0
    fov_horizontal_deg: float = 60.0
    fov_vertical_deg: float = 45.0
    aspect_ratio: float = 1.3333

    def to_dict(self) -> Dict[str, Any]:
        return {
            "focal_length_px": round(self.focal_length_px, 2),
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
            "fov_horizontal_deg": round(self.fov_horizontal_deg, 2),
            "fov_vertical_deg": round(self.fov_vertical_deg, 2),
            "aspect_ratio": round(self.aspect_ratio, 4),
        }


@dataclass
class MountingConfig:
    """Camera physical mounting parameters."""
    mounting_height_m: float = 1.20  # Height above ground in metres
    pitch_angle_deg: float = 0.0     # Downward/upward tilt angle
    yaw_angle_deg: float = 0.0       # Left/right orientation offset
    mounting_position: str = "chest" # "chest", "head", "laptop", "handheld"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mounting_height_m": self.mounting_height_m,
            "pitch_angle_deg": self.pitch_angle_deg,
            "yaw_angle_deg": self.yaw_angle_deg,
            "mounting_position": self.mounting_position,
        }


@dataclass
class CalibrationRecord:
    """Calibration record holding intrinsics, mounting, and reference profile overrides."""
    intrinsics: CameraIntrinsics = field(default_factory=CameraIntrinsics)
    mounting: MountingConfig = field(default_factory=MountingConfig)
    reference_profiles: Dict[str, float] = field(default_factory=dict)
    calibration_timestamp: Optional[float] = None
    calibrated_by: str = "WebcamCalibrator"
    is_calibrated: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intrinsics": self.intrinsics.to_dict(),
            "mounting": self.mounting.to_dict(),
            "reference_profiles": self.reference_profiles,
            "calibration_timestamp": self.calibration_timestamp,
            "calibrated_by": self.calibrated_by,
            "is_calibrated": self.is_calibrated,
        }
