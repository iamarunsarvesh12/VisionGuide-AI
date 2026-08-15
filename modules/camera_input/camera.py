import time
import os
import sys
import logging
from typing import Tuple, Dict, Any, Optional
import cv2
import numpy as np

from modules.camera_input.interface import CameraInterface

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

# Configure module-level logger
logger = logging.getLogger("CameraInput")
logger.setLevel(logging.INFO)
if not logger.handlers:
    # Console handler
    c_handler = logging.StreamHandler(sys.stdout)
    c_handler.setFormatter(logging.Formatter("[%(name)s][%(levelname)s] %(message)s"))
    logger.addHandler(c_handler)
    
    # File handler
    f_handler = logging.FileHandler("logs/camera_input.log", mode="a", encoding="utf-8")
    f_handler.setFormatter(logging.Formatter("%(asctime)s [%(name)s][%(levelname)s] %(message)s"))
    logger.addHandler(f_handler)


class WebcamInput(CameraInterface):
    """
    OpenCV Webcam Input implementation for VisionGuide AI.
    
    Provides reliable, low-latency video frame acquisition from laptop webcams,
    USB video class (UVC) cameras, or external camera feeds.
    Includes real-time FPS calculation, frame capture latency tracking,
    and robust error handling.
    """

    def __init__(
        self,
        camera_index: int = 0,
        width: int = 640,
        height: int = 480,
        target_fps: int = 30,
        backend_preference: str = "CAP_ANY",
    ):
        """
        Initialize standard webcam parameters.
        
        Args:
            camera_index: Operating system video device index (default: 0).
            width: Desired frame width resolution.
            height: Desired frame height resolution.
            target_fps: Target acquisition frames per second.
            backend_preference: OpenCV capture backend flag ('CAP_DSHOW', 'CAP_MSMF', or 'CAP_ANY').
        """
        self.camera_index = camera_index
        self.requested_width = width
        self.requested_height = height
        self.target_fps = target_fps
        self.backend_preference_str = backend_preference

        self._cap: Optional[cv2.VideoCapture] = None
        self._is_opened: bool = False

        # Real-time performance tracking metrics
        self._frame_count: int = 0
        self._start_time: float = 0.0
        self._last_frame_time: float = 0.0
        self._measured_fps: float = 0.0
        self._last_capture_latency_ms: float = 0.0

        # Actual hardware reported dimensions
        self.actual_width: int = 0
        self.actual_height: int = 0

    def _get_backend_enum(self) -> int:
        """Map backend string to OpenCV enum."""
        if self.backend_preference_str == "CAP_DSHOW" and sys.platform == "win32":
            return cv2.CAP_DSHOW
        elif self.backend_preference_str == "CAP_MSMF" and sys.platform == "win32":
            return cv2.CAP_MSMF
        return cv2.CAP_ANY

    def start(self) -> bool:
        """
        Initialize and open the camera device.
        Configures width, height, and target FPS on the hardware capture stream.
        """
        logger.info(f"Initializing camera index {self.camera_index} (Backend: {self.backend_preference_str})")
        
        try:
            backend = self._get_backend_enum()
            self._cap = cv2.VideoCapture(self.camera_index, backend)
            
            if not self._cap.isOpened():
                # Fallback to default backend if specific backend failed
                if backend != cv2.CAP_ANY:
                    logger.warning(f"Failed opening camera index {self.camera_index} with {self.backend_preference_str}. Attempting CAP_ANY fallback.")
                    self._cap = cv2.VideoCapture(self.camera_index, cv2.CAP_ANY)

            if not self._cap.isOpened():
                logger.error(f"Unable to open camera index {self.camera_index}")
                self._is_opened = False
                return False

            # Configure properties
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.requested_width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.requested_height)
            self._cap.set(cv2.CAP_PROP_FPS, self.target_fps)

            # Query actual hardware parameters
            self.actual_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.actual_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Warmup read to verify stream viability
            ret, test_frame = self._cap.read()
            if not ret or test_frame is None:
                logger.error(f"Camera index {self.camera_index} opened but failed initial frame read.")
                self.stop()
                return False

            self._is_opened = True
            self._start_time = time.perf_counter()
            self._last_frame_time = time.perf_counter()
            self._frame_count = 0
            
            logger.info(f"Camera index {self.camera_index} initialized successfully. Resolution: {self.actual_width}x{self.actual_height}")
            return True

        except Exception as e:
            logger.error(f"Exception during camera startup for index {self.camera_index}: {e}")
            self._is_opened = False
            return False

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read the next frame from the open camera stream.
        Calculates frame latency and rolling FPS.
        """
        if not self._is_opened or self._cap is None:
            logger.error("Attempted to read frame from closed or uninitialized camera.")
            return False, None

        t_start = time.perf_counter()
        
        try:
            ret, frame = self._cap.read()
            t_end = time.perf_counter()
            
            self._last_capture_latency_ms = (t_end - t_start) * 1000.0

            if not ret or frame is None:
                logger.warning(f"Frame read failure on camera index {self.camera_index}")
                return False, None

            # Calculate FPS metrics
            self._frame_count += 1
            elapsed = t_end - self._start_time
            if elapsed > 0:
                self._measured_fps = self._frame_count / elapsed

            self._last_frame_time = t_end
            return True, frame

        except Exception as e:
            logger.error(f"Exception reading frame from camera index {self.camera_index}: {e}")
            return False, None

    def stop(self) -> None:
        """
        Stop video capture and safely release camera hardware handles.
        """
        if self._cap is not None:
            try:
                self._cap.release()
                logger.info(f"Camera index {self.camera_index} stopped and hardware resources released.")
            except Exception as e:
                logger.error(f"Error releasing camera index {self.camera_index}: {e}")
        
        self._cap = None
        self._is_opened = False

    def is_opened(self) -> bool:
        """Return True if camera capture handle is active."""
        return self._is_opened and self._cap is not None and self._cap.isOpened()

    def get_properties(self) -> Dict[str, Any]:
        """Return hardware properties and real-time statistics."""
        return {
            "index": self.camera_index,
            "width": self.actual_width,
            "height": self.actual_height,
            "configured_fps": self.target_fps,
            "measured_fps": round(self._measured_fps, 2),
            "capture_latency_ms": round(self._last_capture_latency_ms, 2),
            "is_opened": self.is_opened(),
            "total_frames_read": self._frame_count,
        }
