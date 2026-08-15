from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, Optional
import numpy as np


class CameraInterface(ABC):
    """
    Abstract interface for video input sources in VisionGuide AI.
    
    Decouples visual input acquisition from downstream perception components.
    Allows substituting laptop camera with USB cameras, external webcams,
    video files, or wearable camera interfaces without affecting downstream modules.
    """

    @abstractmethod
    def start(self) -> bool:
        """
        Initialize and open the camera device.
        Returns True if successfully initialized and opened, False otherwise.
        """
        pass

    @abstractmethod
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read the next frame from the camera stream.
        
        Returns:
            Tuple[bool, Optional[np.ndarray]]: 
                - success: True if a valid frame was read, False otherwise.
                - frame: BGR Image matrix (numpy ndarray) if successful, None otherwise.
        """
        pass

    @abstractmethod
    def stop(self) -> None:
        """
        Stop video capture and safely release hardware resources.
        """
        pass

    @abstractmethod
    def is_opened(self) -> bool:
        """
        Check if the camera input stream is currently opened and active.
        """
        pass

    @abstractmethod
    def get_properties(self) -> Dict[str, Any]:
        """
        Retrieve camera hardware properties and real-time operational statistics.
        
        Returns:
            Dict containing:
                - index: Camera index or source URI
                - width: Frame width in pixels
                - height: Frame height in pixels
                - configured_fps: Configured target frames per second
                - measured_fps: Real-time calculated frame rate
                - capture_latency_ms: Average frame capture latency in milliseconds
                - is_opened: Stream status boolean
        """
        pass
