from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np


@dataclass
class Detection:
    """
    Standardized bounding box detection object for VisionGuide AI.
    
    Provides structured attributes decoupling detection results from any specific
    underlying framework (Ultralytics, OpenCV DNN, Torchvision, ONNX Runtime).
    """
    class_id: int
    class_name: str
    confidence: float
    bounding_box: List[float]  # [x1, y1, x2, y2] in pixel coordinates
    center_x: float
    center_y: float
    width: float
    height: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert detection instance to a serializable dictionary."""
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": round(float(self.confidence), 4),
            "bounding_box": [round(float(v), 2) for v in self.bounding_box],
            "center_x": round(float(self.center_x), 2),
            "center_y": round(float(self.center_y), 2),
            "width": round(float(self.width), 2),
            "height": round(float(self.height), 2),
        }


class ObjectDetectorInterface(ABC):
    """
    Abstract interface for object detection models in VisionGuide AI.
    """

    @abstractmethod
    def load_model(
        self,
        model_path: str = "yolov8m.pt",
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
    ) -> bool:
        """
        Load weights and initialize the detection model.
        Returns True if loaded successfully, False otherwise.
        """
        pass

    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Perform object detection on a single BGR OpenCV frame.
        
        Args:
            frame (np.ndarray): Input image matrix (H, W, 3)
            
        Returns:
            List[Detection]: Standardized detection objects
        """
        pass

    @abstractmethod
    def get_class_names(self) -> Dict[int, str]:
        """
        Retrieve class ID to class name mapping dictionary.
        """
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Return metadata regarding the current detector (model name, device, latency, etc.).
        """
        pass

    @abstractmethod
    def release(self) -> None:
        """
        Release model resources and memory handles.
        """
        pass
