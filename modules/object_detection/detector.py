import time
import os
import sys
import logging
from typing import List, Dict, Any, Optional
import numpy as np

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

from modules.object_detection.interface import ObjectDetectorInterface, Detection

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

# Module logger setup
logger = logging.getLogger("ObjectDetection")
logger.setLevel(logging.INFO)
if not logger.handlers:
    c_handler = logging.StreamHandler(sys.stdout)
    c_handler.setFormatter(logging.Formatter("[%(name)s][%(levelname)s] %(message)s"))
    logger.addHandler(c_handler)
    
    f_handler = logging.FileHandler("logs/object_detection.log", mode="a", encoding="utf-8")
    f_handler.setFormatter(logging.Formatter("%(asctime)s [%(name)s][%(levelname)s] %(message)s"))
    logger.addHandler(f_handler)


class YOLOv8mDetector(ObjectDetectorInterface):
    """
    YOLOv8m Object Detector implementation for VisionGuide AI.
    
    Performs real-time CPU object detection on OpenCV video frames, returning
    structured Detection instances containing bounding boxes, centroids, confidences,
    and class names.
    """

    def __init__(
        self,
        model_path: str = "yolov8m.pt",
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        device: str = "cpu",
    ):
        """
        Initialize detector parameters.
        
        Args:
            model_path: Path to model weights file or pretrained name (e.g. 'yolov8m.pt').
            confidence_threshold: Minimum detection confidence threshold (0.0 to 1.0).
            iou_threshold: Non-maximum suppression (NMS) IoU threshold.
            device: Computing device ('cpu' or 'cuda'). Forced to 'cpu' on CPU systems.
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.requested_device = device

        self.model: Optional[Any] = None
        self._is_loaded: bool = False
        self.class_names: Dict[int, str] = {}

        # Performance tracking metrics
        self.last_inference_latency_ms: float = 0.0
        self.model_load_time_ms: float = 0.0
        self.total_frames_processed: int = 0
        self._total_inference_time_s: float = 0.0

    def initialize(self, config_dict: Optional[Dict[str, Any]] = None) -> bool:
        """Initialize and load YOLOv8m model weights."""
        if config_dict:
            y_cfg = config_dict.get("yolo", {})
            m_path = y_cfg.get("model_path", self.model_path)
            conf = y_cfg.get("confidence_threshold", self.confidence_threshold)
            iou = y_cfg.get("iou_threshold", self.iou_threshold)
            return self.load_model(model_path=m_path, confidence_threshold=conf, iou_threshold=iou)
        return self.load_model()

    def load_model(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
    ) -> bool:
        """
        Load YOLOv8m PyTorch weights and initialize Ultralytics model engine.
        """
        if not ULTRALYTICS_AVAILABLE:
            logger.error("Ultralytics package is not installed.")
            return False

        if model_path:
            self.model_path = model_path
        if confidence_threshold is not None:
            self.confidence_threshold = confidence_threshold
        if iou_threshold is not None:
            self.iou_threshold = iou_threshold

        logger.info(f"Loading YOLOv8m model weights from '{self.model_path}' on device '{self.requested_device}'")
        t0 = time.perf_counter()

        try:
            # Initialize YOLOv8 model instance
            self.model = YOLO(self.model_path)
            t1 = time.perf_counter()
            self.model_load_time_ms = (t1 - t0) * 1000.0

            # Retrieve class names mapping dictionary
            if hasattr(self.model, "names") and isinstance(self.model.names, dict):
                self.class_names = {int(k): str(v) for k, v in self.model.names.items()}
            elif hasattr(self.model, "names") and isinstance(self.model.names, (list, tuple)):
                self.class_names = {i: str(v) for i, v in enumerate(self.model.names)}
            else:
                self.class_names = {}

            self._is_loaded = True
            logger.info(f"YOLOv8m loaded successfully in {self.model_load_time_ms:.2f} ms. Total classes: {len(self.class_names)}")
            return True

        except Exception as e:
            logger.error(f"Failed to load YOLOv8m model from '{self.model_path}': {e}")
            self.model = None
            self._is_loaded = False
            return False

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Perform inference on an input OpenCV frame matrix.
        
        Returns a list of structured Detection objects.
        """
        if not self._is_loaded or self.model is None:
            logger.error("Attempted detection before model was successfully loaded.")
            return []

        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            logger.warning("Received invalid, empty, or non-numpy frame for detection.")
            return []

        t_start = time.perf_counter()

        try:
            # Execute Ultralytics inference
            results = self.model.predict(
                source=frame,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                device=self.requested_device,
                verbose=False
            )
            t_end = time.perf_counter()

            self.last_inference_latency_ms = (t_end - t_start) * 1000.0
            self.total_frames_processed += 1
            self._total_inference_time_s += (t_end - t_start)

            detections: List[Detection] = []

            if results and len(results) > 0:
                result = results[0]
                if result.boxes is not None and len(result.boxes) > 0:
                    boxes = result.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
                    confs = result.boxes.conf.cpu().numpy()
                    clss = result.boxes.cls.cpu().numpy().astype(int)

                    for box, conf, cls_id in zip(boxes, confs, clss):
                        x1, y1, x2, y2 = [float(v) for v in box]
                        width = max(0.0, x2 - x1)
                        height = max(0.0, y2 - y1)
                        center_x = x1 + (width / 2.0)
                        center_y = y1 + (height / 2.0)
                        class_name = self.class_names.get(cls_id, f"class_{cls_id}")

                        det = Detection(
                            class_id=int(cls_id),
                            class_name=str(class_name),
                            confidence=float(conf),
                            bounding_box=[x1, y1, x2, y2],
                            center_x=center_x,
                            center_y=center_y,
                            width=width,
                            height=height
                        )
                        detections.append(det)

            return detections

        except Exception as e:
            logger.error(f"Exception during YOLOv8m inference: {e}")
            return []

    def get_class_names(self) -> Dict[int, str]:
        """Return class ID to class name mapping."""
        return self.class_names

    def get_model_info(self) -> Dict[str, Any]:
        """Return model specifications and real-time inference statistics."""
        avg_fps = (self.total_frames_processed / self._total_inference_time_s) if self._total_inference_time_s > 0 else 0.0
        avg_latency = (self._total_inference_time_s * 1000.0 / self.total_frames_processed) if self.total_frames_processed > 0 else 0.0

        return {
            "model_path": self.model_path,
            "model_name": "YOLOv8m",
            "is_loaded": self._is_loaded,
            "device": self.requested_device,
            "confidence_threshold": self.confidence_threshold,
            "iou_threshold": self.iou_threshold,
            "model_load_time_ms": round(self.model_load_time_ms, 2),
            "last_inference_latency_ms": round(self.last_inference_latency_ms, 2),
            "average_inference_latency_ms": round(avg_latency, 2),
            "average_inference_fps": round(avg_fps, 2),
            "total_frames_processed": self.total_frames_processed,
            "total_classes": len(self.class_names),
        }

    def release(self) -> None:
        """Release PyTorch model memory and reset handles."""
        self.model = None
        self._is_loaded = False
        logger.info("YOLOv8m model resources released.")
