"""
VisionGuide AI - Module 02 & 03: Object Detection Subsystem
"""

from modules.object_detection.interface import ObjectDetectorInterface, Detection
from modules.object_detection.detector import YOLOv8mDetector

__all__ = ["ObjectDetectorInterface", "Detection", "YOLOv8mDetector"]
