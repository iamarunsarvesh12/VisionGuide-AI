import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.object_detection.detector import YOLOv8mDetector
from modules.object_detection.interface import Detection


class TestObjectDetection(unittest.TestCase):
    """
    Unit test suite for Module 02/03 — YOLOv8m Object Detection.
    """

    @classmethod
    def setUpClass(cls):
        """Initialize detector instance once for test suite execution."""
        cls.model_path = "yolov8m.pt"
        cls.detector = YOLOv8mDetector(model_path=cls.model_path, confidence_threshold=0.25)
        cls.load_success = cls.detector.load_model()

    @classmethod
    def tearDownClass(cls):
        """Clean up model resources."""
        if cls.detector:
            cls.detector.release()

    def test_01_model_loading(self):
        """Test 1: Validate model loading status and load latency recording."""
        self.assertTrue(self.load_success, "YOLOv8m model failed to load.")
        info = self.detector.get_model_info()
        self.assertTrue(info["is_loaded"])
        self.assertEqual(info["model_name"], "YOLOv8m")
        self.assertGreater(info["model_load_time_ms"], 0.0)

    def test_02_frame_acceptance(self):
        """Test 2: Validate acceptance of standard BGR OpenCV frame."""
        # Create synthetic test frame (480x640x3 BGR uint8)
        synthetic_frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        detections = self.detector.detect(synthetic_frame)
        self.assertIsInstance(detections, list, "detect() must return a list of Detection objects")

    def test_03_detection_output_structure(self):
        """Test 3: Detection output structure validation."""
        synthetic_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        detections = self.detector.detect(synthetic_frame)
        for det in detections:
            self.assertIsInstance(det, Detection)
            self.assertIsInstance(det.class_id, int)
            self.assertIsInstance(det.class_name, str)
            self.assertIsInstance(det.confidence, float)
            self.assertIsInstance(det.bounding_box, list)
            self.assertEqual(len(det.bounding_box), 4)

    def test_04_bounding_box_validity(self):
        """Test 4: Bounding box coordinate logic check (x1 <= x2, y1 <= y2)."""
        synthetic_frame = np.full((480, 640, 3), 200, dtype=np.uint8)
        detections = self.detector.detect(synthetic_frame)
        for det in detections:
            x1, y1, x2, y2 = det.bounding_box
            self.assertLessEqual(x1, x2, f"Invalid bbox coordinates: x1 ({x1}) > x2 ({x2})")
            self.assertLessEqual(y1, y2, f"Invalid bbox coordinates: y1 ({y1}) > y2 ({y2})")
            self.assertGreaterEqual(det.width, 0.0)
            self.assertGreaterEqual(det.height, 0.0)

    def test_05_confidence_range(self):
        """Test 5: Confidence range validation (0.0 <= confidence <= 1.0)."""
        synthetic_frame = np.full((480, 640, 3), 50, dtype=np.uint8)
        detections = self.detector.detect(synthetic_frame)
        for det in detections:
            self.assertGreaterEqual(det.confidence, 0.0)
            self.assertLessEqual(det.confidence, 1.0)
            self.assertGreaterEqual(det.confidence, self.detector.confidence_threshold)

    def test_06_class_name_mapping(self):
        """Test 6: Class names mapping dictionary validation."""
        class_names = self.detector.get_class_names()
        self.assertIsInstance(class_names, dict)
        self.assertGreater(len(class_names), 0, "Class names mapping dictionary must not be empty")
        self.assertIn(0, class_names, "COCO Class ID 0 ('person') must be present")
        self.assertEqual(class_names[0], "person")

    def test_07_invalid_frame_handling(self):
        """Test 7: Robust handling of invalid input frames (None, empty array)."""
        # None input
        dets_none = self.detector.detect(None)
        self.assertEqual(dets_none, [])

        # Empty array input
        dets_empty = self.detector.detect(np.array([]))
        self.assertEqual(dets_empty, [])

        # Non-image input type
        dets_invalid_type = self.detector.detect("not_an_image")  # type: ignore
        self.assertEqual(dets_invalid_type, [])

    def test_08_model_cleanup(self):
        """Test 8: Model resource release cleanup check."""
        temp_detector = YOLOv8mDetector(model_path=self.model_path)
        self.assertTrue(temp_detector.load_model())
        self.assertTrue(temp_detector.get_model_info()["is_loaded"])
        
        temp_detector.release()
        info = temp_detector.get_model_info()
        self.assertFalse(info["is_loaded"])
        
        dets_after_release = temp_detector.detect(np.full((480, 640, 3), 100, dtype=np.uint8))
        self.assertEqual(dets_after_release, [])


if __name__ == "__main__":
    unittest.main()
