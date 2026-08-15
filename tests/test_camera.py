import sys
import os
import unittest
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.camera_input.camera import WebcamInput


class TestCameraInput(unittest.TestCase):
    """
    Unit test suite for Module 01 — Camera Input.
    Validates webcam initialization, frame capture, dimensions, resource cleanup,
    and invalid device index handling.
    """

    def setUp(self):
        """Default setup parameters."""
        self.valid_index = 0
        self.invalid_index = 99
        self.req_width = 640
        self.req_height = 480
        self.target_fps = 30

    def test_01_camera_initialization(self):
        """Test 1: Camera initialization on valid device index."""
        camera = WebcamInput(
            camera_index=self.valid_index,
            width=self.req_width,
            height=self.req_height,
            target_fps=self.target_fps
        )
        success = camera.start()
        self.assertTrue(success, f"Failed to initialize camera on index {self.valid_index}")
        self.assertTrue(camera.is_opened(), "Camera is_opened() returned False after successful start")
        camera.stop()

    def test_02_camera_availability(self):
        """Test 2: Camera availability check."""
        camera = WebcamInput(camera_index=self.valid_index)
        camera.start()
        props = camera.get_properties()
        self.assertEqual(props["index"], self.valid_index)
        self.assertTrue(props["is_opened"])
        camera.stop()
        props_after = camera.get_properties()
        self.assertFalse(props_after["is_opened"])

    def test_03_frame_capture(self):
        """Test 3: Frame capture validity."""
        camera = WebcamInput(camera_index=self.valid_index)
        self.assertTrue(camera.start(), "Failed to start camera for frame capture test")
        
        # Read multiple frames to test continuous acquisition
        for i in range(5):
            ret, frame = camera.read()
            self.assertTrue(ret, f"Frame read failed on iteration {i}")
            self.assertIsNotNone(frame, "Captured frame is None")
            self.assertIsInstance(frame, np.ndarray, "Captured frame is not a numpy ndarray")
        
        camera.stop()

    def test_04_frame_dimensions(self):
        """Test 4: Frame dimensions and channels check."""
        camera = WebcamInput(
            camera_index=self.valid_index,
            width=self.req_width,
            height=self.req_height
        )
        self.assertTrue(camera.start())
        ret, frame = camera.read()
        self.assertTrue(ret)
        self.assertIsNotNone(frame)
        
        height, width, channels = frame.shape
        self.assertGreater(height, 0, "Frame height must be > 0")
        self.assertGreater(width, 0, "Frame width must be > 0")
        self.assertEqual(channels, 3, "Frame must have 3 channels (BGR)")
        
        props = camera.get_properties()
        self.assertEqual(props["width"], width)
        self.assertEqual(props["height"], height)
        camera.stop()

    def test_05_graceful_shutdown(self):
        """Test 5: Graceful camera shutdown and handle release."""
        camera = WebcamInput(camera_index=self.valid_index)
        camera.start()
        self.assertTrue(camera.is_opened())
        
        camera.stop()
        self.assertFalse(camera.is_opened())
        
        # Subsequent read after stop must fail cleanly
        ret, frame = camera.read()
        self.assertFalse(ret)
        self.assertIsNone(frame)

    def test_06_invalid_camera_index(self):
        """Test 6: Invalid camera index handling."""
        camera = WebcamInput(camera_index=self.invalid_index)
        success = camera.start()
        self.assertFalse(success, f"Expected start() to fail for invalid index {self.invalid_index}")
        self.assertFalse(camera.is_opened())
        
        ret, frame = camera.read()
        self.assertFalse(ret)
        self.assertIsNone(frame)
        camera.stop()


if __name__ == "__main__":
    unittest.main()
