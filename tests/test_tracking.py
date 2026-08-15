import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.object_detection.interface import Detection
from modules.object_tracking.tracker import BoTSORTTracker
from modules.object_tracking.interface import Track


class TestObjectTracking(unittest.TestCase):
    """
    Unit test suite for Module 04 — BoT-SORT Multi-Object Tracking.
    """

    def setUp(self):
        """Initialize fresh tracker instance before each test."""
        self.tracker = BoTSORTTracker(iou_threshold=0.3, max_age=5)
        self.tracker.initialize()

    def tearDown(self):
        """Clean up tracker state."""
        self.tracker.reset()

    def test_01_tracker_initialization(self):
        """Test 1: Validate tracker initialization."""
        self.assertTrue(self.tracker._is_initialized)
        info = self.tracker.get_tracker_info()
        self.assertEqual(info["tracker_type"], "BoT-SORT")
        self.assertEqual(info["active_tracks_count"], 0)

    def test_02_detection_input_acceptance(self):
        """Test 2: Validate acceptance of detection list input."""
        det = Detection(
            class_id=0, class_name="person", confidence=0.9,
            bounding_box=[100.0, 100.0, 200.0, 300.0],
            center_x=150.0, center_y=200.0, width=100.0, height=200.0
        )
        tracks = self.tracker.update([det])
        self.assertIsInstance(tracks, list)
        self.assertEqual(len(tracks), 1)

    def test_03_track_creation(self):
        """Test 3: Track creation and initial state verification."""
        det = Detection(
            class_id=1, class_name="chair", confidence=0.85,
            bounding_box=[50.0, 50.0, 150.0, 150.0],
            center_x=100.0, center_y=100.0, width=100.0, height=100.0
        )
        tracks = self.tracker.update([det])
        self.assertEqual(len(tracks), 1)
        track = tracks[0]
        self.assertIsInstance(track, Track)
        self.assertEqual(track.tracking_state, "NEW")
        self.assertEqual(track.class_name, "chair")
        self.assertEqual(track.hits, 1)

    def test_04_track_id_persistence(self):
        """Test 4: Track ID generation and persistent association across frames."""
        det1 = Detection(
            class_id=0, class_name="person", confidence=0.9,
            bounding_box=[100.0, 100.0, 200.0, 300.0],
            center_x=150.0, center_y=200.0, width=100.0, height=200.0
        )
        # Frame 1: Create track
        tracks1 = self.tracker.update([det1])
        initial_id = tracks1[0].track_id

        # Frame 2: Slightly shifted detection (same person)
        det2 = Detection(
            class_id=0, class_name="person", confidence=0.92,
            bounding_box=[105.0, 102.0, 205.0, 302.0],
            center_x=155.0, center_y=202.0, width=100.0, height=200.0
        )
        tracks2 = self.tracker.update([det2])
        self.assertEqual(len(tracks2), 1)
        self.assertEqual(tracks2[0].track_id, initial_id, "Track ID must persist across frames")
        self.assertEqual(tracks2[0].tracking_state, "TRACKED")
        self.assertEqual(tracks2[0].hits, 2)

    def test_05_multiple_object_tracking(self):
        """Test 5: Simultaneous multi-object tracking with distinct track IDs."""
        det_person = Detection(
            class_id=0, class_name="person", confidence=0.95,
            bounding_box=[10.0, 10.0, 80.0, 150.0],
            center_x=45.0, center_y=80.0, width=70.0, height=140.0
        )
        det_chair = Detection(
            class_id=1, class_name="chair", confidence=0.88,
            bounding_box=[300.0, 200.0, 400.0, 350.0],
            center_x=350.0, center_y=275.0, width=100.0, height=150.0
        )
        tracks = self.tracker.update([det_person, det_chair])
        self.assertEqual(len(tracks), 2)
        track_ids = {t.track_id for t in tracks}
        self.assertEqual(len(track_ids), 2, "Each distinct object must receive a unique track_id")

    def test_06_bounding_box_propagation(self):
        """Test 6: Bounding box spatial coordinate propagation."""
        det_orig = Detection(
            class_id=3, class_name="door", confidence=0.8,
            bounding_box=[200.0, 100.0, 350.0, 400.0],
            center_x=275.0, center_y=250.0, width=150.0, height=300.0
        )
        self.tracker.update([det_orig])

        det_moved = Detection(
            class_id=3, class_name="door", confidence=0.82,
            bounding_box=[210.0, 100.0, 360.0, 400.0],
            center_x=285.0, center_y=250.0, width=150.0, height=300.0
        )
        tracks = self.tracker.update([det_moved])
        self.assertEqual(tracks[0].bounding_box, [210.0, 100.0, 360.0, 400.0])

    def test_07_invalid_input_handling(self):
        """Test 7: Robust handling of None or empty detection inputs."""
        tracks_none = self.tracker.update(None)
        self.assertIsInstance(tracks_none, list)

        tracks_empty = self.tracker.update([])
        self.assertIsInstance(tracks_empty, list)

    def test_08_tracker_cleanup(self):
        """Test 8: Tracker reset and memory cleanup."""
        det = Detection(
            class_id=0, class_name="person", confidence=0.9,
            bounding_box=[10.0, 10.0, 50.0, 50.0],
            center_x=30.0, center_y=30.0, width=40.0, height=40.0
        )
        self.tracker.update([det])
        self.assertEqual(len(self.tracker.get_active_tracks()), 1)

        self.tracker.reset()
        self.assertEqual(len(self.tracker.get_active_tracks()), 0)
        info = self.tracker.get_tracker_info()
        self.assertEqual(info["active_tracks_count"], 0)


if __name__ == "__main__":
    unittest.main()
