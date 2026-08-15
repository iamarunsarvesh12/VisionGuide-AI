import sys
import os
import unittest
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.object_tracking.interface import Track
from modules.hazard_memory.memory import PersistentHazardMemory
from modules.hazard_memory.models import HazardMemoryRecord


class TestPersistentHazardMemory(unittest.TestCase):
    """
    Unit test suite for Module 05 — Persistent Hazard Memory Unit (PHMU).
    """

    def setUp(self):
        """Initialize fresh PHMU instance before each test."""
        self.phmu = PersistentHazardMemory(
            memory_timeout_seconds=3.0,
            decay_rate=0.2,
            minimum_memory_confidence=0.1,
            persistence_threshold=0.2
        )
        self.phmu.initialize()
        self.start_time = 1000.0  # Synthetic base timestamp in seconds

    def tearDown(self):
        """Clean up PHMU state."""
        self.phmu.clear()

    def test_01_phmu_initialization(self):
        """Test 1: Validate PHMU initialization and default statistics."""
        self.assertTrue(self.phmu._is_initialized)
        stats = self.phmu.get_statistics()
        self.assertEqual(stats["total_memories_created"], 0)
        self.assertEqual(stats["currently_active_hazards"], 0)
        self.assertEqual(stats["memory_timeout_seconds"], 3.0)

    def test_02_new_track_creates_memory(self):
        """Test 2: Validate new track creation in PHMU (State: ACTIVE)."""
        trk = Track(
            track_id=7, class_id=1, class_name="chair", confidence=0.85,
            bounding_box=[100.0, 100.0, 200.0, 200.0],
            center_x=150.0, center_y=150.0, width=100.0, height=100.0,
            tracking_state="NEW"
        )
        hazards = self.phmu.update([trk], current_time=self.start_time)
        self.assertEqual(len(hazards), 1)
        rec = hazards[0]
        self.assertIsInstance(rec, HazardMemoryRecord)
        self.assertEqual(rec.track_id, 7)
        self.assertEqual(rec.object_class, "chair")
        self.assertEqual(rec.memory_state, "ACTIVE")
        self.assertIsNone(rec.estimated_distance)
        self.assertIsNone(rec.danger_score)

    def test_03_existing_track_updates_memory(self):
        """Test 3: Validate existing track update in PHMU."""
        trk1 = Track(
            track_id=7, class_id=1, class_name="chair", confidence=0.85,
            bounding_box=[100.0, 100.0, 200.0, 200.0],
            center_x=150.0, center_y=150.0, width=100.0, height=100.0,
            tracking_state="NEW"
        )
        self.phmu.update([trk1], current_time=self.start_time)

        # Update position in next frame
        trk2 = Track(
            track_id=7, class_id=1, class_name="chair", confidence=0.90,
            bounding_box=[110.0, 100.0, 210.0, 200.0],
            center_x=160.0, center_y=150.0, width=100.0, height=100.0,
            tracking_state="TRACKED"
        )
        hazards = self.phmu.update([trk2], current_time=self.start_time + 0.1)
        self.assertEqual(len(hazards), 1)
        rec = hazards[0]
        self.assertEqual(rec.center_x, 160.0)
        self.assertEqual(rec.observation_count, 2)
        self.assertEqual(rec.memory_state, "ACTIVE")

    def test_04_multiple_tracks_independent_memories(self):
        """Test 4: Validate independent memory management for multiple tracks."""
        trk_person = Track(
            track_id=1, class_id=0, class_name="person", confidence=0.92,
            bounding_box=[10.0, 10.0, 50.0, 100.0],
            center_x=30.0, center_y=55.0, width=40.0, height=90.0,
            tracking_state="NEW"
        )
        trk_door = Track(
            track_id=2, class_id=3, class_name="door", confidence=0.88,
            bounding_box=[300.0, 50.0, 450.0, 400.0],
            center_x=375.0, center_y=225.0, width=150.0, height=350.0,
            tracking_state="NEW"
        )
        hazards = self.phmu.update([trk_person, trk_door], current_time=self.start_time)
        self.assertEqual(len(hazards), 2)
        ids = {h.track_id for h in hazards}
        self.assertEqual(ids, {1, 2})

    def test_05_missing_track_becomes_occluded_remembered(self):
        """Test 5: Validate transition of missing track to OCCLUDED / REMEMBERED."""
        trk = Track(
            track_id=7, class_id=1, class_name="chair", confidence=0.9,
            bounding_box=[100.0, 100.0, 200.0, 200.0],
            center_x=150.0, center_y=150.0, width=100.0, height=100.0,
            tracking_state="NEW"
        )
        self.phmu.update([trk], current_time=self.start_time)

        # 0.5s later: Track missing from observation
        hazards_05s = self.phmu.update([], current_time=self.start_time + 0.5)
        self.assertEqual(len(hazards_05s), 1)
        self.assertEqual(hazards_05s[0].memory_state, "OCCLUDED")

        # 1.5s later: Absence continues -> REMEMBERED
        hazards_15s = self.phmu.update([], current_time=self.start_time + 1.5)
        self.assertEqual(len(hazards_15s), 1)
        self.assertEqual(hazards_15s[0].memory_state, "REMEMBERED")

    def test_06_temporarily_missing_track_retained(self):
        """Test 6: Validate temporal memory retention during short absence."""
        trk = Track(
            track_id=5, class_id=4, class_name="stairs", confidence=0.8,
            bounding_box=[0.0, 200.0, 640.0, 480.0],
            center_x=320.0, center_y=340.0, width=640.0, height=280.0,
            tracking_state="NEW"
        )
        self.phmu.update([trk], current_time=self.start_time)

        # Missing for 2.0 seconds (less than 3.0s timeout)
        hazards = self.phmu.update([], current_time=self.start_time + 2.0)
        self.assertEqual(len(hazards), 1)
        self.assertEqual(hazards[0].track_id, 5)
        self.assertTrue(hazards[0].memory_state in ("OCCLUDED", "REMEMBERED"))

    def test_07_recovered_track_returns_to_active(self):
        """Test 7: Validate recovered track state transition (REMEMBERED -> RECOVERED -> ACTIVE)."""
        trk = Track(
            track_id=12, class_id=0, class_name="person", confidence=0.95,
            bounding_box=[200.0, 100.0, 300.0, 300.0],
            center_x=250.0, center_y=200.0, width=100.0, height=200.0,
            tracking_state="NEW"
        )
        self.phmu.update([trk], current_time=self.start_time)

        # Missing for 1.5s -> REMEMBERED
        self.phmu.update([], current_time=self.start_time + 1.5)
        rec_remembered = self.phmu.get_hazard(12)
        self.assertIsNotNone(rec_remembered)
        self.assertEqual(rec_remembered.memory_state, "REMEMBERED")

        # Object appears again -> RECOVERED
        trk_reappeared = Track(
            track_id=12, class_id=0, class_name="person", confidence=0.92,
            bounding_box=[205.0, 100.0, 305.0, 300.0],
            center_x=255.0, center_y=200.0, width=100.0, height=200.0,
            tracking_state="TRACKED"
        )
        hazards_rec = self.phmu.update([trk_reappeared], current_time=self.start_time + 1.6)
        self.assertEqual(hazards_rec[0].memory_state, "RECOVERED")

        # Next frame -> ACTIVE
        hazards_act = self.phmu.update([trk_reappeared], current_time=self.start_time + 1.7)
        self.assertEqual(hazards_act[0].memory_state, "ACTIVE")

    def test_08_memory_expires_after_timeout(self):
        """Test 8: Validate memory expiration after timeout threshold (3.0s)."""
        trk = Track(
            track_id=9, class_id=2, class_name="table", confidence=0.8,
            bounding_box=[100.0, 100.0, 200.0, 200.0],
            center_x=150.0, center_y=150.0, width=100.0, height=100.0,
            tracking_state="NEW"
        )
        self.phmu.update([trk], current_time=self.start_time)

        # Missing beyond 3.0s timeout (e.g. 3.5s)
        hazards = self.phmu.update([], current_time=self.start_time + 3.5)
        self.assertEqual(len(hazards), 0, "Hazard memory must expire after timeout")
        self.assertIsNone(self.phmu.get_hazard(9))
        stats = self.phmu.get_statistics()
        self.assertEqual(stats["total_expirations"], 1)

    def test_09_confidence_decay_bounds(self):
        """Test 9: Validate confidence decay calculation within [0.0, 1.0]."""
        initial_conf = 0.9
        decayed_1s = self.phmu.calculate_decayed_confidence(initial_conf, 1.0)
        decayed_5s = self.phmu.calculate_decayed_confidence(initial_conf, 5.0)

        self.assertLessEqual(decayed_1s, initial_conf)
        self.assertLessEqual(decayed_5s, decayed_1s)
        self.assertGreaterEqual(decayed_5s, 0.0)
        self.assertLessEqual(decayed_5s, 1.0)

    def test_10_persistence_score_bounds(self):
        """Test 10: Validate persistence score calculation within [0.0, 1.0]."""
        score1 = self.phmu.calculate_persistence_score(1.0, 10, 30)
        score2 = self.phmu.calculate_persistence_score(0.2, 1, 1)

        self.assertGreaterEqual(score1, 0.0)
        self.assertLessEqual(score1, 1.0)
        self.assertGreaterEqual(score2, 0.0)
        self.assertLessEqual(score2, 1.0)
        self.assertGreater(score1, score2)

    def test_11_invalid_input_handling(self):
        """Test 11: Validate safe handling of None input."""
        hazards = self.phmu.update(None, current_time=self.start_time)
        self.assertIsInstance(hazards, list)

    def test_12_phmu_clear_reset(self):
        """Test 12: Validate PHMU store clear and reset."""
        trk = Track(
            track_id=1, class_id=0, class_name="person", confidence=0.9,
            bounding_box=[10.0, 10.0, 50.0, 50.0],
            center_x=30.0, center_y=30.0, width=40.0, height=40.0,
            tracking_state="NEW"
        )
        self.phmu.update([trk], current_time=self.start_time)
        self.assertEqual(len(self.phmu.get_active_hazards()), 1)

        self.phmu.clear()
        self.assertEqual(len(self.phmu.get_active_hazards()), 0)

    def test_13_core_synthetic_persistence_experiment(self):
        """
        Test 13: Core Synthetic Persistence Experiment.
        Simulates: Frame 1 (Detect) -> Frame 2 (Detect) -> Frame 3 (Missing) -> Frame 4 (Missing) -> Frame 5 (Reappear) -> Frame 6 (Detect)
        Verifies: ACTIVE -> ACTIVE -> REMEMBERED -> REMEMBERED -> RECOVERED -> ACTIVE.
        """
        trk = Track(
            track_id=1, class_id=0, class_name="person", confidence=0.9,
            bounding_box=[100.0, 100.0, 200.0, 300.0],
            center_x=150.0, center_y=200.0, width=100.0, height=200.0,
            tracking_state="NEW"
        )

        # Frame 1: Person ID 1 detected -> ACTIVE
        h1 = self.phmu.update([trk], current_time=self.start_time + 0.0, frame_index=1)
        self.assertEqual(h1[0].memory_state, "ACTIVE")

        # Frame 2: Person ID 1 detected -> ACTIVE
        h2 = self.phmu.update([trk], current_time=self.start_time + 0.1, frame_index=2)
        self.assertEqual(h2[0].memory_state, "ACTIVE")

        # Frame 3: Person ID 1 missing -> OCCLUDED / REMEMBERED
        h3 = self.phmu.update([], current_time=self.start_time + 1.2, frame_index=3)
        self.assertEqual(h3[0].memory_state, "REMEMBERED")

        # Frame 4: Person ID 1 missing -> REMEMBERED
        h4 = self.phmu.update([], current_time=self.start_time + 2.0, frame_index=4)
        self.assertEqual(h4[0].memory_state, "REMEMBERED")

        # Frame 5: Person ID 1 detected again -> RECOVERED
        h5 = self.phmu.update([trk], current_time=self.start_time + 2.1, frame_index=5)
        self.assertEqual(h5[0].memory_state, "RECOVERED")

        # Frame 6: Person ID 1 detected -> ACTIVE
        h6 = self.phmu.update([trk], current_time=self.start_time + 2.2, frame_index=6)
        self.assertEqual(h6[0].memory_state, "ACTIVE")


if __name__ == "__main__":
    unittest.main()
