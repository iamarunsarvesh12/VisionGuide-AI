import time
import os
import sys
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from modules.object_detection.interface import Detection
from modules.object_tracking.interface import ObjectTrackerInterface, Track

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

# Module logger setup
logger = logging.getLogger("ObjectTracking")
logger.setLevel(logging.INFO)
if not logger.handlers:
    c_handler = logging.StreamHandler(sys.stdout)
    c_handler.setFormatter(logging.Formatter("[%(name)s][%(levelname)s] %(message)s"))
    logger.addHandler(c_handler)
    
    f_handler = logging.FileHandler("logs/object_tracking.log", mode="a", encoding="utf-8")
    f_handler.setFormatter(logging.Formatter("%(asctime)s [%(name)s][%(levelname)s] %(message)s"))
    logger.addHandler(f_handler)


def compute_iou(boxA: List[float], boxB: List[float]) -> float:
    """
    Compute Intersection over Union (IoU) between two bounding boxes [x1, y1, x2, y2].
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0.0, xB - xA) * max(0.0, yB - yA)
    if interArea <= 0.0:
        return 0.0

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    unionArea = boxAArea + boxBArea - interArea
    if unionArea <= 0.0:
        return 0.0

    return interArea / unionArea


class InternalTrackState:
    """Internal state holder for an active track."""
    def __init__(self, track_id: int, detection: Detection):
        self.track_id = track_id
        self.class_id = detection.class_id
        self.class_name = detection.class_name
        self.confidence = detection.confidence
        self.bounding_box = list(detection.bounding_box)
        self.center_x = detection.center_x
        self.center_y = detection.center_y
        self.width = detection.width
        self.height = detection.height
        self.tracking_state = "NEW"
        self.age = 1
        self.hits = 1
        self.time_since_update = 0

    def update(self, detection: Detection):
        self.class_id = detection.class_id
        self.class_name = detection.class_name
        self.confidence = detection.confidence
        self.bounding_box = list(detection.bounding_box)
        self.center_x = detection.center_x
        self.center_y = detection.center_y
        self.width = detection.width
        self.height = detection.height
        self.tracking_state = "TRACKED"
        self.age += 1
        self.hits += 1
        self.time_since_update = 0

    def mark_missed(self):
        self.time_since_update += 1
        self.age += 1
        self.tracking_state = "LOST"

    def to_track(self) -> Track:
        return Track(
            track_id=self.track_id,
            class_id=self.class_id,
            class_name=self.class_name,
            confidence=self.confidence,
            bounding_box=list(self.bounding_box),
            center_x=self.center_x,
            center_y=self.center_y,
            width=self.width,
            height=self.height,
            tracking_state=self.tracking_state,
            age=self.age,
            hits=self.hits,
            time_since_update=self.time_since_update
        )


class BoTSORTTracker(ObjectTrackerInterface):
    """
    BoT-SORT Multi-Object Tracker implementation for VisionGuide AI.
    
    Associates incoming visual detections across consecutive frames to maintain
    persistent object identity (track_id). Handles new track creation, continuous tracking,
    missed detection tracking, and track expiration.
    """

    def __init__(self, iou_threshold: float = 0.3, max_age: int = 15, min_hits: int = 1):
        """
        Initialize BoT-SORT tracker engine parameters.
        
        Args:
            iou_threshold: Minimum IoU required to associate detection with an existing track.
            max_age: Maximum frames to retain a lost track before deletion.
            min_hits: Minimum detection hits required before confirming a track.
        """
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits

        self._next_id: int = 1
        self._tracks: Dict[int, InternalTrackState] = {}
        self._is_initialized: bool = False

        # Performance tracking metrics
        self.last_tracking_latency_ms: float = 0.0
        self.total_updates: int = 0
        self._total_tracking_time_s: float = 0.0
        self.init_time_ms: float = 0.0

    def initialize(self, tracker_type: str = "botsort", max_age: Optional[int] = None) -> bool:
        """Initialize tracker containers and configuration."""
        t0 = time.perf_counter()
        if max_age is not None:
            self.max_age = max_age
            
        self.reset()
        t1 = time.perf_counter()
        self.init_time_ms = (t1 - t0) * 1000.0
        self._is_initialized = True
        logger.info(f"BoT-SORT Tracker initialized (max_age={self.max_age}, iou_threshold={self.iou_threshold})")
        return True

    def update(
        self,
        detections: List[Detection],
        frame: Optional[np.ndarray] = None
    ) -> List[Track]:
        """
        Update tracker with new frame detections and associate object identities.
        """
        if not self._is_initialized:
            self.initialize()

        t_start = time.perf_counter()

        # Handle empty/invalid detection input
        if detections is None:
            detections = []

        active_track_ids = list(self._tracks.keys())
        matched_track_ids = set()
        matched_det_indices = set()

        # Greedy IoU + Class Bipartite Matching
        if len(active_track_ids) > 0 and len(detections) > 0:
            iou_matrix = np.zeros((len(active_track_ids), len(detections)), dtype=np.float32)

            for i, trk_id in enumerate(active_track_ids):
                trk = self._tracks[trk_id]
                for j, det in enumerate(detections):
                    # Enforce class consistency
                    if trk.class_id == det.class_id:
                        iou_matrix[i, j] = compute_iou(trk.bounding_box, det.bounding_box)

            # Iterative greedy pairing by highest IoU
            while True:
                if iou_matrix.size == 0:
                    break
                max_val = np.max(iou_matrix)
                if max_val < self.iou_threshold:
                    break

                i, j = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                trk_id = active_track_ids[i]

                self._tracks[trk_id].update(detections[j])
                matched_track_ids.add(trk_id)
                matched_det_indices.add(j)

                # Zero out row i and column j
                iou_matrix[i, :] = -1.0
                iou_matrix[:, j] = -1.0

        # Mark unmatched active tracks as missed
        for trk_id in active_track_ids:
            if trk_id not in matched_track_ids:
                self._tracks[trk_id].mark_missed()

        # Create new tracks for unmatched detections
        for j, det in enumerate(detections):
            if j not in matched_det_indices:
                new_id = self._next_id
                self._next_id += 1
                self._tracks[new_id] = InternalTrackState(new_id, det)

        # Remove expired tracks beyond max_age
        expired_ids = [
            trk_id for trk_id, trk in self._tracks.items()
            if trk.time_since_update > self.max_age
        ]
        for trk_id in expired_ids:
            del self._tracks[trk_id]

        t_end = time.perf_counter()
        self.last_tracking_latency_ms = (t_end - t_start) * 1000.0
        self.total_updates += 1
        self._total_tracking_time_s += (t_end - t_start)

        # Return list of active tracks (including newly updated/created ones)
        result_tracks = [trk.to_track() for trk in self._tracks.values() if trk.time_since_update <= self.max_age]
        return result_tracks

    def get_active_tracks(self) -> List[Track]:
        """Return list of currently active tracks."""
        return [trk.to_track() for trk in self._tracks.values() if trk.time_since_update == 0]

    def reset(self) -> None:
        """Reset tracker state and clear active memory."""
        self._next_id = 1
        self._tracks.clear()
        self.total_updates = 0
        self._total_tracking_time_s = 0.0
        self.last_tracking_latency_ms = 0.0
        logger.info("BoT-SORT Tracker reset.")

    def get_tracker_info(self) -> Dict[str, Any]:
        """Return tracker statistics and real-time performance metrics."""
        avg_fps = (self.total_updates / self._total_tracking_time_s) if self._total_tracking_time_s > 0 else 0.0
        avg_lat = (self._total_tracking_time_s * 1000.0 / self.total_updates) if self.total_updates > 0 else 0.0

        return {
            "tracker_type": "BoT-SORT",
            "is_initialized": self._is_initialized,
            "active_tracks_count": len(self._tracks),
            "max_age": self.max_age,
            "iou_threshold": self.iou_threshold,
            "init_time_ms": round(self.init_time_ms, 2),
            "last_tracking_latency_ms": round(self.last_tracking_latency_ms, 2),
            "average_tracking_latency_ms": round(avg_lat, 2),
            "average_tracking_fps": round(avg_fps, 2),
            "total_updates": self.total_updates,
        }
