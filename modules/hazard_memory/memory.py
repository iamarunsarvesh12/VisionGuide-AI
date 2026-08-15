import time
import math
import os
import sys
import logging
from typing import List, Dict, Any, Optional

from modules.object_tracking.interface import Track
from modules.hazard_memory.models import HazardMemoryRecord
from modules.hazard_memory.interface import HazardMemoryInterface

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

# Module logger setup
logger = logging.getLogger("HazardMemory")
logger.setLevel(logging.INFO)
if not logger.handlers:
    c_handler = logging.StreamHandler(sys.stdout)
    c_handler.setFormatter(logging.Formatter("[%(name)s][%(levelname)s] %(message)s"))
    logger.addHandler(c_handler)
    
    f_handler = logging.FileHandler("logs/hazard_memory.log", mode="a", encoding="utf-8")
    f_handler.setFormatter(logging.Formatter("%(asctime)s [%(name)s][%(levelname)s] %(message)s"))
    logger.addHandler(f_handler)


class PersistentHazardMemory(HazardMemoryInterface):
    """
    Persistent Hazard Memory Unit (PHMU) Implementation for VisionGuide AI.
    
    Maintains short-term memory of navigation hazards across video frames so that
    hazards are temporarily retained when visually occluded or missed by detectors.
    """

    def __init__(
        self,
        memory_timeout_seconds: float = 3.0,
        decay_rate: float = 0.2,
        minimum_memory_confidence: float = 0.1,
        persistence_threshold: float = 0.2,
    ):
        """
        Initialize PHMU memory parameters.
        
        Args:
            memory_timeout_seconds: Duration in seconds to retain an unobserved hazard.
            decay_rate: Exponential confidence decay rate parameter.
            minimum_memory_confidence: Confidence threshold below which memory expires.
            persistence_threshold: Minimum persistence score to retain hazard.
        """
        self.memory_timeout_seconds = memory_timeout_seconds
        self.decay_rate = decay_rate
        self.minimum_memory_confidence = minimum_memory_confidence
        self.persistence_threshold = persistence_threshold

        self._memory_store: Dict[int, HazardMemoryRecord] = {}
        self._is_initialized: bool = False
        self._frame_count: int = 0

        # Statistics counter
        self.total_memories_created: int = 0
        self.total_recoveries: int = 0
        self.total_expirations: int = 0
        self.last_update_latency_ms: float = 0.0

    def initialize(self, config_dict: Optional[Dict[str, Any]] = None) -> bool:
        """Initialize memory storage and configure parameters."""
        t0 = time.perf_counter()
        if config_dict:
            self.memory_timeout_seconds = config_dict.get("memory_timeout_seconds", self.memory_timeout_seconds)
            self.decay_rate = config_dict.get("decay_rate", self.decay_rate)
            self.minimum_memory_confidence = config_dict.get("minimum_memory_confidence", self.minimum_memory_confidence)
            self.persistence_threshold = config_dict.get("persistence_threshold", self.persistence_threshold)

        self.clear()
        self._is_initialized = True
        t1 = time.perf_counter()

        logger.info(
            f"PHMU Initialized (timeout={self.memory_timeout_seconds}s, "
            f"decay_rate={self.decay_rate}, min_conf={self.minimum_memory_confidence})"
        )
        return True

    def calculate_decayed_confidence(self, initial_confidence: float, delta_time: float) -> float:
        """
        Calculate exponential memory confidence decay over time delta (seconds).
        Formula: C_mem = clamp(C_det * exp(-decay_rate * delta_t), 0.0, 1.0)
        """
        decayed = initial_confidence * math.exp(-self.decay_rate * max(0.0, delta_time))
        return max(0.0, min(1.0, float(decayed)))

    def calculate_persistence_score(
        self,
        memory_confidence: float,
        observation_count: int,
        track_age_frames: int
    ) -> float:
        """
        Calculate deterministic hazard persistence score.
        Formula: P_score = clamp(0.4 * C_mem + 0.4 * min(1.0, obs_count / 10) + 0.2 * min(1.0, age / 30), 0.0, 1.0)
        """
        c_factor = 0.4 * memory_confidence
        obs_factor = 0.4 * min(1.0, observation_count / 10.0)
        age_factor = 0.2 * min(1.0, track_age_frames / 30.0)

        p_score = c_factor + obs_factor + age_factor
        return max(0.0, min(1.0, float(p_score)))

    def update(
        self,
        tracks: List[Track],
        current_time: Optional[float] = None,
        frame_index: Optional[int] = None
    ) -> List[HazardMemoryRecord]:
        """
        Update memory state with current frame tracks from BoT-SORT tracker.
        Handles new memory creation, existing memory updates, occlusion retention,
        recovery transitions, confidence decay, and memory expirations.
        """
        if not self._is_initialized:
            self.initialize()

        t_start = time.perf_counter()

        now = time.time() if current_time is None else current_time
        if frame_index is not None:
            self._frame_count = frame_index
        else:
            self._frame_count += 1

        if tracks is None:
            tracks = []

        current_track_ids = {trk.track_id for trk in tracks}

        # 1. Process observed tracks
        for trk in tracks:
            trk_id = trk.track_id
            if trk_id not in self._memory_store:
                # Create NEW hazard memory record
                rec = HazardMemoryRecord(
                    track_id=trk_id,
                    object_class=trk.class_name,
                    class_id=trk.class_id,
                    bounding_box=list(trk.bounding_box),
                    center_x=trk.center_x,
                    center_y=trk.center_y,
                    width=trk.width,
                    height=trk.height,
                    estimated_distance=None,
                    danger_score=None,
                    detection_confidence=trk.confidence,
                    memory_confidence=trk.confidence,
                    last_seen_timestamp=now,
                    last_seen_frame=self._frame_count,
                    observation_count=1,
                    track_age_frames=1,
                    time_since_last_seen=0.0,
                    tracking_state=trk.tracking_state,
                    memory_state="ACTIVE"
                )
                rec.persistence_score = self.calculate_persistence_score(
                    rec.memory_confidence, rec.observation_count, rec.track_age_frames
                )
                self._memory_store[trk_id] = rec
                self.total_memories_created += 1
                logger.info(f"[PHMU] New hazard memory created: ID={trk_id} class={trk.class_name}")

            else:
                # Update existing memory record
                rec = self._memory_store[trk_id]
                prev_state = rec.memory_state

                rec.bounding_box = list(trk.bounding_box)
                rec.center_x = trk.center_x
                rec.center_y = trk.center_y
                rec.width = trk.width
                rec.height = trk.height
                rec.detection_confidence = trk.confidence
                rec.memory_confidence = trk.confidence  # Restored on direct observation
                rec.last_seen_timestamp = now
                rec.last_seen_frame = self._frame_count
                rec.observation_count += 1
                rec.track_age_frames += 1
                rec.time_since_last_seen = 0.0
                rec.tracking_state = trk.tracking_state

                # Transition RECOVERED or ACTIVE
                if prev_state in ("OCCLUDED", "REMEMBERED"):
                    rec.memory_state = "RECOVERED"
                    self.total_recoveries += 1
                    logger.info(f"[PHMU] Hazard recovered: ID={trk_id} class={rec.object_class}")
                else:
                    rec.memory_state = "ACTIVE"

                rec.persistence_score = self.calculate_persistence_score(
                    rec.memory_confidence, rec.observation_count, rec.track_age_frames
                )

        # 2. Process missing/unobserved hazard memories
        missing_track_ids = set(self._memory_store.keys()) - current_track_ids
        for m_id in missing_track_ids:
            rec = self._memory_store[m_id]
            delta_t = now - rec.last_seen_timestamp
            rec.time_since_last_seen = delta_t
            rec.track_age_frames += 1

            # Calculate confidence decay
            rec.memory_confidence = self.calculate_decayed_confidence(rec.detection_confidence, delta_t)
            rec.persistence_score = self.calculate_persistence_score(
                rec.memory_confidence, rec.observation_count, rec.track_age_frames
            )

            # Transition memory states
            if delta_t <= 1.0:
                rec.memory_state = "OCCLUDED"
            else:
                rec.memory_state = "REMEMBERED"

        # 3. Perform memory expirations
        self.expire_memories(now)

        t_end = time.perf_counter()
        self.last_update_latency_ms = (t_end - t_start) * 1000.0

        return self.get_active_hazards()

    def mark_missing(self, track_id: int, current_time: Optional[float] = None) -> None:
        """Mark a track memory explicitly as missing/occluded."""
        now = time.time() if current_time is None else current_time
        if track_id in self._memory_store:
            rec = self._memory_store[track_id]
            delta_t = now - rec.last_seen_timestamp
            rec.time_since_last_seen = delta_t
            rec.memory_confidence = self.calculate_decayed_confidence(rec.detection_confidence, delta_t)
            rec.memory_state = "OCCLUDED" if delta_t <= 1.0 else "REMEMBERED"

    def expire_memories(self, current_time: Optional[float] = None) -> List[int]:
        """
        Check for memories exceeding memory_timeout_seconds or below minimum_memory_confidence.
        Purge expired entries from active memory store.
        """
        now = time.time() if current_time is None else current_time
        expired_ids = []

        for trk_id, rec in list(self._memory_store.items()):
            delta_t = now - rec.last_seen_timestamp
            
            if (
                delta_t > self.memory_timeout_seconds
                or rec.memory_confidence < self.minimum_memory_confidence
            ):
                rec.memory_state = "EXPIRED"
                expired_ids.append(trk_id)
                logger.info(
                    f"[PHMU] Hazard expired: ID={trk_id} class={rec.object_class} "
                    f"(absent {delta_t:.2f}s, conf={rec.memory_confidence:.3f})"
                )
                del self._memory_store[trk_id]
                self.total_expirations += 1

        return expired_ids

    def get_active_hazards(self) -> List[HazardMemoryRecord]:
        """Return list of all non-expired hazard memory records."""
        return list(self._memory_store.values())

    def get_hazard(self, track_id: int) -> Optional[HazardMemoryRecord]:
        """Return hazard memory record for specified track_id."""
        return self._memory_store.get(track_id, None)

    def clear(self) -> None:
        """Reset memory store and counters."""
        self._memory_store.clear()
        self._frame_count = 0
        self.total_memories_created = 0
        self.total_recoveries = 0
        self.total_expirations = 0
        self.last_update_latency_ms = 0.0
        logger.info("[PHMU] Hazard memory store cleared.")

    def get_statistics(self) -> Dict[str, Any]:
        """Return PHMU operational statistics."""
        active_count = len(self._memory_store)
        state_counts = {"ACTIVE": 0, "OCCLUDED": 0, "REMEMBERED": 0, "RECOVERED": 0}
        for rec in self._memory_store.values():
            state_counts[rec.memory_state] = state_counts.get(rec.memory_state, 0) + 1

        return {
            "total_memories_created": self.total_memories_created,
            "currently_active_hazards": active_count,
            "state_breakdown": state_counts,
            "total_recoveries": self.total_recoveries,
            "total_expirations": self.total_expirations,
            "memory_timeout_seconds": self.memory_timeout_seconds,
            "decay_rate": self.decay_rate,
            "last_update_latency_ms": round(self.last_update_latency_ms, 3),
        }
