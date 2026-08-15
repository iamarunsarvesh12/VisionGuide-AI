# Module 05 — Persistent Hazard Memory Unit (PHMU) ⭐ Core Innovation

## 1. Purpose
The **Persistent Hazard Memory Unit (PHMU)** is the core short-term temporal memory subsystem of **VisionGuide AI**. Its primary purpose is to maintain continuous, context-aware memory of navigation-relevant hazards across consecutive video frames so that hazards are temporarily retained when visually occluded, missed by object detectors, or affected by sudden camera movements.

---

## 2. Problem Being Solved
Standard frame-by-frame object detection models treat each frame in isolation. If a person, chair, or open door is temporarily hidden behind a pillar or briefly missed by an object detector for 1–2 seconds, traditional systems instantly forget the hazard. PHMU solves this problem by maintaining temporal hazard persistence:

```text
Current Visual Observation (BoT-SORT Track)
                 +
Previous Hazard Memory State
                 ↓
Persistent Environmental State (PHMU)
```

---

## 3. Relationship with YOLOv8m (Module 03)
* **YOLOv8m** answers: *"WHAT objects are detected in the current frame?"*
* YOLOv8m provides un-tracked bounding box predictions. PHMU does NOT consume raw YOLO detections directly; it relies on BoT-SORT for inter-frame identity tracking.

---

## 4. Relationship with BoT-SORT (Module 04)
* **BoT-SORT** answers: *"WHICH detected object corresponds to the same object across frames?"*
* BoT-SORT assigns persistent temporal `track_id` values. PHMU consumes these `Track` objects as inputs to update hazard memory records and track occlusion lifecycles.

---

## 5. Memory Structure
Each hazard memory entry is represented by [`HazardMemoryRecord`](file:///c:/Users/Admin/Documents/VisionGuide%20AI/modules/hazard_memory/models.py):

* `track_id` (int): Unique persistent identifier.
* `object_class` (str), `class_id` (int): Target navigation object class.
* `bounding_box`, `center_x`, `center_y`, `width`, `height`: Last observed spatial position in image coordinates.
* `estimated_distance` (Optional[float]): Placeholder (`None` until Phase 5 Distance Estimation).
* `danger_score` (Optional[float]): Placeholder (`None` until Phase 6 Danger Mapping).
* `persistence_score` (float in `[0.0, 1.0]`): Deterministic retention strength.
* `memory_confidence` (float in `[0.0, 1.0]`): Exponentially decayed confidence score.
* `detection_confidence` (float): Initial detector confidence.
* `last_seen_timestamp`, `last_seen_frame`: Temporal observation markers.
* `observation_count`, `track_age_frames`, `time_since_last_seen`: Temporal counters.
* `tracking_state`: Tracking state from BoT-SORT.
* `memory_state`: PHMU lifecycle state (`ACTIVE`, `OCCLUDED`, `REMEMBERED`, `RECOVERED`, `EXPIRED`).

---

## 6. Memory Lifecycle

```text
                     ┌─────────────┐
                     │   ACTIVE    │  ──► Object observed in current frame
                     └──────┬──────┘
                            │
                     Track missing from
                      observation
                            ↓
                     ┌─────────────┐
                     │  OCCLUDED   │  ──► Absent <= 1.0 second
                     └──────┬──────┘
                            │
                     Absence continues
                            ↓
                     ┌─────────────┐
                     │  REMEMBERED │  ──► Absent > 1.0 second (Memory retained)
                     └──────┬──────┘
                            │
             ┌──────────────┴──────────────┐
             │                             │
    Object detected again             Timeout reached
             │                        (or C_mem < 0.1)
             ▼                             ▼
      ┌─────────────┐               ┌─────────────┐
      │  RECOVERED  │               │   EXPIRED   │
      └──────┬──────┘               └──────┬──────┘
             │                             │
             ▼                             ▼
      ┌─────────────┐               Removed from Active
      │   ACTIVE    │                  PHMU Store
      └─────────────┘
```

---

## 7. Persistence Score
The persistence score is calculated deterministically to measure hazard retention priority:

$$P_{\text{score}} = \text{clamp}\left(0.4 \cdot C_{\text{mem}} + 0.4 \cdot \min\left(1.0, \frac{\text{obs\_count}}{10}\right) + 0.2 \cdot \min\left(1.0, \frac{\text{track\_age}}{30}\right), 0.0, 1.0\right)$$

---

## 8. Confidence Decay
When a hazard is unobserved ($\Delta t = t_{\text{current}} - t_{\text{last\_seen}} > 0$), its memory confidence decays exponentially:

$$C_{\text{mem}}(t) = \text{clamp}\left(C_{\text{det}} \cdot e^{-\text{decay\_rate} \cdot \Delta t}, 0.0, 1.0\right)$$

* `decay_rate` default = `0.2` per second.
* When observed ($\Delta t = 0$), $C_{\text{mem}} = C_{\text{det}}$.

---

## 9. Timeout
Configured in [`config/config.yaml`](../../config/config.yaml):

```yaml
phmu:
  memory_timeout_seconds: 3.0
  decay_rate: 0.2
  minimum_memory_confidence: 0.1
```

If an object remains missing beyond `memory_timeout_seconds` (3.0s) or its confidence drops below `0.1`, its state transitions to `EXPIRED` and it is safely purged from active memory.

---

## 10. Inputs
* List of [`Track`](../object_tracking/interface.py) records from BoT-SORT (Module 04).

---

## 11. Outputs
* List of active, non-expired [`HazardMemoryRecord`](models.py) entries (`ACTIVE`, `OCCLUDED`, `REMEMBERED`, `RECOVERED`).

---

## 12. API
All implementations inherit from [`HazardMemoryInterface`](interface.py):

```python
class HazardMemoryInterface(ABC):
    def initialize(self, config_dict: Optional[Dict] = None) -> bool: ...
    def update(self, tracks: List[Track], current_time: Optional[float] = None) -> List[HazardMemoryRecord]: ...
    def get_active_hazards(self) -> List[HazardMemoryRecord]: ...
    def get_hazard(self, track_id: int) -> Optional[HazardMemoryRecord]: ...
    def mark_missing(self, track_id: int) -> None: ...
    def expire_memories(self, current_time: Optional[float] = None) -> List[int]: ...
    def clear(self) -> None: ...
    def get_statistics(self) -> Dict[str, Any]: ...
```

---

## 13. Testing
Covered in [`tests/test_hazard_memory.py`](../../tests/test_hazard_memory.py):
1. PHMU initialization.
2. New track creates `ACTIVE` memory.
3. Existing track updates existing memory.
4. Independent multi-track management.
5. Missing track state transitions (`OCCLUDED` / `REMEMBERED`).
6. Temporal memory retention over short disappearances.
7. Track recovery (`RECOVERED` ➔ `ACTIVE`).
8. Memory expiration after timeout (`EXPIRED`).
9. Confidence decay range bounds (`[0.0, 1.0]`).
10. Persistence score range bounds (`[0.0, 1.0]`).
11. Invalid track input safety.
12. Memory clear/reset.

---

## 14. Performance
* **Update Latency**: `< 0.05 ms` per frame update.
* **CPU Overhead**: Zero computational bottleneck (operates at > 20,000 updates/sec).

---

## 15. Limitations
1. **Image-Space Coordinates**: Retains spatial centroids in image pixel coordinates (`center_x`, `center_y`). Does not yet perform 3D world-coordinate or camera egomotion transformation.
2. **Short-Term Memory**: PHMU is designed for short-term obstacle retention (seconds), not long-term indoor SLAM mapping.
3. **Tracker Dependency**: PHMU relies on BoT-SORT to re-identify recovered objects; if BoT-SORT assigns a new `track_id` upon re-emergence, PHMU treats it as a new hazard.

---

## 16. Future Integration with Distance Estimation (Phase 5)
In Phase 5, `estimated_distance` (metres & category) will populate `HazardMemoryRecord.estimated_distance`.

---

## 17. Future Integration with Danger Mapping (Phase 6)
In Phase 6, `danger_score` and danger level (`LOW`, `MEDIUM`, `HIGH`, `CRITICAL`) will populate `HazardMemoryRecord.danger_score`.

---

## 18. Future Integration with Decision Engine (Phase 8)
The Decision Engine (Module 09) will consume active hazard memories from PHMU to ensure navigation commands (`LEFT`, `RIGHT`, `FORWARD`, `STOP`) account for remembered hazards that are currently occluded from direct visual view.

---

## Technical Contribution & Invention Note

### Existing Technologies
* **YOLOv8m**: Pretrained deep learning detection framework.
* **BoT-SORT**: Bipartite graph association & tracking framework.
* **OpenCV / PyTorch**: Computer vision runtime libraries.

### Proposed System Mechanism (VisionGuide AI Contribution)
* Short-term Persistent Hazard Memory representation (`HazardMemoryRecord`).
* Deterministic 5-stage hazard memory lifecycle (`ACTIVE` ➔ `OCCLUDED` ➔ `REMEMBERED` ➔ `RECOVERED` ➔ `EXPIRED`).
* Exponential memory confidence decay & persistence scoring integration.
* Fusion of temporal hazard memory with downstream danger mapping and navigation decision engine.
