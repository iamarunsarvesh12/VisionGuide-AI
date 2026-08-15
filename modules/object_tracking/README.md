# Module 04 — BoT-SORT Multi-Object Tracking

## 1. Purpose of BoT-SORT
The **Object Tracking Module** implements **BoT-SORT (Robust Multi-Object Tracking)** to maintain temporal continuity of navigation-relevant objects across consecutive video frames. By assigning persistent `track_id` identities to detected visual objects, the system avoids treating each frame in isolation and tracks spatial movement over time.

---

## 2. Input from YOLOv8m
* Receives a list of structured [`Detection`](../object_detection/interface.py) instances (`class_id`, `class_name`, `confidence`, `bounding_box`, `center_x`, `center_y`, `width`, `height`) produced per frame by Module 03 (YOLOv8m Object Detection).

---

## 3. Output Tracking Records
* List of structured [`Track`](file:///c:/Users/Admin/Documents/VisionGuide%20AI/modules/object_tracking/interface.py) instances containing:
  - `track_id` (int): Unique persistent identifier maintained across video frames.
  - `class_id`, `class_name` (int, str)
  - `confidence` (float)
  - `bounding_box` (`[x1, y1, x2, y2]`)
  - `center_x`, `center_y`, `width`, `height` (spatial attributes)
  - `tracking_state` (`"NEW"`, `"TRACKED"`, `"LOST"`, `"REMOVED"`)
  - `age` (total frame age)
  - `hits` (successful detection associations)
  - `time_since_update` (consecutive missed detection frames)

---

## 4. Track ID Lifecycle
```text
 New Object Detected
         │
         ▼
 ┌───────────────┐
 │ State: "NEW"  │  ──► Assign new unique track_id
 └───────┬───────┘
         │
  Next Frame Match
         │
         ▼
 ┌───────────────┐
 │State: "TRACKED"│  ──► Maintain track_id & update spatial state
 └───────┬───────┘
         │
  Frame Missed (IoU < threshold)
         │
         ▼
 ┌───────────────┐
 │ State: "LOST" │  ──► Retain identity for max_age frames
 └───────┬───────┘
         │
 ┌───────┴─────────────────┐
 │                         │
Re-detected            Missed > max_age
 │                         │
 ▼                         ▼
"TRACKED"              "REMOVED" (Exits Active State)
```

---

## 5. Relationship Between Detection and Tracking
* **Detection (Module 03)**: Instantaneous spatial frame inference ("WHAT objects are in this frame?").
* **Tracking (Module 04)**: Inter-frame identity association ("WHICH detected object in frame $N$ corresponds to object ID $K$ from frame $N-1$?").

---

## 6. Current CPU Performance
* **Tracking Latency**: Extremely lightweight computation (< 1.5 ms per frame), creating zero CPU bottleneck.
* **Pipeline Bottleneck**: The overall pipeline FPS remains constrained by YOLOv8m CPU inference (~1.40 FPS / 712.81 ms).

---

## 7. Limitations
* **Occlusion Boundary**: BoT-SORT maintains tracks during short-term frame misses (e.g. `time_since_update <= max_age`). However, once `max_age` expires or an object remains occluded beyond the tracking window, BoT-SORT deletes the track.
* **Hazard Persistence Gap**: Pure object tracking does NOT maintain semantic hazard memory once visual evidence disappears completely.

---

## 8. Integration Boundary with Future PHMU (Phase 4)

> [!IMPORTANT]
> **Architectural Boundary Division**:
> * **YOLOv8m (Module 03)** answers: *"WHAT objects are detected?"*
> * **BoT-SORT (Module 04)** answers: *"WHICH detected object corresponds to the same object across frames?"*
> * **PHMU (Module 05 — Phase 4)** answers: *"WHAT relevant hazards should the system temporarily remember when visual observations are interrupted or occluded?"*

BoT-SORT provides the raw temporal `track_id` inputs into the **Persistent Hazard Memory Unit (PHMU)**, which manages hazard memory states (`ACTIVE`, `OCCLUDED`, `REMEMBERED`, `RECOVERED`, `EXPIRED`), decay rates, and safety persistence scoring.
