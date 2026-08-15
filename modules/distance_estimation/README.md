# Module 06 — Monocular Approximate Distance Estimation

## 1. Purpose
The **Distance Estimation Module** provides approximate relative proximity categorization (`NEAR`, `MEDIUM`, `FAR`) and monocular metric distance approximations for tracked navigation hazards in **VisionGuide AI**.

---

## 2. Problem Being Solved
Object detection and multi-object tracking provide spatial bounding boxes in 2D image pixel coordinates (`x1, y1, x2, y2`). However, safe navigation decision-making requires understanding relative proximity. This module estimates distance using monocular pinhole optics geometry and class-specific reference profiles without requiring expensive stereo cameras or LiDAR hardware.

---

## 3. Inputs
* Structured [`Track`](../object_tracking/interface.py) records from BoT-SORT (Module 04) or [`HazardMemoryRecord`](../hazard_memory/models.py) entries from PHMU (Module 05).

---

## 4. Outputs
* List of structured [`DistanceResult`](models.py) instances containing:
  - `track_id` (int)
  - `class_name` (str)
  - `distance_category` (`"NEAR"`, `"MEDIUM"`, `"FAR"`, `"UNKNOWN"`)
  - `distance_confidence` (float in `[0.0, 1.0]`)
  - `estimated_distance_m` (Optional[float] in metres)
  - `bounding_box` (`[x1, y1, x2, y2]`)
  - `estimation_method` (`"monocular_bbox"`)
  - `distance_status` (`"MEASURED"` or `"LAST_OBSERVED"`)

---

## 5. Estimation Method
* **Monocular Bounding Box Pinhole Model (`"monocular_bbox"`)**: Uses the relationship between physical object height $H_{\text{ref\_m}}$, camera focal length $f_{\text{px}}$, and observed bounding box pixel height $h_{\text{bbox\_px}}$:

$$d_{\text{approx}} = \frac{H_{\text{ref\_m}} \times f_{\text{px}}}{h_{\text{bbox\_px}}}$$

---

## 6. Proximity Categorization & Thresholds
Categorization thresholds configured in [`config/config.yaml`](../../config/config.yaml):

```yaml
distance_estimation:
  enabled: true
  method: "monocular_bbox"
  focal_length_px: 600.0
  near_threshold_m: 1.5
  medium_threshold_m: 3.0
```

* $d_{\text{approx}} \le 1.5\text{m} \implies \mathbf{NEAR}$
* $1.5\text{m} < d_{\text{approx}} \le 3.0\text{m} \implies \mathbf{MEDIUM}$
* $d_{\text{approx}} > 3.0\text{m} \implies \mathbf{FAR}$

---

## 7. Calibration Mechanism
Supports dynamic calibration for specific object classes via `calibrate_class()`:

$$\text{Calibrated } H_{\text{ref\_m}} = \frac{d_{\text{known\_m}} \times h_{\text{observed\_px}}}{f_{\text{px}}}$$

---

## 8. Configuration Profiles
Default class reference heights stored in [`config/config.yaml`](../../config/config.yaml):

```yaml
distance_profiles:
  person: 1.70m
  door: 2.00m
  chair: 0.85m
  table: 0.75m
  stairs: 1.20m
  default: 1.00m
```

---

## 9. Class-Specific Assumptions
Apparent pixel height varies significantly by object category. For example, a 200px tall `person` is farther away than a 200px tall `chair`. Class-specific reference profiles normalize height measurements across visual object types.

---

## 10. PHMU Integration & Remembered-Object Policy
When processing a remembered hazard (`memory_state == "OCCLUDED"` or `"REMEMBERED"`):
* The estimator does **NOT** fabricate a new physical measurement when visual observations are missing.
* It retains the `last_known_distance` and sets `distance_status = "LAST_OBSERVED"`.
* Its `distance_confidence` is set to the decayed PHMU `memory_confidence`.

---

## 11. Testing
Covered in [`tests/test_distance_estimation.py`](../../tests/test_distance_estimation.py):
1. Module initialization.
2. Valid tracking input.
3. Bounding-box height processing.
4. Distance category generation (`NEAR`, `MEDIUM`, `FAR`).
5. Near classification (< 1.5m).
6. Medium classification (1.5m – 3.0m).
7. Far classification (> 3.0m).
8. Invalid bounding box handling (`h <= 0`).
9. Missing track handling (`None`).
10. Multiple-object distance estimation.
11. Confidence bounds (`0.0 <= conf <= 1.0`).
12. PHMU integration.
13. Remembered-object handling (`LAST_OBSERVED` status policy).
14. Reset and profile calibration.
15. Core Synthetic Distance Experiment (Large, Medium, Small bbox consistency).

---

## 12. Performance
* **Estimation Latency**: `< 0.01 ms` per object.
* **CPU Overhead**: Zero computational bottleneck (> 100,000 estimations/sec).

---

## 13. Explicit Limitations
> [!WARNING]
> 1. Single-camera monocular distance estimation is inherently an approximation.
> 2. Bounding-box height is affected by physical object dimension variations.
> 3. Object pose, sitting vs. standing, and partial cropping alter apparent box height.
> 4. Camera pitch tilt and perspective alter ground-plane distance mapping.
> 5. Bounding box detector jitter affects per-frame distance stability.
> 6. Metric distance accuracy cannot be assumed without explicit camera calibration.
> 7. Remembered hazards do not have a newly measured distance unless currently observed (`LAST_OBSERVED`).

---

## 14. Accuracy Considerations
This prototype module prioritizes relative proximity categorization (`NEAR`, `MEDIUM`, `FAR`) for safety reasoning over millimeter-level metric depth.

---

## 15. Future Upgrade Possibilities
* Integration of lightweight monocular depth estimation neural networks (e.g. MiDaS / Depth Anything ONNX).
* Incorporating vertical ground-plane position (`center_y`) and camera mounting height.
