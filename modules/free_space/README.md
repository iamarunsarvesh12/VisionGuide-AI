# Module 08 — Image-Space Free-Space Analysis

## 1. Purpose
The **Image-Space Free-Space Analysis Module** evaluates horizontal scene regions (`LEFT`, `CENTER`, `RIGHT`) to determine regional occupancy states (`CLEAR`, `BLOCKED`, `UNCERTAIN`), regional occupancy scores ($0.0 \le \text{occupancy\_score} \le 1.0$), and safe-space scores ($1.0 - \text{occupancy\_score}$).

---

## 2. Problem Statement
Visual object detection alone indicates *what* objects are present, but does not explicitly quantify *which regions of the walking corridor are currently open or obstructed*. Free-Space Analysis aggregates object detection, multi-object tracking, PHMU temporal memory states, monocular distance estimates, and context-aware danger scores into structured regional traversability metrics.

---

## 3. Architecture
```text
  Camera Input (Module 01)
            │
            ▼
  YOLOv8m Object Detection (Module 03)
            │
            ▼
  BoT-SORT Multi-Object Tracking (Module 04)
            │
            ▼
  Persistent Hazard Memory Unit - PHMU (Module 05)
            │
            ▼
  Monocular Distance Estimation (Module 06)
            │
            ▼
  Context-Aware Danger Mapping (Module 07)
            │
            ▼
┌───────────────────────────────────────┐
│  ImageSpaceFreeSpaceAnalyzer (Mod 08) │
└───────────────────┬───────────────────┘
                    │
                    ▼
    Regional Traversability & Occupancy  ──► [State, Score, Safe-Space, Objects]
                    │
                    ▼
      Context-Aware Decision Engine (Module 09 - Future Phase)
```

> [!IMPORTANT]
> **Architectural Boundary Rule**: Free-Space Analysis outputs ONLY structured regional occupancy objects (`FreeSpaceAnalysisResult`, `RegionOccupancy`). It does **NOT** generate navigation commands (`LEFT`, `RIGHT`, `FORWARD`, `STOP`). Command generation is strictly reserved for Module 09 (Decision Engine).

---

## 4. Inputs
* List of [`DangerAssessment`](../danger_mapping/models.py) objects from Module 07, or [`HazardMemoryRecord`](../hazard_memory/models.py) entries from Module 05.

---

## 5. Outputs
* Structured [`FreeSpaceAnalysisResult`](models.py) object containing:
  - `regions`: Dictionary mapping `"LEFT"`, `"CENTER"`, `"RIGHT"` to [`RegionOccupancy`](models.py):
    - `region_name` (str)
    - `occupancy_state` (`"CLEAR"`, `"BLOCKED"`, `"UNCERTAIN"`)
    - `occupancy_score` (float in `[0.0, 1.0]`)
    - `safe_space_score` (float in `[0.0, 1.0]`, $1.0 - \text{occupancy\_score}$)
    - `blocked_object_ids` (List[int])
    - `dominant_danger_level` (`"NONE"`, `"LOW"`, `"MODERATE"`, `"HIGH"`, `"CRITICAL"`)
    - `confidence` (float in `[0.0, 1.0]`)
    - `reasoning` (str)
  - `total_hazards_assessed` (int)
  - `overall_traversability` (`"CLEAR"`, `"PARTIALLY_BLOCKED"`, `"BLOCKED"`)

---

## 6. Region Model
Normalized horizontal camera boundaries:
* `LEFT`: $[0.00, 0.33]$
* `CENTER`: $[0.33, 0.67]$
* `RIGHT`: $[0.67, 1.00]$

---

## 7. Bounding-Box Overlap
Horizontal interval overlap $O_{\text{frac}}$ between normalized bounding box $[x1_{\text{norm}}, x2_{\text{norm}}]$ and region $[R_{\text{start}}, R_{\text{end}}]$:
$$O_{\text{frac}} = \frac{\max(0, \min(x2_{\text{norm}}, R_{\text{end}}) - \max(x1_{\text{norm}}, R_{\text{start}}))}{R_{\text{end}} - R_{\text{start}}}$$

Wide objects crossing multiple boundaries contribute proportionally to all overlapped regions.

---

## 8. Distance Integration
Distance category multiplier $F_{\text{dist}}$:
* `NEAR` $\implies 1.00$
* `MEDIUM` $\implies 0.50$
* `FAR` $\implies 0.20$
* `UNKNOWN` $\implies 0.10$

---

## 9. Danger Integration
Multiplies by `danger_score` ($0.0 \le D_{\text{score}} \le 1.0$) calculated by Module 07.

---

## 10. PHMU Integration
Memory state factor $M_{\text{state\_factor}}$:
* `ACTIVE` / `RECOVERED` $\implies 1.00$
* `OCCLUDED` $\implies 0.80$
* `REMEMBERED` $\implies 0.60$ (Contributes blocking evidence with decayed confidence $C_{\text{mem}}$)
* `EXPIRED` $\implies 0.00$ (Ignored completely)

---

## 11. Occupancy Model
Per-object contribution:
$$S_{k, R} = O_{\text{frac}, k, R} \times F_{\text{dist}, k} \times D_{\text{score}, k} \times C_{\text{mem}, k} \times M_{\text{state\_factor}, k} \times V_{\text{boost}, k}$$

Regional accumulation:
$$\text{Occupancy Score}_R = \text{clamp}\left(\sum_k S_{k, R}, 0.0, 1.0\right)$$

State Mapping:
* $\text{Occupancy Score}_R \le 0.25 \implies \mathbf{CLEAR}$
* $0.25 < \text{Occupancy Score}_R < 0.60 \implies \mathbf{UNCERTAIN}$
* $\text{Occupancy Score}_R \ge 0.60 \implies \mathbf{BLOCKED}$

---

## 12. Safe-Space Calculation
$$\text{Safe Space Score}_R = 1.0 - \text{Occupancy Score}_R$$

---

## 13. Uncertainty Handling
Regions with intermediate occupancy ($0.25 < \text{score} < 0.60$) or decayed memory observations are categorized as `UNCERTAIN` to prevent false confidence.

---

## 14. Configuration
Defined in [`config/config.yaml`](../../config/config.yaml) under `free_space`.

---

## 15. Testing
Covered in [`tests/test_free_space.py`](../../tests/test_free_space.py) (25 unit tests):
1. Analyzer initialization.
2. Valid input processing.
3. LEFT zoning.
4. CENTER zoning.
5. RIGHT zoning.
6. Empty scene handling (`CLEAR` unverified space).
7. Single center obstacle.
8. Left obstacle.
9. Right obstacle.
10. Near obstacle.
11. Medium obstacle.
12. Far obstacle.
13. High danger obstacle.
14. Low danger obstacle.
15. Multiple obstacles accumulating occupancy.
16. Cross-region bounding box (wide table overlapping `CENTER` and `RIGHT`).
17. ACTIVE PHMU observation.
18. REMEMBERED PHMU observation (`UNCERTAIN` / decayed occupancy).
19. EXPIRED PHMU exclusion.
20. Low-confidence input handling.
21. Missing distance category (`UNKNOWN`).
22. Occupancy score bounds (`[0.0, 1.0]`).
23. Safe space score bounds (`[0.0, 1.0]`).
24. Region state thresholds (`CLEAR`, `BLOCKED`, `UNCERTAIN`).
25. **Core Synthetic Free-Space Experiments** (Scenarios A through H).

---

## 16. Performance
* **Analysis Latency**: `< 0.02 ms` per frame pass.
* **CPU Overhead**: Zero computational bottleneck (> 50,000 passes/sec).

---

## 17. Explicit Limitations
> [!WARNING]
> 1. This is image-space free-space estimation, not true 3D spatial metric mapping.
> 2. A clear image region does not guarantee a physically safe 3D walking path.
> 3. Monocular distance estimation is approximate.
> 4. 2D bounding boxes do not provide exact 3D obstacle geometry.
> 5. Camera motion affects apparent image-space object position.
> 6. Unobserved regions must not automatically be considered physically safe (**ABSENCE OF DETECTION ≠ PROOF OF SAFETY**).
> 7. PHMU remembered hazards represent historical context, not current observations.
> 8. No ground-plane segmentation model is implemented in this phase.
> 9. No SLAM is implemented.
> 10. No global navigation planning is implemented.
> 11. No movement commands (`LEFT`, `RIGHT`, `FORWARD`, `STOP`) are generated by this module.

---

## 18. Integration with Decision Engine (Phase 8)
In Phase 8, the Decision Engine (Module 09) will consume `FreeSpaceAnalysisResult` alongside `DangerAssessment` to evaluate immediate safety rules and generate vocal navigation instructions.
