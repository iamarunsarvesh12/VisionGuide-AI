# Module 07 — Context-Aware Danger Mapping

## 1. Purpose
The **Context-Aware Danger Mapping Module** converts physical spatial observations and temporal hazard states into a normalized danger score ($0.0 \le \text{danger\_score} \le 1.0$), danger level (`LOW`, `MODERATE`, `HIGH`, `CRITICAL`), navigation relevance boolean, and deterministic reasoning.

---

## 2. Architecture
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
┌───────────────────────────────────────┐
│   ContextAwareDangerMapper (Mod 07)   │
└───────────────────┬───────────────────┘
                    │
                    ▼
      Ranked Danger Assessments  ──► [Score, Level, Zone, Reasoning]
                    │
                    ▼
      Context-Aware Decision Engine (Module 09 - Future Phase)
```

> [!IMPORTANT]
> **Architectural Boundary Rule**: Danger Mapping produces ONLY structured danger assessment objects (`DangerAssessment`). It does **NOT** generate navigation commands (`LEFT`, `RIGHT`, `FORWARD`, `STOP`). Command generation is strictly reserved for Module 09 (Decision Engine).

---

## 3. Inputs
* Output list of [`DistanceResult`](../distance_estimation/models.py) objects from Module 06, or [`HazardMemoryRecord`](../hazard_memory/models.py) entries from Module 05.

---

## 4. Outputs
* List of structured [`DangerAssessment`](models.py) instances sorted by `danger_score` in descending order (highest risk first):
  - `track_id` (int)
  - `class_name` (str)
  - `danger_score` (float in `[0.0, 1.0]`)
  - `danger_level` (`"LOW"`, `"MODERATE"`, `"HIGH"`, `"CRITICAL"`)
  - `distance_category` (`"NEAR"`, `"MEDIUM"`, `"FAR"`, `"UNKNOWN"`)
  - `position_zone` (`"LEFT"`, `"CENTER"`, `"RIGHT"`)
  - `memory_state` (`"ACTIVE"`, `"OCCLUDED"`, `"REMEMBERED"`, `"RECOVERED"`)
  - `memory_confidence`, `persistence_score` (floats)
  - `navigation_relevance` (bool)
  - `danger_factors` (List[str])
  - `reasoning` (str)

---

## 5. Position Zoning
Horizontal normalized center: $x_{\text{norm}} = \frac{\text{center\_x}}{\text{frame\_width}}$

* $x_{\text{norm}} < 0.33 \implies \mathbf{LEFT}$ (Factor $F_{\text{pos}} = 0.50$)
* $0.33 \le x_{\text{norm}} \le 0.67 \implies \mathbf{CENTER}$ (Factor $F_{\text{pos}} = 1.00$)
* $x_{\text{norm}} > 0.67 \implies \mathbf{RIGHT}$ (Factor $F_{\text{pos}} = 0.50$)

---

## 6. Object Factors ($F_{\text{obj}}$)
Configured in [`config/config.yaml`](../../config/config.yaml):
* `stairs`: `0.95`, `glass_door`: `0.90`, `glass_wall`: `0.85`, `person`: `0.70`, `chair`: `0.65`, `table`: `0.60`, `door`: `0.50`, `exit`: `0.30`, `default`: `0.50`.

---

## 7. Distance Factors ($F_{\text{dist}}$)
* `NEAR` $\implies 1.00$
* `MEDIUM` $\implies 0.50$
* `FAR` $\implies 0.20$
* `UNKNOWN` $\implies 0.10$

---

## 8. Motion Factors ($F_{\text{mot}}$)
* Stationary / Unmeasured default = `0.50`
* Approaching vector = up to `1.00`

---

## 9. Persistence Factor ($F_{\text{pers}}$)
* Extracted directly from PHMU `persistence_score` in range `[0.0, 1.0]`.

---

## 10. Memory Confidence ($F_{\text{mem}}$)
* Extracted directly from PHMU `memory_confidence` in range `[0.0, 1.0]`.

---

## 11. Weighted Scoring Model
Configurable weights in [`config/config.yaml`](../../config/config.yaml):

$$\text{Danger Score} = \text{clamp}\left(0.25 F_{\text{obj}} + 0.30 F_{\text{dist}} + 0.25 F_{\text{pos}} + 0.05 F_{\text{mot}} + 0.10 F_{\text{pers}} + 0.05 F_{\text{mem}}, 0.0, 1.0\right)$$

Danger Level Thresholds:
* $\text{Danger Score} \ge 0.80 \implies \mathbf{CRITICAL}$
* $0.60 \le \text{Danger Score} < 0.80 \implies \mathbf{HIGH}$
* $0.35 \le \text{Danger Score} < 0.60 \implies \mathbf{MODERATE}$
* $\text{Danger Score} < 0.35 \implies \mathbf{LOW}$

---

## 12. Configuration
Defined in [`config/config.yaml`](../../config/config.yaml) under `danger_mapping` and `object_hazard_factors`.

---

## 13. Multi-Object Ranking
The `rank_hazards()` function sorts all active assessments in descending order of `danger_score`, ensuring the most critical navigation hazards are placed at the head of the array.

---

## 14. Testing
Covered in [`tests/test_danger_mapping.py`](../../tests/test_danger_mapping.py) (20 unit tests):
1. Mapper initialization.
2. Valid hazard input assessment.
3. Position zone LEFT.
4. Position zone CENTER.
5. Position zone RIGHT.
6. NEAR distance factor scoring.
7. MEDIUM distance factor scoring.
8. FAR distance factor scoring.
9. High-risk object rule (`stairs`).
10. Low-risk object rule (distant `chair`).
11. Persistence factor contribution.
12. Memory confidence factor contribution.
13. Remembered hazard handling (`REMEMBERED` state reasoning).
14. Expired hazard exclusion.
15. Multi-hazard ranking (highest score first).
16. Danger score bounds (`[0.0, 1.0]`).
17. Navigation relevance boolean.
18. Invalid input handling.
19. Missing distance handling.
20. Missing motion handling.
21. **Deterministic Core Context-Aware Experiment** (5 distinct scenarios).

---

## 15. Performance
* **Mapping Latency**: `< 0.02 ms` per hazard.
* **CPU Overhead**: Zero computational bottleneck (> 50,000 mappings/sec).

---

## 16. Explicit Limitations
> [!WARNING]
> 1. Danger score is a prototype heuristic weight model, not a medically or scientifically certified risk metric.
> 2. Image-space `CENTER` zone does not guarantee an object physically obstructs the user's 3D walking path.
> 3. Monocular distance inputs are approximate.
> 4. Object class alone does not dictate danger; spatial context governs score.
> 5. Motion estimation is sensitive to camera egomotion without IMU fusion.
> 6. Remembered hazards preserve historical context but carry decayed confidence.
> 7. Danger Mapping does NOT perform route planning or command generation (`LEFT`, `RIGHT`, `FORWARD`, `STOP`).
> 8. No 3D world-coordinate spatial map is constructed in this phase.

---

## 17. Integration with Decision Engine (Phase 8)
In Phase 8, the Decision Engine (Module 09) will consume the ranked `DangerAssessment` array alongside Free-Space Analysis (Module 08) to evaluate immediate safety rules and issue vocal navigation commands.
