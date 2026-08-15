# Image-Space Free-Space Analysis Performance Report — Module 08

**Project**: VisionGuide AI  
**Date**: 2026-08-09  
**Hardware Environment**: Windows 11 Laptop (CPU-Only, 4 Physical Cores / 8 Logical Processors, 16 GB RAM)  
**Free-Space Model**: Interval Bounding-Box Overlap & Occupancy Accumulator (`ImageSpaceFreeSpaceAnalyzer`)  
**Integrated 7-Stage Pipeline**: Camera ➔ YOLOv8m ➔ BoT-SORT ➔ PHMU ➔ Distance ➔ Danger ➔ Free-Space  

---

## 1. Measured Empirical Performance Summary

| Metric | Measured Value | Unit / Scale | Performance Evaluation |
| :--- | :--- | :--- | :--- |
| **Max Hazards Stress-Tested** | 100 simultaneous hazards | Records in batch | Synthetic stress-test |
| **Total Batch Analysis Latency** | `10.983 ms` | 100 hazard records | Ultra-fast accumulation |
| **Average Per-Hazard Latency** | **0.10983 ms** | Per hazard entry | Instantaneous calculation |
| **Integrated Frame Latency** | **0.743 ms** | Per live frame pass | Negligible CPU footprint |
| **YOLOv8m CPU Inference** | `837.12 ms` | Per frame pass | Main CPU bottleneck |
| **Camera Read Latency** | `2.90 ms` | Per frame pass | Webcam acquisition |
| **BoT-SORT Tracking Latency** | `0.20 ms` | Per frame pass | Lightweight tracking |
| **PHMU Hazard Memory Latency** | `0.187 ms` | Per frame pass | Lightweight memory |
| **Distance Estimation Latency**| `1.218 ms` | Per frame pass | Lightweight distance |
| **Context Danger Mapping Latency**| `0.850 ms` | Per frame pass | Lightweight risk mapping |
| **Total End-to-End Latency** | **843.21 ms** | 7-stage pipeline time | Governed by YOLO CPU pass |
| **End-to-End Pipeline Rate** | **1.19 FPS** | Live integrated stream | Synchronous CPU pass |
| **RAM Memory Consumption** | `555.25 MB` | Resident Set Size | Stable memory footprint |

---

## 2. Horizontal Region Occupancy Model

### Region Boundaries (Normalized $x_{\text{norm}}$)
* `LEFT`: $[0.00, 0.33]$
* `CENTER`: $[0.33, 0.67]$
* `RIGHT`: $[0.67, 1.00]$

### Configured State Thresholds
* $\text{Occupancy Score} \le 0.20 \implies \mathbf{CLEAR}$
* $0.20 < \text{Occupancy Score} < 0.45 \implies \mathbf{UNCERTAIN}$
* $\text{Occupancy Score} \ge 0.45 \implies \mathbf{BLOCKED}$

---

## 3. Core Synthetic Free-Space Experiment Verification

| Scenario | Description | LEFT State | CENTER State | RIGHT State | Overall Traversability |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Scenario A** | Completely Unobstructed Scene | **CLEAR** (0.00) | **CLEAR** (0.00) | **CLEAR** (0.00) | **CLEAR** |
| **Scenario B** | Single NEAR Center Obstacle | **CLEAR** (0.00) | **BLOCKED** (0.60) | **CLEAR** (0.00) | **PARTIALLY_BLOCKED** |
| **Scenario C** | Single NEAR Left Obstacle | **BLOCKED** (0.72) | **CLEAR** (0.00) | **CLEAR** (0.00) | **PARTIALLY_BLOCKED** |
| **Scenario D** | Single NEAR Right Obstacle | **CLEAR** (0.00) | **CLEAR** (0.00) | **BLOCKED** (0.81) | **PARTIALLY_BLOCKED** |
| **Scenario E** | Multiple Obstacles (L, C, R) | **BLOCKED** (0.77) | **BLOCKED** (0.57) | **CLEAR** (0.01) | **PARTIALLY_BLOCKED** |
| **Scenario F** | REMEMBERED Hazard (Decayed) | **CLEAR** (0.00) | **UNCERTAIN** / **BLOCKED** | **CLEAR** (0.00) | **PARTIALLY_BLOCKED** |
| **Scenario G** | EXPIRED Hazard | **CLEAR** (0.00) | **CLEAR** (0.00) | **CLEAR** (0.00) | **CLEAR** |
| **Scenario H** | Wide Object Crossing (C + R) | **CLEAR** (0.00) | **BLOCKED** (0.59) | **BLOCKED** (0.80) | **PARTIALLY_BLOCKED** |

---

## 4. Key Engineering Conclusions

1. **Near-Zero Latency Overhead**: Module 08 adds only **0.74 ms** per frame in the live pipeline, operating with an average per-hazard processing time of **~0.11 ms**.
2. **Absence of Detection Principle**: Unobstructed scenes return `CLEAR` with explicit reasoning noting unverified open space (`"no detected obstacles; unverified open space"`).
3. **Strict Separation of Concerns**: Module 08 produces purely regional traversability assessments (`FreeSpaceAnalysisResult`), leaving vocal command generation (`LEFT`, `RIGHT`, `FORWARD`, `STOP`) exclusively to Module 09.
