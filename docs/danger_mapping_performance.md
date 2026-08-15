# Context-Aware Danger Mapping Performance Report — Module 07

**Project**: VisionGuide AI  
**Date**: 2026-08-09  
**Hardware Environment**: Windows 11 Laptop (CPU-Only, 4 Physical Cores / 8 Logical Processors, 16 GB RAM)  
**Risk Assessment Model**: Multi-Factor Weighted Danger Scoring Model (`ContextAwareDangerMapper`)  
**Integrated 6-Stage Pipeline**: Camera ➔ YOLOv8m ➔ BoT-SORT ➔ PHMU ➔ Distance ➔ Danger Mapping  

---

## 1. Measured Empirical Performance Summary

| Metric | Measured Value | Unit / Scale | Performance Evaluation |
| :--- | :--- | :--- | :--- |
| **Max Hazards Stress-Tested** | 100 simultaneous hazards | Records in batch | Micro-benchmark stress-test |
| **Total Batch Assessment & Ranking** | `20.562 ms` | 100 hazard records | Ultra-fast ranking |
| **Average Per-Hazard Mapping**| **0.20562 ms** | Per hazard entry | Instantaneous calculation |
| **Integrated Frame Latency** | **0.490 ms** | Per live frame pass | Negligible CPU footprint |
| **YOLOv8m CPU Inference** | `1,385.78 ms` | Per frame pass | Main CPU bottleneck |
| **Camera Read Latency** | `3.30 ms` | Per frame pass | Webcam acquisition |
| **BoT-SORT Tracking Latency** | `0.26 ms` | Per frame pass | Lightweight tracking |
| **PHMU Hazard Memory Latency** | `0.139 ms` | Per frame pass | Lightweight memory |
| **Distance Estimation Latency**| `0.697 ms` | Per frame pass | Lightweight distance |
| **Total End-to-End Latency** | **1,390.67 ms** | 6-stage pipeline time | Governed by YOLO CPU pass |
| **End-to-End Pipeline Rate** | **0.72 FPS** | Live integrated stream | Synchronous CPU pass |
| **RAM Memory Consumption** | `538.47 MB` | Resident Set Size | Stable memory footprint |

---

## 2. Multi-Factor Danger Scoring Model

$$\text{Danger Score} = \text{clamp}\left(0.25 F_{\text{obj}} + 0.30 F_{\text{dist}} + 0.25 F_{\text{pos}} + 0.05 F_{\text{mot}} + 0.10 F_{\text{pers}} + 0.05 F_{\text{mem}}, 0.0, 1.0\right)$$

### Configured Danger Level Thresholds
* $\text{Danger Score} \ge 0.85 \implies \mathbf{CRITICAL}$
* $0.70 \le \text{Danger Score} < 0.85 \implies \mathbf{HIGH}$
* $0.55 \le \text{Danger Score} < 0.70 \implies \mathbf{MODERATE}$
* $\text{Danger Score} < 0.55 \implies \mathbf{LOW}$

---

## 3. Core Context-Aware Experiment Verification

| Scenario | Object | Distance | Position Zone | Memory State | Danger Score | Danger Level | Navigation Relevance |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Scenario A** | Chair | FAR | LEFT | ACTIVE | `0.51` | **LOW** | False |
| **Scenario B** | Chair | NEAR | CENTER | ACTIVE | `0.88` | **CRITICAL** | True |
| **Scenario C** | Stairs | NEAR | CENTER | ACTIVE | `0.96` | **CRITICAL** | True |
| **Scenario D (Stat)**| Person | NEAR | CENTER | ACTIVE (Stat.) | `0.87` | **CRITICAL** | True |
| **Scenario D (Move)**| Person | NEAR | CENTER | ACTIVE (Move) | `0.92` | **CRITICAL** | True |
| **Scenario E** | Chair | NEAR | CENTER | REMEMBERED | `0.86` | **CRITICAL** | True |

---

## 4. Key Engineering Conclusions

1. **Near-Zero CPU Cost**: Module 07 adds only **0.49 ms** per frame in the live pipeline, operating with an average per-hazard processing time of **~0.21 ms**.
2. **Contextual Intelligence**: Danger levels are dynamically calculated based on proximity, walking corridor position, object hazard factor, and PHMU memory confidence rather than crude class-only lookup.
3. **Strict Separation of Concerns**: Module 07 produces purely risk assessments and rankings (`DangerAssessment`), leaving decision command generation (`LEFT`, `RIGHT`, `FORWARD`, `STOP`) exclusively to Module 09.
