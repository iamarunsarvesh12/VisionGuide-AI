# Persistent Hazard Memory Unit (PHMU) Performance Report — Module 05

**Project**: VisionGuide AI  
**Date**: 2026-08-09  
**Hardware Environment**: Windows 11 Laptop (CPU-Only, 4 Physical Cores / 8 Logical Processors, 16 GB RAM)  
**Core Innovation Subsystem**: Persistent Hazard Memory Unit (PHMU)  
**Integrated Pipeline**: Camera Input ➔ YOLOv8m Detection ➔ BoT-SORT Tracking ➔ PHMU Hazard Memory  

---

## 1. Measured Empirical Performance Summary

| Metric | Measured Value | Unit / Scale | Performance Evaluation |
| :--- | :--- | :--- | :--- |
| **Max Active Hazards Benchmark**| 100 simultaneous hazards | Records in memory | Stress-test baseline |
| **Memory Creation Latency** | `0.2605 ms` | Per hazard record | Instantaneous instantiation |
| **Memory Lookup Latency** | `0.00077 ms` (0.77 µs) | Per lookup call | O(1) Hash Map access |
| **Memory Expiration Latency** | `0.028 ms` | Per purge cycle | Fast expiration cleanup |
| **Average PHMU Update Latency**| **0.346 ms** | Per live frame pass | Negligible CPU footprint |
| **BoT-SORT Tracking Latency** | `0.59 ms` | Per frame pass | Lightweight tracking |
| **YOLOv8m CPU Inference** | `1,455.50 ms` | Per frame pass | Main CPU bottleneck |
| **Camera Read Latency** | `8.04 ms` | Per frame pass | Nominal webcam acquisition |
| **Total End-to-End Latency** | **1,464.48 ms** | Total per-frame time | Governed by YOLO CPU pass |
| **End-to-End Pipeline Rate** | **0.68 FPS** | Live integrated stream | Synchronous CPU pass |
| **RAM Memory Consumption** | `542.36 MB` | Resident Set Size | Stable memory footprint |

---

## 2. Temporal Persistence Verification

### Synthetic Occlusion Experiment
```text
Frame 1: Person ID 1 detected        ──► Memory State: ACTIVE (obs_count=1)
Frame 2: Person ID 1 detected        ──► Memory State: ACTIVE (obs_count=2)
Frame 3: Person ID 1 missing (0.5s)  ──► Memory State: OCCLUDED (Memory Retained)
Frame 4: Person ID 1 missing (1.5s)  ──► Memory State: REMEMBERED (Memory Retained)
Frame 5: Person ID 1 reappears       ──► Memory State: RECOVERED (obs_count=3)
Frame 6: Person ID 1 detected        ──► Memory State: ACTIVE (obs_count=4)
```

### Timeout Expiration Experiment
```text
Absence Duration: 0.0s to 1.0s   ──► Memory State: OCCLUDED (C_mem decayed via exp(-0.2 * t))
Absence Duration: 1.0s to 3.0s   ──► Memory State: REMEMBERED (C_mem > 0.1)
Absence Duration: > 3.0s (Timeout) ──► Memory State: EXPIRED (Purged from active memory)
```

---

## 3. Mathematical Metrics Implemented

### Memory Confidence Decay
$$C_{\text{mem}}(t) = \text{clamp}\left(C_{\text{det}} \cdot e^{-0.2 \cdot \Delta t}, 0.0, 1.0\right)$$

### Persistence Score
$$P_{\text{score}} = \text{clamp}\left(0.4 \cdot C_{\text{mem}} + 0.4 \cdot \min\left(1.0, \frac{\text{obs\_count}}{10}\right) + 0.2 \cdot \min\left(1.0, \frac{\text{track\_age}}{30}\right), 0.0, 1.0\right)$$

---

## 4. Key Engineering Conclusions

1. **Near-Zero CPU Cost**: PHMU adds less than **0.35 ms** to total frame latency, proving that temporal hazard memory retention can be executed with virtually zero computational overhead.
2. **Context Retention**: When visual perception drops an object due to occlusion or detector failure, PHMU retains the hazard's last-known image coordinates and confidence for up to **3.0 seconds**, bridging perception gaps.
