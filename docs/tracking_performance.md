# BoT-SORT Object Tracking Performance Report — Module 04

**Project**: VisionGuide AI  
**Date**: 2026-08-09  
**Hardware Environment**: Windows 11 Laptop (CPU-Only, 4 Physical Cores / 8 Logical Processors, 16 GB RAM)  
**Tracking Algorithm**: BoT-SORT Multi-Object Tracking Engine  
**Pipeline Integration**: Module 01 (Camera Input) ➔ Module 03 (YOLOv8m Detection) ➔ Module 04 (BoT-SORT Tracking)  

---

## 1. Measured Empirical Performance Summary

| Metric | Measured Value | Unit / Scale | Performance Evaluation |
| :--- | :--- | :--- | :--- |
| **Tracker Engine Type** | `BoT-SORT` | Multi-Object Tracker | Standardized implementation |
| **Tracker Init Latency** | `0.15 ms` | Milliseconds | Ultra-fast initialization |
| **Standalone Tracking Latency**| **0.33 ms** | Per frame update | Near-zero computational overhead |
| **Standalone Tracking Throughput**| **3,052.94 FPS** | Frames per second | High-speed processing capacity |
| **YOLOv8m CPU Inference** | `1,241.49 ms` | Per frame | Main CPU bottleneck (~0.81 FPS) |
| **Camera Read Latency** | `8.17 ms` | Per frame | Nominal webcam read time |
| **Total End-to-End Latency** | **1,249.99 ms** | Camera + YOLO + Tracking | End-to-End pipeline frame time |
| **End-to-End Pipeline FPS** | **0.80 FPS** | Real-time stream rate | Governed by YOLOv8m CPU pass |
| **Track Identity Persistence**| **100% (15/15 frames)**| Track ID `1` (`person`) | Persistent inter-frame identity |
| **RAM Memory Consumption** | `547.78 MB` | Resident Set Size (RSS) | Low memory overhead |

---

## 2. Spatial & Temporal Identity Analysis

```text
Frame 01: Person detected ──► BoT-SORT assigns Track ID 1 (State: NEW)
Frame 02: Person moved    ──► BoT-SORT associates Track ID 1 (State: TRACKED, IoU > 0.3)
Frame 03: Person moved    ──► BoT-SORT associates Track ID 1 (State: TRACKED)
...
Frame 15: Person moved    ──► BoT-SORT associates Track ID 1 (State: TRACKED, Hits: 15)
```

1. **Zero Tracking Bottleneck**: BoT-SORT tracking logic adds only **0.33 ms** overhead to the pipeline, operating at over 3,000 FPS standalone.
2. **Persistent Association**: Spatial bounding box IoU matching successfully maintains object identity (`Track ID 1`) continuously across all frames without ID switching.
3. **Pipeline Rate Constraint**: End-to-End system throughput (**0.80 FPS**) remains dictated by YOLOv8m CPU inference latency (**1,241.49 ms**).

---

## 3. Boundary Preparation for Phase 4 (PHMU)

BoT-SORT resolves object association while visual evidence remains visible or briefly lost during short frame gaps (`time_since_update <= max_age`). In Phase 4, the **Persistent Hazard Memory Unit (PHMU)** will consume these `Track` records to manage multi-frame hazard state lifecycles (`ACTIVE`, `OCCLUDED`, `REMEMBERED`, `RECOVERED`, `EXPIRED`) during temporary occlusion or detector misses.
