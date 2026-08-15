# Monocular Distance Estimation Performance Report — Module 06

**Project**: VisionGuide AI  
**Date**: 2026-08-09  
**Hardware Environment**: Windows 11 Laptop (CPU-Only, 4 Physical Cores / 8 Logical Processors, 16 GB RAM)  
**Distance Subsystem**: Monocular Pinhole Geometry Bounding-Box Estimator (`monocular_bbox`)  
**Integrated 5-Stage Pipeline**: Camera ➔ YOLOv8m ➔ BoT-SORT ➔ PHMU ➔ Distance Estimation  

---

## 1. Measured Empirical Performance Summary

| Metric | Measured Value | Unit / Scale | Performance Evaluation |
| :--- | :--- | :--- | :--- |
| **Max Objects Tested** | 100 simultaneous hazards | Records in batch | Synthetic stress-test |
| **Total Batch Latency** | `33.372 ms` | 100 objects | Instantaneous execution |
| **Average Per-Object Latency** | **0.33372 ms** | Per object record | Ultra-fast calculation |
| **Integrated Frame Latency** | **0.756 ms** | Per live frame pass | Negligible CPU footprint |
| **YOLOv8m CPU Inference** | `1,635.46 ms` | Per frame pass | Main CPU bottleneck |
| **BoT-SORT Tracking Latency** | `0.35 ms` | Per frame pass | Lightweight tracking |
| **PHMU Memory Latency** | `0.138 ms` | Per frame pass | Lightweight memory |
| **Camera Read Latency** | `3.65 ms` | Per frame pass | Webcam acquisition |
| **Total End-to-End Latency** | **1,640.36 ms** | 5-stage pipeline time | Governed by YOLO CPU pass |
| **End-to-End Pipeline Rate** | **0.61 FPS** | Live integrated stream | Synchronous CPU pass |
| **RAM Memory Consumption** | `547.52 MB` | Resident Set Size | Stable memory footprint |

---

## 2. Distance Classification & Proximity Verification

### Synthetic Calibration Experiment Results

```text
Object: Person (Reference Height: 1.70m, Focal Length: 600px)

1. Large Bounding Box (700px height)  ──► d_est = 1.46m  ──► Category: NEAR (< 1.5m)
2. Medium Bounding Box (350px height) ──► d_est = 2.91m  ──► Category: MEDIUM (1.5m - 3.0m)
3. Small Bounding Box (100px height)  ──► d_est = 10.20m ──► Category: FAR (> 3.0m)
```

---

## 3. Remembered Hazard Distance Policy Verification

```text
1. Active Frame Observation     ──► distance_status: "MEASURED"     (Fresh pinhole geometry calculation)
2. Unobserved (REMEMBERED/OCCLUDED) ──► distance_status: "LAST_OBSERVED" (Preserves last known distance & decayed confidence)
```

---

## 4. Key Engineering Conclusions

1. **Zero Computational Overhead**: Module 06 adds less than **0.76 ms** to total frame latency, operating with an average per-object estimation time of **~0.33 ms**.
2. **Robust Proximity Categorization**: Successfully categorizes hazards into `NEAR`, `MEDIUM`, and `FAR` proximity zones for downstream danger mapping (Phase 6).
3. **Strict Policy Compliance**: Unobserved remembered hazards maintain `"LAST_OBSERVED"` status without fabricating synthetic current measurements.
