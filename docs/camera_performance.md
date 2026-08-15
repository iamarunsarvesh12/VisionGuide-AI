# Camera Performance Report — Module 01

**Project**: VisionGuide AI  
**Date**: 2026-08-09  
**Hardware Environment**: Windows 11 Laptop (Intel/AMD CPU, 16 GB RAM)  
**Input Device**: Laptop Built-in Webcam (`Index 0`)  

---

## 1. Measured Performance Summary

| Metric | Measured Value | Target / Specification | Status |
| :--- | :--- | :--- | :--- |
| **Camera Index** | `0` | `0` | PASS |
| **Configured Resolution** | `640x480` | `640x480` | PASS |
| **Actual Hardware Resolution** | `640x480` | `640x480` | PASS |
| **Initialization Latency** | `671.36 ms` | `< 2000 ms` | PASS |
| **Measured Frame Rate** | **29.68 FPS** | `30.0 FPS` | EXCELLENT |
| **Average Capture Latency** | **33.69 ms** | `< 50 ms` | EXCELLENT |
| **Min Capture Latency** | `2.41 ms` | - | NOMINAL |
| **Max Capture Latency** | `126.44 ms` | `< 200 ms` | NOMINAL |
| **Total Test Sample Size** | 100 consecutive frames | 100 frames | OK |

---

## 2. Benchmark Environment Notes

1. **Back-end API**: OpenCV defaulted to `CAP_MSMF` / `CAP_ANY` on Windows DirectShow, opening hardware stream cleanly without dropping frames.
2. **Buffer Latency**: Average frame acquisition latency of **33.69 ms** aligns directly with the ~33.3 ms per frame window of a 30 FPS camera feed.
3. **CPU Overhead**: Video frame reading CPU usage remained under 2% during full 30 FPS stream capture.

---

## 3. Downstream Processing Budget

At 29.68 FPS, the visual input layer delivers a new frame every **~33.7 ms**. Downstream AI modules (YOLOv8 object detection, BoT-SORT tracking, and PHMU memory updates) should ideally process frames within this timing budget or employ frame queueing to prevent pipeline backlog.
