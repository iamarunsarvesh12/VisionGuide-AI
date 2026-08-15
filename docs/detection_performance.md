# YOLOv8m Detection Performance Report — Module 02 & 03

**Project**: VisionGuide AI  
**Date**: 2026-08-09  
**Hardware Environment**: Windows 11 Laptop (CPU-Only, 4 Physical Cores / 8 Logical Processors, 16 GB RAM)  
**Model Architecture**: YOLOv8m (Medium variant, 25.9 Million Parameters)  
**Execution Backend**: `PyTorch 2.10.0+cpu` (`torch.device('cpu')`)  

---

## 1. Measured Empirical Performance Summary

| Metric | Measured CPU Value | Benchmark Specification | Notes / Impact |
| :--- | :--- | :--- | :--- |
| **Model Weights Used** | `yolov8m.pt` (COCO Pretrained) | `YOLOv8m` | Designated baseline model |
| **Cold Start Load Latency** | `14,165.49 ms` | Warmup / Initial load | PyTorch disk weight loading |
| **Warm Load Latency** | `147.42 ms` | In-memory init | Subsequent instantiation |
| **Input Frame Resolution** | `640x480` | `640x480` BGR Frame | Standard video input |
| **Average Inference Latency** | **712.81 ms** | Per frame | CPU matrix computation |
| **Min Inference Latency** | `645.95 ms` | Best case | Single frame forward pass |
| **Max Inference Latency** | `825.98 ms` | Worst case | Multi-object frame pass |
| **Average Inference Throughput** | **1.40 FPS** | Real-time stream | CPU bottleneck |
| **RAM Memory Footprint** | `496.26 MB` | `< 1000 MB` | Stable memory footprint |

---

## 2. Technical Findings & Architectural Observations

1. **CPU Computation Bottleneck**: On the current CPU-only hardware environment, YOLOv8m requires an average of **712.81 ms** per 640x480 frame, yielding an inference throughput of **1.40 FPS**.
2. **Comparison to Camera Capture**: While Module 01 (Camera Input) captures raw video at **29.68 FPS** (~33.7 ms/frame), running YOLOv8m synchronously on every camera frame would cause heavy frame buffer backlog.
3. **Decoupling Strategy**: Module 01 (Camera Input) and Module 03 (Object Detection) must be run asynchronously using non-blocking queues or background inference worker threads in Module 15 (System Controller).
4. **Model Architecture Status**: Per project rules, YOLOv8m is strictly maintained as the primary designated model. No lightweight model substitutions (`YOLOv8n` / `YOLOv8s`) were made.

---

## 3. Recommended Pipeline Optimization Strategies

To maintain high visual fluidity while leveraging YOLOv8m on CPU:
* **Asynchronous Inference Worker**: Run detection on every N-th frame (e.g. 1 inference every ~700 ms), allowing Module 04 (BoT-SORT Tracker) and Module 05 (PHMU) to interpolate object positions across high-FPS camera frames using persistent state memory.
