# VisionGuide AI — Offline Multimodal Persistent-Hazard-Memory Navigation System

> **An offline, real-time AI-powered navigation assistance prototype designed for visually impaired individuals, powered by local computer vision, persistent spatial hazard tracking, and low-latency audio guidance.**

---

## 📌 Executive Summary

**VisionGuide AI** is an intelligent, privacy-first computer vision navigation system built to assist visually impaired users in navigating complex physical environments safely. Unlike cloud-dependent solutions, VisionGuide AI runs **100% offline** on consumer hardware (Laptop CPU + Webcam + Windows Audio / Bluetooth Headphones), eliminating network latency and privacy risks.

The central breakthrough of VisionGuide AI is the **Persistent Hazard Memory Unit (PHMU)** — the proposed core innovation of the system — which maintains temporal memory of previously detected obstacles even when temporarily occluded, blurred, or outside the immediate camera field-of-view.

---

## 🎯 Problem Statement

Visually impaired individuals face daily navigation challenges due to:
1. **Dynamic and Static Obstacles**: Sudden obstacles (doors, chairs, pedestrians, stairs) requiring immediate reaction.
2. **Occlusion Loss**: Standard real-time object detectors "forget" hazards as soon as the user turns their head or an obstacle is momentarily blocked.
3. **Cloud Latency & Privacy Risks**: Relying on internet connectivity for cloud-based AI introduces dangerous latency and exposes camera feeds to privacy breaches.
4. **Complex Setup Requirements**: Many assistive technologies require specialized smart glasses or expensive depth sensors.

VisionGuide AI addresses all four challenges with an offline, CPU-optimized processing pipeline that runs on standard laptop webcams.

---

## 💡 Core Innovation — Persistent Hazard Memory Unit (PHMU)

> *PHMU is the proposed core innovation of the system.*

In traditional frame-by-frame object detection systems, if a hazard (such as a table or stairs) is partially occluded by a passing person or falls slightly out of frame due to user motion, the system loses track of it. 

The **Persistent Hazard Memory Unit (PHMU)** solves this by maintaining a spatial-temporal hazard state table:
- **Spatial State Retention**: Retains hazard positions, velocities, bounding geometry, and danger levels across consecutive frames.
- **Occlusion Resilience**: Applies exponential decay to confidence scores during temporary detection drops, keeping hazards active for up to 3.0 seconds.
- **False Positive Filtering**: Requires multi-frame verification before declaring new hazards while instantly persisting critical threats (e.g., stairs, approaching vehicles).

---

## 🔄 Core Processing Pipeline

```text
               Laptop Webcam (USB / Built-in)
                             │
                             ▼
              [Module 01: Camera Input]
                             │
                             ▼
         [Module 02/03: YOLOv8m Object Detection]
                             │
                             ▼
         [Module 04: BoT-SORT Multi-Object Tracking]
                             │
                             ▼
     [Module 05: Persistent Hazard Memory Unit (PHMU)] ★ Core Innovation
                             │
                             ▼
        [Module 06: Monocular Distance Estimation]
                             │
                             ▼
        [Module 07: Context-Aware Danger Mapping]
                             │
                             ▼
         [Module 08: Image-Space Free-Space Analysis]
                             │
                             ▼
        [Module 09: Context-Aware Decision Engine]
                             │
                             ▼
         [Module 10: Offline Audio Guidance Engine]
                             │
                             ▼
       Laptop Speakers / Connected Bluetooth Headphones
```

---

## 🧩 Module Breakdown

VisionGuide AI is structured into 10 decoupled, unit-tested core modules:

| Module | Name | Function / Responsibility |
| :--- | :--- | :--- |
| **Module 01** | Camera Input | Captures live video stream (640x480 @ 30 FPS) with auto-reconnection and thread safety. |
| **Module 02/03** | YOLOv8m Object Detection | Detects 80 COCO classes + specialized indoor hazards (stairs, doors, chairs, tables). |
| **Module 04** | BoT-SORT Multi-Object Tracking | Tracks obstacles across frames, computing motion vectors and ID stability. |
| **Module 05** | Persistent Hazard Memory Unit | Maintains temporal memory of hazards across temporary occlusions and motion drops. |
| **Module 06** | Monocular Distance Estimation | Computes pinhole geometry-based distance estimation (in meters) from bounding box heights. |
| **Module 07** | Context-Aware Danger Mapping | Calculates composite danger scores (0.0 to 1.0) based on class priority, distance, and vector trajectory. |
| **Module 08** | Image-Space Free-Space Analysis | Divides view into LEFT, CENTER, RIGHT sectors to evaluate clear navigation paths. |
| **Module 09** | Context-Aware Decision Engine | Determines optimal navigation direction (`FORWARD`, `LEFT`, `RIGHT`, `STOP`) with rate-limiting. |
| **Module 10** | Offline Audio Guidance | Generates low-latency voice prompts via `pyttsx3` with emergency priority overrides. |
| **Integration** | System Integration | Orchestrates multithreaded execution across all 10 modules under a single main loop. |

---

## 💻 Hardware Setup for Current Working Prototype

The current working prototype uses standard off-the-shelf hardware:

- **Camera**: Standard Laptop Built-in Webcam (or USB Webcam)
- **Compute**: Laptop CPU (Intel Core i5/i7/i9 or AMD Ryzen 5/7/9; no discrete GPU required)
- **Audio Output**: Laptop Speaker or standard Bluetooth Headphones
- **OS**: Windows 10 / 11 (Linux and macOS supported)

> *Note: Smart glasses are NOT required for the current working prototype.*

---

## 🛠️ Tech Stack & Dependencies

- **Language**: Python 3.8+
- **Deep Learning / Vision**: PyTorch (CPU), Ultralytics (YOLOv8m), OpenCV (`opencv-python`)
- **Scientific Computing**: NumPy, SciPy
- **Tracking & Geometry**: BoT-SORT algorithm, Pinhole Camera Geometry
- **Audio & TTS**: `pyttsx3` (Offline Text-to-Speech), `sounddevice`, `SpeechRecognition`
- **System & Monitoring**: `psutil`, `pyyaml`

---

## 🚀 Quick Start Guide

### Option 1: One-Command Python Startup (Recommended)

Simply run the unified launcher script:

```powershell
python run_visionguide.py
```

The automated launcher handles all initialization steps automatically:
1. Verifies Python version compatibility.
2. Creates `.venv` virtual environment if missing.
3. Automatically installs required dependencies from `requirements.txt`.
4. Checks and automatically downloads `yolov8m.pt` model weights if missing.
5. Verifies webcam accessibility and audio devices.
6. Starts the VisionGuide AI real-time navigation interface.

### Option 2: Windows Double-Click Launcher

On Windows systems, double-click:
```text
run_visionguide.bat
```

---

## 🧪 Testing & Validation

VisionGuide AI includes an extensive suite of **183 automated unit tests** and benchmark scripts.

To run the complete test suite:

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests
```

### Benchmark & Validation Coverage:
- **System Integration**: End-to-end pipeline latency and frame delivery tests.
- **PHMU Persistence**: Temporal hazard retention across simulated occlusions.
- **Free-Space Analysis**: Sector availability and occupancy mapping.
- **Audio Priority**: Emergency `STOP` command interruption validation.
- **Resource Usage**: CPU and RAM performance profiling.

---

## 📁 Repository Structure

```text
VisionGuide-AI/
├── README.md                           # Comprehensive project overview & documentation
├── HOW_TO_RUN.md                      # Detailed user guide & troubleshooting
├── ENVIRONMENT_REPORT.md              # System environment diagnostics & verification
├── QUICK_START.md                     # Quick reference installation guide
├── requirements.txt                   # Production Python dependencies
├── .gitignore                         # Configured Git exclusions (venv, weights, logs)
├── .env.example                       # Environment configuration template
├── run_visionguide.py                 # Automated Python launcher script
├── run_visionguide.bat                # Windows double-click batch launcher
│
├── config/                            # YAML Configuration files
│   ├── config.yaml                    # System pipeline configuration
│   ├── classes.yaml                   # Hazard class mapping & weights
│   ├── audio.yaml                     # TTS & audio priority settings
│   └── calibration.yaml               # Camera calibration & distance focal length
│
├── modules/                           # Core 10-Module Implementation
│   ├── camera_input/                  # Module 01: Video capture & frame preprocessing
│   ├── object_detection/              # Module 02/03: YOLOv8m detection engine
│   ├── object_tracking/               # Module 04: BoT-SORT multi-object tracker
│   ├── hazard_memory/                 # Module 05: PHMU persistent memory unit
│   ├── distance_estimation/           # Module 06: Monocular geometry distance estimator
│   ├── danger_mapping/                # Module 07: Context-aware danger calculator
│   ├── free_space/                    # Module 08: Sector free-space analyzer
│   ├── decision_engine/               # Module 09: Navigation decision engine
│   ├── audio_guidance/                # Module 10: Offline TTS audio output
│   ├── system_integration/            # Pipeline orchestrator & main control loop
│   └── system_monitor/                # Real-time resource & latency monitor
│
├── tests/                             # Automated Test & Benchmark Suite (183 tests)
│   ├── test_camera.py
│   ├── test_detection.py
│   ├── test_tracking.py
│   ├── test_hazard_memory.py
│   ├── test_distance_estimation.py
│   ├── test_danger_mapping.py
│   ├── test_free_space.py
│   ├── test_decision_engine.py
│   ├── test_audio_guidance.py
│   ├── test_system_integration.py
│   ├── test_safety_failures.py
│   ├── benchmark_*.py                 # Performance & latency benchmark scripts
│   └── validate_*.py                  # End-to-end validation scripts
│
└── docs/                              # Performance reports & documentation
    ├── camera_performance.md
    ├── detection_performance.md
    ├── tracking_performance.md
    ├── phmu_performance.md
    ├── distance_estimation_performance.md
    ├── danger_mapping_performance.md
    ├── free_space_performance.md
    ├── decision_engine_performance.md
    ├── audio_guidance_performance.md
    └── system_integration_performance.md
```

---

## ⚠️ Prototype Limitations & Future Scope

Current prototype limitations:
- **Monocular Distance Approximation**: Uses pinhole camera geometry assuming standard object dimensions; accuracy varies on non-standard obstacle geometry.
- **Lighting Sensitivity**: Performance relies on ambient room/outdoor illumination for optimal webcam frame capture.
- **Future Enhancements**: Integration with smart glasses hardware, stereo vision/depth cameras, and haptic feedback integration.

---

## 📜 License & Citation

Developed for research, patent evaluation, and accessibility technological advancement.

```text
VisionGuide AI — Offline Multimodal Persistent-Hazard-Memory Navigation System
Repository: https://github.com/iamarunsarvesh12/VisionGuide-AI
```
