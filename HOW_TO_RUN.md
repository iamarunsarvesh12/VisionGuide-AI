# VisionGuide AI — Complete System Setup & Execution Guide

Welcome to **VisionGuide AI**, an offline, multimodal, AI-based assistive navigation prototype designed to assist visually impaired individuals in navigating indoor and outdoor environments safely.

This document provides a comprehensive, step-by-step guide explaining how to install dependencies, verify hardware, configure settings, run unit tests, execute performance benchmarks, view real-time pipeline visualizers, connect Bluetooth audio, and launch the complete working prototype on a Windows environment.

---

# VisionGuide AI — One Command Run

## First Time

Open PowerShell inside the project folder and run:

```powershell
python run_visionguide.py
```

That's it.

The launcher automatically:
- Creates the virtual environment (`.venv`)
- Switches execution to `.venv` automatically
- Installs required dependencies from `requirements.txt`
- Prepares YOLOv8m model weights (`yolov8m.pt`)
- Checks webcam hardware and audio output
- Initializes all 10 modules and boots VisionGuide AI

## Every Next Time

Run:

```powershell
python run_visionguide.py
```

OR double-click: `run_visionguide.bat`

> For full beginner instructions, see [QUICK_START.md](file:///c:/Users/Admin/Documents/VisionGuide%20AI/QUICK_START.md).

---

## 1. Project Overview

**VisionGuide AI** processes visual data from a laptop webcam to identify environmental hazards, estimate object distances, maintain short-term hazard memory when objects are occluded or temporarily out of view, evaluate traversable walking space across three directional zones (LEFT, CENTER, RIGHT), make safety-first navigation decisions with hysteresis, and communicate real-time audio guidance to the user.

### Core Architectural Pipeline

```text
Laptop Webcam
     │
     ▼
YOLOv8m Object Detection (PyTorch CPU)
     │
     ▼
BoT-SORT Multi-Object Tracking
     │
     ▼
Persistent Hazard Memory Unit (PHMU)
     │
     ▼
Monocular Geometry Distance Estimation
     │
     ▼
Context-Aware Danger Mapping
     │
     ▼
Free-Space Region Analysis (LEFT / CENTER / RIGHT)
     │
     ▼
Safety-First Decision Engine (with Hysteresis & Priority Override)
     │
     ▼
Threaded Offline Audio Guidance Engine (Windows SAPI5 / pyttsx3)
     │
     ▼
Bluetooth Audio Headphones / Speakers
```

### Key System Characteristics

* **Visual Input**: Standard integrated or external laptop webcam (`cv2.VideoCapture(0)`).
* **Local CPU AI Processing**: PyTorch-based neural detection and spatial tracking executed completely on local laptop CPU hardware.
* **100% Offline Execution**: Zero dependency on internet connections, cloud vision APIs, or external server infrastructures.
* **Audio Guidance**: Real-time spoken navigation instructions dispatched via local Windows SAPI5 offline Text-to-Speech (TTS).
* **Bluetooth Compatible**: Direct audio output through Windows default audio device (e.g., Bluetooth earbuds or portable speakers).
* **Target Hardware Context**: Designed for local Windows laptop evaluation (wearable smart-glasses and mobile form factors are future architectural targets and are wrapped cleanly behind modular hardware interfaces).

---

## 2. System Requirements

The current prototype has been verified and tested under the following environment:

| Requirement / Component | Verified Specification |
| :--- | :--- |
| **Operating System** | Windows 11 64-bit (Home / Pro / Enterprise) |
| **Python Version** | Python `3.14.6` 64-bit (Compatible with Python 3.10+) |
| **PyTorch Architecture** | PyTorch `2.10.0+cpu` (CPU execution mode) |
| **OpenCV Computer Vision** | OpenCV `4.10.0` (`opencv-python`) |
| **Object Detection Engine** | Ultralytics `8.4.19` (YOLOv8m pretrained COCO model) |
| **Numerical & Scientific** | NumPy `2.3.5`, SciPy `1.16.3` |
| **Configuration Parser** | PyYAML `6.0.3` |
| **Text-to-Speech (TTS)** | `pyttsx3` with native Windows SAPI5 TTS engine (`TTS_MS_EN-US_DAVID_11.0`, `TTS_MS_EN-US_ZIRA_11.0`) |
| **Audio I/O & Monitoring** | `sounddevice` `0.5.5`, `SpeechRecognition` `3.15.1`, `psutil` `7.0.0` |
| **Video Device** | Integrated Laptop Webcam (Camera index `0`, `640x480` @ 30 FPS target) |
| **Audio Endpoint** | Bluetooth Headphones or Laptop Built-in Speakers (Windows Default Playback Endpoint) |

---

## 3. Project Directory Structure

Below is the project directory structure representing all implemented modules, configuration files, launchers, test suites, documentation, and log files:

```text
VisionGuide AI/
│
├── config/                        # YAML System Configuration Files
│   ├── config.yaml                # Master system pipeline & module hyperparameter settings
│   ├── classes.yaml               # Object class mapping & hazard classification definitions
│   ├── audio.yaml                 # Offline audio guidance & TTS engine settings
│   └── calibration.yaml           # Camera pinhole geometry & focal length calibration settings
│
├── modules/                       # 10 Core Architectural Pipeline Modules (+ Calibration & Monitor)
│   ├── camera_input/              # Module 01: OpenCV Laptop Webcam Interface & Frame Capture
│   ├── object_detection/          # Module 02: PyTorch YOLOv8m Neural Object Detection Engine
│   ├── object_tracking/           # Module 03: BoT-SORT Multi-Object Tracking & Spatial History
│   ├── hazard_memory/             # Module 04: Persistent Hazard Memory Unit (PHMU)
│   ├── distance_estimation/       # Module 05: Monocular Bounding-Box Geometry Distance Estimator
│   ├── danger_mapping/            # Module 06: Multi-Factor Context-Aware Danger Assessment
│   ├── free_space/                # Module 07: 3-Region Traversable Free-Space & Occupancy Analyzer
│   ├── decision_engine/           # Module 08: Context-Aware Navigation Decision Engine with Hysteresis
│   ├── audio_guidance/            # Module 09/10: Offline Threaded SAPI5 Audio Guidance Engine
│   ├── system_integration/        # System Integration Pipeline Wrapper & System State Models
│   ├── system_monitor/            # CPU, Memory, and Telemetry Resource Monitor
│   └── calibration/               # Camera Pinhole Focal Length Calibration Utilities
│
├── tests/                         # Comprehensive Unit, Integration, Benchmark, and Viewer Suite
│   ├── test_*.py                  # Automated unit and integration test scripts (183 tests)
│   ├── benchmark_*.py             # Empirical per-module and end-to-end performance benchmarks
│   ├── view_*.py                  # Interactive OpenCV GUI visualizer scripts
│   └── validate_*.py              # Scenario reasoning, distance, & PHMU validation scripts
│
├── docs/                          # Empirical Performance Benchmark & Validation Documentation
│   ├── system_integration_performance.md  # Unified system end-to-end latency & FPS report
│   ├── phase11_validation_report.md       # Final validation and stability verification report
│   └── *_performance.md           # Per-module empirical performance reports
│
├── logs/                          # System Runtime Log Files (Per-module & pipeline logs)
│   ├── camera_input.log
│   ├── object_detection.log
│   ├── object_tracking.log
│   ├── hazard_memory.log
│   ├── distance_estimation.log
│   ├── danger_mapping.log
│   ├── free_space.log
│   ├── decision_engine.log
│   ├── audio_guidance.log
│   └── system_integration.log
│
├── experiments/                   # Standalone Experimental & Validation Artifacts
├── ENVIRONMENT_REPORT.md          # Hardware & Software Environment Inspection Summary
├── yolov8m.pt                     # PyTorch Pretrained YOLOv8m COCO Model Weights (52.1 MB)
├── run_visionguide.py             # Primary Master System Executable Launcher
└── HOW_TO_RUN.md                  # Complete System Setup, Configuration, & Execution Guide
```

### Purpose of Major Directories

* **`config/`**: Contains all externalized configuration YAML files, allowing tuning of detection thresholds, danger weighting, PHMU retention times, camera settings, and audio TTS parameters without modifying python code.
* **`modules/`**: Contains the decoupled 10 core pipeline modules. Each module maintains strict isolation behind clean interface abstractions (`interface.py`, `models.py`).
* **`tests/`**: Contains automated unit tests (`test_*.py`), performance benchmark scripts (`benchmark_*.py`), interactive visual GUI inspection windows (`view_*.py`), and validation suites (`validate_*.py`).
* **`docs/`**: Stores empirical performance metrics, resource benchmarks, and test coverage verification reports generated during system validation phases.
* **`logs/`**: Automatically captures runtime logs for each subsystem and the integrated system wrapper.

---

## 4. Installation Guide

Follow these steps to set up the VisionGuide AI execution environment on Windows PowerShell.

### Step 1 — Open the Project Directory

Open Windows PowerShell or Command Prompt and navigate to the project directory:

```powershell
cd "C:\Users\Admin\Documents\VisionGuide AI"
```

> **Note**: Replace `"C:\Users\Admin\Documents\VisionGuide AI"` with your actual local repository path if different.

### Step 2 — Verify Python and Pip Installation

Ensure Python 3.10+ and `pip` are accessible from your terminal:

```powershell
python --version
python -m pip --version
```

Expected output should display Python version `3.10` or higher (e.g., `Python 3.14.6`).

### Step 3 — Create and Activate Virtual Environment

Create a dedicated Python virtual environment `.venv`:

```powershell
python -m venv .venv
```

Activate the virtual environment:

```powershell
.venv\Scripts\Activate.ps1
```

> **Troubleshooting PowerShell Execution Policy**:
> If PowerShell displays a script execution policy restriction error (`...cannot be loaded because running scripts is disabled on this system`), run the following command to temporarily permit script execution in the current session:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
> ```
> Then run `.venv\Scripts\Activate.ps1` again.

### Step 4 — Install Core Dependencies

Install PyTorch CPU edition followed by the required computer vision and audio libraries:

```powershell
# Install PyTorch CPU Build
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install Computer Vision, Audio, and Utility Dependencies
python -m pip install opencv-python ultralytics pyttsx3 sounddevice SpeechRecognition pyyaml psutil scipy
```

### Step 5 — Verify Package Installation

Run the following verification commands to ensure all key packages import cleanly:

```powershell
python -c "import cv2; print('OpenCV OK:', cv2.__version__)"
python -c "import torch; print('PyTorch OK:', torch.__version__)"
python -c "import ultralytics; print('Ultralytics OK:', ultralytics.__version__)"
python -c "import pyttsx3; print('TTS Engine OK')"
python -c "import psutil; print('psutil OK:', psutil.__version__)"
```

Every command should output `OK` with the respective version number without throwing import errors.

---

## 5. Camera Setup

The system relies on OpenCV's `VideoCapture` interface to stream live visual frames from the laptop webcam.

### Interface Details

* **Module Location**: `modules/camera_input/camera.py`
* **Default Camera Index**: `0` (Standard Windows laptop webcam)
* **Default Frame Resolution**: `640 x 480` pixels
* **Target Frame Rate**: `30 FPS`

### Testing the Camera Stream

To test the webcam hardware connection and view the live video feed with basic HUD information, run:

```powershell
python tests/view_camera.py
```

* A window titled **"VisionGuide AI — Camera Input Module Test"** will open displaying your live webcam stream.
* Press **`q`** or **`ESC`** with the window selected to exit cleanly.

### Camera Configuration & Troubleshooting

If camera index `0` does not open or if you are using an external USB camera, update the camera index in `config/config.yaml`:

```yaml
camera:
  index: 0          # Change to 1 or 2 if using an external USB webcam
  width: 640
  height: 480
  target_fps: 30
  cap_api_preference: "CAP_ANY"
```

To run the automated unit test for camera initialization and frame retrieval:

```powershell
python -m unittest tests/test_camera.py
```

---

## 6. YOLOv8m Model Setup

VisionGuide AI uses the **YOLOv8m (Medium)** object detection model powered by PyTorch for identifying obstacles and navigate-relevant objects in real time.

### Model Weights & Path

* **Model File**: `yolov8m.pt` (approx. 52.1 MB) located in the project root directory.
* **Weights Source**: Pretrained COCO dataset weights.
* **Automatic Download**: If `yolov8m.pt` is missing from the project root, the `ultralytics` framework will automatically download it on first run.

### Detection Hyperparameters (`config/config.yaml`)

```yaml
yolo:
  model_path: "yolov8m.pt"
  confidence_threshold: 0.35   # Filters low-confidence noise
  iou_threshold: 0.45          # Non-Maximum Suppression (NMS) threshold
  device: "cpu"                # Force CPU execution for Windows compatibility
```

### Supported Object Classes (`config/classes.yaml`)

The system maps object detections into navigation-critical categories:
`person`, `chair`, `table`, `door`, `stairs`, `glass_door`, `glass_wall`, `cabinet`, `corridor`, `exit`, and `drinking_water_source`.

To verify detection functionality on live webcam video:

```powershell
q
```

---

## 7. Configuration

All operational parameters across the 10 pipeline modules are configured through human-readable YAML files in the `config/` directory.

### Overview of Configuration Files

| Configuration File | Controlled Subsystems | User-Editable Parameters |
| :--- | :--- | :--- |
| **`config/config.yaml`** | Camera, YOLO, PHMU, Distance, Danger, Free-Space, Decision Engine | Confidence thresholds, camera resolution, memory decay rates, region boundaries, decision hysteresis |
| **`config/classes.yaml`** | Object Detection & Class Mapping | Object class IDs and hazard category mappings |
| **`config/audio.yaml`** | Audio Guidance & Offline TTS Engine | TTS voice rate, volume, cooldown intervals, and audio output endpoint |
| **`config/calibration.yaml`** | Monocular Distance Geometry | Camera focal length ($f_x, f_y$), optical center ($c_x, c_y$), and pinhole geometry values |

### Key Parameters in `config/config.yaml`

* **PHMU Retention (`phmu`)**:
  * `memory_timeout_seconds: 3.0`: Retains memory of occluded or temporarily lost hazards for up to 3 seconds.
  * `decay_rate: 0.2`: Rate at which memory confidence decays per second when an object is not detected.
* **Distance Thresholds (`distance_estimation`)**:
  * `near_threshold_m: 1.5`: Objects within 1.5 meters are classified as `NEAR` (High Proximity Danger).
  * `medium_threshold_m: 3.0`: Objects between 1.5m and 3.0m are classified as `MEDIUM`.
* **Decision Engine Hysteresis (`decision_engine`)**:
  * `switching_margin: 0.10`: Prevents rapid command flickering between directions.
  * `min_command_hold_duration_sec: 0.5`: Minimum duration to hold a command before switching.
  * `forward_safe_space_threshold: 0.70`: Free-space threshold required to recommend `FORWARD`.

---

## 8. Running Individual Modules

Each of the 10 system modules can be independently tested, benchmarked, and visually inspected using dedicated scripts in the `tests/` directory.

### Module 01 — Camera Input (`modules/camera_input`)
* **Unit Test**: `python -m unittest tests/test_camera.py`
* **Benchmark**: `python tests/benchmark_camera.py`
* **Visual Inspector**: `python tests/view_camera.py`

### Module 02 — Object Detection (`modules/object_detection`)
* **Unit Test**: `python -m unittest tests/test_detection.py`
* **Benchmark**: `python tests/benchmark_detection.py`
* **Visual Inspector**: `python tests/view_detection.py`

### Module 03 — Object Tracking (`modules/object_tracking`)
* **Unit Test**: `python -m unittest tests/test_tracking.py`
* **Benchmark**: `python tests/benchmark_tracking.py`
* **Visual Inspector**: `python tests/view_tracking.py`

### Module 04 — Persistent Hazard Memory Unit (PHMU) (`modules/hazard_memory`)
* **Unit Test**: `python -m unittest tests/test_hazard_memory.py`
* **Benchmark**: `python tests/benchmark_hazard_memory.py`
* **Visual Inspector**: `python tests/view_hazard_memory.py`

### Module 05 — Distance Estimation (`modules/distance_estimation`)
* **Unit Test**: `python -m unittest tests/test_distance_estimation.py`
* **Benchmark**: `python tests/benchmark_distance_estimation.py`
* **Visual Inspector**: `python tests/view_distance_estimation.py`

### Module 06 — Danger Mapping (`modules/danger_mapping`)
* **Unit Test**: `python -m unittest tests/test_danger_mapping.py`
* **Benchmark**: `python tests/benchmark_danger_mapping.py`
* **Visual Inspector**: `python tests/view_danger_mapping.py`

### Module 07 — Free-Space Analysis (`modules/free_space`)
* **Unit Test**: `python -m unittest tests/test_free_space.py`
* **Benchmark**: `python tests/benchmark_free_space.py`
* **Visual Inspector**: `python tests/view_free_space.py`

### Module 08 — Decision Engine (`modules/decision_engine`)
* **Unit Test**: `python -m unittest tests/test_decision_engine.py`
* **Benchmark**: `python tests/benchmark_decision_engine.py`
* **Visual Inspector**: `python tests/view_decision_engine.py`

### Module 09 & 10 — Offline Audio Guidance (`modules/audio_guidance`)
* **Unit Test**: `python -m unittest tests/test_audio_guidance.py tests/test_audio_integration.py`
* **Benchmark**: `python tests/benchmark_audio_guidance.py`
* **Visual Inspector**: `python tests/view_audio_guidance.py`

---

## 9. Full System Validation

To run the complete automated test suite across all modules and system integration pipelines, use Python's built-in `unittest` runner.

### Running the Complete Unit & Integration Suite (183 Tests)

```powershell
python -m unittest discover -s tests
```

Expected output:
```text
----------------------------------------------------------------------
Ran 183 tests in ~36.4s

OK
```

### Running Specific Validation & Verification Suites

1. **System Integration Test**:
   ```powershell
   python -m unittest tests/test_system_integration.py
   ```
2. **End-to-End Navigation Scenario Validation**:
   ```powershell
   python tests/validate_end_to_end.py
   ```
3. **PHMU Hazard Persistence & Memory Retention Test**:
   ```powershell
   python tests/validate_phmu_persistence.py
   ```
4. **Monocular Distance Estimation Accuracy Test**:
   ```powershell
   python tests/validate_distance_accuracy.py
   ```
5. **Navigation Decision Reasoning Verification**:
   ```powershell
   python tests/validate_navigation_reasoning.py
   ```
6. **Safety Failure & Emergency Override Test**:
   ```powershell
   python -m unittest tests/test_safety_failures.py
   ```

---

# RUN THE COMPLETE VISIONGUIDE AI SYSTEM

To launch the unified VisionGuide AI system prototype with live camera capture, AI detection, hazard tracking, free-space reasoning, HUD visual overlay, and SAPI5 offline audio guidance:

```powershell
python run_visionguide.py
```

### System Execution Sequence

```text
               Execute `python run_visionguide.py`
                               │
                               ▼
            [1/11] Load Configuration Files (YAML)
                               │
                               ▼
             [2/11] Initialize Camera Capture (Index 0)
                               │
                               ▼
            [3/11] Load PyTorch YOLOv8m Model (CPU Mode)
                               │
                               ▼
           [4/11] Initialize BoT-SORT Object Tracker
                               │
                               ▼
        [5/11] Initialize Persistent Hazard Memory Unit (PHMU)
                               │
                               ▼
       [6/11] Initialize Monocular Distance Estimator
                               │
                               ▼
         [7/11] Initialize Context-Aware Danger Mapper
                               │
                               ▼
        [8/11] Initialize Free-Space Region Analyzer
                               │
                               ▼
       [9/11] Initialize Safety Decision Engine (Hysteresis)
                               │
                               ▼
     [10/11] Initialize Offline Audio Guidance (SAPI5 Engine)
                               │
                               ▼
       [11/11] Start Main Processing Loop (Live Video Feed)
                               │
                               ▼
       Continuous Visual HUD Rendering & Offline Spoken Audio
```

### Stopping the Application

* Click on the OpenCV visual window and press **`q`** or **`ESC`**.
* Alternatively, press **`Ctrl+C`** in the PowerShell terminal.
* The system will gracefully release the webcam hardware, terminate background TTS audio threads, print session statistics, and exit cleanly.

---

## 11. Expected Runtime Behavior

When VisionGuide AI is processing frames, the system evaluates scene hazards and emits real-time visual commands and spoken audio cues based on environmental conditions:

| Environmental Scenario | Decision Command | Spoken Audio Output | Reason & Behavior |
| :--- | :--- | :--- | :--- |
| **Path Clear Ahead** | `FORWARD` | *"Forward"* | All regions clear, safe-space confidence > 0.70. |
| **Obstacle in Left Zone** | `RIGHT` | *"Right"* | Left zone occupied; right zone offers clear traversable space. |
| **Obstacle in Right Zone** | `LEFT` | *"Left"* | Right zone occupied; left zone offers clear traversable space. |
| **Obstacle in Center Zone** | `LEFT` / `RIGHT` | *"Left"* or *"Right"* | Evaluates left vs. right traversability score and guides detour around center hazard. |
| **Critical Hazard / Path Blocked** | `STOP` | *"Stop"* | Critical hazard within <1.5m or all directional paths blocked. Emergency STOP queue override triggered immediately (Priority 100). |
| **Temporarily Hidden Hazard** | Held Command | Spoken Warning | Object occluded or out of frame; PHMU memory holds hazard active for up to 3 seconds to prevent walking into remembered hazards. |

---

## 12. Bluetooth Audio Setup

VisionGuide AI uses offline Windows SAPI5 Speech Synthesis (`pyttsx3`) for audio feedback. Audio is automatically routed through the active Windows Default Audio Device.

### How to Connect and Route Audio to Bluetooth Headphones

1. **Pair Headphones**: Open Windows Settings (`Win + I`) -> **Bluetooth & devices** -> Add your Bluetooth earbuds, headphones, or portable speaker.
2. **Set Playback Endpoint**: Click the sound icon in the Windows taskbar system tray and set your Bluetooth headphones as the **Default Playback Device**.
3. **Verify Audio**: Test sound playback in Windows to ensure audio is routed to your Bluetooth headphones.
4. **Launch VisionGuide AI**: Run `python run_visionguide.py`. Spoken navigation guidance (*"Forward"*, *"Left"*, *"Right"*, *"Stop"*) will play directly through your Bluetooth headphones.

### Audio Configuration Settings (`config/audio.yaml`)

```yaml
tts:
  engine: "pyttsx3"
  rate: 175          # Speech speed in words per minute (WPM)
  volume: 1.0        # Volume (0.0 to 1.0)
  voice_id: null     # Set to specific Windows voice ID or null for default

device:
  output_device_name: null  # Uses default Windows output (Bluetooth device)
  cooldown_seconds: 1.5     # Cooldown interval between repeated audio commands
  enable_speech_queue: true
```

---

## 13. Live Visualization

VisionGuide AI includes an interactive visual display HUD for demonstration, debugging, and real-time monitoring.

### Launching the Interactive Pipeline Visualizer

To launch the real-time visualizer with full bounding box overlays, track IDs, PHMU memory states, region masks, and decision indicators:

```powershell
python tests/view_system_integration.py
```

### Launching the Comprehensive Validation Dashboard

To open the real-time telemetry validation dashboard showing multi-window processing statistics:

```powershell
python tests/view_validation_dashboard.py
```

### Display Elements in the Visual HUD Overlay

```text
┌──────────────────────────────────────────────────────────────────────────┐
│ VISIONGUIDE AI — UNIFIED LIVE SYSTEM PIPELINE                             │
│ FPS: 1.4 | Latency: 737.5ms | Memory: 483MB RAM | CPU: 81%               │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  [Track ID: 1] PERSON                                                   │
│  ┌────────────────────────┐                                              │
│  │ Bounding Box           │  Distance: NEAR (1.20m)                      │
│  │                        │  Danger Score: 0.82 [HIGH]                   │
│  │                        │  PHMU State: ACTIVE                          │
│  └────────────────────────┘                                              │
│                                                                          │
│  ──────────────────────────────────────────────────────────────────────  │
│    LEFT ZONE (CLEAR)   │   CENTER ZONE (BLOCKED)  │  RIGHT ZONE (CLEAR) │
│    Free Space: 1.00    │   Free Space: 0.15       │  Free Space: 0.85   │
│  ──────────────────────────────────────────────────────────────────────  │
│                                                                          │
│  RECOMMENDED COMMAND: [ RIGHT ]                                          │
│  SPOKEN AUDIO QUEUE: "Right"                                            │
└──────────────────────────────────────────────────────────────────────────┘
```

* **Color-Coded Bounding Boxes**:
  * 🔴 **Red**: Critical Danger ($>0.85$) or Emergency STOP.
  * 🟡 **Yellow**: Moderate/High Hazard ($0.55 - 0.85$).
  * 🟢 **Green**: Safe / Low Hazard ($<0.55$).
  * 🔵 **Cyan**: PHMU `REMEMBERED` Hazard (Occluded or temporarily lost object retained in memory).
* **Directional Region Overlay**: Vertical grid lines denoting **LEFT (0-33%)**, **CENTER (33-67%)**, and **RIGHT (67-100%)** visual zones.
* **Top Telemetry Bar**: Shows live pipeline frame rate (FPS), frame processing latency (ms), system RAM footprint (MB), and CPU utilization (%).

---

## 14. Performance Expectations

The system performance has been empirically measured and documented on the reference Windows laptop hardware environment (Intel/AMD 4-Core CPU, 16 GB RAM, CPU Execution Mode).

### Empirical Benchmark Summary (`docs/system_integration_performance.md`)

```text
================================================================
                    PER-MODULE LATENCY BREAKDOWN                
================================================================
  - CAMERA      :    0.00 ms  (  0.0%)  [Frame Buffer Retrieve]
  - YOLO        :  736.36 ms  ( 99.8%)  [YOLOv8m CPU Inference]
  - TRACKING    :    0.02 ms  (  0.0%)  [BoT-SORT IoU & Identity]
  - PHMU        :    0.02 ms  (  0.0%)  [PHMU Memory Decay]
  - DISTANCE    :    0.00 ms  (  0.0%)  [Pinhole Monocular Geometry]
  - DANGER      :    0.00 ms  (  0.0%)  [Multi-Factor Context Danger]
  - FREE_SPACE  :    0.76 ms  (  0.1%)  [3-Region Traversability]
  - DECISION    :    0.32 ms  (  0.0%)  [Safety & Stability Hysteresis]
  - AUDIO       :    0.05 ms  (  0.0%)  [Threaded SAPI5 Queue Dispatch]
----------------------------------------------------------------
  TOTAL END-TO-END LATENCY : 737.53 ms
  SYSTEM PIPELINE FPS       : 1.36 FPS
  SYSTEM RAM CONSUMPTION    : 483.35 MB
  SYSTEM CPU UTILIZATION    : 81.2 %
----------------------------------------------------------------

PRIMARY SYSTEM BOTTLENECK : YOLOv8m Object Detection CPU Inference (736.36 ms / frame)
================================================================
```

### Architectural Performance Insights

1. **Sub-Millisecond Pipeline Overhead**: Non-vision modules (Modules 03 through 10 combined) process in just **1.17 ms** per frame (< 0.2% of total pipeline latency).
2. **Primary System Bottleneck**: PyTorch CPU inference for YOLOv8m accounts for **99.8%** of processing latency.
3. **Prototype Status**: VisionGuide AI is a **functional research/proof-of-concept prototype**. Its current CPU execution rate (~1.36 FPS) provides deterministic navigation proof-of-concept verification on standard laptops, while hardware accelerators (such as CUDA GPUs, Intel OpenVINO, or NPU units) represent future optimization paths.

---

## 15. Troubleshooting

| Problem / Error | Possible Cause | Solution |
| :--- | :--- | :--- |
| **`python: command not found`** | Python is not installed or not added to system `PATH` | Reinstall Python 3.10+ and check the box *"Add Python to PATH"* during installation. |
| **PowerShell script activation error** | Windows script execution policy restriction | Run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process` in PowerShell, then activate `.venv\Scripts\Activate.ps1`. |
| **`ModuleNotFoundError: No module named 'cv2'`** | Dependencies not installed in active environment | Ensure virtual environment is active (`.venv`) and run `pip install opencv-python ultralytics pyttsx3 sounddevice SpeechRecognition pyyaml psutil scipy`. |
| **`[ERROR] Failed to start camera stream`** | Webcam in use by another app or incorrect camera index | Close apps using camera (Zoom, Teams, Camera app). Check `camera.index` in `config/config.yaml` (try `0`, `1`, or `2`). |
| **YOLO model fails to load** | Missing or corrupted `yolov8m.pt` weights file | Delete corrupted `yolov8m.pt` if present; `ultralytics` will auto-download a clean copy on next startup. |
| **No audio output heard** | Windows default playback device muted or misrouted | Verify Windows audio volume and check that `config/audio.yaml` has `device.output_device_name: null` (uses Windows default endpoint). |
| **Bluetooth audio latency / disconnect** | Bluetooth power saving or disconnected headphones | Re-pair Bluetooth headphones in Windows Settings and set them as the default output device before launching `run_visionguide.py`. |
| **Audio commands repeat too rapidly** | Cooldown settings too low | Increase `audio.cooldown_seconds` in `config/config.yaml` (e.g., set to `2.0` seconds). |
| **Low Frame Rate (~1.3 - 1.5 FPS)** | Expected CPU processing bottleneck for YOLOv8m | This is normal behavior for medium-sized PyTorch neural model inference on laptop CPU. |

---

## 16. System Logs

VisionGuide AI generates isolated subsystem logs stored in the `logs/` directory for runtime inspection and post-session analysis:

```text
logs/
├── camera_input.log        # Camera frame capture events & hardware initialization status
├── object_detection.log    # YOLO model load time, inference latency, & detection counts
├── object_tracking.log     # BoT-SORT track creation, update logs, & ID assignments
├── hazard_memory.log       # PHMU memory creation, decay events, & retention expiry
├── distance_estimation.log # Monocular geometry calculations & proximity category assignments
├── danger_mapping.log      # Danger score calculations & critical hazard flags
├── free_space.log          # 3-Region occupancy calculations & traversability scores
├── decision_engine.log     # Navigation command decisions, score comparisons, & hysteresis switches
├── audio_guidance.log      # Audio queue dispatches, SAPI5 utterance completion, & priority overrides
└── system_integration.log  # Unified system state pipeline initialization & frame telemetry
```

Log outputs can be inspected during runtime using PowerShell:

```powershell
Get-Content logs/system_integration.log -Tail 20 -Wait
```

---

## 17. Recommended Demonstration Procedure

Follow this 5-stage demonstration procedure for live presentation and Robo Expo evaluation:

### Scenario 1 — Clear Path Demonstration (`FORWARD`)
1. Point webcam down an open, unobstructed hallway or room path.
2. **Observe Output**: Visual HUD displays `COMMAND: FORWARD` in green; spoken audio announces *"Forward"*.

### Scenario 2 — Center Obstacle Avoidance (`LEFT` / `RIGHT` Detour)
1. Place a chair or obstacle directly in the center of the camera field of view (< 2.0 meters away).
2. **Observe Output**: System detects center hazard, evaluates side zones, displays `COMMAND: LEFT` or `RIGHT` in yellow, and speaks *"Left"* or *"Right"*.

### Scenario 3 — Temporary Occlusion & Memory Persistence (PHMU Demo)
1. Place an object in view so it registers as an active hazard.
2. Momentarily block or obscure the camera view of the object with a sheet of paper or hand.
3. **Observe Output**: Object bounding box turns **Cyan** (`REMEMBERED` state); PHMU retains the hazard in memory for up to 3 seconds, preserving the safety detour command even while visually occluded.

### Scenario 4 — Completely Blocked Path Emergency (`STOP`)
1. Block both left, center, and right regions or stand directly in front of a wall/large object (< 1.5 meters).
2. **Observe Output**: Visual HUD flashes `COMMAND: STOP` in red; system instantly overrides audio queue with priority 100 and speaks *"Stop"*.

### Scenario 5 — Wireless Bluetooth Audio Verification
1. Pair Bluetooth earbuds to the Windows laptop.
2. Walk away from the laptop keyboard while listening to real-time audio guidance (*"Forward"*, *"Left"*, *"Right"*, *"Stop"*) spoken clearly through the Bluetooth headset.

---

## 18. Safety Disclaimer

> [!WARNING]
> **Safety & Research Disclaimer**
> 
> **VisionGuide AI** is currently a research, educational, and proof-of-concept assistive technology prototype. It is **not** a certified medical device or replacement for trained white cane mobility, guide dogs, or human assistance. The system operates using monocular distance approximation and CPU-bound neural detection which have inherent physical and environmental limitations. Do not rely solely on this prototype for real-world navigation in hazardous or safety-critical mobility environments.

---

## 19. Developer & Maintenance Guidelines

### How to Modify Hyperparameters
All operational thresholds (e.g., detection confidence, distance profiles, danger weights, free-space region boundaries, decision switching margins) are externalized in `config/config.yaml`. Edit YAML values directly without modifying core Python code.

### Module Architecture & Isolation
Each subsystem resides in its own isolated directory inside `modules/`. When introducing enhancements or new hardware abstractions, maintain the decoupled design pattern by updating `interface.py` and `models.py` within the respective module directory.

### Running Automated Test Suites After Edits
Always verify codebase integrity after making code or configuration changes by running the automated unit test suite:

```powershell
python -m unittest discover -s tests
```

---

## 20. Final Verification Checklist

Before running a live demonstration, confirm the following verification steps:

* [x] **Python Environment**: Python 3.10+ virtual environment activated.
* [x] **Dependencies Installed**: PyTorch CPU, OpenCV, Ultralytics, pyttsx3, sounddevice, psutil installed.
* [x] **Hardware Check**: Webcam connected and index verified (`python tests/view_camera.py`).
* [x] **Model Check**: `yolov8m.pt` present in project root.
* [x] **Audio Check**: Windows audio default endpoint set to speakers or Bluetooth headphones.
* [x] **Unit Test Suite**: All 183 automated tests passing (`python -m unittest discover -s tests`).
* [x] **Master Executable**: Pipeline boots cleanly via `python run_visionguide.py`.
