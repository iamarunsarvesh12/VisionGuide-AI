# VisionGuide AI — Quick Start Guide

> Simple instructions to run the VisionGuide AI working prototype using a laptop webcam and Bluetooth headphones.

```text
Laptop Webcam
      ↓
VisionGuide AI
      ↓
AI Perception & Reasoning
      ↓
Navigation Decision
      ↓
Bluetooth Headphones
```

---

## Required Hardware

```text
1. Laptop / PC (Windows 10/11)
2. Laptop webcam (built-in or USB)
3. Bluetooth headphones / earbuds
```

> **NOTE**:
> The current working prototype does **NOT** require smart glasses, an external camera array, a Raspberry Pi, or dedicated AI hardware accelerators.

---

## Before You Start

Make sure:
- [x] Laptop is powered on
- [x] Laptop webcam is available and uncovered
- [x] Bluetooth headphones are connected to Windows (or default speakers)
- [x] Project folder `VisionGuide AI` is available

---

# Run VisionGuide AI — One Command

Open Windows PowerShell or Command Prompt and run:

```powershell
cd "C:\Users\Admin\Documents\VisionGuide AI"
python run_visionguide.py
```

> **This single command automatically prepares and starts the complete VisionGuide AI prototype.**

The launcher automatically:
- Creates the `.venv` virtual environment if missing
- Switches execution to `.venv` automatically
- Installs required packages from `requirements.txt` if missing
- Prepares/downloads the `yolov8m.pt` model weights if missing
- Verifies the webcam hardware and Windows default audio output
- Initializes all 10 modules and boots the VisionGuide AI stream

```text
Camera
  ↓
YOLOv8m Object Detection
  ↓
BoT-SORT Tracking
  ↓
PHMU Hazard Memory
  ↓
Distance Estimation
  ↓
Danger Mapping
  ↓
Free-Space Analysis
  ↓
Decision Engine
  ↓
Offline Audio Guidance
  ↓
Bluetooth / Speaker Audio
```

---

## Easiest Method for Robo Expo

You can also simply double-click `run_visionguide.bat` to start VisionGuide AI.

```text
VisionGuide AI/
│
├── run_visionguide.bat   ← DOUBLE-CLICK THIS TO START
├── run_visionguide.py
├── requirements.txt
├── yolov8m.pt
├── modules/
├── tests/
└── config/
```

> **Recommended for live Robo Expo presentations, mentors, and judges.**

---

## What Happens After Starting?

When `python run_visionguide.py` (or `run_visionguide.bat`) is launched, the main launcher automatically performs 6 pre-flight diagnostic checks:

```text
========================================
        VISIONGUIDE AI
   One-Command Prototype Launcher
========================================

[1/6] Checking environment...       OK
[2/6] Virtual environment...         READY
[3/6] Checking dependencies...        READY
[4/6] Checking YOLOv8m model...       READY
[5/6] Checking hardware...            
      Laptop Webcam...               READY
      Audio Output...                READY (Bluetooth / Speaker)
[6/6] Checking configuration...       READY

========================================
       VISIONGUIDE AI READY
========================================

Starting VisionGuide AI...
```

Once all checks pass, the system initializes all 10 modules and opens the live navigation window.

---

## How to Demonstrate the Prototype

### Demonstration 1 — Clear Path
- Point webcam at an open hallway or clear walking area.
- **Visual Display**: `FORWARD`
- **Audio Output**: `"Forward"`

### Demonstration 2 — Obstacle Avoidance
- Place a chair or obstacle on one side of the camera view.
- The system evaluates safe free space and steers around it.
- **Visual Display**: `LEFT` or `RIGHT`
- **Audio Output**: `"Left"` or `"Right"`

### Demonstration 3 — Persistent Hazard Memory ⭐
- **⭐ CORE INNOVATION — Persistent Hazard Memory Unit (PHMU)**
  1. Detect an object in front of the camera.
  2. Temporarily obscure or shift the object outside active vision.
  3. PHMU retains the short-term hazard in memory (cyan bounding box).
  4. The remembered hazard continues influencing navigation reasoning for up to 3 seconds.

### Demonstration 4 — Unsafe Environment / Emergency Stop
- Stand directly in front of the camera or block all walking paths.
- **Visual Display**: `STOP` (flashes red)
- **Audio Output**: `"Stop"` (immediate priority override)

---

## How to Stop

To terminate VisionGuide AI gracefully:
- Press **`Q`** or **`ESC`** in the OpenCV window
- Or press **`Ctrl+C`** in the terminal window

The system will safely release camera and audio hardware handles.

---

## If the Project Does Not Start

### Problem: Webcam not detected
1. Make sure your laptop camera is enabled and uncovered.
2. Close Camera, Teams, Zoom, or other apps using the webcam.
3. Run `python run_visionguide.py` again.

### Problem: Bluetooth audio not working
1. Connect your Bluetooth headphones.
2. Open Windows Sound Settings and set them as the default output device.
3. Run `python run_visionguide.py` again.

### Problem: Missing Python package
1. Make sure you have internet access for initial setup.
2. Run `python run_visionguide.py` again (automatic repair will retry pip install).

---

## Every Normal Run

The normal process for every run is simply:

```text
1. Connect Bluetooth headphones (optional)
2. Open VisionGuide AI folder
3. Double-click run_visionguide.bat
4. VisionGuide AI starts
```

**OR using PowerShell**:

```powershell
cd "C:\Users\Admin\Documents\VisionGuide AI"
python run_visionguide.py
```

---

## IMPORTANT: Do Not Run Individual Modules

> For normal prototype execution, you do **NOT** need to run individual python files inside `modules/` or `tests/`.

The main system launcher automatically manages and executes all modules in sequence.

```text
DO NOT RUN INDIVIDUALLY FOR NORMAL DEMO:
❌ camera.py
❌ detector.py
❌ tracker.py
❌ memory.py
❌ estimator.py
❌ mapper.py
❌ analyzer.py
❌ engine.py
❌ guidance.py

RUN INSTEAD:
✅ python run_visionguide.py
or
✅ double-click run_visionguide.bat
```

---

## System Architecture

```text
                    VISIONGUIDE AI
                         │
                         ▼
                  Laptop Webcam
                         │
                         ▼
                    YOLOv8m
                         │
                         ▼
                    BoT-SORT
                         │
                         ▼
                      PHMU ⭐
                         │
                         ▼
               Distance Estimation
                         │
                         ▼
                 Danger Mapping
                         │
                         ▼
               Free-Space Analysis
                         │
                         ▼
                Decision Engine
                         │
                         ▼
              Offline Audio Guidance
                         │
                         ▼
              Bluetooth Headphones
```
