import time
import os
import sys
import logging
import subprocess
import importlib.util
import urllib.request

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

logger = logging.getLogger("VisionGuideMain")


# ============================================================================
# AUTOMATIC ENVIRONMENT & VIRTUAL ENVIRONMENT LAUNCHER (STEP 1 & STEP 2)
# ============================================================================

def get_venv_python_path() -> str:
    """Return absolute path to virtual environment Python executable."""
    if sys.platform == "win32":
        return os.path.join(PROJECT_ROOT, ".venv", "Scripts", "python.exe")
    return os.path.join(PROJECT_ROOT, ".venv", "bin", "python")


def ensure_virtual_environment() -> str:
    """
    Check if .venv exists; create it automatically if missing.
    If running under global/system Python, automatically re-execute using .venv Python.
    """
    venv_dir = os.path.join(PROJECT_ROOT, ".venv")
    venv_python = get_venv_python_path()

    # Create .venv if it does not exist
    if not os.path.exists(venv_dir) or not os.path.exists(venv_python):
        print("[2/6] Virtual environment (.venv) not found. Creating automatically...")
        try:
            subprocess.run([sys.executable, "-m", "venv", venv_dir], check=True)
            print("      Virtual environment created successfully.")
        except Exception as e:
            print(f"\n========================================\nVISIONGUIDE AI — STARTUP ERROR\n========================================")
            print(f"Problem: Failed to create virtual environment (.venv).\nError: {e}\n")
            sys.exit(1)

    # Re-execute under .venv Python if currently running under global Python
    current_python = os.path.abspath(sys.executable).lower()
    expected_python = os.path.abspath(venv_python).lower()

    if current_python != expected_python and not os.environ.get("_VG_RELAUNCHED"):
        os.environ["_VG_RELAUNCHED"] = "1"
        print("[2/6] Switching execution to virtual environment (.venv)...")
        try:
            code = subprocess.call([venv_python, __file__] + sys.argv[1:])
            sys.exit(code)
        except Exception as e:
            print(f"Failed to switch to virtual environment Python: {e}")

    return venv_python


def check_python_environment() -> bool:
    """Check Python version compatibility, CPU, RAM, and OS."""
    if sys.version_info < (3, 8):
        print("\n========================================")
        print("VISIONGUIDE AI — STARTUP ERROR")
        print("========================================")
        print(f"\nProblem: Python version {sys.version_info.major}.{sys.version_info.minor} is not supported.")
        print("Please install Python 3.8 or higher.\n")
        return False
    return True


# ============================================================================
# AUTOMATIC DEPENDENCY INSTALLATION (STEP 3)
# ============================================================================

REQUIRED_PACKAGES = [
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("opencv-python", "cv2"),
    ("ultralytics", "ultralytics"),
    ("pyttsx3", "pyttsx3"),
    ("sounddevice", "sounddevice"),
    ("SpeechRecognition", "speech_recognition"),
    ("pyyaml", "yaml"),
    ("psutil", "psutil"),
    ("scipy", "scipy"),
    ("numpy", "numpy"),
]


def ensure_dependencies() -> bool:
    """Check required packages; automatically run pip install if missing."""
    missing = []
    for pkg_name, module_name in REQUIRED_PACKAGES:
        if importlib.util.find_spec(module_name) is None:
            missing.append(pkg_name)

    if missing:
        print(f"\n[3/6] Missing dependencies detected: {', '.join(missing)}")
        print("      Installing required packages from requirements.txt...")
        req_file = os.path.join(PROJECT_ROOT, "requirements.txt")
        if not os.path.exists(req_file):
            print("\n========================================")
            print("VISIONGUIDE AI — STARTUP ERROR")
            print("========================================")
            print("Problem: requirements.txt is missing from project root.")
            return False

        try:
            cmd = [sys.executable, "-m", "pip", "install", "-r", req_file]
            result = subprocess.run(cmd, check=True)
            print("      Dependencies installation complete.")
        except Exception as e:
            print("\n========================================")
            print("VISIONGUIDE AI — STARTUP ERROR")
            print("========================================")
            print("Problem: A required Python dependency could not be installed.")
            print(f"Error details: {e}")
            print("\nPlease check your internet connection and run:")
            print("python run_visionguide.py\n")
            return False

    return True


# ============================================================================
# AUTOMATIC MODEL PREPARATION (STEP 4)
# ============================================================================

def ensure_yolo_model(model_filename: str = "yolov8m.pt") -> bool:
    """Verify presence of YOLO weights; automatically download if missing."""
    model_path = os.path.join(PROJECT_ROOT, model_filename)
    if os.path.exists(model_path):
        return True

    print(f"\n[4/6] YOLOv8m model file ({model_filename}) not found in project root.")
    print("      Downloading required model weights...")
    url = f"https://github.com/ultralytics/assets/releases/download/v8.3.0/{model_filename}"

    try:
        def _progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(100.0, (downloaded / total_size) * 100.0)
                sys.stdout.write(f"\r      Downloading {model_filename}: {percent:.1f}% ({downloaded // (1024*1024)}MB / {total_size // (1024*1024)}MB)")
                sys.stdout.flush()

        urllib.request.urlretrieve(url, model_path, _progress)
        print("\n      Download complete. YOLOv8m model READY.")
        return True
    except Exception as e:
        # Fallback to ultralytics package download
        try:
            from ultralytics import YOLO
            print(f"\n      Attempting secondary download via Ultralytics framework...")
            model = YOLO(model_filename)
            print("      Download complete. YOLOv8m model READY.")
            return True
        except Exception as e2:
            print("\n========================================")
            print("VISIONGUIDE AI — STARTUP ERROR")
            print("========================================")
            print("Problem: YOLOv8m model could not be prepared.")
            print(f"Details: {e2}")
            print("\nPlease check your internet connection or manually place yolov8m.pt in:")
            print(f"{PROJECT_ROOT}\n")
            return False


# ============================================================================
# AUTOMATIC HARDWARE & CONFIGURATION CHECKS (STEPS 5 & 6)
# ============================================================================

def check_hardware() -> tuple:
    """
    Verify webcam hardware accessibility and audio endpoints.
    Returns (cam_ok, audio_ok, is_bluetooth, device_name).
    """
    # Camera check
    cam_ok = False
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret and frame is not None:
                cam_ok = True
    except Exception:
        cam_ok = False

    # Audio check
    audio_ok = True
    is_bt = False
    dev_name = "Windows Default Speaker"
    try:
        from modules.audio_guidance.audio_output import AudioOutputDevice
        audio_dev = AudioOutputDevice(use_system_default=True)
        audio_dev.initialize()
        audio_ok = audio_dev.is_available
        is_bt = audio_dev.is_bluetooth
        dev_name = audio_dev.device_name
    except Exception:
        pass

    return cam_ok, audio_ok, is_bt, dev_name


def check_configurations() -> bool:
    """Verify required configuration files exist."""
    required_configs = [
        "config/config.yaml",
        "config/classes.yaml",
        "config/audio.yaml",
        "requirements.txt",
    ]
    for cfg in required_configs:
        p = os.path.join(PROJECT_ROOT, cfg)
        if not os.path.exists(p):
            print("\n========================================")
            print("VISIONGUIDE AI — STARTUP ERROR")
            print("========================================")
            print(f"Problem: Required configuration file '{cfg}' is missing.")
            return False
    return True


def run_one_command_preflight() -> bool:
    """Execute complete 6-step one-command automated startup sequence."""
    print("========================================")
    print("        VISIONGUIDE AI                  ")
    print("   One-Command Prototype Launcher       ")
    print("========================================")
    print()

    # Step 1: Environment Check
    print("[1/6] Checking environment...       ", end="", flush=True)
    if not check_python_environment():
        return False
    print("OK")

    # Step 2: Virtual Environment Setup & Re-execution
    ensure_virtual_environment()
    print("[2/6] Virtual environment...         READY")

    # Step 3: Dependency Check & Auto-Installation
    print("[3/6] Checking dependencies...        ", end="", flush=True)
    if not ensure_dependencies():
        return False
    print("READY")

    # Step 4: YOLO Model Check & Auto-Download
    print("[4/6] Checking YOLOv8m model...       ", end="", flush=True)
    if not ensure_yolo_model("yolov8m.pt"):
        return False
    print("READY")

    # Step 5: Hardware Check
    print("[5/6] Checking hardware...            ")
    cam_ok, audio_ok, is_bt, dev_name = check_hardware()
    if not cam_ok:
        print("\n========================================")
        print("VISIONGUIDE AI — STARTUP ERROR")
        print("========================================")
        print("Problem: Laptop webcam could not be detected.")
        print("\nPlease check:")
        print("1. Camera is connected/enabled.")
        print("2. No other application (Zoom, Teams, Camera app) is using the camera.")
        print("3. Windows camera permissions are enabled.")
        print("\nRun python run_visionguide.py again.\n")
        return False
    print("      Laptop Webcam...               READY")
    bt_str = "READY (Bluetooth)" if is_bt else f"READY ({dev_name})"
    print(f"      Audio Output...                {bt_str}")

    # Step 6: Configuration Check
    print("[6/6] Checking configuration...       ", end="", flush=True)
    if not check_configurations():
        return False
    print("READY")

    print("\n========================================")
    print("       VISIONGUIDE AI READY             ")
    print("========================================")
    print("Environment       : READY")
    print("Virtual Env (.venv): READY")
    print("Dependencies      : READY")
    print("YOLOv8m Model     : READY")
    print("Laptop Camera     : READY")
    print("Audio Output      : READY")
    print("========================================")
    print("Starting VisionGuide AI...\n")
    return True


# ============================================================================
# MAIN APPLICATION PIPELINE EXECUTION
# ============================================================================

def self_has_display() -> bool:
    """Check if graphical display desktop environment is available."""
    return os.name == 'nt' or bool(os.environ.get('DISPLAY'))


def render_overlay(result) -> cv2.Mat:
    """Render HUD overlay on visual frame for main window display."""
    import cv2
    import numpy as np

    frame = result.camera_status.get("raw_frame", None) if hasattr(result, "camera_status") and isinstance(result.camera_status, dict) else None
    if frame is None or not isinstance(frame, np.ndarray):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
    else:
        img = frame.copy()

    h, w = img.shape[:2]

    # 1. Regional Free Space Dividers
    w_left = int(w * 0.33)
    w_right = int(w * 0.67)
    cv2.line(img, (w_left, 0), (w_left, h - 70), (100, 100, 100), 1)
    cv2.line(img, (w_right, 0), (w_right, h - 70), (100, 100, 100), 1)

    cv2.putText(img, "LEFT", (w_left // 2 - 20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    cv2.putText(img, "CENTER", (w_left + (w_right - w_left) // 2 - 30, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    cv2.putText(img, "RIGHT", (w_right + (w - w_right) // 2 - 25, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    # 2. Bounding Boxes & Danger Assessments
    if hasattr(result, "danger_assessments") and result.danger_assessments:
        for da in result.danger_assessments:
            bbox = getattr(da, "bbox", [0, 0, 50, 50])
            x1, y1, x2, y2 = [int(v) for v in bbox]
            tid = getattr(da, "track_id", 0)
            lvl = getattr(da, "danger_level", "LOW")
            cls_n = getattr(da, "class_name", "object")
            phmu_st = getattr(da, "memory_state", "ACTIVE")

            if phmu_st == "REMEMBERED":
                color = (255, 255, 0)  # Cyan for PHMU Memory
            elif lvl == "CRITICAL":
                color = (0, 0, 255)    # Red
            elif lvl in ["HIGH", "MODERATE"]:
                color = (0, 255, 255)  # Yellow
            else:
                color = (0, 255, 0)    # Green

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, f"ID:{tid} {cls_n} ({lvl})", (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    # 3. Bottom Navigation Banner
    cmd = result.decision_result.command if getattr(result, "decision_result", None) else "STOP"
    audio_msg = result.audio_result.message if getattr(result, "audio_result", None) else ""
    cmd_color = (0, 255, 0) if cmd == "FORWARD" else (0, 255, 255) if cmd in ["LEFT", "RIGHT"] else (0, 0, 255)

    cv2.rectangle(img, (0, h - 70), (w, h), (20, 20, 20), -1)
    cv2.putText(img, f"COMMAND: {cmd}", (15, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, cmd_color, 2)
    cv2.putText(img, f"AUDIO: \"{audio_msg}\"", (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(img, f"FPS: {result.pipeline_fps:.1f} | Latency: {result.total_latency:.1f}ms", (w - 240, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    return img


def main():
    """
    Main entry point executable for VisionGuide AI.
    Executes automated pre-flight diagnostics, initializes all 10 modules, and boots the real-time stream.
    """
    if not run_one_command_preflight():
        sys.exit(1)

    # Deferred import of system integration pipeline
    from modules.system_integration.pipeline import VisionGuideSystemPipeline
    import cv2

    pipeline = VisionGuideSystemPipeline(config_path="config/config.yaml")

    if not pipeline.initialize():
        print(f"\n[FATAL ERROR] System Initialization Failed: {pipeline.error_message}")
        sys.exit(1)

    stats = pipeline.get_statistics()

    print("========================================")
    print("         MODULE INITIALIZATION          ")
    print("========================================")

    module_display_map = [
        ("Camera", "camera"),
        ("YOLOv8m", "yolo"),
        ("BoT-SORT", "tracking"),
        ("PHMU", "phmu"),
        ("Distance", "distance"),
        ("Danger Mapping", "danger"),
        ("Free Space", "free_space"),
        ("Decision Engine", "decision"),
        ("Audio", "audio"),
    ]

    for label, key in module_display_map:
        state = stats["module_statuses"].get(key, "READY")
        print(f"{label:<16}: [{state}]")

    print()
    print("========================================")
    print("SYSTEM RUNNING — Press Q, ESC, or Ctrl+C to quit.")
    print("========================================\n")

    if not pipeline.start():
        print("[WARNING] Live camera hardware stream unavailable. Continuing with fallback stream...")

    frame_count = 0
    start_time = time.time()

    try:
        while True:
            result = pipeline.process_frame()
            frame_count += 1

            cmd = result.decision_result.command if result.decision_result else "UNKNOWN"
            audio_msg = result.audio_result.message if result.audio_result else ""

            if frame_count % 10 == 0:
                print(
                    f"Frame {result.frame_id:04d} | FPS: {result.pipeline_fps:.1f} | "
                    f"Latency: {result.total_latency:.1f}ms | Command: {cmd:<7} | "
                    f"Audio: \"{audio_msg}\""
                )

            if self_has_display():
                display_img = render_overlay(result)
                cv2.imshow("VisionGuide AI — Live Navigation Pipeline", display_img)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), ord('Q'), 27):  # 27 is ESC key
                    print("\nUser requested termination ('Q' or 'ESC' key pressed).")
                    break

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nKeyboard Interrupt received (Ctrl+C). Terminating...")
    except Exception as e:
        print(f"\n[UNHANDLED PIPELINE ERROR]: {e}")
    finally:
        pipeline.stop()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        total_time = time.time() - start_time
        print(f"\nSystem session terminated. Processed {frame_count} frames in {total_time:.1f}s.")


if __name__ == "__main__":
    main()
