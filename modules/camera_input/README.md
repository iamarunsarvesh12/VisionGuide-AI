# Module 01 — Camera Input

## 1. Purpose
The **Camera Input Module** provides a standardized, hardware-decoupled visual acquisition subsystem for **VisionGuide AI**. Its primary objective is to continuously capture live video frames from the input device, track real-time frame rates and acquisition latency, and feed raw BGR frame matrices into downstream perception modules (Object Detection, Tracking, and PHMU).

---

## 2. Hardware
* **Prototype Input Device**: Built-in laptop webcam / USB video device class (UVC) camera.
* **Target Interface**: Standardized video acquisition pipeline via OpenCV (`cv2.VideoCapture`).
* **Verified Operating Index**: Device index `0` on Windows 11.

---

## 3. Camera Configuration
System-wide settings are defined in [`config/config.yaml`](../../config/config.yaml):

```yaml
camera:
  index: 0
  width: 640
  height: 480
  target_fps: 30
  cap_api_preference: "CAP_ANY"
```

---

## 4. Input
* Physical video stream from laptop camera device index `0`.

---

## 5. Output
* Standardized 3-channel OpenCV BGR image matrix (`numpy.ndarray`) of dimensions `(height, width, 3)` with uint8 values.

---

## 6. Interface
All visual input implementations inherit from the abstract base class [`CameraInterface`](file:///c:/Users/Admin/Documents/VisionGuide%20AI/modules/camera_input/interface.py):

```python
class CameraInterface(ABC):
    def start(self) -> bool: ...
    def read(self) -> Tuple[bool, Optional[np.ndarray]]: ...
    def stop(self) -> None: ...
    def is_opened(self) -> bool: ...
    def get_properties(self) -> Dict[str, Any]: ...
```

---

## 7. Error Handling
* **Device Initialization Failure**: Detects when camera fails to open or returns invalid initial frames, logging detailed error context and returning `False`.
* **Backend Fallback**: Attempts requested backend (e.g., `CAP_DSHOW`), automatically falling back to `CAP_ANY` if backend initialization fails.
* **Stream Disconnection / Frame Read Dropping**: Handles sudden frame capture drops gracefully without crashing downstream processing.
* **Resource Leaks**: Implements safe hardware stream release inside `stop()` with exception handling.

---

## 8. Performance
* **Capture Latency**: Measures exact per-frame read time in milliseconds (`capture_latency_ms`).
* **Real-time FPS**: Calculates rolling frame rate based on total elapsed time and captured frame count.
* **Zero AI Processing Leakage**: Perception logic (YOLO, BoT-SORT, PHMU) is strictly excluded from this module to ensure maximal frame acquisition throughput.

---

## 9. Testing
Unit testing is implemented in [`tests/test_camera.py`](../../tests/test_camera.py) covering:
1. Camera initialization on device index 0.
2. Device availability checks.
3. Live frame capture and matrix validation.
4. Frame height, width, and channel dimension validation.
5. Graceful camera release and hardware shutdown.
6. Invalid camera index handling (e.g., index 99).

---

## 10. Future Hardware Migration
The modular design strictly isolates camera acquisition behind `CameraInterface`. Future migration to wearable smart glass cameras, wide-angle USB cameras, or Android smartphone camera streams will only require implementing a corresponding class (e.g. `WearableCameraInput` or `AndroidCameraStream`) implementing `CameraInterface` without modifying any downstream object detection, tracking, or PHMU reasoning components:

```text
Laptop Webcam  ──────┐
USB Camera     ──────┼───► [CameraInterface] ──► raw BGR Frame ──► YOLOv8 / PHMU
Wearable Glass ──────┘
```
