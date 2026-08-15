# Module 02 & 03 — YOLOv8m Object Detection

## 1. Purpose
The **Object Detection Module** performs real-time visual perception by receiving raw video frames from Module 01 (Camera Input) and extracting bounding boxes, confidences, spatial centroids, and class names for navigation-relevant objects.

---

## 2. YOLOv8m Role
* **Primary AI Perception Engine**: Uses Ultralytics **YOLOv8m** (Medium variant, ~25.9M parameters) as the baseline object detector.
* **Current Weights Baseline**: Official pretrained COCO weights (`yolov8m.pt`), covering standard navigation classes (`person`, `chair`, `dining table`, `door` equivalent, `backpack`, `bottle`, etc.).
* **Fine-Tuning Architecture**: Structured to support custom-trained weights (`models/best.pt`) once specialized navigation fine-tuning is completed.

---

## 3. Input
* BGR image matrix (`numpy.ndarray`) of dimensions `(height, width, 3)` supplied per frame by [`CameraInterface`](../camera_input/interface.py).

---

## 4. Output
* List of structured [`Detection`](file:///c:/Users/Admin/Documents/VisionGuide%20AI/modules/object_detection/interface.py) objects containing:
  - `class_id` (int)
  - `class_name` (str)
  - `confidence` (float in range 0.0 – 1.0)
  - `bounding_box` (`[x1, y1, x2, y2]` in pixel coordinates)
  - `center_x`, `center_y` (centroid spatial coordinates)
  - `width`, `height` (bounding box dimensions)

---

## 5. Model Configuration
Configured in [`config/config.yaml`](../../config/config.yaml):

```yaml
yolo:
  model_path: "yolov8m.pt"
  confidence_threshold: 0.50
  iou_threshold: 0.45
```

---

## 6. Detection Pipeline
```text
  Camera Input (Module 01)
            │
      OpenCV BGR Frame
            │
            ▼
┌───────────────────────┐
│   YOLOv8mDetector     │  <── device: 'cpu'
└───────────┬───────────┘
            │
            ▼
  Structured Detections  ──► [Class, Conf, BBox, Centroid]
            │
            ▼
  BoT-SORT Tracker (Module 04 - Future Phase)
```

---

## 7. CPU Limitations
* **Execution Architecture**: CPU-only PyTorch execution (`torch.device('cpu')`) on Windows 11.
* **Latency vs. Accuracy Balance**: YOLOv8m delivers high perception accuracy but requires significant CPU computation per frame compared to lightweight variants.
* **Pipeline Synchronization**: To prevent camera stream lagging behind inference, frame queueing / async threading will be used in System Controller (Module 15).

---

## 8. Performance
Measured empirical metrics on host CPU:
* **Model Load Latency**: ~14.1 seconds (initial PyTorch weights instantiation).
* **Per-Frame CPU Inference Latency**: Measured via [`tests/benchmark_detection.py`](../../tests/benchmark_detection.py).
* **Average Inference FPS**: Measured via empirical CPU benchmarks.

---

## 9. Testing
Unit tests in [`tests/test_detection.py`](../../tests/test_detection.py) cover:
1. Model loading and weight verification.
2. OpenCV frame acceptance.
3. Detection output structure validation (`Detection` dataclass attributes).
4. Bounding-box boundary checks (`x1 <= x2`, `y1 <= y2`).
5. Confidence value ranges (`0.0 <= conf <= 1.0`).
6. Class-name dictionary mapping.
7. Invalid frame handling (`None`, empty array).
8. Model cleanup and resource release.

---

## 10. Integration with Camera Input
The detector operates completely independently from camera capture. The application acquires frames via `CameraInterface.read()` and passes the frame to `YOLOv8mDetector.detect(frame)`:

```python
camera = WebcamInput(camera_index=0)
detector = YOLOv8mDetector(model_path="yolov8m.pt")
detector.load_model()

ret, frame = camera.read()
if ret:
    detections = detector.detect(frame)
```

---

## 11. Future Integration with BoT-SORT
In Phase 3, the list of `Detection` instances produced per frame by `YOLOv8mDetector` will be fed directly into `BoTSORTTracker`, which will assign persistent temporal `track_id` values to each detected object before passing tracks into the **Persistent Hazard Memory Unit (PHMU)**.
