# Camera Calibration Module (`modules/calibration/`)

## Overview

The **Camera Calibration Module** provides offline intrinsic parameter estimation, focal length calibration, mounting angle setup, and object reference height profile customization for **VisionGuide AI**.

---

## Calibration Principle & Optics Model

Monocular distance estimation relies on pinhole camera optics:

$$d = \frac{H_{real} \cdot f_{px}}{h_{px}}$$

Where:
- $d$: Estimated distance to object in metres.
- $H_{real}$: Physical reference height of object class in metres.
- $f_{px}$: Camera focal length in pixels.
- $h_{px}$: Bounding box pixel height on the $640 \times 480$ image matrix.

### Focal Length Calibration Formula

To calibrate $f_{px}$ empirically using a known target:

$$f_{px} = \frac{h_{px} \cdot d_{ground\_truth}}{H_{real}}$$

---

## Calibration Procedure

1. **Setup Known Calibration Object**: Place a target of known physical height $H_{real}$ (e.g., a chair $0.85\text{ m}$ tall or door $2.00\text{ m}$ tall) at a measured distance $d_{ground\_truth}$ (e.g., $2.00\text{ m}$).
2. **Measure Bounding Box Height**: Record the pixel height $h_{px}$ detected by the system.
3. **Execute Focal Length Calculation**:
   ```python
   calibrator = WebcamCalibrator()
   calibrator.initialize()
   f_px = calibrator.calibrate_focal_length(object_height_px=255.0, known_distance_m=2.00, known_height_m=0.85)
   calibrator.save_calibration("config/calibration.yaml")
   ```
4. **Persist Settings**: Calibration parameters are saved to `config/calibration.yaml` and loaded automatically by the pipeline.

---

## Limitations

1. **Pinhole Model Approximations**: Ignores radial and tangential lens distortions.
2. **Bounding Box Noise**: Bounding box pixel heights vary depending on lighting, partial occlusion, and detection jitter.
3. **Static Focal Length Assumption**: Assumes fixed optical focus without hardware optical zoom.
