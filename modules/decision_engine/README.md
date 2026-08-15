# Module 09 — Context-Aware Decision Engine

## Overview

The **Context-Aware Decision Engine** is Module 09 of **VisionGuide AI** (Offline Multimodal Persistent-Hazard-Memory Navigation System).

It receives structured perception and spatial-temporal reasoning outputs from upstream modules:
1. **Camera Input** (RGB frames)
2. **YOLOv8m** (Object detection)
3. **BoT-SORT** (Multi-object tracking)
4. **Persistent Hazard Memory Unit (PHMU)** (Temporal hazard retention & decay)
5. **Monocular Distance Estimation** (Distance categories & meters)
6. **Context-Aware Danger Mapping** (Danger levels & numerical risk scores)
7. **Image-Space Free-Space Analysis** (Regional occupancy states & safe-space scores)

The Decision Engine synthesizes these multi-dimensional inputs into a **simple, deterministic navigation directive**:
* `FORWARD`
* `LEFT`
* `RIGHT`
* `STOP`

---

## Architectural Boundary

> [!IMPORTANT]
> The Decision Engine produces **structured decision data objects** (`DecisionResult`). It does **NOT** perform audio synthesis, text-to-speech (TTS), speech recognition, voice input, Bluetooth audio output, GPS navigation, SLAM, or hardware motor control. Audio output is strictly handled downstream in Module 10.

---

## Decision Logic & Formula

### Regional Scoring Formula

For candidate navigation regions (`LEFT`, `CENTER`, `RIGHT`), the engine computes a normalized decision score:

$$\text{Decision Score} = W_{\text{safe}} \cdot \text{SafeSpace} + W_{\text{confidence}} \cdot \text{Confidence} + W_{\text{stability}} \cdot \text{Stability} - W_{\text{danger}} \cdot \text{Danger} - W_{\text{uncertainty}} \cdot \text{Uncertainty}$$

Where prototype weights are configured in `config/config.yaml`:
* $W_{\text{safe}} = 0.40$
* $W_{\text{danger}} = 0.30$
* $W_{\text{confidence}} = 0.15$
* $W_{\text{stability}} = 0.15$ (subtle tie-breaker)
* $W_{\text{uncertainty}} = 0.25$

---

## Safety Override Rules

> **WHEN THE SYSTEM CANNOT CONFIDENTLY IDENTIFY A SAFE DIRECTION, IT MUST PREFER `STOP` OVER AN UNSUPPORTED MOVEMENT COMMAND.**

1. **All Blocked Override**: If `LEFT`, `CENTER`, and `RIGHT` regions are `BLOCKED` or contain `CRITICAL` hazards ($\ge 0.85$), output `STOP`.
2. **Critical Center Override**: If `CENTER` contains a `CRITICAL` hazard and neither `LEFT` nor `RIGHT` has sufficient safe space ($\ge 0.50$), output `STOP`.
3. **Environmental Uncertainty Override**: If all regions have low safe space ($< 0.30$) or are `UNCERTAIN`, output `STOP`.

---

## Directional Command Rules

1. **`FORWARD`**: Selected when `CENTER` is `CLEAR`, safe space $\ge 0.70$ (`forward_safe_space_threshold`), danger $< 0.70$, and `CENTER` is not `BLOCKED`.
2. **`LEFT`**: Selected when `CENTER` is blocked or unsafe, `LEFT` is `CLEAR` / safe space $\ge 0.50$, and `LEFT` has lower danger than `RIGHT`.
3. **`RIGHT`**: Selected when `CENTER` is blocked or unsafe, `RIGHT` is `CLEAR` / safe space $\ge 0.50$, and `RIGHT` has lower danger than `LEFT`.
4. **`STOP`**: Selected when no directional region meets minimum safety confidence.

---

## Hysteresis & Command Stability

To prevent command flickering (e.g. oscillating between `LEFT` and `RIGHT` across rapid frames):
* Retains `previous_command` and tracks command timestamps.
* Switching from an active directional command to a different directional command requires a **candidate score improvement exceeding the `switching_margin`** ($0.10$).
* Enforces a minimum command hold duration ($0.5\text{ s}$).
* `STOP` commands override directional commands immediately when safety overrides trigger.

---

## File Structure

```text
modules/
└── decision_engine/
    ├── __init__.py
    ├── interface.py
    ├── models.py
    ├── engine.py
    └── README.md
```

---

## API Usage Example

```python
from modules.decision_engine.engine import ContextAwareDecisionEngine
from modules.decision_engine.models import DecisionInput

engine = ContextAwareDecisionEngine("config/config.yaml")
engine.initialize()

d_input = DecisionInput(
    timestamp=time.time(),
    frame_id=1,
    regions=free_space_result.regions,
    hazards=danger_assessments,
)

decision = engine.decide(d_input)

print(f"Command: {decision.command}")
print(f"Region: {decision.selected_region}")
print(f"Confidence: {decision.confidence:.2f}")
print(f"Reason: {decision.reason}")
```

---

## Verification & Benchmarking

* **Unit Tests**: `python -m unittest tests/test_decision_engine.py` (26 tests + 12 Core Decision Experiments)
* **Benchmark Suite**: `python tests/benchmark_decision_engine.py`
* **Live Visualizer**: `python tests/view_decision_engine.py`
