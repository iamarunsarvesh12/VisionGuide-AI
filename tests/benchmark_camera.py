import time
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from modules.camera_input.camera import WebcamInput

def run_benchmark():
    print("Starting Camera Performance Measurement...")
    t0 = time.perf_counter()
    cam = WebcamInput(camera_index=0, width=640, height=480, target_fps=30)
    ok = cam.start()
    t1 = time.perf_counter()
    init_ms = (t1 - t0) * 1000.0

    if not ok:
        print("Camera start failed!")
        sys.exit(1)

    latencies = []
    t_start = time.perf_counter()
    for _ in range(100):
        tf0 = time.perf_counter()
        ret, frame = cam.read()
        tf1 = time.perf_counter()
        if ret:
            latencies.append((tf1 - tf0) * 1000.0)
    t_end = time.perf_counter()

    total_s = t_end - t_start
    fps = len(latencies) / total_s
    avg_lat = sum(latencies) / len(latencies) if latencies else 0.0
    min_lat = min(latencies) if latencies else 0.0
    max_lat = max(latencies) if latencies else 0.0

    props = cam.get_properties()
    cam.stop()

    print("=== CAMERA PERFORMANCE RESULTS ===")
    print(f"Camera Index: {props['index']}")
    print("Requested Resolution: 640x480")
    print(f"Actual Resolution: {props['width']}x{props['height']}")
    print(f"Initialization Time: {init_ms:.2f} ms")
    print(f"Total Frames Read: {len(latencies)}")
    print(f"Total Benchmark Duration: {total_s:.2f} s")
    print(f"Measured FPS: {fps:.2f}")
    print(f"Average Capture Latency: {avg_lat:.2f} ms")
    print(f"Min Capture Latency: {min_lat:.2f} ms")
    print(f"Max Capture Latency: {max_lat:.2f} ms")

    return {
        "index": props['index'],
        "width": props['width'],
        "height": props['height'],
        "init_ms": round(init_ms, 2),
        "fps": round(fps, 2),
        "avg_lat_ms": round(avg_lat, 2),
        "min_lat_ms": round(min_lat, 2),
        "max_lat_ms": round(max_lat, 2),
    }

if __name__ == "__main__":
    run_benchmark()
