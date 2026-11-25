# Spatial Awareness (Embedded)

Real-time object detection and tracking with stable object IDs, targeting the
Kria KV260 (detection on the DPU, everything else on the ARM CPU). Currently
runs on a host PC with YOLO11n via ultralytics.

## Pipeline

```
camera -> detector.py -> tracker.py -> memory.py -> main.py (OpenCV window)
```

| Module        | Role |
|---------------|------|
| `detector.py` | `Detector` interface: BGR frame -> `Detections` (xyxy boxes, scores, class ids, names). `UltralyticsDetector` runs ultralytics; `OnnxDetector` runs an ONNX export through onnxruntime with its own NumPy letterbox, box decoding and NMS (no ultralytics), the template for the DPU backend. |
| `tracker.py`  | `ByteTracker`: self-contained ByteTrack (Kalman filter + two-stage IoU matching with `lap`, after the MIT-licensed original), same settings as ultralytics' `bytetrack.yaml`. No ultralytics dependency. |
| `memory.py`   | `ObjectMemory`: maps short-lived tracker IDs to stable object IDs, reconnecting within `MAX_DISTANCE` px and `MAX_GAP` s. |
| `main.py`     | Wires it together and draws boxes plus the object-memory panel. |

## Setup

Requires Python 3.10+ and a webcam.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The YOLO weights (`yolo11n.pt`) download automatically on first run.

## Run

```bash
python main.py                    # webcam 0, press q to quit
python main.py --source 1         # another camera
python main.py --source clip.mp4  # video file
```

Other flags: `--weights`, `--max-frames N`, `--no-display`, `--report-every SEC`.

### ONNX backend

Export once (needs `pip install onnx onnxslim`), then select the backend:

```bash
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt').export(format='onnx', imgsz=640, simplify=True, opset=17)"
python main.py --backend onnx --source clip.mp4   # uses yolov8n.onnx by default
```

Every 5 s the pipeline prints average ms per stage (pre / inference / post /
track / memory) and FPS, plus an `OVERALL` line on exit.
