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
| `detector.py` | `Detector` interface: BGR frame -> `Detections` (xyxy boxes, scores, class ids, names). `UltralyticsDetector` runs ultralytics (dev only); `OnnxDetector` runs an ONNX export with onnxruntime; `DpuDetector` runs a Vitis AI `.xmodel` on the KV260 DPU. The last two share NumPy letterbox, decoding and NMS (`decode_predictions`). |
| `tracker.py`  | `ByteTracker`: self-contained ByteTrack (Kalman filter + two-stage IoU matching with `lap`, after the MIT-licensed original), same settings as ultralytics' `bytetrack.yaml`. No ultralytics dependency. |
| `memory.py`   | `ObjectMemory`: maps short-lived tracker IDs to stable object IDs, reconnecting within `MAX_DISTANCE` px and `MAX_GAP` s. |
| `camera.py`   | `CameraThread`: one capture thread per source; live cameras keep only the newest frame, video files deliver every frame. |
| `stream.py`   | `MjpegServer`: serves the annotated view as MJPEG over HTTP (standard library only). |
| `main.py`     | Wires it together (one tracker per camera, one shared `ObjectMemory`) and draws boxes plus the object-memory panel. |

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
python main.py --source 0 1       # two cameras, shown side by side
```

Other flags: `--weights`, `--max-frames N`, `--no-display`, `--report-every SEC`,
`--stream` / `--stream-port PORT` (MJPEG viewer, default port 8080).

With several cameras, object IDs come from one shared memory, but tracking and
reconnection are per camera: the same object seen by two cameras gets two IDs.

### ONNX backend

Export once (needs `pip install onnx onnxslim`), then select the backend:

```bash
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt').export(format='onnx', imgsz=640, simplify=True, opset=17)"
python main.py --backend onnx --source clip.mp4   # uses yolov8n.onnx by default
```

Every 5 s the pipeline prints average ms per stage (pre / inference / post /
track / memory) and FPS, plus an `OVERALL` line on exit.

## KV260 (ARM, Ubuntu 22.04, headless)

The board runs the ONNX backend only: no torch or ultralytics.

```bash
# on the laptop: export the model, then copy the repo + yolov8n.onnx to the board
# on the board:
./setup_board.sh                  # creates .venv from requirements-board.txt
.venv/bin/python main.py --backend onnx --no-display --stream --source 0
```

Then open `http://<board-ip>:8080/` in a laptop browser. `requirements-board.txt`
pins versions that ship Python 3.10 aarch64 wheels (numpy 2.2.6, onnxruntime
1.23.2); newer numpy/onnxruntime releases no longer support Python 3.10.

### DPU backend (KV260)

```bash
.venv/bin/python main.py --backend dpu --no-display --stream --source 0                        # DPU loaded via xmutil, VART directly
.venv/bin/python main.py --backend dpu --dpu-bitstream dpu.bit --no-display --stream --source 0  # Kria-PYNQ: pynq_dpu loads the overlay
```

`DpuDetector` expects a YOLOv8/YOLO11 `.xmodel` (default `yolov8n.xmodel`)
quantized and compiled with Vitis AI for the KV260 DPU, with the Detect head
cut before the DFL: NHWC input (1, 640, 640, 3) with RGB scaled to [0, 1]
before quantization, and per-stride raw outputs (1, H, W, 64 + nc), or box and
class maps separately. DFL softmax, sigmoid and anchor decoding run on the CPU
(`yolov8_raw_to_pred`), then the same `decode_predictions` as the ONNX backend.
Vitis AI (`vart`, `xir`, or `pynq_dpu`) comes with the board image, not pip,
and is imported only when the DPU backend is used.

## Tests

```bash
python -m unittest discover -s tests -v
```

Covers letterbox, NMS, `decode_predictions`, YOLOv8 head decoding and
`DpuDetector` with a fake VART runner; runs on the board venv (no torch). With
torch, ultralytics and `yolov8n.pt` present it also checks the head decoding
against the real YOLOv8n output.
