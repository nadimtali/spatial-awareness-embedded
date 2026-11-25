"""Object detection backends.

Every backend maps a BGR frame to a `Detections` object, so the rest of the
pipeline (tracker, memory, UI) never depends on where detection runs:

    UltralyticsDetector  ultralytics on the host (dev only)
    OnnxDetector         onnxruntime on CPU (host or board)
    DpuDetector          KV260 DPU via VART / pynq_dpu (board only)

OnnxDetector and DpuDetector share pre-processing (`letterbox`) and
post-processing (`decode_predictions`); DpuDetector additionally decodes the
raw YOLOv8 head (`yolov8_raw_to_pred`), which the DPU cannot run. Board-only
modules (vart, xir, pynq_dpu) are imported inside DpuDetector only, so this
file imports and its shared code is testable anywhere.
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Detections:
    """Detections for one frame, in original-frame pixel coordinates.

    boxes:     (N, 4) float32, xyxy
    scores:    (N,)   float32, confidence in [0, 1]
    class_ids: (N,)   int
    names:     class id -> label
    """

    boxes: np.ndarray
    scores: np.ndarray
    class_ids: np.ndarray
    names: dict

    def __len__(self):
        return len(self.boxes)

    @property
    def labels(self):
        return [self.names[int(c)] for c in self.class_ids]


class Detector(ABC):
    #: Milliseconds spent in each stage of the most recent detect() call.
    timings: dict = {"pre": 0.0, "inference": 0.0, "post": 0.0}

    @abstractmethod
    def detect(self, frame: np.ndarray) -> Detections:
        """Run detection on a single BGR frame."""


class UltralyticsDetector(Detector):
    """YOLO detection via ultralytics.

    conf=0.1 matches what model.track() uses internally: ByteTrack needs the
    low-confidence detections for its second association stage.
    """

    def __init__(self, weights="yolo11n.pt", conf=0.1, iou=0.7, imgsz=640):
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz

    def detect(self, frame):
        result = self.model.predict(
            frame,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            verbose=False,
        )[0]
        self.timings = {
            "pre": result.speed["preprocess"],
            "inference": result.speed["inference"],
            "post": result.speed["postprocess"],
        }
        boxes = result.boxes.cpu().numpy()
        return Detections(
            boxes=boxes.xyxy.astype(np.float32),
            scores=boxes.conf.astype(np.float32),
            class_ids=boxes.cls.astype(int),
            names=result.names,
        )


def letterbox(frame, size=640, pad_value=114):
    """Resize keeping aspect ratio, then pad to size x size (centered).

    Returns (image, gain, (pad_left, pad_top)), matching ultralytics' LetterBox
    rounding so boxes line up with what the ultralytics backend produces.
    """
    h, w = frame.shape[:2]
    gain = min(size / h, size / w)
    new_w, new_h = round(w * gain), round(h * gain)
    dw, dh = (size - new_w) / 2, (size - new_h) / 2

    if (new_w, new_h) != (w, h):
        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    frame = cv2.copyMakeBorder(frame, top, bottom, left, right,
                               cv2.BORDER_CONSTANT, value=(pad_value,) * 3)
    return frame, gain, (left, top)


def nms(boxes, scores, iou_thresh):
    """Greedy NMS. boxes (N, 4) xyxy, scores (N,). Returns kept indices, best first."""
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        rest = order[1:]
        iw = np.maximum(0.0, np.minimum(x2[i], x2[rest]) - np.maximum(x1[i], x1[rest]))
        ih = np.maximum(0.0, np.minimum(y2[i], y2[rest]) - np.maximum(y1[i], y1[rest]))
        inter = iw * ih
        iou = inter / (areas[i] + areas[rest] - inter + 1e-7)
        order = rest[iou <= iou_thresh]
    return np.array(keep, dtype=np.int64)


# ---------------------------------------------------------------------------
# Shared post-processing (OnnxDetector, DpuDetector)
# ---------------------------------------------------------------------------

MAX_WH = 7680       # per-class box offset for class-aware NMS in one pass
MAX_NMS_IN = 30000  # cap on candidates fed to NMS
MAX_DET = 300


def decode_predictions(pred, conf, iou, gain, pad, frame_hw, names):
    """YOLO predictions -> Detections in original-frame pixels.

    pred:     (anchors, 4 + nc): cx, cy, w, h in letterboxed-input pixels, then
              per-class scores in [0, 1] (the ONNX export's output, transposed).
    gain/pad: from letterbox(); frame_hw: original (h, w).
    """
    cls_scores = pred[:, 4:]
    class_ids = cls_scores.argmax(1)
    scores = cls_scores[np.arange(len(pred)), class_ids]

    mask = scores > conf
    boxes, scores, class_ids = pred[mask, :4], scores[mask], class_ids[mask]
    if len(scores) > MAX_NMS_IN:
        top = scores.argsort()[::-1][:MAX_NMS_IN]
        boxes, scores, class_ids = boxes[top], scores[top], class_ids[top]

    # cx, cy, w, h -> x1, y1, x2, y2 (letterboxed input coordinates)
    xyxy = np.empty_like(boxes)
    xyxy[:, :2] = boxes[:, :2] - boxes[:, 2:] / 2
    xyxy[:, 2:] = boxes[:, :2] + boxes[:, 2:] / 2

    # Class-aware NMS: shift each class into its own region so boxes of
    # different classes never overlap.
    keep = nms(xyxy + class_ids[:, None] * MAX_WH, scores, iou)[:MAX_DET]
    xyxy, scores, class_ids = xyxy[keep], scores[keep], class_ids[keep]

    # Undo letterbox -> original frame pixels
    pad_x, pad_y = pad
    h, w = frame_hw
    xyxy[:, [0, 2]] = ((xyxy[:, [0, 2]] - pad_x) / gain).clip(0, w)
    xyxy[:, [1, 3]] = ((xyxy[:, [1, 3]] - pad_y) / gain).clip(0, h)

    return Detections(
        boxes=xyxy.astype(np.float32),
        scores=scores.astype(np.float32),
        class_ids=class_ids.astype(int),
        names=names,
    )


def yolov8_raw_to_pred(maps, imgsz, reg_max=16):
    """Decode raw YOLOv8/YOLO11 Detect-head maps into `decode_predictions` input.

    maps: per-stride feature maps, NHWC without batch, i.e. (H, W, C) arrays,
          in any order. Either one map per stride with C = 4 * reg_max + nc
          (box DFL logits first, then class logits), or two per stride (box
          map with 4 * reg_max channels and class map with nc channels).
    Returns (anchors, 4 + nc): cx, cy, w, h in input pixels and sigmoid class
    scores, anchors ordered stride 8, 16, 32 and row-major within a map,
    exactly like the ONNX export.
    """
    nbox = 4 * reg_max
    by_size = {}
    for m in maps:
        by_size.setdefault(m.shape[:2], []).append(m)

    bins = np.arange(reg_max, dtype=np.float32)
    preds = []
    for (h, w), parts in sorted(by_size.items(), key=lambda kv: -kv[0][0]):
        parts.sort(key=lambda m: m.shape[2] != nbox)  # box map first
        m = np.concatenate(parts, axis=2).reshape(h * w, -1).astype(np.float32)
        stride = imgsz / h

        # DFL: softmax over reg_max bins -> expected distance per side (l, t, r, b)
        logits = m[:, :nbox].reshape(-1, 4, reg_max)
        logits = logits - logits.max(axis=2, keepdims=True)
        p = np.exp(logits)
        dist = (p / p.sum(axis=2, keepdims=True)) @ bins          # (h*w, 4)

        gy, gx = np.mgrid[0:h, 0:w]
        ax, ay = gx.ravel() + 0.5, gy.ravel() + 0.5
        x1, y1 = ax - dist[:, 0], ay - dist[:, 1]
        x2, y2 = ax + dist[:, 2], ay + dist[:, 3]
        box = np.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], axis=1) * stride

        scores = 1.0 / (1.0 + np.exp(-m[:, nbox:]))
        preds.append(np.concatenate([box, scores], axis=1))
    return np.concatenate(preds, axis=0)


def rgb_unit(img):
    """Letterboxed BGR uint8 -> RGB float32 in [0, 1], HWC."""
    return img[:, :, ::-1].astype(np.float32) / 255.0


# ---------------------------------------------------------------------------
# ONNX Runtime (CPU)
# ---------------------------------------------------------------------------

class OnnxDetector(Detector):
    """YOLOv8/YOLO11 detection from an ONNX export, via onnxruntime.

    Pre-processing (letterbox), box decoding and NMS are plain NumPy/OpenCV,
    with no ultralytics dependency. Expects the standard export: input
    (1, 3, S, S) RGB float in [0, 1], output (1, 4 + num_classes, num_anchors)
    with rows cx, cy, w, h, class scores.

    conf=0.1 for the same reason as UltralyticsDetector: ByteTrack's second
    association stage needs the low-confidence boxes.
    """

    def __init__(self, weights="yolov8n.onnx", conf=0.1, iou=0.7, names=None):
        import ast

        import onnxruntime as ort

        self.session = ort.InferenceSession(weights, providers=["CPUExecutionProvider"])
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.imgsz = int(inp.shape[2])
        self.conf = conf
        self.iou = iou

        meta = self.session.get_modelmeta().custom_metadata_map
        if names is None and "names" in meta:
            names = ast.literal_eval(meta["names"])
        num_classes = self.session.get_outputs()[0].shape[1] - 4
        self.names = names or default_names(num_classes)

    def detect(self, frame):
        t0 = time.perf_counter()
        img, gain, pad = letterbox(frame, self.imgsz)
        blob = np.ascontiguousarray(rgb_unit(img).transpose(2, 0, 1)[None])  # NCHW

        t1 = time.perf_counter()
        out = self.session.run(None, {self.input_name: blob})[0]

        t2 = time.perf_counter()
        dets = decode_predictions(out[0].T, self.conf, self.iou, gain, pad,
                                  frame.shape[:2], self.names)
        t3 = time.perf_counter()

        self.timings = {
            "pre": (t1 - t0) * 1e3,
            "inference": (t2 - t1) * 1e3,
            "post": (t3 - t2) * 1e3,
        }
        return dets


# ---------------------------------------------------------------------------
# KV260 DPU (Vitis AI)
# ---------------------------------------------------------------------------

COCO_NAMES = (
    "person bicycle car motorcycle airplane bus train truck boat traffic_light "
    "fire_hydrant stop_sign parking_meter bench bird cat dog horse sheep cow "
    "elephant bear zebra giraffe backpack umbrella handbag tie suitcase frisbee "
    "skis snowboard sports_ball kite baseball_bat baseball_glove skateboard "
    "surfboard tennis_racket bottle wine_glass cup fork knife spoon bowl banana "
    "apple sandwich orange broccoli carrot hot_dog pizza donut cake chair couch "
    "potted_plant bed dining_table toilet tv laptop mouse remote keyboard "
    "cell_phone microwave oven toaster sink refrigerator book clock vase "
    "scissors teddy_bear hair_drier toothbrush"
).split()


def default_names(num_classes):
    if num_classes == len(COCO_NAMES):
        return {i: n.replace("_", " ") for i, n in enumerate(COCO_NAMES)}
    return {i: str(i) for i in range(num_classes)}


def _tensor_format(tensor):
    """(numpy dtype, scale) for a VART tensor.

    Quantized DPU tensors are int8 with a power-of-two scale given by the
    `fix_point` attribute: real = int8 * 2**-fix_point. Float tensors (some
    pynq_dpu builds) need no scaling.
    """
    if "float" in str(getattr(tensor, "dtype", "")).lower():
        return np.float32, 1.0
    return np.int8, 2.0 ** tensor.get_attr("fix_point")


def create_dpu_runner(xmodel, bitstream=None):
    """VART runner for the DPU subgraph of `xmodel`. Board-only imports live here.

    bitstream given: load the overlay with pynq_dpu (Kria-PYNQ), which also
    loads the model. Otherwise the DPU must already be loaded (e.g. with
    `sudo xmutil loadapp <app>` on Ubuntu) and VART is used directly.
    """
    try:
        if bitstream:
            from pynq_dpu import DpuOverlay
        else:
            import vart
            import xir
    except ImportError as e:
        raise SystemExit(
            f"DPU backend needs Vitis AI runtime ({e.name} not found): run on the "
            "KV260 with VART/pynq_dpu installed, or use --backend onnx"
        ) from e

    if bitstream:
        overlay = DpuOverlay(bitstream)
        overlay.load_model(xmodel)
        return overlay.runner

    graph = xir.Graph.deserialize(xmodel)
    subgraphs = [
        s for s in graph.get_root_subgraph().toposort_child_subgraph()
        if s.has_attr("device") and s.get_attr("device").upper() == "DPU"
    ]
    if len(subgraphs) != 1:
        raise RuntimeError(f"{xmodel}: expected 1 DPU subgraph, found {len(subgraphs)}")
    return vart.Runner.create_runner(subgraphs[0], "run")


class DpuDetector(Detector):
    """YOLOv8/YOLO11 on the KV260 DPU (DPUCZDX8G) via VART.

    Expected .xmodel (Vitis AI quantized + compiled for the KV260 DPU):
      input   (1, S, S, 3) NHWC, RGB scaled to [0, 1] before quantization
      outputs the raw Detect-head maps per stride, NHWC: (1, S/s, S/s,
              4 * reg_max + nc), or box and class maps separately. The DFL
              softmax, sigmoid and anchor decoding run on the CPU here.
    Post-processing is the same `decode_predictions` OnnxDetector uses.

    `runner` can be any object with the VART runner interface
    (get_input_tensors, get_output_tensors, execute_async, wait); tests pass a
    fake one, so everything except `create_dpu_runner` runs without a board.
    """

    def __init__(self, weights="yolov8n.xmodel", conf=0.1, iou=0.7, names=None,
                 bitstream=None, runner=None, reg_max=16):
        self.runner = runner if runner is not None else create_dpu_runner(weights, bitstream)
        self.conf = conf
        self.iou = iou
        self.reg_max = reg_max

        in_tensor = self.runner.get_input_tensors()[0]
        in_dims = tuple(in_tensor.dims)
        if len(in_dims) != 4 or in_dims[3] != 3 or in_dims[1] != in_dims[2]:
            raise ValueError(f"expected square NHWC RGB input, got {in_dims}")
        self.imgsz = in_dims[1]
        in_dtype, self.in_scale = _tensor_format(in_tensor)
        self._inputs = [np.empty(in_dims, dtype=in_dtype, order="C")]

        self._outputs, self._out_scales = [], []
        for t in self.runner.get_output_tensors():
            dtype, scale = _tensor_format(t)
            self._outputs.append(np.empty(tuple(t.dims), dtype=dtype, order="C"))
            self._out_scales.append(scale)

        nc = sum(o.shape[3] for o in self._outputs if o.shape[1] == self._outputs[0].shape[1])
        self.names = names or default_names(nc - 4 * reg_max)

    def detect(self, frame):
        t0 = time.perf_counter()
        img, gain, pad = letterbox(frame, self.imgsz)
        x = rgb_unit(img)
        buf = self._inputs[0]
        if buf.dtype == np.int8:
            np.clip(np.round(x * self.in_scale), -128, 127, out=x)
        buf[0] = x

        t1 = time.perf_counter()
        job = self.runner.execute_async(self._inputs, self._outputs)
        self.runner.wait(job)

        t2 = time.perf_counter()
        maps = [o[0].astype(np.float32) / s for o, s in zip(self._outputs, self._out_scales)]
        pred = yolov8_raw_to_pred(maps, self.imgsz, self.reg_max)
        dets = decode_predictions(pred, self.conf, self.iou, gain, pad,
                                  frame.shape[:2], self.names)
        t3 = time.perf_counter()

        self.timings = {
            "pre": (t1 - t0) * 1e3,
            "inference": (t2 - t1) * 1e3,
            "post": (t3 - t2) * 1e3,
        }
        return dets
