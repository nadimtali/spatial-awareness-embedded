"""Object detection backends.

Every backend maps a BGR frame to a `Detections` object, so the rest of the
pipeline (tracker, memory, UI) never depends on where detection runs. Today
that is ultralytics on the host; on the KV260 it will be a DPU backend.
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


class OnnxDetector(Detector):
    """YOLOv8/YOLO11 detection from an ONNX export, via onnxruntime.

    Pre-processing (letterbox), box decoding and NMS are plain NumPy/OpenCV,
    with no ultralytics dependency, so this is the template for the DPU backend.
    Expects the standard export: input (1, 3, S, S) RGB float in [0, 1],
    output (1, 4 + num_classes, num_anchors) with rows cx, cy, w, h, class scores.

    conf=0.1 for the same reason as UltralyticsDetector: ByteTrack's second
    association stage needs the low-confidence boxes.
    """

    MAX_WH = 7680      # per-class box offset for class-aware NMS in one pass
    MAX_NMS_IN = 30000  # cap on candidates fed to NMS
    MAX_DET = 300

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
        self.names = names or {i: str(i) for i in range(num_classes)}

    def detect(self, frame):
        t0 = time.perf_counter()
        img, gain, (pad_x, pad_y) = letterbox(frame, self.imgsz)
        blob = img[:, :, ::-1].transpose(2, 0, 1)[None]  # BGR HWC -> RGB NCHW
        blob = np.ascontiguousarray(blob, dtype=np.float32) / 255.0

        t1 = time.perf_counter()
        out = self.session.run(None, {self.input_name: blob})[0]

        t2 = time.perf_counter()
        dets = self._postprocess(out[0], gain, pad_x, pad_y, frame.shape[:2])
        t3 = time.perf_counter()

        self.timings = {
            "pre": (t1 - t0) * 1e3,
            "inference": (t2 - t1) * 1e3,
            "post": (t3 - t2) * 1e3,
        }
        return dets

    def _postprocess(self, pred, gain, pad_x, pad_y, frame_hw):
        pred = pred.T                       # (anchors, 4 + nc)
        cls_scores = pred[:, 4:]
        class_ids = cls_scores.argmax(1)
        scores = cls_scores[np.arange(len(pred)), class_ids]

        mask = scores > self.conf
        boxes, scores, class_ids = pred[mask, :4], scores[mask], class_ids[mask]
        if len(scores) > self.MAX_NMS_IN:
            top = scores.argsort()[::-1][: self.MAX_NMS_IN]
            boxes, scores, class_ids = boxes[top], scores[top], class_ids[top]

        # cx, cy, w, h -> x1, y1, x2, y2 (letterboxed input coordinates)
        xyxy = np.empty_like(boxes)
        xyxy[:, :2] = boxes[:, :2] - boxes[:, 2:] / 2
        xyxy[:, 2:] = boxes[:, :2] + boxes[:, 2:] / 2

        # Class-aware NMS: shift each class into its own region so boxes of
        # different classes never overlap.
        keep = nms(xyxy + class_ids[:, None] * self.MAX_WH, scores, self.iou)
        keep = keep[: self.MAX_DET]
        xyxy, scores, class_ids = xyxy[keep], scores[keep], class_ids[keep]

        # Undo letterbox -> original frame pixels
        h, w = frame_hw
        xyxy[:, [0, 2]] = ((xyxy[:, [0, 2]] - pad_x) / gain).clip(0, w)
        xyxy[:, [1, 3]] = ((xyxy[:, [1, 3]] - pad_y) / gain).clip(0, h)

        return Detections(
            boxes=xyxy.astype(np.float32),
            scores=scores.astype(np.float32),
            class_ids=class_ids.astype(int),
            names=self.names,
        )
