"""Object detection backends.

Every backend maps a BGR frame to a `Detections` object, so the rest of the
pipeline (tracker, memory, UI) never depends on where detection runs. Today
that is ultralytics on the host; on the KV260 it will be a DPU backend.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

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
        boxes = result.boxes.cpu().numpy()
        return Detections(
            boxes=boxes.xyxy.astype(np.float32),
            scores=boxes.conf.astype(np.float32),
            class_ids=boxes.cls.astype(int),
            names=result.names,
        )
