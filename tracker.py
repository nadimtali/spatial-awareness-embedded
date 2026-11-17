"""ByteTrack over plain detections.

Wraps ultralytics' BYTETracker with the same settings as its bytetrack.yaml,
so tracking behaves exactly as it did under model.track(), but the input is
our own `Detections` rather than an ultralytics Results object.
"""

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
from ultralytics.trackers.byte_tracker import BYTETracker

from detector import Detections

# Defaults from ultralytics/cfg/trackers/bytetrack.yaml
BYTETRACK_CFG = dict(
    tracker_type="bytetrack",
    track_high_thresh=0.25,
    track_low_thresh=0.1,
    new_track_thresh=0.25,
    track_buffer=30,
    match_thresh=0.8,
    fuse_score=True,
)


@dataclass
class Track:
    track_id: int
    box: tuple  # (x1, y1, x2, y2) ints
    class_id: int
    label: str
    score: float


class _DetArray:
    """Minimal duck-typed stand-in for ultralytics Boxes, as BYTETracker reads it."""

    def __init__(self, xyxy, conf, cls):
        self.xyxy = xyxy
        self.conf = conf
        self.cls = cls

    @property
    def xywh(self):
        xy = (self.xyxy[:, :2] + self.xyxy[:, 2:]) / 2
        wh = self.xyxy[:, 2:] - self.xyxy[:, :2]
        return np.concatenate([xy, wh], axis=1)

    def __len__(self):
        return len(self.conf)

    def __getitem__(self, idx):
        return _DetArray(self.xyxy[idx], self.conf[idx], self.cls[idx])


class ByteTracker:
    def __init__(self, **overrides):
        self._tracker = BYTETracker(SimpleNamespace(**{**BYTETRACK_CFG, **overrides}))

    def update(self, dets: Detections, frame=None):
        """Feed one frame of detections; return confirmed, currently-tracked objects."""
        arr = _DetArray(
            dets.boxes.astype(np.float32).reshape(-1, 4),
            dets.scores.astype(np.float32),
            dets.class_ids.astype(np.float32),
        )
        rows = self._tracker.update(arr, frame)

        # rows: [x1, y1, x2, y2, track_id, score, cls, idx]
        # Kalman-smoothed boxes can leave the image; clip like ultralytics does.
        if len(rows) and frame is not None:
            h, w = frame.shape[:2]
            rows[:, [0, 2]] = rows[:, [0, 2]].clip(0, w)
            rows[:, [1, 3]] = rows[:, [1, 3]].clip(0, h)
        tracks = []
        for x1, y1, x2, y2, tid, score, cls, _ in rows:
            cls = int(cls)
            tracks.append(Track(
                track_id=int(tid),
                box=(int(x1), int(y1), int(x2), int(y2)),
                class_id=cls,
                label=dets.names[cls],
                score=float(score),
            ))
        return tracks

    def reset(self):
        self._tracker.reset()
