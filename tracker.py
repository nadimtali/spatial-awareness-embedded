"""ByteTrack over plain detections (NumPy + lap, no ultralytics).

Tracking logic follows the original ByteTrack (Zhang et al., "ByteTrack:
Multi-Object Tracking by Associating Every Detection Box", ECCV 2022),
https://github.com/ifzhang/ByteTrack, MIT License, Copyright (c) 2021 Yifu
Zhang. The Kalman filter is written here from the standard constant-velocity
model on (cx, cy, aspect, h) with the noise weights used by ByteTrack.

Settings match ultralytics' bytetrack.yaml (what the pipeline used before),
including separate high/low/new-track thresholds and per-track class labels.
"""

from dataclasses import dataclass

import lap
import numpy as np

from detector import Detections

BYTETRACK_CFG = dict(
    track_high_thresh=0.25,  # first association uses score > this
    track_low_thresh=0.1,    # second association uses low < score < high
    new_track_thresh=0.25,   # unmatched detection must reach this to start a track
    track_buffer=30,         # frames a lost track is kept (at 30 fps)
    match_thresh=0.8,        # max (1 - IoU) cost in the first association
    fuse_score=True,         # weight IoU by detection score in first association
    frame_rate=30,
)


@dataclass
class Track:
    track_id: int
    box: tuple  # (x1, y1, x2, y2) ints
    class_id: int
    label: str
    score: float


# --------------------------------------------------------------------------
# Kalman filter
# --------------------------------------------------------------------------

class KalmanFilterXYAH:
    """Constant-velocity Kalman filter on z = (cx, cy, a, h), a = w / h.

    State is (cx, cy, a, h, vcx, vcy, va, vh). Position/velocity noise scales
    with box height so uncertainty is relative to object size.
    """

    STD_POS = 1.0 / 20
    STD_VEL = 1.0 / 160

    def __init__(self):
        self.F = np.eye(8)
        self.F[:4, 4:] = np.eye(4)   # dt = 1 frame
        self.H = np.eye(4, 8)

    def initiate(self, z):
        mean = np.r_[z, np.zeros(4)]
        h, p, v = z[3], self.STD_POS, self.STD_VEL
        std = [2 * p * h, 2 * p * h, 1e-2, 2 * p * h,
               10 * v * h, 10 * v * h, 1e-5, 10 * v * h]
        return mean, np.diag(np.square(std))

    def multi_predict(self, means, covs):
        """Predict N states at once. means (N, 8), covs (N, 8, 8)."""
        h, p, v = means[:, 3], self.STD_POS, self.STD_VEL
        small = np.ones_like(h)
        std = np.stack([p * h, p * h, 1e-2 * small, p * h,
                        v * h, v * h, 1e-5 * small, v * h], axis=1)
        Q = np.zeros_like(covs)
        idx = np.arange(8)
        Q[:, idx, idx] = std ** 2
        means = means @ self.F.T
        covs = self.F @ covs @ self.F.T + Q
        return means, covs

    def update(self, mean, cov, z):
        h, p = mean[3], self.STD_POS
        R = np.diag(np.square([p * h, p * h, 1e-1, p * h]))
        S = self.H @ cov @ self.H.T + R           # innovation covariance (4, 4)
        PHt = cov @ self.H.T                      # (8, 4)
        K = np.linalg.solve(S, PHt.T).T           # gain = P H^T S^-1
        mean = mean + K @ (z - self.H @ mean)
        cov = cov - K @ S @ K.T
        return mean, cov


# --------------------------------------------------------------------------
# Single track
# --------------------------------------------------------------------------

class TrackState:
    New, Tracked, Lost, Removed = range(4)


def xyxy_to_xyah(b):
    w, h = b[2] - b[0], b[3] - b[1]
    return np.array([b[0] + w / 2, b[1] + h / 2, w / h, h])


class STrack:
    def __init__(self, xyxy, score, cls):
        self._xyxy = np.asarray(xyxy, dtype=np.float64)
        self.score = float(score)
        self.cls = int(cls)
        self.mean = self.covariance = None
        self.kf = None
        self.track_id = 0
        self.state = TrackState.New
        self.is_activated = False
        self.frame_id = 0
        self.start_frame = 0
        self.tracklet_len = 0

    @property
    def xyxy(self):
        if self.mean is None:
            return self._xyxy.copy()
        cx, cy, a, h = self.mean[:4]
        w = a * h
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])

    def activate(self, kf, frame_id, track_id):
        self.kf = kf
        self.track_id = track_id
        self.mean, self.covariance = kf.initiate(xyxy_to_xyah(self._xyxy))
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        # A new track is confirmed only on the very first frame; otherwise it
        # must be matched again next frame (see unconfirmed handling).
        self.is_activated = frame_id == 1
        self.frame_id = self.start_frame = frame_id

    def _absorb(self, det, frame_id):
        self.mean, self.covariance = self.kf.update(
            self.mean, self.covariance, xyxy_to_xyah(det._xyxy))
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        self.score = det.score
        self.cls = det.cls

    def update(self, det, frame_id):
        self.tracklet_len += 1
        self._absorb(det, frame_id)

    def re_activate(self, det, frame_id):
        self.tracklet_len = 0
        self._absorb(det, frame_id)


def multi_predict(kf, tracks):
    if not tracks:
        return
    means = np.stack([t.mean for t in tracks]).copy()
    covs = np.stack([t.covariance for t in tracks])
    for i, t in enumerate(tracks):
        if t.state != TrackState.Tracked:
            means[i, 7] = 0  # freeze height velocity while lost
    means, covs = kf.multi_predict(means, covs)
    for t, m, c in zip(tracks, means, covs):
        t.mean, t.covariance = m, c


# --------------------------------------------------------------------------
# Association
# --------------------------------------------------------------------------

def iou_matrix(a, b):
    """Pairwise IoU between (N, 4) and (M, 4) xyxy arrays."""
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(br - tl, 0, None).prod(2)
    area_a = (a[:, 2:] - a[:, :2]).prod(1)
    area_b = (b[:, 2:] - b[:, :2]).prod(1)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-7)


def iou_distance(tracks, dets):
    if not tracks or not dets:
        return np.zeros((len(tracks), len(dets)))
    return 1 - iou_matrix(np.stack([t.xyxy for t in tracks]),
                          np.stack([d.xyxy for d in dets]))


def fuse_score(cost, dets):
    if cost.size == 0:
        return cost
    scores = np.array([d.score for d in dets])
    return 1 - (1 - cost) * scores[None, :]


def linear_assignment(cost, thresh):
    """Min-cost matching with pairs above thresh disallowed.

    Returns (matches (K, 2), unmatched rows, unmatched cols).
    """
    if cost.size == 0:
        return (np.empty((0, 2), dtype=int),
                np.arange(cost.shape[0]), np.arange(cost.shape[1]))
    _, x, y = lap.lapjv(cost, extend_cost=True, cost_limit=thresh)
    matches = np.array([[i, j] for i, j in enumerate(x) if j >= 0], dtype=int)
    return matches.reshape(-1, 2), np.where(x < 0)[0], np.where(y < 0)[0]


def joint_stracks(a, b):
    seen = {t.track_id for t in a}
    return a + [t for t in b if t.track_id not in seen]


def sub_stracks(a, b):
    drop = {t.track_id for t in b}
    return [t for t in a if t.track_id not in drop]


def remove_duplicate_stracks(a, b, frame_id):
    """Where a tracked and a lost track overlap heavily, keep the older one."""
    dist = iou_distance(a, b)
    dup_a, dup_b = set(), set()
    for p, q in zip(*np.where(dist < 0.15)):
        age_p = frame_id - a[p].start_frame
        age_q = frame_id - b[q].start_frame
        if age_p > age_q:
            dup_b.add(q)
        else:
            dup_a.add(p)
    return ([t for i, t in enumerate(a) if i not in dup_a],
            [t for i, t in enumerate(b) if i not in dup_b])


# --------------------------------------------------------------------------
# ByteTrack
# --------------------------------------------------------------------------

class BYTETracker:
    MAX_REMOVED = 1000  # bound memory of the removed-track list

    def __init__(self, track_high_thresh, track_low_thresh, new_track_thresh,
                 track_buffer, match_thresh, fuse_score, frame_rate):
        self.high = track_high_thresh
        self.low = track_low_thresh
        self.new_thresh = new_track_thresh
        self.match_thresh = match_thresh
        self.fuse = fuse_score
        self.max_time_lost = int(frame_rate / 30.0 * track_buffer)
        self.kf = KalmanFilterXYAH()
        self.reset()

    def reset(self):
        self.tracked_stracks, self.lost_stracks, self.removed_stracks = [], [], []
        self.frame_id = 0
        self._next_id = 0

    def _new_id(self):
        self._next_id += 1
        return self._next_id

    def update(self, boxes, scores, classes):
        """One frame of xyxy boxes, scores, class ids -> confirmed STracks."""
        self.frame_id += 1
        activated, refind, lost, removed = [], [], [], []

        high = scores > self.high
        low = (scores > self.low) & (scores < self.high)
        dets = [STrack(b, s, c) for b, s, c in zip(boxes[high], scores[high], classes[high])]
        dets_low = [STrack(b, s, c) for b, s, c in zip(boxes[low], scores[low], classes[low])]

        unconfirmed = [t for t in self.tracked_stracks if not t.is_activated]
        confirmed = [t for t in self.tracked_stracks if t.is_activated]

        # Step 1: confirmed + lost tracks vs high-score detections
        pool = joint_stracks(confirmed, self.lost_stracks)
        multi_predict(self.kf, pool)
        dists = iou_distance(pool, dets)
        if self.fuse:
            dists = fuse_score(dists, dets)
        matches, u_track, u_det = linear_assignment(dists, self.match_thresh)
        for it, idet in matches:
            track, det = pool[it], dets[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated.append(track)
            else:
                track.re_activate(det, self.frame_id)
                refind.append(track)

        # Step 2: remaining tracked tracks vs low-score detections
        r_tracked = [pool[i] for i in u_track if pool[i].state == TrackState.Tracked]
        dists = iou_distance(r_tracked, dets_low)
        matches, u_track, _ = linear_assignment(dists, 0.5)
        for it, idet in matches:
            track, det = r_tracked[it], dets_low[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated.append(track)
            else:
                track.re_activate(det, self.frame_id)
                refind.append(track)
        for it in u_track:
            track = r_tracked[it]
            if track.state != TrackState.Lost:
                track.state = TrackState.Lost
                lost.append(track)

        # Step 3: unconfirmed tracks vs leftover high-score detections
        dets = [dets[i] for i in u_det]
        dists = iou_distance(unconfirmed, dets)
        if self.fuse:
            dists = fuse_score(dists, dets)
        matches, u_unconfirmed, u_det = linear_assignment(dists, 0.7)
        for it, idet in matches:
            unconfirmed[it].update(dets[idet], self.frame_id)
            activated.append(unconfirmed[it])
        for it in u_unconfirmed:
            unconfirmed[it].state = TrackState.Removed
            removed.append(unconfirmed[it])

        # Step 4: start new tracks
        for idet in u_det:
            det = dets[idet]
            if det.score < self.new_thresh:
                continue
            det.activate(self.kf, self.frame_id, self._new_id())
            activated.append(det)

        # Step 5: drop tracks lost for too long
        for track in self.lost_stracks:
            if self.frame_id - track.frame_id > self.max_time_lost:
                track.state = TrackState.Removed
                removed.append(track)

        self.tracked_stracks = [t for t in self.tracked_stracks
                                if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks = (self.removed_stracks + removed)[-self.MAX_REMOVED:]
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(
            self.tracked_stracks, self.lost_stracks, self.frame_id)

        return [t for t in self.tracked_stracks if t.is_activated]


class ByteTracker:
    def __init__(self, **overrides):
        self._tracker = BYTETracker(**{**BYTETRACK_CFG, **overrides})

    def update(self, dets: Detections, frame=None):
        """Feed one frame of detections; return confirmed, currently-tracked objects."""
        boxes = dets.boxes.astype(np.float64).reshape(-1, 4)
        scores = dets.scores.astype(np.float64).reshape(-1)
        classes = dets.class_ids.astype(int).reshape(-1)
        stracks = self._tracker.update(boxes, scores, classes)

        if frame is not None:
            h, w = frame.shape[:2]
        tracks = []
        for t in stracks:
            x1, y1, x2, y2 = t.xyxy
            # Kalman-smoothed boxes can leave the image; clip to the frame.
            if frame is not None:
                x1, x2 = np.clip([x1, x2], 0, w)
                y1, y2 = np.clip([y1, y2], 0, h)
            tracks.append(Track(
                track_id=t.track_id,
                box=(int(x1), int(y1), int(x2), int(y2)),
                class_id=t.cls,
                label=dets.names[t.cls],
                score=t.score,
            ))
        return tracks

    def reset(self):
        self._tracker.reset()
