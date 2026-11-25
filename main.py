"""Spatial awareness pipeline: camera -> detector -> ByteTrack -> object memory -> UI."""

import argparse
import time

import cv2

from detector import OnnxDetector, UltralyticsDetector
from memory import ObjectMemory
from tracker import ByteTracker

WINDOW = "Spatial Awareness - Stable IDs"
STAGES = ("pre", "inference", "post", "track", "memory")
DEFAULT_WEIGHTS = {"ultralytics": "yolo11n.pt", "onnx": "yolov8n.onnx"}


class StageTimer:
    """Accumulates per-stage milliseconds and reports averages plus FPS."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.totals = dict.fromkeys(STAGES, 0.0)
        self.frames = 0
        self.start = time.perf_counter()

    def add(self, ms_by_stage):
        for stage, ms in ms_by_stage.items():
            self.totals[stage] += ms
        self.frames += 1

    def elapsed(self):
        return time.perf_counter() - self.start

    def summary(self):
        n = max(self.frames, 1)
        stages = " ".join(f"{s} {self.totals[s] / n:6.1f}" for s in STAGES)
        total = sum(self.totals.values()) / n
        fps = self.frames / max(self.elapsed(), 1e-9)
        return f"[ms/frame] {stages} | total {total:6.1f} | FPS {fps:5.1f} ({self.frames} frames)"


def draw(frame, detections, memory, now):
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"ID {det['object_id']}: {det['label']}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (0, 255, 0), 2,
        )

    # Object-memory side panel
    h, w = frame.shape[:2]
    display = cv2.copyMakeBorder(
        frame, 0, 0, 0, 340, cv2.BORDER_CONSTANT, value=(35, 35, 35)
    )
    cv2.putText(
        display, "OBJECT MEMORY", (w + 10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
    )

    for i, (obj_id, obj) in enumerate(list(memory.objects.items())[-8:]):
        y = 65 + 45 * i
        if y + 20 >= h:
            break

        status = "VISIBLE" if obj["visible"] else "LAST SEEN"
        cv2.putText(
            display, f"#{obj_id} {obj['label']} - {status}", (w + 10, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 1,
        )
        age = now - obj["last_seen"]
        cv2.putText(
            display, f"Position {obj['position']} | {age:.1f}s", (w + 10, y + 18),
            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1,
        )

    return display


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0", help="camera index or video path")
    parser.add_argument("--backend", choices=DEFAULT_WEIGHTS, default="ultralytics")
    parser.add_argument("--weights", default=None,
                        help="model file (default: yolo11n.pt for ultralytics, "
                             "yolov8n.onnx for onnx)")
    parser.add_argument("--report-every", type=float, default=5.0,
                        help="seconds between timing reports")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="stop after N frames (0 = run until 'q')")
    parser.add_argument("--no-display", action="store_true")
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open source {args.source!r}")

    weights = args.weights or DEFAULT_WEIGHTS[args.backend]
    if args.backend == "onnx":
        detector = OnnxDetector(weights)
    else:
        detector = UltralyticsDetector(weights)
    tracker = ByteTracker()
    memory = ObjectMemory()

    frames = 0
    window, overall = StageTimer(), StageTimer()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            now = time.monotonic()

            dets = detector.detect(frame)
            t0 = time.perf_counter()
            tracks = tracker.update(dets, frame)
            t1 = time.perf_counter()
            detections = memory.update(tracks, now)
            t2 = time.perf_counter()

            ms = {**detector.timings,
                  "track": (t1 - t0) * 1e3, "memory": (t2 - t1) * 1e3}
            window.add(ms)
            overall.add(ms)
            if window.elapsed() >= args.report_every:
                print(window.summary(), flush=True)
                window.reset()

            frames += 1
            if args.max_frames and frames >= args.max_frames:
                break
            if args.no_display:
                continue

            cv2.imshow(WINDOW, draw(frame.copy(), detections, memory, now))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if overall.frames:
            print(f"OVERALL {args.backend} ({weights}): {overall.summary()}")


if __name__ == "__main__":
    main()
