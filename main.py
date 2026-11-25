"""Spatial awareness pipeline: camera(s) -> detector -> ByteTrack -> object memory -> UI.

One capture thread and one tracker per camera; all cameras share the detector
and one ObjectMemory. The view (camera images side by side plus the memory
panel) goes to an OpenCV window and/or an MJPEG stream on port 8080.
"""

import argparse
import time

import cv2
import numpy as np

from camera import CameraThread, parse_source
from detector import OnnxDetector, UltralyticsDetector
from memory import ObjectMemory
from tracker import ByteTracker

WINDOW = "Spatial Awareness - Stable IDs"
STAGES = ("pre", "inference", "post", "track", "memory")
DEFAULT_WEIGHTS = {"ultralytics": "yolo11n.pt", "onnx": "yolov8n.onnx"}
PANEL_WIDTH = 340


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


def draw_tracks(frame, detections):
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
    return frame


def compose(frames, memory, now):
    """Camera images side by side (scaled to one height) plus the memory panel."""
    frames = [f for f in frames if f is not None]
    h = frames[0].shape[0]
    frames = [f if f.shape[0] == h else
              cv2.resize(f, (round(f.shape[1] * h / f.shape[0]), h)) for f in frames]
    view = frames[0] if len(frames) == 1 else np.hstack(frames)
    multi_cam = len(frames) > 1

    w = view.shape[1]
    display = cv2.copyMakeBorder(
        view, 0, 0, 0, PANEL_WIDTH, cv2.BORDER_CONSTANT, value=(35, 35, 35)
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
        cam = f" cam{obj['camera']}" if multi_cam else ""
        cv2.putText(
            display, f"#{obj_id} {obj['label']}{cam} - {status}", (w + 10, y),
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
    parser.add_argument("--source", nargs="+", default=["0"],
                        help="one or more camera indices or video paths, e.g. --source 0 1")
    parser.add_argument("--backend", choices=DEFAULT_WEIGHTS, default="ultralytics")
    parser.add_argument("--weights", default=None,
                        help="model file (default: yolo11n.pt for ultralytics, "
                             "yolov8n.onnx for onnx)")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="stop after N processed frames, all cameras (0 = run until 'q')")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--stream", action="store_true",
                        help="serve the annotated view as MJPEG over HTTP")
    parser.add_argument("--stream-port", type=int, default=8080)
    parser.add_argument("--report-every", type=float, default=5.0,
                        help="seconds between timing reports")
    args = parser.parse_args()

    weights = args.weights or DEFAULT_WEIGHTS[args.backend]
    if args.backend == "onnx":
        detector = OnnxDetector(weights)
    else:
        detector = UltralyticsDetector(weights)

    cameras = [CameraThread(parse_source(s), name=f"cam{i}")
               for i, s in enumerate(args.source)]
    trackers = [ByteTracker() for _ in cameras]
    memory = ObjectMemory()

    server = None
    if args.stream:
        from stream import MjpegServer
        server = MjpegServer(port=args.stream_port)
        print(f"Streaming on http://<this-host>:{server.port}/", flush=True)

    for cam in cameras:
        cam.start()

    annotated = [None] * len(cameras)
    active = list(range(len(cameras)))
    window, overall = StageTimer(), StageTimer()
    show = not args.no_display
    try:
        while active:
            for i in list(active):
                frame = cameras[i].read()
                if frame is None:
                    active.remove(i)
                    continue
                now = time.monotonic()

                dets = detector.detect(frame)
                t0 = time.perf_counter()
                tracks = trackers[i].update(dets, frame)
                t1 = time.perf_counter()
                detections = memory.update(tracks, now, camera=i)
                t2 = time.perf_counter()

                ms = {**detector.timings,
                      "track": (t1 - t0) * 1e3, "memory": (t2 - t1) * 1e3}
                window.add(ms)
                overall.add(ms)
                if window.elapsed() >= args.report_every:
                    print(window.summary(), flush=True)
                    window.reset()

                if show or server:
                    # Cheap; JPEG encoding is skipped while nobody is watching.
                    annotated[i] = draw_tracks(frame, detections)

            if args.max_frames and overall.frames >= args.max_frames:
                break
            if not active or not any(f is not None for f in annotated):
                continue
            view = compose(annotated, memory, time.monotonic())
            if server:
                server.publish(view)
            if show:
                try:
                    cv2.imshow(WINDOW, view)
                except cv2.error as e:
                    raise SystemExit(
                        "OpenCV cannot open a window here (headless build or no "
                        "display); run with --no-display, optionally --stream"
                    ) from e
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        for cam in cameras:
            cam.stop()
        if server:
            server.close()
        if show:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass  # headless OpenCV: no windows to close
        if overall.frames:
            print(f"OVERALL {args.backend} ({weights}), {len(cameras)} source(s): "
                  f"{overall.summary()}")
            dropped = [cam.dropped for cam in cameras if cam.live]
            if any(dropped):
                print(f"Live frames dropped (pipeline slower than camera): {dropped}")


if __name__ == "__main__":
    main()
