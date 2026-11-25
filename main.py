"""Spatial awareness pipeline: camera -> detector -> ByteTrack -> object memory -> UI."""

import argparse
import time

import cv2

from detector import UltralyticsDetector
from memory import ObjectMemory
from tracker import ByteTracker

WINDOW = "Spatial Awareness - Stable IDs"


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
    parser.add_argument("--weights", default="yolo11n.pt")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="stop after N frames (0 = run until 'q')")
    parser.add_argument("--no-display", action="store_true")
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open source {args.source!r}")

    detector = UltralyticsDetector(args.weights)
    tracker = ByteTracker()
    memory = ObjectMemory()

    frames = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            now = time.monotonic()

            dets = detector.detect(frame)
            tracks = tracker.update(dets, frame)
            detections = memory.update(tracks, now)

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


if __name__ == "__main__":
    main()
