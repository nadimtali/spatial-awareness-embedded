
import cv2
import time
import math
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

# Our own persistent object records
memory = {}
tracker_to_object = {}
next_object_id = 1

# Initial matching parameters
MAX_DISTANCE = 130  # Pixels
MAX_GAP = 8          # Seconds

results = model.track(
    source=0,
    stream=True,
    persist=True,
    tracker="bytetrack.yaml",
    verbose=False
)

for result in results:
    now = time.monotonic()
    frame = result.orig_img.copy()

    # Mark objects missing until observed again
    for obj in memory.values():
        obj["visible"] = False

    detections = []
    boxes = result.boxes

    if boxes is not None and boxes.id is not None:
        for tid, cls, coords in zip(
            boxes.id.int().cpu().tolist(),
            boxes.cls.int().cpu().tolist(),
            boxes.xyxy.cpu().tolist()
        ):
            x1, y1, x2, y2 = map(int, coords)
            center = ((x1 + x2) // 2, (y1 + y2) // 2)

            detections.append({
                "track_id": tid,
                "label": result.names[cls],
                "box": (x1, y1, x2, y2),
                "position": center
            })

    # First, preserve existing tracker mappings
    used_ids = set()
    unmatched = []

    for det in detections:
        tid = det["track_id"]
        obj_id = tracker_to_object.get(tid)

        if (obj_id in memory
                and memory[obj_id]["label"] == det["label"]
                and obj_id not in used_ids):
            det["object_id"] = obj_id
            used_ids.add(obj_id)
        else:
            tracker_to_object.pop(tid, None)
            unmatched.append(det)

    # Try to reconnect newly assigned tracker IDs
    for det in unmatched:
        best_id = None
        best_distance = MAX_DISTANCE + 1

        for obj_id, obj in memory.items():
            if obj_id in used_ids:
                continue
            if obj["label"] != det["label"]:
                continue
            if now - obj["last_seen"] > MAX_GAP:
                continue

            distance = math.dist(
                det["position"], obj["position"]
            )

            if distance < best_distance:
                best_distance = distance
                best_id = obj_id

        if best_id is not None:
            obj_id = best_id

            # Retire previous tracker mapping
            for old_tid, old_id in list(tracker_to_object.items()):
                if old_id == obj_id:
                    del tracker_to_object[old_tid]

            print(f"RECONNECTED: object #{obj_id}")
        else:
            obj_id = next_object_id
            next_object_id += 1
            print(f"NEW OBJECT: #{obj_id}")

        tracker_to_object[det["track_id"]] = obj_id
        det["object_id"] = obj_id
        used_ids.add(obj_id)

    # Update our memory and draw our own IDs
    for det in detections:
        obj_id = det["object_id"]

        memory[obj_id] = {
            "label": det["label"],
            "position": det["position"],
            "last_seen": now,
            "visible": True
        }

        x1, y1, x2, y2 = det["box"]

        cv2.rectangle(
            frame, (x1, y1), (x2, y2),
            (0, 255, 0), 2
        )

        cv2.putText(
            frame,
            f"ID {obj_id}: {det['label']}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (0, 255, 0), 2
        )

    # Display a small object-memory panel
    h, w = frame.shape[:2]
    display = cv2.copyMakeBorder(
        frame, 0, 0, 0, 340,
        cv2.BORDER_CONSTANT,
        value=(35, 35, 35)
    )

    cv2.putText(
        display, "OBJECT MEMORY",
        (w + 10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6, (255, 255, 255), 2
    )

    for i, (obj_id, obj) in enumerate(
        list(memory.items())[-8:]
    ):
        y = 65 + 45 * i

        if y + 20 >= h:
            break

        status = "VISIBLE" if obj["visible"] else "LAST SEEN"
        cv2.putText(
            display,
            f"#{obj_id} {obj['label']} - {status}",
            (w + 10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43, (255, 255, 255), 1
        )

        age = now - obj["last_seen"]
        cv2.putText(
            display,
            f"Position {obj['position']} | {age:.1f}s",
            (w + 10, y + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38, (180, 180, 180), 1
        )

    cv2.imshow("Spatial Awareness - Stable IDs", display)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()


