"""Persistent object memory: stable IDs on top of short-lived tracker IDs.

ByteTrack IDs change when an object is lost for too long or re-detected. We
keep our own object records and reconnect a new tracker ID to a recently seen
object of the same label within MAX_DISTANCE pixels and MAX_GAP seconds.
"""

import math

MAX_DISTANCE = 130  # Pixels
MAX_GAP = 8         # Seconds


class ObjectMemory:
    def __init__(self, max_distance=MAX_DISTANCE, max_gap=MAX_GAP):
        self.max_distance = max_distance
        self.max_gap = max_gap
        self.objects = {}           # object_id -> {label, position, last_seen, visible}
        self.tracker_to_object = {}  # tracker id -> object_id
        self.next_object_id = 1

    def update(self, tracks, now):
        """Assign a stable object ID to every track and refresh memory.

        Returns a list of dicts: track_id, label, box, position, object_id.
        """
        # Mark objects missing until observed again
        for obj in self.objects.values():
            obj["visible"] = False

        detections = []
        for t in tracks:
            x1, y1, x2, y2 = t.box
            detections.append({
                "track_id": t.track_id,
                "label": t.label,
                "box": t.box,
                "position": ((x1 + x2) // 2, (y1 + y2) // 2),
            })

        # First, preserve existing tracker mappings
        used_ids = set()
        unmatched = []

        for det in detections:
            tid = det["track_id"]
            obj_id = self.tracker_to_object.get(tid)

            if (obj_id in self.objects
                    and self.objects[obj_id]["label"] == det["label"]
                    and obj_id not in used_ids):
                det["object_id"] = obj_id
                used_ids.add(obj_id)
            else:
                self.tracker_to_object.pop(tid, None)
                unmatched.append(det)

        # Try to reconnect newly assigned tracker IDs
        for det in unmatched:
            best_id = None
            best_distance = self.max_distance + 1

            for obj_id, obj in self.objects.items():
                if obj_id in used_ids:
                    continue
                if obj["label"] != det["label"]:
                    continue
                if now - obj["last_seen"] > self.max_gap:
                    continue

                distance = math.dist(det["position"], obj["position"])

                if distance < best_distance:
                    best_distance = distance
                    best_id = obj_id

            if best_id is not None:
                obj_id = best_id

                # Retire previous tracker mapping
                for old_tid, old_id in list(self.tracker_to_object.items()):
                    if old_id == obj_id:
                        del self.tracker_to_object[old_tid]

                print(f"RECONNECTED: object #{obj_id}")
            else:
                obj_id = self.next_object_id
                self.next_object_id += 1
                print(f"NEW OBJECT: #{obj_id}")

            self.tracker_to_object[det["track_id"]] = obj_id
            det["object_id"] = obj_id
            used_ids.add(obj_id)

        # Update memory
        for det in detections:
            self.objects[det["object_id"]] = {
                "label": det["label"],
                "position": det["position"],
                "last_seen": now,
                "visible": True,
            }

        return detections
