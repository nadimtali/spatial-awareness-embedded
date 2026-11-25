"""Threaded frame capture, one thread per camera or video file.

Live sources (camera index, /dev/video*, rtsp:// ...) keep only the newest
frame, so a slow pipeline processes fresh frames instead of building a backlog.
Video files hand over every frame in order (the reader waits for the pipeline),
so a recorded clip gives the same result however fast the board is.
"""

import os
import queue
import threading

import cv2


def parse_source(text):
    """'0' -> camera index 0, anything else -> path/URL."""
    return int(text) if text.isdigit() else text


class CameraThread(threading.Thread):
    def __init__(self, source, name="cam"):
        super().__init__(daemon=True, name=name)
        self.source = source
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise SystemExit(f"Could not open source {source!r}")
        self.live = not (isinstance(source, str) and os.path.isfile(source))
        self.dropped = 0  # live frames replaced before the pipeline took them
        self._queue = queue.Queue(maxsize=1)
        self._halt = threading.Event()

    def run(self):
        try:
            while not self._halt.is_set():
                ok, frame = self.cap.read()
                if not ok:
                    break
                if self.live:
                    try:
                        self._queue.get_nowait()  # discard the stale frame
                        self.dropped += 1
                    except queue.Empty:
                        pass
                    self._queue.put_nowait(frame)  # we are the only producer
                else:
                    self._put(frame)
        finally:
            self.cap.release()
            self._put(None)  # end-of-stream marker

    def _put(self, item):
        while not self._halt.is_set():
            try:
                self._queue.put(item, timeout=0.1)
                return
            except queue.Full:
                pass

    def read(self):
        """Next frame (blocks), or None once the source has ended."""
        return self._queue.get()

    def stop(self):
        self._halt.set()
        self.join(timeout=2)
