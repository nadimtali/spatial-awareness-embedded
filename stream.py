"""MJPEG-over-HTTP viewer for headless boards (standard library + OpenCV only).

    http://<board-ip>:8080/              page with the live view
    http://<board-ip>:8080/stream        multipart/x-mixed-replace MJPEG stream
    http://<board-ip>:8080/snapshot.jpg  one JPEG

Frames are JPEG-encoded only while at least one client is connected.
"""

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

BOUNDARY = b"frame"
PAGE = b"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Spatial Awareness</title>
<style>body{margin:0;background:#111}img{display:block;max-width:100%;margin:auto}</style>
</head><body><img src="/stream" alt="live view"></body></html>"""


class MjpegServer:
    def __init__(self, port=8080, host="0.0.0.0", quality=80):
        self.quality = quality
        self._cond = threading.Condition()
        self._jpeg = None
        self._seq = 0
        self._clients = 0
        self._closed = False

        server = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/":
                    self._send(200, "text/html; charset=utf-8", PAGE)
                elif self.path == "/stream":
                    server._serve_stream(self)
                elif self.path == "/snapshot.jpg":
                    jpeg = server._next_jpeg(timeout=10)
                    if jpeg is None:
                        self._send(503, "text/plain", b"no frame yet\n")
                    else:
                        self._send(200, "image/jpeg", jpeg)
                else:
                    self._send(404, "text/plain", b"not found\n")

            def _send(self, code, ctype, body):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass  # keep the console for timing reports

        self._httpd = ThreadingHTTPServer((host, port), Handler)
        self._httpd.daemon_threads = True
        self._thread = threading.Thread(target=self._httpd.serve_forever,
                                        name="mjpeg", daemon=True)
        self._thread.start()
        self.port = port

    @property
    def has_clients(self):
        return self._clients > 0

    def publish(self, frame):
        """Encode and hand a BGR frame to connected clients (no-op without clients)."""
        if not self._clients:
            return
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        if not ok:
            return
        with self._cond:
            self._jpeg = buf.tobytes()
            self._seq += 1
            self._cond.notify_all()

    def _next_jpeg(self, timeout):
        """Wait for the next published frame; None on timeout or shutdown."""
        with self._cond:
            self._clients += 1
            try:
                seen = self._seq
                self._cond.wait_for(lambda: self._closed or self._seq != seen, timeout)
                if self._closed or self._seq == seen:
                    return None
                return self._jpeg
            finally:
                self._clients -= 1

    def _serve_stream(self, handler):
        handler.send_response(200)
        handler.send_header("Content-Type",
                            "multipart/x-mixed-replace; boundary=" + BOUNDARY.decode())
        handler.send_header("Cache-Control", "no-cache")
        handler.end_headers()
        with self._cond:
            self._clients += 1
            seen = self._seq  # start from the next frame encoded for us
        try:
            while True:
                with self._cond:
                    self._cond.wait_for(lambda: self._closed or self._seq != seen, 5)
                    if self._closed:
                        return
                    if self._seq == seen:
                        continue
                    jpeg, seen = self._jpeg, self._seq
                handler.wfile.write(
                    b"--" + BOUNDARY + b"\r\nContent-Type: image/jpeg\r\n"
                    + f"Content-Length: {len(jpeg)}\r\n\r\n".encode() + jpeg + b"\r\n")
                handler.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass  # browser tab closed
        finally:
            with self._cond:
                self._clients -= 1

    def close(self):
        with self._cond:
            self._closed = True
            self._cond.notify_all()
        self._httpd.shutdown()
        self._httpd.server_close()
