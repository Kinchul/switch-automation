from __future__ import annotations

import json
import sys
import threading
import time
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from socket import timeout as SocketTimeout
from typing import Any, Callable, ClassVar

from ..capture import encode_rgb_frame
from .base import FrameSink


HTML_PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Switch Automation Camera Feed</title>
  <style>
    body { background: #111; color: #eee; font-family: sans-serif; margin: 0; padding: 1rem; }
    img { max-width: 100%; height: auto; border: 1px solid #333; }
    code { background: #222; padding: 0.1rem 0.3rem; }
  </style>
</head>
<body>
  <h1>Switch Automation Camera Feed</h1>
  <p>MJPEG stream: <code>/stream.mjpg</code></p>
  <img src="/stream.mjpg" alt="Camera feed">
</body>
</html>
"""


class _PreviewRequestHandler(BaseHTTPRequestHandler):
    server_version = "SwitchAutomationPreview/1.0"
    sink: ClassVar["MjpegSink"]

    def do_GET(self) -> None:
        if self.path in {"/", "/index.html"}:
            data = HTML_PAGE.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        if self.path == "/frame.jpg":
            jpeg = self.sink.get_latest_jpeg(wait=True)
            if not jpeg:
                self.send_error(HTTPStatus.SERVICE_UNAVAILABLE, "Preview frame not ready")
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(jpeg)))
            self.end_headers()
            self.wfile.write(jpeg)
            return

        if self.path == "/health":
            jpeg = self.sink.get_latest_jpeg(wait=False)
            data = json.dumps(
                {
                    "ready": bool(jpeg),
                    "bytes": len(jpeg),
                    "last_error": self.sink._last_error,
                }
            ).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        if self.path == "/stream.mjpg":
            self.send_response(HTTPStatus.OK)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            self.connection.settimeout(5.0)

            try:
                while not self.sink.stop_event.is_set():
                    jpeg = self.sink.get_latest_jpeg(wait=True)
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii"))
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, SocketTimeout):
                return
            return

        print(f"Preview 404 for path: {self.path}", file=sys.stderr)
        self.send_error(HTTPStatus.NOT_FOUND)

    def log_message(self, format: str, *args) -> None:
        return


@dataclass(slots=True)
class MjpegSink(FrameSink):
    """HTTP MJPEG output sink.

    Encodes the rendered frame to JPEG at the requested ``target_fps`` and
    serves it on ``host:port``. Pipeline ticks faster than ``target_fps`` are
    silently skipped — encoding is the expensive step.
    """

    name: str = "mjpeg"
    host: str = "0.0.0.0"
    port: int = 8080
    target_fps: float = 5.0
    quality: int = 80
    overlay_transform: Callable[[Any], Any] | None = None
    _latest_jpeg: bytes = field(init=False, default=b"", repr=False)
    _latest_lock: threading.Lock = field(init=False, default_factory=threading.Lock, repr=False)
    _frame_ready: threading.Event = field(init=False, default_factory=threading.Event, repr=False)
    stop_event: threading.Event = field(init=False, default_factory=threading.Event, repr=False)
    _http_server: ThreadingHTTPServer | None = field(init=False, default=None, repr=False)
    _http_thread: threading.Thread | None = field(init=False, default=None, repr=False)
    _last_error: str | None = field(init=False, default=None, repr=False)
    _last_encoded_at: float = field(init=False, default=0.0, repr=False)
    _interval: float = field(init=False, default=0.2, repr=False)

    def start(self) -> None:
        if self._http_server is not None:
            # Already started — the runner pre-binds the socket to resolve
            # EADDRINUSE conflicts before handing the sink to the pipeline,
            # so a second start() from OutputPipeline.start() must be a no-op.
            return
        handler = type("PreviewHandler", (_PreviewRequestHandler,), {})
        handler.sink = self
        self._http_server = ThreadingHTTPServer((self.host, self.port), handler)
        self._http_server.daemon_threads = True
        self._http_thread = threading.Thread(target=self._http_server.serve_forever, daemon=True)
        self._http_thread.start()
        self._interval = 1.0 / self.target_fps if self.target_fps > 0 else 0.2

    def transform_overlay(self, overlay):
        if self.overlay_transform is None:
            return None
        return self.overlay_transform(overlay)

    def consume(self, frame: Any) -> None:
        now = time.monotonic()
        if (now - self._last_encoded_at) < self._interval:
            return
        try:
            jpeg = encode_rgb_frame(frame, quality=self.quality)
        except Exception as exc:
            message = f"MJPEG encode error: {exc}"
            if message != self._last_error:
                print(message, file=sys.stderr)
                self._last_error = message
            return
        with self._latest_lock:
            self._latest_jpeg = jpeg
            self._last_error = None
        self._frame_ready.set()
        self._last_encoded_at = now

    def close(self) -> None:
        self.stop_event.set()
        self._frame_ready.set()
        if self._http_server is not None:
            self._http_server.shutdown()
            self._http_server.server_close()
        if self._http_thread is not None:
            self._http_thread.join(timeout=2.0)
        self._http_server = None
        self._http_thread = None

    def get_latest_jpeg(self, *, wait: bool = False) -> bytes:
        deadline = time.monotonic() + 5.0 if wait else time.monotonic()
        while True:
            if wait and not self._frame_ready.is_set():
                self._frame_ready.wait(timeout=0.5)
            with self._latest_lock:
                jpeg = self._latest_jpeg
            if jpeg or not wait or time.monotonic() >= deadline:
                return jpeg
            time.sleep(0.05)
