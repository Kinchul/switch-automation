from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from socket import timeout as SocketTimeout
from typing import Callable, ClassVar
import json

from .capture import CameraCapture, encode_rgb_frame


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
    preview_server: ClassVar[MjpegPreviewServer]

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
            jpeg = self.preview_server.get_latest_jpeg(wait=True)
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
            jpeg = self.preview_server.get_latest_jpeg(wait=False)
            data = json.dumps(
                {
                    "ready": bool(jpeg),
                    "bytes": len(jpeg),
                    "last_error": self.preview_server._last_error,
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
                while not self.preview_server.stop_event.is_set():
                    jpeg = self.preview_server.get_latest_jpeg(wait=True)
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
class OverlayBox:
    x: int
    y: int
    width: int
    height: int
    label: str | None = None
    outline: tuple[int, int, int, int] = (255, 215, 0, 255)
    fill: tuple[int, int, int, int] | None = (255, 215, 0, 48)


@dataclass(slots=True)
class OverlayState:
    lines: list[str] = field(default_factory=list)
    top_left_lines: list[str] = field(default_factory=list)
    bottom_left_lines: list[str] = field(default_factory=list)
    top_right_lines: list[str] = field(default_factory=list)
    boxes: list[OverlayBox] = field(default_factory=list)
    bottom_right_lines: list[str] = field(default_factory=list)


@dataclass(slots=True)
class MjpegPreviewServer:
    capture: CameraCapture
    host: str = "0.0.0.0"
    port: int = 8080
    fps: float = 5.0
    quality: int = 80
    overlay_lines_fn: Callable[[], list[str]] | None = None
    overlay_state_fn: Callable[[], OverlayState] | None = None
    _latest_jpeg: bytes = field(init=False, default=b"", repr=False)
    _latest_lock: threading.Lock = field(init=False, default_factory=threading.Lock, repr=False)
    _frame_ready: threading.Event = field(init=False, default_factory=threading.Event, repr=False)
    stop_event: threading.Event = field(init=False, default_factory=threading.Event, repr=False)
    _http_server: ThreadingHTTPServer = field(init=False, repr=False)
    _http_thread: threading.Thread = field(init=False, repr=False)
    _capture_thread: threading.Thread = field(init=False, repr=False)
    _last_error: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        handler = type("PreviewHandler", (_PreviewRequestHandler,), {})
        handler.preview_server = self
        self._http_server = ThreadingHTTPServer((self.host, self.port), handler)
        self._http_server.daemon_threads = True
        self._http_thread = threading.Thread(target=self._http_server.serve_forever, daemon=True)
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)

    def start(self) -> MjpegPreviewServer:
        self._http_thread.start()
        self._capture_thread.start()
        return self

    def close(self) -> None:
        self.stop_event.set()
        self._http_server.shutdown()
        self._http_server.server_close()
        self._frame_ready.set()
        self._http_thread.join(timeout=2.0)
        self._capture_thread.join(timeout=2.0)

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

    def _capture_loop(self) -> None:
        interval = 1.0 / self.fps if self.fps > 0 else 0.2
        while not self.stop_event.is_set():
            started = time.monotonic()
            try:
                frame = self.capture.get_frame()
                overlay = self._current_overlay_state()
                frame = _draw_overlay(frame, overlay)
                jpeg = encode_rgb_frame(frame, quality=self.quality)
                with self._latest_lock:
                    self._latest_jpeg = jpeg
                    self._last_error = None
                self._frame_ready.set()
            except Exception as exc:
                message = f"Preview capture error: {exc}"
                if message != self._last_error:
                    print(message, file=sys.stderr)
                    self._last_error = message
                time.sleep(0.5)
                continue

            elapsed = time.monotonic() - started
            delay = max(0.0, interval - elapsed)
            if delay > 0:
                time.sleep(delay)

    def _current_overlay_state(self) -> OverlayState:
        if self.overlay_state_fn is not None:
            return self.overlay_state_fn()
        if self.overlay_lines_fn is not None:
            return OverlayState(lines=self.overlay_lines_fn())
        return OverlayState()


def _draw_overlay(frame, overlay: OverlayState):
    if (
        not overlay.lines
        and not overlay.top_left_lines
        and not overlay.bottom_left_lines
        and not overlay.top_right_lines
        and not overlay.boxes
        and not overlay.bottom_right_lines
    ):
        return frame

    from PIL import Image, ImageDraw, ImageFont
    import numpy as np

    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image, "RGBA")
    font_size = max(24, frame.shape[1] // 55)
    font = _load_overlay_font(font_size)
    small_font = _load_overlay_font(max(18, font_size - 4))

    if overlay.boxes:
        _draw_boxes(draw, overlay.boxes, font=small_font, frame_width=frame.shape[1], frame_height=frame.shape[0])

    if overlay.lines:
        _draw_corner_lines(
            draw,
            overlay.lines,
            font=font,
            frame_width=frame.shape[1],
            frame_height=frame.shape[0],
            anchor="bottom_left",
        )

    if overlay.top_left_lines:
        _draw_corner_lines(
            draw,
            overlay.top_left_lines,
            font=font,
            frame_width=frame.shape[1],
            frame_height=frame.shape[0],
            anchor="top_left",
        )

    if overlay.bottom_left_lines:
        _draw_corner_lines(
            draw,
            overlay.bottom_left_lines,
            font=font,
            frame_width=frame.shape[1],
            frame_height=frame.shape[0],
            anchor="bottom_left",
        )

    if overlay.top_right_lines:
        _draw_corner_lines(
            draw,
            overlay.top_right_lines,
            font=font,
            frame_width=frame.shape[1],
            frame_height=frame.shape[0],
            anchor="top_right",
        )

    if overlay.bottom_right_lines:
        _draw_corner_lines(
            draw,
            overlay.bottom_right_lines,
            font=font,
            frame_width=frame.shape[1],
            frame_height=frame.shape[0],
            anchor="bottom_right",
        )

    return np.ascontiguousarray(image)


def _draw_corner_lines(draw, lines: list[str], *, font, frame_width: int, frame_height: int, anchor: str) -> None:
    if not lines:
        return

    font_size = getattr(font, "size", 24)
    padding_x = max(14, font_size // 2)
    padding_y = max(10, font_size // 3)
    line_gap = max(6, font_size // 5)

    line_boxes = [draw.textbbox((0, 0), line, font=font) for line in lines]
    text_width = max((box[2] - box[0]) for box in line_boxes)
    text_height = sum((box[3] - box[1]) for box in line_boxes) + line_gap * (len(lines) - 1)
    box_width = text_width + padding_x * 2
    box_height = text_height + padding_y * 2

    if anchor == "bottom_right":
        box = (
            max(12, frame_width - box_width - 12),
            max(12, frame_height - box_height - 12),
            frame_width - 12,
            frame_height - 12,
        )
    elif anchor == "bottom_left":
        box = (
            12,
            max(12, frame_height - box_height - 12),
            12 + box_width,
            frame_height - 12,
        )
    elif anchor == "top_right":
        box = (
            max(12, frame_width - box_width - 12),
            12,
            frame_width - 12,
            12 + box_height,
        )
    else:
        box = (12, 12, 12 + box_width, 12 + box_height)

    draw.rounded_rectangle(box, radius=10, fill=(0, 0, 0, 160), outline=(255, 255, 255, 64))

    y = box[1] + padding_y
    for line, bbox in zip(lines, line_boxes, strict=False):
        draw.text((box[0] + padding_x, y), line, font=font, fill=(255, 255, 255, 255))
        y += (bbox[3] - bbox[1]) + line_gap


def _draw_boxes(draw, boxes: list[OverlayBox], *, font, frame_width: int, frame_height: int) -> None:
    for box in boxes:
        left = max(0, min(frame_width - 1, box.x))
        top = max(0, min(frame_height - 1, box.y))
        right = max(left + 1, min(frame_width, box.x + box.width))
        bottom = max(top + 1, min(frame_height, box.y + box.height))

        if box.fill is not None:
            draw.rectangle((left, top, right, bottom), fill=box.fill)
        draw.rectangle((left, top, right, bottom), outline=box.outline, width=4)

        if not box.label:
            continue

        label_bbox = draw.textbbox((0, 0), box.label, font=font)
        label_width = label_bbox[2] - label_bbox[0]
        label_height = label_bbox[3] - label_bbox[1]
        label_left = left
        label_top = max(0, top - label_height - 12)
        label_right = min(frame_width, label_left + label_width + 16)
        label_bottom = min(frame_height, label_top + label_height + 10)
        draw.rounded_rectangle(
            (label_left, label_top, label_right, label_bottom),
            radius=8,
            fill=(0, 0, 0, 170),
            outline=box.outline,
        )
        draw.text((label_left + 8, label_top + 4), box.label, font=font, fill=(255, 255, 255, 255))


def _load_overlay_font(size: int):
    from PIL import ImageFont

    font_candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in font_candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()
