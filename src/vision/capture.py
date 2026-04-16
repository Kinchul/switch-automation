from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from threading import Event, Lock, Thread
from time import sleep
from typing import Any


def _to_rgb_frame(frame: Any):
    if getattr(frame, "ndim", None) == 3 and getattr(frame, "shape", (0, 0, 0))[2] >= 3:
        try:
            import numpy as np  # type: ignore
        except ModuleNotFoundError:
            return frame[:, :, [2, 1, 0]].copy()
        return np.ascontiguousarray(frame[:, :, [2, 1, 0]])
    return frame


def _save_rgb_frame(frame: Any, output_path: Path, quality: int = 95) -> None:
    jpeg = encode_rgb_frame(frame, quality=quality)
    output_path.write_bytes(jpeg)


def encode_rgb_frame(frame: Any, quality: int = 95) -> bytes:
    try:
        import simplejpeg  # type: ignore
    except ModuleNotFoundError:
        try:
            from PIL import Image  # type: ignore
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Saving camera frames requires either `simplejpeg` or `Pillow`."
            ) from exc
        from io import BytesIO

        buffer = BytesIO()
        Image.fromarray(frame).save(buffer, format="JPEG", quality=quality)
        return buffer.getvalue()

    return simplejpeg.encode_jpeg(frame, quality=quality, colorspace="RGB")


@dataclass(slots=True)
class CameraCapture:
    camera_index: int = 0
    width: int = 1920
    height: int = 1080
    fps: int = 20
    warmup: float = 2.0
    _camera: Any | None = field(init=False, default=None, repr=False)
    _frame_lock: Lock = field(init=False, default_factory=Lock, repr=False)
    _latest_frame: Any | None = field(init=False, default=None, repr=False)
    _reader_thread: Thread | None = field(init=False, default=None, repr=False)
    _stop_event: Event = field(init=False, default_factory=Event, repr=False)
    _frame_ready: Event = field(init=False, default_factory=Event, repr=False)
    _last_error: str | None = field(init=False, default=None, repr=False)

    def start(self) -> CameraCapture:
        if self._camera is not None:
            return self

        try:
            from picamera2 import Picamera2  # type: ignore
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Picamera2 is not installed. Install `python3-picamera2` on the Raspberry Pi."
            ) from exc

        camera_info = Picamera2.global_camera_info()
        if not camera_info:
            raise RuntimeError(
                "No Picamera2 cameras were detected. Confirm `rpicam-hello --list-cameras` "
                "shows the sensor and that the current environment can access `/dev/media*`."
            )
        if not (0 <= self.camera_index < len(camera_info)):
            raise RuntimeError(
                f"Camera index {self.camera_index} is out of range. "
                f"Detected cameras: {len(camera_info)}."
            )

        frame_duration_us = max(1, int(1_000_000 / self.fps))
        camera = Picamera2(self.camera_index)
        config = camera.create_video_configuration(
            main={"size": (self.width, self.height), "format": "RGB888"},
            controls={"FrameDurationLimits": (frame_duration_us, frame_duration_us)},
            buffer_count=4,
            queue=False,
        )
        camera.configure(config)
        camera.start()

        if self.warmup > 0:
            sleep(self.warmup)

        self._camera = camera
        self._stop_event.clear()
        self._frame_ready.clear()
        self._latest_frame = None
        self._reader_thread = Thread(target=self._reader_loop, name="camera-capture", daemon=True)
        self._reader_thread.start()
        return self

    def get_frame(self):
        self._require_started()
        if not self._frame_ready.wait(timeout=max(1.0, self.warmup + 1.0)):
            detail = self._last_error or "Timed out waiting for the first camera frame."
            raise RuntimeError(detail)
        with self._frame_lock:
            frame = self._latest_frame
        if frame is None:
            detail = self._last_error or "Camera frame cache is empty."
            raise RuntimeError(detail)
        return frame

    def save_frame(self, output_path: str | Path, quality: int = 95) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        _save_rgb_frame(self.get_frame(), path, quality=quality)
        return path

    def close(self) -> None:
        if self._camera is None:
            return

        self._stop_event.set()
        self._frame_ready.set()
        reader = self._reader_thread
        self._reader_thread = None
        if reader is not None:
            reader.join(timeout=2.0)

        camera = self._camera
        self._camera = None
        try:
            camera.stop()
        finally:
            camera.close()
        with self._frame_lock:
            self._latest_frame = None

    def _require_started(self) -> None:
        if self._camera is None:
            raise RuntimeError("Camera is not running. Call start() first.")

    def _reader_loop(self) -> None:
        retry_delay = 0.05
        while not self._stop_event.is_set():
            camera = self._camera
            if camera is None:
                return
            try:
                frame = _to_rgb_frame(camera.capture_array("main"))
                with self._frame_lock:
                    self._latest_frame = frame
                self._last_error = None
                self._frame_ready.set()
            except Exception as exc:
                self._last_error = f"Camera capture error: {exc}"
                sleep(retry_delay)


def open_capture(
    camera_index: int = 0,
    width: int = 1920,
    height: int = 1080,
    fps: int = 20,
    warmup: float = 2.0,
) -> CameraCapture:
    return CameraCapture(
        camera_index=camera_index,
        width=width,
        height=height,
        fps=fps,
        warmup=warmup,
    ).start()
