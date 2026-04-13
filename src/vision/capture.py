from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
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
    _lock: Lock = field(init=False, default_factory=Lock, repr=False)

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
        return self

    def get_frame(self):
        self._require_started()
        with self._lock:
            return _to_rgb_frame(self._camera.capture_array("main"))

    def save_frame(self, output_path: str | Path, quality: int = 95) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        _save_rgb_frame(self.get_frame(), path, quality=quality)
        return path

    def close(self) -> None:
        if self._camera is None:
            return

        camera = self._camera
        self._camera = None
        try:
            camera.stop()
        finally:
            camera.close()

    def _require_started(self) -> None:
        if self._camera is None:
            raise RuntimeError("Camera is not running. Call start() first.")


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
