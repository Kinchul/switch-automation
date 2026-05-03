from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from threading import Event, Lock, Thread
from time import sleep
from typing import Any

from .sources import (
    BlackFrameSource,
    CameraSource,
    CsiCameraSource,
    UsbCameraSource,
)


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
    lock_auto_controls: bool = True
    controls_path: Path | None = None
    preferred_sources: tuple[str, ...] = ("usb", "csi", "black")
    _source: CameraSource | None = field(init=False, default=None, repr=False)
    _frame_lock: Lock = field(init=False, default_factory=Lock, repr=False)
    _latest_frame: Any | None = field(init=False, default=None, repr=False)
    _reader_thread: Thread | None = field(init=False, default=None, repr=False)
    _stop_event: Event = field(init=False, default_factory=Event, repr=False)
    _frame_ready: Event = field(init=False, default_factory=Event, repr=False)
    _last_error: str | None = field(init=False, default=None, repr=False)

    @property
    def source_name(self) -> str | None:
        return self._source.name if self._source is not None else None

    def start(self) -> CameraCapture:
        if self._source is not None:
            return self

        source = self._select_source()
        self._source = source
        self._stop_event.clear()
        self._frame_ready.clear()
        self._latest_frame = None
        self._reader_thread = Thread(target=self._reader_loop, name="camera-capture", daemon=True)
        self._reader_thread.start()
        return self

    def _build_source(self, name: str) -> CameraSource:
        if name == "usb":
            return UsbCameraSource(width=self.width, height=self.height, fps=self.fps)
        if name == "csi":
            return CsiCameraSource(
                camera_index=self.camera_index,
                width=self.width,
                height=self.height,
                fps=self.fps,
                warmup=self.warmup,
                lock_auto_controls=self.lock_auto_controls,
                controls_path=self.controls_path,
            )
        if name == "black":
            return BlackFrameSource(width=self.width, height=self.height, fps=self.fps)
        raise ValueError(f"Unknown camera source: {name!r}")

    def _select_source(self) -> CameraSource:
        errors: list[str] = []
        for name in self.preferred_sources:
            candidate = self._build_source(name)
            try:
                candidate.start()
            except Exception as exc:
                errors.append(f"{name}: {exc}")
                try:
                    candidate.close()
                except Exception:
                    pass
                print(f"Camera source '{name}' unavailable: {exc}")
                continue
            print(f"Camera source '{name}' active.")
            return candidate
        raise RuntimeError(
            "No camera source could be started. Tried: " + "; ".join(errors)
        )

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
        if self._source is None:
            return

        self._stop_event.set()
        self._frame_ready.set()
        reader = self._reader_thread
        self._reader_thread = None
        if reader is not None:
            reader.join(timeout=2.0)

        source = self._source
        self._source = None
        try:
            source.close()
        finally:
            with self._frame_lock:
                self._latest_frame = None

    def _require_started(self) -> None:
        if self._source is None:
            raise RuntimeError("Camera is not running. Call start() first.")

    def _reader_loop(self) -> None:
        retry_delay = 0.05
        while not self._stop_event.is_set():
            source = self._source
            if source is None:
                return
            try:
                frame = source.read_frame()
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
    lock_auto_controls: bool = True,
    controls_path: str | Path | None = None,
    preferred_sources: tuple[str, ...] = ("usb", "csi", "black"),
) -> CameraCapture:
    return CameraCapture(
        camera_index=camera_index,
        width=width,
        height=height,
        fps=fps,
        warmup=warmup,
        lock_auto_controls=lock_auto_controls,
        controls_path=None if controls_path in (None, "") else Path(controls_path),
        preferred_sources=preferred_sources,
    ).start()
