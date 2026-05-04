from __future__ import annotations

import shutil
import subprocess
from time import monotonic, sleep
from typing import Any

from .base import CameraSource, to_rgb_frame


DEFAULT_V4L2_CONTROLS: dict[str, int] = {
    "brightness": 35,
    "contrast": 50,
    "saturation": 40,
}


class UsbCameraSource(CameraSource):
    """USB UVC capture device (e.g. MS2130 HDMI grabber) read via OpenCV V4L2."""

    name = "usb"

    def __init__(
        self,
        *,
        width: int,
        height: int,
        fps: int,
        device: str = "/dev/video0",
        warmup: float = 1.0,
        controls: dict[str, int] | None = None,
    ) -> None:
        self.width = width
        self.height = height
        self.fps = fps
        self.device = device
        self.warmup = warmup
        self.controls = DEFAULT_V4L2_CONTROLS if controls is None else dict(controls)
        self._capture: Any | None = None

    def start(self) -> None:
        try:
            import cv2  # type: ignore
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "OpenCV (cv2) is not installed. Install `python3-opencv` on the Raspberry Pi."
            ) from exc

        device = self._resolve_device_index(self.device)
        capture = cv2.VideoCapture(device, cv2.CAP_V4L2)
        if not capture.isOpened():
            raise RuntimeError(f"Could not open USB capture device {self.device!r}.")

        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        capture.set(cv2.CAP_PROP_FOURCC, fourcc)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        capture.set(cv2.CAP_PROP_FPS, self.fps)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._apply_v4l2_controls()

        deadline = monotonic() + max(self.warmup, 0.5)
        first_frame = None
        while monotonic() < deadline:
            ok, frame = capture.read()
            if ok and frame is not None and frame.size > 0:
                first_frame = frame
                break
            sleep(0.05)

        if first_frame is None:
            try:
                capture.release()
            except Exception:
                pass
            raise RuntimeError(
                f"USB capture device {self.device!r} opened but produced no frames "
                f"at {self.width}x{self.height}@{self.fps} MJPG."
            )

        actual_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = capture.get(cv2.CAP_PROP_FPS)
        print(
            f"USB capture {self.device} negotiated {actual_w}x{actual_h}@{actual_fps:.0f} MJPG "
            f"(requested {self.width}x{self.height}@{self.fps})."
        )

        self._capture = capture

    def read_frame(self) -> Any:
        capture = self._capture
        if capture is None:
            raise RuntimeError("USB camera is not started.")
        ok, frame = capture.read()
        if not ok or frame is None:
            raise RuntimeError("USB capture read failed.")
        return to_rgb_frame(frame)

    def close(self) -> None:
        capture = self._capture
        self._capture = None
        if capture is None:
            return
        try:
            capture.release()
        except Exception:
            pass

    def _apply_v4l2_controls(self) -> None:
        if not self.controls:
            return
        binary = shutil.which("v4l2-ctl")
        if binary is None:
            print("v4l2-ctl not found; skipping USB capture color controls.")
            return
        applied: list[str] = []
        for name, value in self.controls.items():
            try:
                subprocess.run(
                    [binary, "--device", self.device, f"--set-ctrl={name}={value}"],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                applied.append(f"{name}={value}")
            except subprocess.CalledProcessError as exc:
                stderr = (exc.stderr or "").strip()
                print(f"v4l2-ctl could not set {name}={value} on {self.device}: {stderr}")
        if applied:
            print(f"USB capture controls: {' '.join(applied)}")

    @staticmethod
    def _resolve_device_index(device: str) -> Any:
        if device.startswith("/dev/video"):
            tail = device[len("/dev/video"):]
            if tail.isdigit():
                return int(tail)
        return device
