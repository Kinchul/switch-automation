from __future__ import annotations

from typing import Any

from .base import CameraSource


class UsbCameraSource(CameraSource):
    """USB capture device (e.g. HDMI grabber). Implementation pending."""

    name = "usb"

    def __init__(self, *, width: int, height: int, fps: int) -> None:
        self.width = width
        self.height = height
        self.fps = fps

    def start(self) -> None:
        # TODO: open the USB HDMI grabber here (e.g. via OpenCV V4L2 or PyAV).
        # While unimplemented, raise so CameraCapture falls back to the next source.
        raise RuntimeError("USB camera source is not implemented yet.")

    def read_frame(self) -> Any:
        raise RuntimeError("USB camera source is not implemented yet.")

    def close(self) -> None:
        return
