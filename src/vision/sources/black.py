from __future__ import annotations

from time import sleep
from typing import Any

from .base import CameraSource


class BlackFrameSource(CameraSource):
    """Synthetic black-frame fallback when no camera backend is available."""

    name = "black"

    def __init__(self, *, width: int, height: int, fps: int) -> None:
        self.width = width
        self.height = height
        self.fps = max(1, fps)
        self._frame: Any | None = None
        self._interval = 1.0 / self.fps

    def start(self) -> None:
        try:
            import numpy as np  # type: ignore
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Black-frame fallback requires numpy."
            ) from exc
        self._frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def read_frame(self) -> Any:
        if self._frame is None:
            raise RuntimeError("Black-frame source is not started.")
        sleep(self._interval)
        return self._frame

    def close(self) -> None:
        self._frame = None
