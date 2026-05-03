from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class CameraSource(ABC):
    """Backend that produces RGB frames for CameraCapture."""

    name: str = "camera"

    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def read_frame(self) -> Any: ...

    @abstractmethod
    def close(self) -> None: ...


def to_rgb_frame(frame: Any):
    """Convert a BGR frame coming from a backend into a contiguous RGB frame."""
    if getattr(frame, "ndim", None) == 3 and getattr(frame, "shape", (0, 0, 0))[2] >= 3:
        try:
            import numpy as np  # type: ignore
        except ModuleNotFoundError:
            return frame[:, :, [2, 1, 0]].copy()
        return np.ascontiguousarray(frame[:, :, [2, 1, 0]])
    return frame
