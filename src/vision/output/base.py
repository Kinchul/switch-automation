from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class FrameSink(ABC):
    """Consumer of overlay-rendered RGB frames produced by OutputPipeline."""

    name: str = "sink"
    target_fps: float = 30.0

    def start(self) -> None:
        return

    @abstractmethod
    def consume(self, frame: Any) -> None: ...

    def close(self) -> None:
        return
