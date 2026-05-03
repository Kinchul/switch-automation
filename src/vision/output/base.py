from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class FrameSink(ABC):
    """Consumer of overlay-rendered RGB frames produced by OutputPipeline.

    A sink may declare ``target_size = (width, height)`` to ask the pipeline to
    pre-scale (letterbox) the capture frame to that size *before* the HUD is
    drawn. That keeps text and ROI boxes legible on small panels — drawing at
    capture resolution and shrinking the result downstream makes the HUD
    nearly invisible. Sinks that want raw capture-resolution output leave
    ``target_size`` as ``None``.
    """

    name: str = "sink"
    target_fps: float = 30.0
    target_size: tuple[int, int] | None = None

    def start(self) -> None:
        return

    @abstractmethod
    def consume(self, frame: Any) -> None: ...

    def close(self) -> None:
        return
