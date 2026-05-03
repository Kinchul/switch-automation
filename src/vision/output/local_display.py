from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Any

from .base import FrameSink


@dataclass(slots=True)
class FramebufferSink(FrameSink):
    """Render frames to a local SDL/framebuffer panel (e.g. JT3.5TR on /dev/fb1).

    Failures during ``start`` (no pygame, no /dev/fb*, no SDL) are reported but
    do not raise — the pipeline simply skips this sink.
    """

    name: str = "framebuffer"
    target_fps: float = 30.0
    fbdev: str = "/dev/fb1"
    width: int = 480
    height: int = 320
    _pygame: Any = field(init=False, default=None, repr=False)
    _screen: Any = field(init=False, default=None, repr=False)
    _disabled_reason: str | None = field(init=False, default=None, repr=False)

    def start(self) -> None:
        try:
            import pygame  # type: ignore
        except ModuleNotFoundError:
            self._disabled_reason = "pygame is not installed"
            print(f"Local display disabled: {self._disabled_reason}", file=sys.stderr)
            return

        try:
            if self.fbdev and os.path.exists(self.fbdev):
                os.environ.setdefault("SDL_VIDEODRIVER", "fbcon")
                os.environ["SDL_FBDEV"] = self.fbdev
                os.environ.setdefault("SDL_NOMOUSE", "1")
            elif "SDL_VIDEODRIVER" not in os.environ:
                # No /dev/fb* — let SDL pick (useful for desktop testing).
                pass

            pygame.display.init()
            screen = pygame.display.set_mode((self.width, self.height))
            pygame.mouse.set_visible(False)
        except Exception as exc:
            self._disabled_reason = str(exc)
            print(f"Local display disabled: {exc}", file=sys.stderr)
            try:
                pygame.display.quit()
            except Exception:
                pass
            return

        self._pygame = pygame
        self._screen = screen

    def consume(self, frame: Any) -> None:
        if self._screen is None or self._pygame is None:
            return
        try:
            scaled = self._fit_letterbox(frame, self.width, self.height)
            surface = self._pygame.image.frombuffer(
                scaled.tobytes(), (self.width, self.height), "RGB"
            )
            self._screen.blit(surface, (0, 0))
            self._pygame.display.flip()
        except Exception as exc:
            print(f"Local display render error: {exc}", file=sys.stderr)

    def close(self) -> None:
        if self._pygame is None:
            return
        try:
            self._pygame.display.quit()
            self._pygame.quit()
        except Exception:
            pass
        self._pygame = None
        self._screen = None

    @staticmethod
    def _fit_letterbox(frame, target_w: int, target_h: int):
        """Scale-to-fit with letterboxing, preserving aspect ratio. Pure numpy."""
        import numpy as np  # type: ignore

        src_h, src_w = frame.shape[:2]
        if (src_w, src_h) == (target_w, target_h):
            return frame
        scale = min(target_w / src_w, target_h / src_h)
        new_w = max(1, int(src_w * scale))
        new_h = max(1, int(src_h * scale))

        x_idx = (np.arange(new_w) * src_w / new_w).astype(np.int32)
        y_idx = (np.arange(new_h) * src_h / new_h).astype(np.int32)
        resized = frame[y_idx[:, None], x_idx[None, :]]

        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        off_x = (target_w - new_w) // 2
        off_y = (target_h - new_h) // 2
        canvas[off_y : off_y + new_h, off_x : off_x + new_w] = resized
        return canvas
