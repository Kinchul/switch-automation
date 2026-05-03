from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Sequence

from ..capture import CameraCapture
from ..hud import OverlayState, draw_overlay
from .base import FrameSink


@dataclass(slots=True)
class OutputPipeline:
    """Single render loop: capture once, draw HUD once, fan out to all sinks.

    Each sink declares its own ``target_fps``; the pipeline runs at the highest
    requested rate. Sinks are responsible for any internal pacing if they want
    to emit slower than the pipeline (e.g. an MJPEG sink streaming at 5 fps
    while the local display draws at 30 fps).
    """

    capture: CameraCapture
    sinks: Sequence[FrameSink]
    overlay_state_fn: Callable[[], OverlayState] | None = None
    stop_event: threading.Event = field(init=False, default_factory=threading.Event, repr=False)
    _thread: threading.Thread = field(init=False, repr=False)
    _last_error: str | None = field(init=False, default=None, repr=False)
    _started_sinks: list[FrameSink] = field(init=False, default_factory=list, repr=False)

    def __post_init__(self) -> None:
        self._thread = threading.Thread(target=self._run, name="output-pipeline", daemon=True)

    @property
    def fps(self) -> float:
        if not self.sinks:
            return 1.0
        return max((sink.target_fps for sink in self.sinks), default=1.0) or 1.0

    def start(self) -> OutputPipeline:
        for sink in self.sinks:
            try:
                sink.start()
                self._started_sinks.append(sink)
            except Exception as exc:
                print(f"Output sink '{sink.name}' failed to start: {exc}", file=sys.stderr)
        self._thread.start()
        return self

    def close(self) -> None:
        self.stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
        for sink in self._started_sinks:
            try:
                sink.close()
            except Exception as exc:
                print(f"Output sink '{sink.name}' close error: {exc}", file=sys.stderr)
        self._started_sinks.clear()

    def _run(self) -> None:
        interval = 1.0 / self.fps if self.fps > 0 else 0.2
        while not self.stop_event.is_set():
            started = time.monotonic()
            try:
                frame = self.capture.get_frame()
                overlay = self._current_overlay_state()
                rendered = draw_overlay(frame, overlay)
                self._last_error = None
            except Exception as exc:
                message = f"Pipeline render error: {exc}"
                if message != self._last_error:
                    print(message, file=sys.stderr)
                    self._last_error = message
                time.sleep(0.5)
                continue

            for sink in self._started_sinks:
                try:
                    sink.consume(rendered)
                except Exception as exc:
                    print(f"Sink '{sink.name}' consume error: {exc}", file=sys.stderr)

            elapsed = time.monotonic() - started
            delay = interval - elapsed
            if delay > 0:
                time.sleep(delay)

    def _current_overlay_state(self) -> OverlayState:
        if self.overlay_state_fn is not None:
            return self.overlay_state_fn()
        return OverlayState()
