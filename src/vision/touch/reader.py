from __future__ import annotations

import json
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class TouchEvent:
    """A debounced "tap" event with calibrated panel coordinates.

    ``x`` and ``y`` are in panel-local pixel space (post-rotation), so a tap on
    the visible top-left of the screen is always (0, 0) regardless of how the
    framebuffer is rotated.
    """

    x: int
    y: int
    when_monotonic: float


@dataclass(slots=True)
class TouchCalibration:
    """Maps raw ADS7846 ABS_X/ABS_Y to panel pixel coordinates.

    The transform happens in two steps:
      1. Map raw → normalized (0..1) using min/max from a 4-corner calibration.
      2. Apply axis swap / inversion / panel rotation to land in panel pixel
         space (where the framebuffer actually puts pixels after software
         rotation in FramebufferSink).

    ``rotation_quarter_turns`` is the FramebufferSink rotation in 90° steps;
    the touch reader uses it to compose the final coordinate so taps line up
    with what the user sees.
    """

    raw_x_min: int = 200
    raw_x_max: int = 3900
    raw_y_min: int = 200
    raw_y_max: int = 3900
    swap_xy: bool = False
    invert_x: bool = False
    invert_y: bool = False
    panel_width: int = 480
    panel_height: int = 320
    rotation_quarter_turns: int = 0

    def map(self, raw_x: int, raw_y: int) -> tuple[int, int]:
        nx = _clamp01((raw_x - self.raw_x_min) / max(1, self.raw_x_max - self.raw_x_min))
        ny = _clamp01((raw_y - self.raw_y_min) / max(1, self.raw_y_max - self.raw_y_min))
        if self.invert_x:
            nx = 1.0 - nx
        if self.invert_y:
            ny = 1.0 - ny
        if self.swap_xy:
            nx, ny = ny, nx

        # nx/ny are in the panel's *unrotated* coordinate frame (the raw fbtft
        # frame, 480x320 here). Apply the same rotation the sink applies to
        # pixels so taps at the visible (0,0) map to (0,0) regardless.
        rot = self.rotation_quarter_turns % 4
        if rot == 0:
            px, py = nx, ny
            w, h = self.panel_width, self.panel_height
        elif rot == 1:  # 90° clockwise
            px, py = 1.0 - ny, nx
            w, h = self.panel_height, self.panel_width
        elif rot == 2:
            px, py = 1.0 - nx, 1.0 - ny
            w, h = self.panel_width, self.panel_height
        else:  # 270°
            px, py = ny, 1.0 - nx
            w, h = self.panel_height, self.panel_width

        return int(px * (w - 1)), int(py * (h - 1))

    def to_json(self) -> dict[str, Any]:
        return {
            "raw_x_min": self.raw_x_min,
            "raw_x_max": self.raw_x_max,
            "raw_y_min": self.raw_y_min,
            "raw_y_max": self.raw_y_max,
            "swap_xy": self.swap_xy,
            "invert_x": self.invert_x,
            "invert_y": self.invert_y,
            "panel_width": self.panel_width,
            "panel_height": self.panel_height,
            "rotation_quarter_turns": self.rotation_quarter_turns,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> TouchCalibration:
        return cls(
            raw_x_min=int(data.get("raw_x_min", 200)),
            raw_x_max=int(data.get("raw_x_max", 3900)),
            raw_y_min=int(data.get("raw_y_min", 200)),
            raw_y_max=int(data.get("raw_y_max", 3900)),
            swap_xy=bool(data.get("swap_xy", False)),
            invert_x=bool(data.get("invert_x", False)),
            invert_y=bool(data.get("invert_y", False)),
            panel_width=int(data.get("panel_width", 480)),
            panel_height=int(data.get("panel_height", 320)),
            rotation_quarter_turns=int(data.get("rotation_quarter_turns", 0)),
        )


@dataclass
class TouchReader:
    """Background thread that reads ADS7846 events and emits debounced taps.

    Strategy:
      - Read ABS_X / ABS_Y / BTN_TOUCH from the evdev device.
      - On press (BTN_TOUCH=1) start tracking. On release (BTN_TOUCH=0), if
        the press lasted at least ``min_press_ms`` and didn't drift past
        ``drag_threshold_px``, emit a tap at the last raw coordinate.
      - Calibration may be hot-swapped via ``set_calibration``; useful for the
        in-app calibration menu.

    Failures (device missing, evdev not installed) are logged and the reader
    silently no-ops — the rest of the system stays alive.
    """

    device_path: str = "/dev/input/event4"
    on_tap: Callable[[TouchEvent], None] | None = None
    on_press: Callable[[int, int], None] | None = None
    on_raw_press: Callable[[int, int], None] | None = None
    on_any_activity: Callable[[], None] | None = None
    min_press_ms: int = 10
    drag_threshold_px: int = 30

    _calibration: TouchCalibration = field(default_factory=TouchCalibration)
    _calib_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _stop: threading.Event = field(default_factory=threading.Event, repr=False)
    _thread: threading.Thread | None = field(default=None, repr=False)
    _device: Any = field(default=None, repr=False)
    _raw_listener_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _raw_listener: Callable[[int, int], None] | None = field(default=None, repr=False)

    def calibration(self) -> TouchCalibration:
        with self._calib_lock:
            return TouchCalibration(**self._calibration.to_json())

    def set_calibration(self, calibration: TouchCalibration) -> None:
        with self._calib_lock:
            self._calibration = calibration

    def set_raw_listener(self, listener: Callable[[int, int], None] | None) -> None:
        """Register a one-off listener that receives every raw (x, y) press.

        Used by the calibration UI to capture corner taps without going through
        the calibrated tap path.
        """
        with self._raw_listener_lock:
            self._raw_listener = listener

    def start(self) -> None:
        try:
            import evdev  # type: ignore
        except ImportError as exc:
            print(
                f"Touch input disabled: python-evdev not installed ({exc}).",
                file=sys.stderr,
            )
            return
        try:
            self._device = evdev.InputDevice(self.device_path)
            try:
                self._device.grab()
            except Exception as exc:
                # Non-fatal — grab failure just means another consumer can also
                # see the events. Continue.
                print(
                    f"Touch reader: could not exclusive-grab {self.device_path}: {exc}",
                    file=sys.stderr,
                )
        except Exception as exc:
            print(
                f"Touch input disabled: could not open {self.device_path}: {exc}",
                file=sys.stderr,
            )
            return
        capabilities = self._device.capabilities(verbose=False)
        print(
            f"Touch input active on {self.device_path} ({self._device.name}). "
            f"caps={ {k: v for k, v in capabilities.items() if k in (1, 3)} }"
        )
        self._thread = threading.Thread(target=self._run, name="touch-reader", daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        if self._device is not None:
            try:
                self._device.close()
            except Exception:
                pass
            self._device = None
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def _run(self) -> None:
        import evdev  # type: ignore

        ABS_X = evdev.ecodes.ABS_X
        ABS_Y = evdev.ecodes.ABS_Y
        ABS_PRESSURE = evdev.ecodes.ABS_PRESSURE
        BTN_TOUCH = evdev.ecodes.BTN_TOUCH
        SYN_REPORT = evdev.ecodes.SYN_REPORT

        raw_x = 0
        raw_y = 0
        pressure = 0
        # Pressed = either BTN_TOUCH=1 or pressure>0 was seen since last release.
        btn_pressed = False
        pressure_pressed = False
        press_started_at: float | None = None
        press_start_raw: tuple[int, int] | None = None
        max_drift = 0
        last_event_at = time.monotonic()
        # If neither BTN_TOUCH=0 nor pressure=0 arrives but events stop coming
        # in, treat that idle period as a release.
        idle_release_ms = 80

        while not self._stop.is_set() and self._device is not None:
            try:
                for event in self._device.read_loop():
                    if self._stop.is_set():
                        return
                    last_event_at = time.monotonic()
                    if event.type == evdev.ecodes.EV_ABS:
                        if event.code == ABS_X:
                            raw_x = event.value
                        elif event.code == ABS_Y:
                            raw_y = event.value
                        elif event.code == ABS_PRESSURE:
                            pressure = event.value
                            new_pressed = pressure > 0
                            if new_pressed and not pressure_pressed:
                                pressure_pressed = True
                                if not btn_pressed:
                                    self._begin_press(raw_x, raw_y)
                                    press_started_at = time.monotonic()
                                    press_start_raw = (raw_x, raw_y)
                                    max_drift = 0
                            elif not new_pressed and pressure_pressed:
                                pressure_pressed = False
                                if not btn_pressed:
                                    self._handle_release(
                                        raw_x, raw_y, press_started_at, press_start_raw, max_drift
                                    )
                                    press_started_at = None
                                    press_start_raw = None
                                    max_drift = 0
                    elif event.type == evdev.ecodes.EV_KEY and event.code == BTN_TOUCH:
                        if event.value == 1 and not btn_pressed:
                            btn_pressed = True
                            if not pressure_pressed:
                                self._begin_press(raw_x, raw_y)
                                press_started_at = time.monotonic()
                                press_start_raw = (raw_x, raw_y)
                                max_drift = 0
                        elif event.value == 0 and btn_pressed:
                            btn_pressed = False
                            if not pressure_pressed:
                                self._handle_release(
                                    raw_x, raw_y, press_started_at, press_start_raw, max_drift
                                )
                                press_started_at = None
                                press_start_raw = None
                                max_drift = 0
                    elif event.type == SYN_REPORT:
                        if (btn_pressed or pressure_pressed) and press_start_raw is not None:
                            dx = abs(raw_x - press_start_raw[0])
                            dy = abs(raw_y - press_start_raw[1])
                            max_drift = max(max_drift, dx + dy)
            except OSError:
                # Device closed.
                return
            except Exception as exc:
                print(f"Touch reader error: {exc}", file=sys.stderr)
                time.sleep(0.5)

    def _begin_press(self, raw_x: int, raw_y: int) -> None:
        if self.on_any_activity is not None:
            try:
                self.on_any_activity()
            except Exception as exc:
                print(f"on_any_activity error: {exc}", file=sys.stderr)
        if self.on_press is not None:
            with self._calib_lock:
                px, py = self._calibration.map(raw_x, raw_y)
            try:
                self.on_press(px, py)
            except Exception as exc:
                print(f"on_press error: {exc}", file=sys.stderr)

    def _handle_release(
        self,
        raw_x: int,
        raw_y: int,
        press_started_at: float | None,
        press_start_raw: tuple[int, int] | None,
        max_drift: int,
    ) -> None:
        if press_started_at is None or press_start_raw is None:
            return
        duration_ms = (time.monotonic() - press_started_at) * 1000
        if duration_ms < self.min_press_ms:
            return
        # Drift comparison uses raw axis units. ADS7846 raw is 0..4095, so a
        # ~20-unit threshold is well below any deliberate swipe.
        drag_raw_threshold = self.drag_threshold_px * 8
        if max_drift > drag_raw_threshold:
            return

        with self._raw_listener_lock:
            raw_listener = self._raw_listener
        if raw_listener is not None:
            try:
                raw_listener(press_start_raw[0], press_start_raw[1])
            except Exception as exc:
                print(f"raw_listener error: {exc}", file=sys.stderr)
            return

        if self.on_tap is None:
            return
        with self._calib_lock:
            x, y = self._calibration.map(press_start_raw[0], press_start_raw[1])
        try:
            self.on_tap(TouchEvent(x=x, y=y, when_monotonic=time.monotonic()))
        except Exception as exc:
            print(f"on_tap error: {exc}", file=sys.stderr)


def load_calibration(path: Path) -> TouchCalibration:
    if not path.exists():
        return TouchCalibration()
    try:
        return TouchCalibration.from_json(json.loads(path.read_text(encoding="utf-8")))
    except Exception as exc:
        print(f"Could not load touch calibration from {path}: {exc}", file=sys.stderr)
        return TouchCalibration()


def save_calibration(path: Path, calibration: TouchCalibration) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(calibration.to_json(), indent=2), encoding="utf-8")


def _clamp01(v: float) -> float:
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v
