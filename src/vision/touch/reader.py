from __future__ import annotations

import json
import os
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


_DEBUG = os.getenv("SWITCH_TOUCH_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}


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
    """Maps raw ADS7846 ABS_X/ABS_Y directly to panel pixel coordinates via an
    affine transform.

        panel_x = a*rx + b*ry + c
        panel_y = d*rx + e*ry + f

    Solved from the four corner taps captured during calibration. The affine
    representation handles axis swap, inversion, *and* the framebuffer rotation
    in a single step — the user taps the visible corners and we directly fit
    the matrix.

    The default values are an identity-ish guess that maps a typical ADS7846
    range to a 480x320 panel without rotation. They are only used until the
    user runs calibration, after which we always have a proper fit.

    ``panel_width`` / ``panel_height`` are kept as bounds for clamping and so
    the rotation cycler can rebuild a sensible default if calibration is
    discarded. ``rotation_quarter_turns`` is informational — the affine
    already encodes any orientation.
    """

    a: float = 480.0 / 3700.0
    b: float = 0.0
    c: float = -480.0 * 200.0 / 3700.0
    d: float = 0.0
    e: float = 320.0 / 3700.0
    f: float = -320.0 * 200.0 / 3700.0
    panel_width: int = 480
    panel_height: int = 320
    rotation_quarter_turns: int = 0

    def map(self, raw_x: int, raw_y: int) -> tuple[int, int]:
        px = self.a * raw_x + self.b * raw_y + self.c
        py = self.d * raw_x + self.e * raw_y + self.f
        # Clamp to panel bounds so out-of-range raws don't draw off-screen
        # ripples or trigger wrong-corner button hits.
        px_i = max(0, min(self.panel_width - 1, int(round(px))))
        py_i = max(0, min(self.panel_height - 1, int(round(py))))
        return px_i, py_i

    def to_json(self) -> dict[str, Any]:
        return {
            "a": self.a,
            "b": self.b,
            "c": self.c,
            "d": self.d,
            "e": self.e,
            "f": self.f,
            "panel_width": self.panel_width,
            "panel_height": self.panel_height,
            "rotation_quarter_turns": self.rotation_quarter_turns,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> TouchCalibration:
        # Accept new affine format. The legacy min/max/swap/invert format is
        # not migrated — any saved file from before this change will fall back
        # to defaults and the user will be prompted to recalibrate.
        if "a" not in data:
            return cls(
                panel_width=int(data.get("panel_width", 480)),
                panel_height=int(data.get("panel_height", 320)),
                rotation_quarter_turns=int(data.get("rotation_quarter_turns", 0)),
            )
        return cls(
            a=float(data.get("a", 0.0)),
            b=float(data.get("b", 0.0)),
            c=float(data.get("c", 0.0)),
            d=float(data.get("d", 0.0)),
            e=float(data.get("e", 0.0)),
            f=float(data.get("f", 0.0)),
            panel_width=int(data.get("panel_width", 480)),
            panel_height=int(data.get("panel_height", 320)),
            rotation_quarter_turns=int(data.get("rotation_quarter_turns", 0)),
        )

    @classmethod
    def from_corners(
        cls,
        *,
        raw_corners: list[tuple[int, int]],
        panel_targets: list[tuple[float, float]],
        panel_width: int,
        panel_height: int,
        rotation_quarter_turns: int = 0,
    ) -> TouchCalibration:
        """Fit an affine raw→panel map from N>=3 tapped corners.

        ``raw_corners`` and ``panel_targets`` must be the same length and in
        matching order. Solves two independent least-squares problems for X
        and Y panel coordinates.
        """
        if len(raw_corners) != len(panel_targets) or len(raw_corners) < 3:
            raise ValueError("Need at least 3 matching raw/panel pairs to fit.")

        # 6-param affine = two independent linear regressions:
        #   panel_x = a*rx + b*ry + c
        #   panel_y = d*rx + e*ry + f
        # Solve both via the normal equations on the same design matrix.
        n = len(raw_corners)
        sum_rx = sum(rx for rx, _ in raw_corners)
        sum_ry = sum(ry for _, ry in raw_corners)
        sum_rx2 = sum(rx * rx for rx, _ in raw_corners)
        sum_ry2 = sum(ry * ry for _, ry in raw_corners)
        sum_rxry = sum(rx * ry for rx, ry in raw_corners)

        # Design matrix M^T M (3x3 symmetric). Columns of M are [rx, ry, 1].
        m = [
            [sum_rx2, sum_rxry, sum_rx],
            [sum_rxry, sum_ry2, sum_ry],
            [sum_rx, sum_ry, float(n)],
        ]

        def _solve(targets: list[float]) -> tuple[float, float, float]:
            v = [
                sum(rx * t for (rx, _), t in zip(raw_corners, targets, strict=True)),
                sum(ry * t for (_, ry), t in zip(raw_corners, targets, strict=True)),
                sum(targets),
            ]
            return _solve3(m, v)

        targets_x = [tx for tx, _ in panel_targets]
        targets_y = [ty for _, ty in panel_targets]
        a, b, c = _solve(targets_x)
        d, e, f = _solve(targets_y)
        return cls(
            a=a, b=b, c=c, d=d, e=e, f=f,
            panel_width=panel_width,
            panel_height=panel_height,
            rotation_quarter_turns=rotation_quarter_turns,
        )


def _solve3(m: list[list[float]], v: list[float]) -> tuple[float, float, float]:
    """Solve a 3x3 linear system m @ x = v via Cramer's rule. Stable enough for
    the well-conditioned matrices produced by 4 corner taps spread over the
    panel."""
    def _det(rows: list[list[float]]) -> float:
        a, b, c = rows[0]
        d, e, f = rows[1]
        g, h, i = rows[2]
        return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)

    det = _det(m)
    if abs(det) < 1e-9:
        raise ValueError("Calibration matrix is singular — corner taps are colinear?")

    def _replace_col(col: int, vec: list[float]) -> list[list[float]]:
        out = [row.copy() for row in m]
        for i in range(3):
            out[i][col] = vec[i]
        return out

    x0 = _det(_replace_col(0, v)) / det
    x1 = _det(_replace_col(1, v)) / det
    x2 = _det(_replace_col(2, v)) / det
    return x0, x1, x2


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
        """Process evdev events frame-by-frame.

        ADS7846 emits each touch sample as a sequence terminated by SYN_REPORT:
            EV_ABS ABS_X v
            EV_ABS ABS_Y v
            [EV_ABS ABS_PRESSURE v]
            [EV_KEY BTN_TOUCH 0|1]
            EV_SYN SYN_REPORT 0

        We accumulate the values for the current frame in ``frame_*`` and only
        commit them at SYN_REPORT. That guarantees the press coordinates we
        report belong to the *same* frame as the BTN_TOUCH=1 event, instead of
        carrying over stale values from the previous tap.
        """
        import evdev  # type: ignore

        ABS_X = evdev.ecodes.ABS_X
        ABS_Y = evdev.ecodes.ABS_Y
        ABS_PRESSURE = evdev.ecodes.ABS_PRESSURE
        BTN_TOUCH = evdev.ecodes.BTN_TOUCH
        SYN_REPORT = evdev.ecodes.SYN_REPORT
        EV_ABS = evdev.ecodes.EV_ABS
        EV_KEY = evdev.ecodes.EV_KEY
        EV_SYN = evdev.ecodes.EV_SYN

        # Per-frame staging.
        frame_x: int | None = None
        frame_y: int | None = None
        frame_pressure: int | None = None
        frame_btn_touch: int | None = None

        # Tap-tracking state (committed at SYN_REPORT).
        last_x = 0
        last_y = 0
        pressed = False
        press_started_at: float | None = None
        press_start_raw: tuple[int, int] | None = None
        max_drift = 0

        while not self._stop.is_set() and self._device is not None:
            try:
                for event in self._device.read_loop():
                    if self._stop.is_set():
                        return
                    if event.type == EV_ABS:
                        if event.code == ABS_X:
                            frame_x = event.value
                        elif event.code == ABS_Y:
                            frame_y = event.value
                        elif event.code == ABS_PRESSURE:
                            frame_pressure = event.value
                    elif event.type == EV_KEY and event.code == BTN_TOUCH:
                        frame_btn_touch = event.value
                    elif event.type == EV_SYN and event.code == SYN_REPORT:
                        # Commit the frame.
                        if frame_x is not None:
                            last_x = frame_x
                        if frame_y is not None:
                            last_y = frame_y

                        # A frame indicates a press if BTN_TOUCH=1 or pressure>0.
                        # It indicates release if BTN_TOUCH=0 or pressure=0.
                        # If neither fires, the existing state is unchanged.
                        is_pressed = pressed
                        if frame_btn_touch == 1:
                            is_pressed = True
                        elif frame_btn_touch == 0:
                            is_pressed = False
                        if frame_pressure is not None:
                            if frame_pressure > 0:
                                is_pressed = True
                            else:
                                is_pressed = False

                        if _DEBUG:
                            print(
                                f"touch frame: x={last_x} y={last_y} btn={frame_btn_touch} "
                                f"p={frame_pressure} pressed={pressed}->{is_pressed}",
                                file=sys.stderr,
                            )

                        if is_pressed and not pressed:
                            press_started_at = time.monotonic()
                            press_start_raw = (last_x, last_y)
                            max_drift = 0
                            self._begin_press(last_x, last_y)
                        elif not is_pressed and pressed:
                            self._handle_release(
                                last_x, last_y, press_started_at, press_start_raw, max_drift
                            )
                            press_started_at = None
                            press_start_raw = None
                            max_drift = 0
                        elif is_pressed and pressed and press_start_raw is not None:
                            dx = abs(last_x - press_start_raw[0])
                            dy = abs(last_y - press_start_raw[1])
                            max_drift = max(max_drift, dx + dy)
                            self._notify_move(last_x, last_y)

                        pressed = is_pressed
                        # Reset per-frame stage; we keep last_x/last_y rolling
                        # because some frames omit them.
                        frame_x = None
                        frame_y = None
                        frame_pressure = None
                        frame_btn_touch = None
            except OSError:
                # Device closed.
                return
            except Exception as exc:
                print(f"Touch reader error: {exc}", file=sys.stderr)
                time.sleep(0.5)

    def _notify_move(self, raw_x: int, raw_y: int) -> None:
        """No-op hook for future drag tracking; kept here so callers can subclass."""
        return

    def _begin_press(self, raw_x: int, raw_y: int) -> None:
        if self.on_any_activity is not None:
            try:
                self.on_any_activity()
            except Exception as exc:
                print(f"on_any_activity error: {exc}", file=sys.stderr)
        with self._calib_lock:
            px, py = self._calibration.map(raw_x, raw_y)
        if _DEBUG:
            print(
                f"touch press: raw=({raw_x},{raw_y}) panel=({px},{py})",
                file=sys.stderr,
            )
        if self.on_press is not None:
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
