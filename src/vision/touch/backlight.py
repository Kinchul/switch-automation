from __future__ import annotations

import sys
import threading
import time
from typing import Any


class BacklightController:
    """Hold the SPI panel's backlight FET on GPIO18 via libgpiod v2.

    The piscreen2r overlay does not register a working /sys/class/backlight node
    (the one that appears under fb_ili9486 is a stub: max_brightness is 0 and
    bl_power writes are no-ops). The actual backlight is wired to GPIO18 on
    these Waveshare/Kuman 3.5" resistive clones; pulling it low cuts power to
    the LEDs, pulling it high turns them on.

    libgpiod v2 ships on Bookworm. The line must be held by a live process —
    releasing it (gpiod request goes out of scope) lets the line float, which
    on this hardware leaves the backlight at its previous level but can be
    flaky. We keep the request alive for the controller's lifetime.
    """

    def __init__(
        self,
        *,
        chip: str = "/dev/gpiochip0",
        line: int = 18,
        pwm_frequency_hz: float = 200.0,
    ) -> None:
        self._chip_path = chip
        self._line = line
        self._request: Any = None
        self._gpiod: Any = None
        self._line_value_on: Any = None
        self._line_value_off: Any = None
        self._on = True
        self._available = False
        self._fallback_blackout = False
        # Software-PWM brightness. ``brightness`` is in [0.0, 1.0]; 1.0 means
        # the line is held high continuously (no PWM thread needed). Values
        # < 1.0 spin a thread that toggles the line at ``pwm_frequency_hz``.
        self._brightness = 1.0
        self._pwm_period_s = 1.0 / max(1.0, pwm_frequency_hz)
        self._pwm_thread: threading.Thread | None = None
        self._pwm_stop = threading.Event()
        self._pwm_lock = threading.Lock()

    def start(self) -> None:
        try:
            import gpiod  # type: ignore
        except ImportError as exc:
            print(
                f"Backlight control disabled: python3-gpiod not installed ({exc}). "
                "Falling back to black-frame blanking.",
                file=sys.stderr,
            )
            self._fallback_blackout = True
            return

        try:
            line_settings = gpiod.LineSettings(
                direction=gpiod.line.Direction.OUTPUT,
                output_value=gpiod.line.Value.ACTIVE,
            )
            self._request = gpiod.request_lines(
                self._chip_path,
                consumer="switch-automation-backlight",
                config={self._line: line_settings},
            )
            self._line_value_on = gpiod.line.Value.ACTIVE
            self._line_value_off = gpiod.line.Value.INACTIVE
        except Exception as exc:
            print(
                f"Backlight control disabled: could not request GPIO{self._line} on {self._chip_path}: {exc}. "
                "Falling back to black-frame blanking.",
                file=sys.stderr,
            )
            self._fallback_blackout = True
            return

        self._gpiod = gpiod
        self._available = True
        self._on = True
        print(f"Backlight control active on GPIO{self._line} ({self._chip_path}).")

    @property
    def available(self) -> bool:
        return self._available

    @property
    def fallback_blackout(self) -> bool:
        """True when GPIO control failed and the caller should blank pixels instead."""
        return self._fallback_blackout

    @property
    def is_on(self) -> bool:
        return self._on

    @property
    def brightness(self) -> float:
        return self._brightness

    def set_on(self, on: bool) -> None:
        if on == self._on:
            return
        self._on = on
        self._apply_state()

    def set_brightness(self, brightness: float) -> None:
        """Set backlight brightness in [0.0, 1.0].

        At 1.0 the line is held high continuously. Below 1.0 a software PWM
        thread toggles the line at ~200 Hz (well above the visible flicker
        threshold for an LED backlight, with negligible CPU cost).
        """
        clamped = max(0.0, min(1.0, brightness))
        self._brightness = clamped
        self._apply_state()

    def _apply_state(self) -> None:
        if not self._available or self._request is None:
            return
        # Stop any running PWM thread before changing direction.
        self._stop_pwm()
        if not self._on:
            try:
                self._request.set_value(self._line, self._line_value_off)
            except Exception as exc:
                print(f"Backlight set_value(off) failed: {exc}", file=sys.stderr)
            return
        if self._brightness >= 0.999:
            try:
                self._request.set_value(self._line, self._line_value_on)
            except Exception as exc:
                print(f"Backlight set_value(on) failed: {exc}", file=sys.stderr)
            return
        if self._brightness <= 0.001:
            try:
                self._request.set_value(self._line, self._line_value_off)
            except Exception as exc:
                print(f"Backlight set_value(off) failed: {exc}", file=sys.stderr)
            return
        self._start_pwm()

    def _start_pwm(self) -> None:
        with self._pwm_lock:
            self._pwm_stop = threading.Event()
            self._pwm_thread = threading.Thread(
                target=self._pwm_loop,
                name="backlight-pwm",
                daemon=True,
            )
            self._pwm_thread.start()

    def _stop_pwm(self) -> None:
        with self._pwm_lock:
            thread = self._pwm_thread
            self._pwm_thread = None
        if thread is not None:
            self._pwm_stop.set()
            thread.join(timeout=0.5)

    def _pwm_loop(self) -> None:
        period = self._pwm_period_s
        on_time = period * self._brightness
        off_time = period - on_time
        while not self._pwm_stop.is_set():
            # Re-read brightness each cycle so changes take effect within one
            # period without restarting the thread.
            on_time = period * self._brightness
            off_time = period - on_time
            try:
                self._request.set_value(self._line, self._line_value_on)
            except Exception:
                return
            if self._pwm_stop.wait(on_time):
                break
            try:
                self._request.set_value(self._line, self._line_value_off)
            except Exception:
                return
            if self._pwm_stop.wait(off_time):
                break

    def close(self) -> None:
        self._stop_pwm()
        if self._request is None:
            return
        try:
            # Restore backlight ON before releasing — otherwise the line floats
            # and on some boards the FET stays in whatever state it was last in.
            if self._line_value_on is not None:
                self._request.set_value(self._line, self._line_value_on)
        except Exception:
            pass
        try:
            self._request.release()
        except Exception:
            pass
        self._request = None
        self._available = False
