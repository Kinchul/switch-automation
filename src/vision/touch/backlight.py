from __future__ import annotations

import sys
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

    def __init__(self, *, chip: str = "/dev/gpiochip0", line: int = 18) -> None:
        self._chip_path = chip
        self._line = line
        self._request: Any = None
        self._gpiod: Any = None
        self._line_value_on: Any = None
        self._line_value_off: Any = None
        self._on = True
        self._available = False
        self._fallback_blackout = False

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

    def set_on(self, on: bool) -> None:
        if on == self._on:
            return
        self._on = on
        if not self._available or self._request is None:
            return
        try:
            value = self._line_value_on if on else self._line_value_off
            self._request.set_value(self._line, value)
        except Exception as exc:
            print(f"Backlight set_on({on}) failed: {exc}", file=sys.stderr)

    def close(self) -> None:
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
