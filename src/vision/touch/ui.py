from __future__ import annotations

import json
import sys
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..hud import OverlayButton, OverlayState
from .backlight import BacklightController
from .reader import TouchCalibration, TouchReader, load_calibration, save_calibration


# Overlay modes that the panel cycles between when the user taps outside any
# button. Only these two are part of the cycle — the button overlays are
# entered explicitly and only the dedicated Exit button leaves them.
_CYCLE_MODES: tuple[str, ...] = ("none", "hud_only")


@dataclass(slots=True)
class TouchUiConfig:
    panel_width: int = 480
    panel_height: int = 320
    state_file: Path = Path("debug/camera/display.json")
    sequences_dir: Path = Path("sequences")
    initial_mode: str = "hud_only"


@dataclass(slots=True)
class _DisplayState:
    rotation_quarter_turns: int = 0
    calibration: TouchCalibration = field(default_factory=TouchCalibration)


class TouchUi:
    """Owns the local panel's overlay state machine and dispatches touch input.

    Modes:
        none           — capture only, no HUD, no buttons. Tap → cycle.
        hud_only       — capture + full HUD (same content as MJPEG). Tap → cycle.
        home           — capture + 3 big buttons (Action, Display, Exit). HUD hidden.
        action         — Action menu. HUD top-left visible (sequence/status/step).
        display        — Display menu (Off, Calibrate, Rotate, Back).
        seq_picker     — Scrollable list of sequences from sequences_dir.
        display_off    — Backlight off / black frame. Any tap returns to hud_only.
        calibrate      — Capture 4 corner taps and persist them.

    The class is thread-safe: ``handle_tap`` runs on the touch reader thread,
    ``transform_overlay`` on the output pipeline thread.
    """

    def __init__(
        self,
        *,
        config: TouchUiConfig,
        backlight: BacklightController,
        reader: TouchReader,
        loop_control,
        request_select_sequence: Callable[[str], None],
        list_sequences: Callable[[], list[str]],
    ) -> None:
        self.config = config
        self.backlight = backlight
        self.reader = reader
        self.loop_control = loop_control
        self.request_select_sequence = request_select_sequence
        self.list_sequences = list_sequences

        self._lock = threading.RLock()
        self._mode = config.initial_mode if config.initial_mode in _CYCLE_MODES + ("home",) else "hud_only"
        self._display_state = _load_display_state(config.state_file)
        self._previous_mode_before_off: str = "hud_only"
        self._toast: tuple[str, float] | None = None
        self._calibration_step = 0
        self._calibration_raws: list[tuple[int, int]] = []
        self._seq_picker_offset = 0

        self.reader.set_calibration(self._display_state.calibration)
        self.reader.on_tap = self._on_tap
        self.reader.on_any_activity = self._on_any_activity

    # ---- public API ----------------------------------------------------

    def transform_overlay(self, base: OverlayState) -> OverlayState:
        """Build the local panel's overlay for the current mode.

        ``base`` is the shared overlay state (what MJPEG gets). We pick
        elements from it depending on mode rather than mutate it.
        """
        with self._lock:
            mode = self._mode
            toast = self._current_toast_locked()

        if mode == "display_off":
            # Independent of backlight availability: blackout for safety so
            # the panel goes dark even if GPIO control failed.
            return OverlayState(blackout=True)

        if mode == "none":
            state = OverlayState()
        elif mode == "hud_only":
            state = OverlayState(
                top_left_lines=list(base.top_left_lines),
                bottom_left_lines=list(base.bottom_left_lines),
                top_right_lines=list(base.top_right_lines),
                bottom_right_lines=list(base.bottom_right_lines),
                boxes=list(base.boxes),
                banner=base.banner,
            )
        elif mode == "home":
            state = OverlayState(buttons=list(self._home_buttons()), banner=base.banner)
        elif mode == "action":
            state = OverlayState(
                top_left_lines=list(base.top_left_lines),
                buttons=list(self._action_buttons()),
                banner=base.banner,
            )
        elif mode == "display":
            state = OverlayState(buttons=list(self._display_buttons()), banner=base.banner)
        elif mode == "seq_picker":
            state = OverlayState(buttons=list(self._sequence_buttons()), banner=base.banner)
        elif mode == "calibrate":
            state = self._calibration_overlay()
        else:
            state = OverlayState()

        if toast:
            # Show toast in bottom-right of panel as a small line.
            state.bottom_right_lines = [toast] + list(state.bottom_right_lines)
        return state

    def close(self) -> None:
        # Persist state on shutdown.
        with self._lock:
            _save_display_state(self.config.state_file, self._display_state)

    @property
    def rotation_quarter_turns(self) -> int:
        with self._lock:
            return self._display_state.rotation_quarter_turns

    # ---- callbacks from TouchReader -----------------------------------

    def _on_any_activity(self) -> None:
        """Wake-from-off — runs at press start, before tap is finalized."""
        with self._lock:
            if self._mode == "display_off":
                self._wake_locked()

    def _on_tap(self, event) -> None:
        x, y = event.x, event.y
        with self._lock:
            if self._mode == "display_off":
                # The wake already happened in _on_any_activity. Swallow this
                # tap — don't let it count as a button press on the screen
                # the user can't see yet.
                return
            self._dispatch_tap_locked(x, y)

    # ---- mode handling -------------------------------------------------

    def _dispatch_tap_locked(self, x: int, y: int) -> None:
        mode = self._mode
        if mode in _CYCLE_MODES:
            self._advance_cycle_locked()
            return
        if mode == "home":
            self._handle_button_tap_locked(self._home_buttons(), x, y)
            return
        if mode == "action":
            self._handle_button_tap_locked(self._action_buttons(), x, y)
            return
        if mode == "display":
            self._handle_button_tap_locked(self._display_buttons(), x, y)
            return
        if mode == "seq_picker":
            self._handle_button_tap_locked(self._sequence_buttons(), x, y)
            return
        if mode == "calibrate":
            # Calibration uses raw listener path; ignore calibrated taps here.
            return

    def _advance_cycle_locked(self) -> None:
        cycle = (*_CYCLE_MODES, "home")
        try:
            idx = cycle.index(self._mode)
        except ValueError:
            self._mode = cycle[0]
            return
        self._mode = cycle[(idx + 1) % len(cycle)]

    def _handle_button_tap_locked(
        self, buttons: Iterable[OverlayButton], x: int, y: int
    ) -> None:
        for btn in buttons:
            if btn.x <= x < btn.x + btn.width and btn.y <= y < btn.y + btn.height:
                self._on_button_locked(btn.button_id)
                return
        # No-op for taps outside any button in button overlays.

    def _on_button_locked(self, button_id: str) -> None:
        if button_id == "exit":
            self._mode = "hud_only"
            return
        if button_id == "back_home":
            self._mode = "home"
            return

        if button_id == "menu_action":
            self._mode = "action"
            return
        if button_id == "menu_display":
            self._mode = "display"
            return

        if button_id in {"act_stop", "act_restart", "act_reset", "act_pair"}:
            command = button_id.split("_", 1)[1]
            try:
                self.loop_control.set_command(command)
                self._set_toast_locked(f"{command} sent")
            except Exception as exc:
                self._set_toast_locked(f"{command} failed")
                print(f"TouchUi: could not send command {command}: {exc}", file=sys.stderr)
            return

        if button_id == "act_sequence":
            self._seq_picker_offset = 0
            self._mode = "seq_picker"
            return

        if button_id == "disp_off":
            self._sleep_locked()
            return
        if button_id == "disp_calibrate":
            self._begin_calibration_locked()
            return
        if button_id == "disp_rotate":
            self._cycle_rotation_locked()
            return

        if button_id.startswith("seq_pick:"):
            sequence_id = button_id.split(":", 1)[1]
            try:
                self.request_select_sequence(sequence_id)
                self._set_toast_locked(f"seq: {sequence_id}")
            except Exception as exc:
                self._set_toast_locked("select failed")
                print(f"TouchUi: select_sequence({sequence_id}) failed: {exc}", file=sys.stderr)
            self._mode = "action"
            return

        if button_id == "seq_prev":
            self._seq_picker_offset = max(0, self._seq_picker_offset - _SEQ_PAGE_SIZE)
            return
        if button_id == "seq_next":
            self._seq_picker_offset += _SEQ_PAGE_SIZE
            return

    # ---- sleep / wake --------------------------------------------------

    def _sleep_locked(self) -> None:
        self._previous_mode_before_off = "hud_only"
        self._mode = "display_off"
        self.backlight.set_on(False)

    def _wake_locked(self) -> None:
        self.backlight.set_on(True)
        self._mode = self._previous_mode_before_off

    # ---- rotation ------------------------------------------------------

    def _cycle_rotation_locked(self) -> None:
        # Step by 2 quarter-turns so the rotated render still matches the
        # panel's (W, H) — 90/270 would require pre-rendering at (H, W) which
        # we don't support yet. 0° and 180° (flip) cover the practical case
        # of the panel being mounted upside down.
        self._display_state.rotation_quarter_turns = (
            self._display_state.rotation_quarter_turns + 2
        ) % 4
        # Persist rotation into calibration so the touch reader maps to the
        # new orientation immediately.
        self._display_state.calibration = TouchCalibration(
            **{
                **self._display_state.calibration.to_json(),
                "rotation_quarter_turns": self._display_state.rotation_quarter_turns,
            }
        )
        self.reader.set_calibration(self._display_state.calibration)
        _save_display_state(self.config.state_file, self._display_state)
        self._set_toast_locked(f"rot: {self._display_state.rotation_quarter_turns * 90}°")

    # ---- calibration ---------------------------------------------------

    _CALIBRATION_CORNERS: tuple[tuple[str, tuple[float, float]], ...] = (
        ("top-left", (0.05, 0.05)),
        ("top-right", (0.95, 0.05)),
        ("bottom-right", (0.95, 0.95)),
        ("bottom-left", (0.05, 0.95)),
    )

    def _begin_calibration_locked(self) -> None:
        self._mode = "calibrate"
        self._calibration_step = 0
        self._calibration_raws = []
        self.reader.set_raw_listener(self._on_calibration_tap)

    def _on_calibration_tap(self, raw_x: int, raw_y: int) -> None:
        with self._lock:
            self._calibration_raws.append((raw_x, raw_y))
            self._calibration_step += 1
            if self._calibration_step >= len(self._CALIBRATION_CORNERS):
                self._finalize_calibration_locked()
                self.reader.set_raw_listener(None)

    def _finalize_calibration_locked(self) -> None:
        if len(self._calibration_raws) != 4:
            self._mode = "display"
            return
        tl, tr, br, bl = self._calibration_raws

        # Average corner raws to derive min/max for each axis. We assume the
        # device is roughly axis-aligned (the panel rotation is software-only;
        # the touch IC reports its native frame). Auto-detect swap/invert.
        x_left = (tl[0] + bl[0]) / 2
        x_right = (tr[0] + br[0]) / 2
        y_top = (tl[1] + tr[1]) / 2
        y_bottom = (bl[1] + br[1]) / 2

        # Detect swap_xy: if X axis varies less between left/right than between
        # top/bottom, the IC has X and Y swapped relative to our prompts.
        x_axis_range = abs(x_right - x_left)
        y_axis_range = abs(y_bottom - y_top)
        swap_xy = x_axis_range < y_axis_range / 2 or y_axis_range < x_axis_range / 2 and (
            abs((tl[1] + tr[1]) / 2 - (bl[1] + br[1]) / 2) < abs((tl[0] + bl[0]) / 2 - (tr[0] + br[0]) / 2)
        )

        if swap_xy:
            x_left = (tl[1] + bl[1]) / 2
            x_right = (tr[1] + br[1]) / 2
            y_top = (tl[0] + tr[0]) / 2
            y_bottom = (bl[0] + br[0]) / 2

        invert_x = x_left > x_right
        invert_y = y_top > y_bottom

        raw_x_min = int(min(x_left, x_right))
        raw_x_max = int(max(x_left, x_right))
        raw_y_min = int(min(y_top, y_bottom))
        raw_y_max = int(max(y_top, y_bottom))

        # Inset prompts were at 5%/95% of panel; expand the raw range so 0..1
        # maps to the *full* panel.
        x_span = (raw_x_max - raw_x_min) / 0.9
        y_span = (raw_y_max - raw_y_min) / 0.9
        raw_x_min = int(raw_x_min - x_span * 0.05)
        raw_x_max = int(raw_x_max + x_span * 0.05)
        raw_y_min = int(raw_y_min - y_span * 0.05)
        raw_y_max = int(raw_y_max + y_span * 0.05)

        self._display_state.calibration = TouchCalibration(
            raw_x_min=raw_x_min,
            raw_x_max=raw_x_max,
            raw_y_min=raw_y_min,
            raw_y_max=raw_y_max,
            swap_xy=swap_xy,
            invert_x=invert_x,
            invert_y=invert_y,
            panel_width=self.config.panel_width,
            panel_height=self.config.panel_height,
            rotation_quarter_turns=self._display_state.rotation_quarter_turns,
        )
        self.reader.set_calibration(self._display_state.calibration)
        _save_display_state(self.config.state_file, self._display_state)
        self._set_toast_locked("calibrated")
        self._mode = "display"

    def _calibration_overlay(self) -> OverlayState:
        if self._calibration_step >= len(self._CALIBRATION_CORNERS):
            return OverlayState(banner="calibrating...")
        corner_name, (fx, fy) = self._CALIBRATION_CORNERS[self._calibration_step]
        # Draw a small target as a button (so the renderer handles it). It is
        # not actually tappable — the raw listener bypasses the button hit-test.
        target_size = 60
        cx = int(self.config.panel_width * fx) - target_size // 2
        cy = int(self.config.panel_height * fy) - target_size // 2
        target = OverlayButton(
            x=cx,
            y=cy,
            width=target_size,
            height=target_size,
            label="+",
            button_id="calib_target",
        )
        step = self._calibration_step + 1
        return OverlayState(
            buttons=[target],
            top_left_lines=[f"calibrate {step}/4", f"tap {corner_name}"],
        )

    # ---- button layouts ------------------------------------------------

    def _home_buttons(self) -> list[OverlayButton]:
        return _grid(
            self.config.panel_width,
            self.config.panel_height,
            [
                ("Action", "menu_action"),
                ("Display", "menu_display"),
                ("Exit", "exit"),
            ],
            cols=3,
            rows=1,
        )

    def _action_buttons(self) -> list[OverlayButton]:
        # Top-left HUD takes ~40% of width, ~50% of height. Lay out buttons in
        # a 3x2 grid below/right of it.
        return _grid(
            self.config.panel_width,
            self.config.panel_height,
            [
                ("Stop", "act_stop"),
                ("Restart", "act_restart"),
                ("Reset", "act_reset"),
                ("Pair", "act_pair"),
                ("Sequence", "act_sequence"),
                ("Back", "back_home"),
            ],
            cols=3,
            rows=2,
            top_inset_px=110,  # leave room for HUD top-left card
        )

    def _display_buttons(self) -> list[OverlayButton]:
        return _grid(
            self.config.panel_width,
            self.config.panel_height,
            [
                ("Off", "disp_off"),
                ("Calibrate", "disp_calibrate"),
                ("Rotate", "disp_rotate"),
                ("Back", "back_home"),
            ],
            cols=2,
            rows=2,
        )

    def _sequence_buttons(self) -> list[OverlayButton]:
        try:
            sequences = self.list_sequences()
        except Exception as exc:
            print(f"TouchUi: list_sequences failed: {exc}", file=sys.stderr)
            sequences = []

        page = sequences[self._seq_picker_offset : self._seq_picker_offset + _SEQ_PAGE_SIZE]
        items: list[tuple[str, str]] = [(seq, f"seq_pick:{seq}") for seq in page]

        # Reserve last slot of 3x2 for navigation row: prev / next / back
        nav: list[tuple[str, str]] = []
        if self._seq_picker_offset > 0:
            nav.append(("<", "seq_prev"))
        if self._seq_picker_offset + _SEQ_PAGE_SIZE < len(sequences):
            nav.append((">", "seq_next"))
        nav.append(("Back", "back_home"))

        # Pad items to fit a clean 2-row, 3-col grid + nav row at bottom.
        while len(items) < _SEQ_PAGE_SIZE:
            items.append(("", "seq_noop"))

        all_items = items + _pad_nav(nav)
        return _grid(
            self.config.panel_width,
            self.config.panel_height,
            all_items,
            cols=3,
            rows=3,
            skip_blank_labels=True,
        )

    # ---- toasts --------------------------------------------------------

    def _set_toast_locked(self, message: str, *, duration_seconds: float = 1.5) -> None:
        self._toast = (message, time.monotonic() + duration_seconds)

    def _current_toast_locked(self) -> str | None:
        if self._toast is None:
            return None
        message, deadline = self._toast
        if time.monotonic() > deadline:
            self._toast = None
            return None
        return message


_SEQ_PAGE_SIZE = 6


def _grid(
    panel_w: int,
    panel_h: int,
    items: list[tuple[str, str]],
    *,
    cols: int,
    rows: int,
    margin_px: int = 12,
    gap_px: int = 8,
    top_inset_px: int = 0,
    skip_blank_labels: bool = False,
) -> list[OverlayButton]:
    inner_w = panel_w - 2 * margin_px
    inner_h = panel_h - 2 * margin_px - top_inset_px
    cell_w = (inner_w - gap_px * (cols - 1)) // cols
    cell_h = (inner_h - gap_px * (rows - 1)) // rows

    buttons: list[OverlayButton] = []
    for idx, (label, button_id) in enumerate(items[: cols * rows]):
        if skip_blank_labels and not label:
            continue
        row = idx // cols
        col = idx % cols
        x = margin_px + col * (cell_w + gap_px)
        y = margin_px + top_inset_px + row * (cell_h + gap_px)
        buttons.append(
            OverlayButton(
                x=x,
                y=y,
                width=cell_w,
                height=cell_h,
                label=label,
                button_id=button_id,
            )
        )
    return buttons


def _pad_nav(nav: list[tuple[str, str]]) -> list[tuple[str, str]]:
    while len(nav) < 3:
        nav.insert(0, ("", "seq_noop"))
    return nav


def _load_display_state(path: Path) -> _DisplayState:
    if not path.exists():
        return _DisplayState()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"TouchUi: could not load {path}: {exc}", file=sys.stderr)
        return _DisplayState()
    rotation = int(data.get("rotation_quarter_turns", 0)) % 4
    calib_raw = data.get("calibration") or {}
    calibration = TouchCalibration.from_json(calib_raw) if isinstance(calib_raw, dict) else TouchCalibration()
    # Force calibration's rotation field to the persisted display rotation.
    calibration = TouchCalibration(
        **{
            **calibration.to_json(),
            "rotation_quarter_turns": rotation,
        }
    )
    return _DisplayState(rotation_quarter_turns=rotation, calibration=calibration)


def _save_display_state(path: Path, state: _DisplayState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "rotation_quarter_turns": state.rotation_quarter_turns,
        "calibration": state.calibration.to_json(),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
