from __future__ import annotations

import json
import sys
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..hud import OverlayButton, OverlayRipple, OverlayState
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
        self._display_state, has_calibration = _load_display_state(config.state_file)
        self._previous_mode_before_off: str = "hud_only"
        self._toast: tuple[str, float] | None = None
        self._calibration_step = 0
        self._calibration_raws: list[tuple[int, int]] = []
        self._seq_picker_offset = 0
        self._ripples: list[OverlayRipple] = []

        self.reader.set_calibration(self._display_state.calibration)
        self.reader.on_tap = self._on_tap
        self.reader.on_press = self._on_press
        self.reader.on_any_activity = self._on_any_activity

        # Force calibration on first run (no saved state file). The user can't
        # interact reliably without it, so make this the very first overlay.
        if not has_calibration:
            self._mode = "calibrate"
            self._calibration_step = 0
            self._calibration_raws = []
            self.reader.set_raw_listener(self._on_calibration_tap)
        else:
            self._mode = (
                config.initial_mode
                if config.initial_mode in _CYCLE_MODES + ("home",)
                else "hud_only"
            )

    # ---- public API ----------------------------------------------------

    def transform_overlay(self, base: OverlayState) -> OverlayState:
        """Build the local panel's overlay for the current mode.

        ``base`` is the shared overlay state (what MJPEG gets). We pick
        elements from it depending on mode rather than mutate it.
        """
        now = time.monotonic()
        with self._lock:
            mode = self._mode
            toast = self._current_toast_locked()
            cutoff = now - _RIPPLE_DURATION_S
            self._ripples = [r for r in self._ripples if r.started_monotonic >= cutoff]
            ripples = list(self._ripples)

        rotation = self._display_state.rotation_quarter_turns
        ripples = [self._rotate_ripple(r, rotation) for r in ripples]

        if mode == "display_off":
            # Independent of backlight availability: blackout for safety so
            # the panel goes dark even if GPIO control failed.
            return OverlayState(blackout=True, ripples=ripples)

        if mode == "none":
            # Keep the NO IMAGE banner — it's part of the camera placeholder,
            # not a HUD element, and should always be visible when the camera
            # has fallen back to the black source.
            state = OverlayState(banner=base.banner)
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
        if ripples:
            state.ripples = ripples
        if rotation == 2 and state.buttons:
            state.buttons = [self._rotate_button(b) for b in state.buttons]
        return state

    def _rotate_ripple(self, ripple: OverlayRipple, rotation: int) -> OverlayRipple:
        if rotation % 4 != 2:
            return ripple
        return OverlayRipple(
            x=self.config.panel_width - 1 - ripple.x,
            y=self.config.panel_height - 1 - ripple.y,
            started_monotonic=ripple.started_monotonic,
        )

    def _rotate_button(self, btn: OverlayButton) -> OverlayButton:
        # Pre-flip button positions so sink rotation lands them where the user
        # tapped (taps are calibrated against the visible frame).
        return OverlayButton(
            x=self.config.panel_width - btn.x - btn.width,
            y=self.config.panel_height - btn.y - btn.height,
            width=btn.width,
            height=btn.height,
            label=btn.label,
            button_id=btn.button_id,
        )

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

    def _on_press(self, x: int, y: int) -> None:
        """Record a tap ripple in panel coordinates for visual feedback."""
        now = time.monotonic()
        with self._lock:
            self._ripples.append(OverlayRipple(x=x, y=y, started_monotonic=now))
            # Expire old ripples and keep the list bounded.
            cutoff = now - _RIPPLE_DURATION_S
            self._ripples = [r for r in self._ripples if r.started_monotonic >= cutoff][-8:]

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
            # Home is part of the cycle: tapping outside any button advances
            # to the next mode (back to "none"). Buttons themselves capture
            # their taps.
            if not self._handle_button_tap_locked(self._home_buttons(), x, y):
                self._advance_cycle_locked()
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
    ) -> bool:
        """Returns True if the tap hit a button."""
        for btn in buttons:
            if btn.x <= x < btn.x + btn.width and btn.y <= y < btn.y + btn.height:
                self._on_button_locked(btn.button_id)
                return True
        return False

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
        self._display_state.calibration.rotation_quarter_turns = (
            self._display_state.rotation_quarter_turns
        )
        _save_display_state(self.config.state_file, self._display_state)
        # The pixel rotation flips the panel — taps now arrive at flipped
        # coordinates relative to what the user sees. Force re-calibration.
        self._set_toast_locked(f"rot: {self._display_state.rotation_quarter_turns * 90}° — recalibrate")
        self._mode = "calibrate"
        self._calibration_step = 0
        self._calibration_raws = []
        self.reader.set_raw_listener(self._on_calibration_tap)

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
        if len(self._calibration_raws) != len(self._CALIBRATION_CORNERS):
            self._mode = "display"
            return

        # Build the panel-space targets that match the on-screen prompts.
        panel_targets: list[tuple[float, float]] = []
        for _label, (fx, fy) in self._CALIBRATION_CORNERS:
            panel_targets.append(
                (fx * (self.config.panel_width - 1), fy * (self.config.panel_height - 1))
            )

        try:
            calibration = TouchCalibration.from_corners(
                raw_corners=list(self._calibration_raws),
                panel_targets=panel_targets,
                panel_width=self.config.panel_width,
                panel_height=self.config.panel_height,
                rotation_quarter_turns=self._display_state.rotation_quarter_turns,
            )
        except ValueError as exc:
            print(f"Calibration fit failed: {exc}", file=sys.stderr)
            self._set_toast_locked("calibration failed")
            self._mode = "display"
            return

        self._display_state.calibration = calibration
        self.reader.set_calibration(calibration)
        _save_display_state(self.config.state_file, self._display_state)
        self._set_toast_locked("calibrated")
        # Print the fit so the user can sanity-check from the journal.
        print(
            f"Touch calibration: a={calibration.a:.4f} b={calibration.b:.4f} c={calibration.c:.2f} "
            f"d={calibration.d:.4f} e={calibration.e:.4f} f={calibration.f:.2f}"
        )
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
        # Sized to ~50% of the panel height so there's an obvious tap-outside
        # zone above and below — tapping outside cycles back to hud_only.
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
            top_inset_px=self.config.panel_height // 4,
            bottom_inset_px=self.config.panel_height // 4,
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
            top_inset_px=self.config.panel_height // 5,
            bottom_inset_px=self.config.panel_height // 5,
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
_RIPPLE_DURATION_S = 0.6


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
    bottom_inset_px: int = 0,
    skip_blank_labels: bool = False,
) -> list[OverlayButton]:
    inner_w = panel_w - 2 * margin_px
    inner_h = panel_h - 2 * margin_px - top_inset_px - bottom_inset_px
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


def _load_display_state(path: Path) -> tuple[_DisplayState, bool]:
    """Load persisted display state.

    Returns ``(state, has_calibration)``. ``has_calibration`` is False when no
    file exists or the file lacks a calibration entry — used by callers to
    force the calibration UI on first start.
    """
    if not path.exists():
        return _DisplayState(), False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"TouchUi: could not load {path}: {exc}", file=sys.stderr)
        return _DisplayState(), False
    rotation = int(data.get("rotation_quarter_turns", 0)) % 4
    calib_raw = data.get("calibration")
    # Only treat the file as calibrated when the affine "a" key is present —
    # legacy min/max calibrations are not migrated and need a fresh fit.
    has_calibration = isinstance(calib_raw, dict) and "a" in calib_raw
    if has_calibration:
        calibration = TouchCalibration.from_json(calib_raw)
    else:
        calibration = TouchCalibration()
    calibration.rotation_quarter_turns = rotation
    return _DisplayState(rotation_quarter_turns=rotation, calibration=calibration), has_calibration


def _save_display_state(path: Path, state: _DisplayState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "rotation_quarter_turns": state.rotation_quarter_turns,
        "calibration": state.calibration.to_json(),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
