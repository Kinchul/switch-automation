from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from statistics import median

from control import Button, ControllerBackend, ControllerConnectCancelled
from vision import CameraCapture, MatchResult, Roi, encode_rgb_frame
from vision.detector import StaticImageDetector
from vision.stream import OverlayBox, OverlayState

from .persistence import LoopStatsSnapshot, PersistentLoopControl, PersistentLoopStatsStore
from .sequence import (
    ActionSpec,
    SequenceConfigError,
    SequenceRuntime,
    StateSpec,
    build_runtime,
    load_sequences,
)

NotifyCallback = Callable[[str, str, list[Path]], None]


@dataclass(slots=True)
class LoopOutcome:
    status: str
    detail: str


@dataclass(slots=True)
class CameraLoopConfig:
    sequences_dir: Path = Path("sequences")
    default_sequence: str | None = None
    debug_dir: Path = Path("debug/camera")
    stats_file: Path = Path("debug/camera/loop_stats.json")
    control_file: Path = Path("debug/camera/loop_control.json")
    control_poll_interval: float = 0.5
    match_poll_interval: float = 0.1
    stats_checkpoint_interval: float = 5.0


@dataclass(slots=True)
class StateMatch:
    state_name: str
    detector: object | None
    result: MatchResult | None
    decision_score: float | None = None
    static_threshold: float | None = None
    dynamic_threshold: float | None = None
    effective_threshold: float | None = None
    decision_reason: str | None = None


@dataclass(slots=True)
class StateTransition:
    next_state: str
    match: StateMatch
    detail: str | None = None
    held_ms: int = 0


def _trimmed_mean(buf: deque, outlier_ratio: float = 0.25) -> float:
    if not buf:
        return 1.0
    sorted_vals = sorted(buf)
    trim = int(len(sorted_vals) * outlier_ratio)
    kept = sorted_vals[:len(sorted_vals) - trim] if trim > 0 else sorted_vals
    return sum(kept) / len(kept)


def _match_score(match: StateMatch | None) -> float:
    if match is None:
        return float("inf")
    if match.decision_score is not None:
        return match.decision_score
    if match.result is not None:
        return match.result.score
    return float("inf")


def _effective_threshold(match: StateMatch | None) -> float | None:
    if match is None:
        return None
    if match.effective_threshold is not None:
        return match.effective_threshold
    if match.static_threshold is not None:
        return match.static_threshold
    return None


class StopRequested(Exception):
    pass


class RestartRequested(Exception):
    pass


class ResetRequested(Exception):
    def __init__(self, *, timers_printed: bool = False) -> None:
        super().__init__()
        self.timers_printed = timers_printed


class CameraLoopRunner:
    def __init__(
        self,
        *,
        controller: ControllerBackend,
        capture: CameraCapture,
        config: CameraLoopConfig,
        notify_cb: NotifyCallback | None = None,
    ) -> None:
        self.controller = controller
        self.capture = capture
        self.config = config
        self.notify_cb = notify_cb
        self.stats = PersistentLoopStatsStore.load(config.stats_file)
        self.control = PersistentLoopControl.load(config.control_file)
        self._last_checkpoint_monotonic = time.monotonic()
        self._controller_connected = False
        self._preview_lock = threading.Lock()
        self._preview_step = "idle"
        self._preview_detail: str | None = None
        self._preview_boxes: list[OverlayBox] = []
        self._preview_buttons: tuple[str, ...] = ()
        self._preview_buttons_until_monotonic = 0.0
        self._current_sequence_id: str | None = None
        self._latest_outcome_frame_path: Path | None = None
        self._latest_state_roi_path: Path | None = None
        self._timeout_reset_count = 0
        self._entered_initial_state_once = False
        self._frame_buffer: deque = deque(maxlen=5)
        self._current_loop_decision_scores: dict[str, dict[str, float]] = {}
        self._previous_loop_decision_scores: dict[str, dict[str, float]] = {}
        self._failed_loop_score_history: dict[str, dict[str, deque[float]]] = {}

    def initialize(self) -> LoopStatsSnapshot:
        self.control.set_command("noop")
        self.stats.normalize_on_startup()
        sequence_id = self._ensure_selected_sequence()
        self._current_sequence_id = sequence_id
        return self.stats.snapshot(sequence_id)

    def connect(self) -> bool:
        if self._controller_connected:
            return False
        print("Connecting controller...")
        self._mark_current_status("connecting")
        self._set_preview_state("connecting", "Initializing controller connection...", [])

        def _on_status(state: str) -> None:
            print(f"Controller state: {state}")
            normalized = str(state).strip().lower().replace(" ", "_")
            status = "pairing" if normalized in {"reconnecting", "pairing"} else "connecting"
            if normalized == "connected":
                status = "paired"
            if "l+r" in str(state).lower():
                self._note_button_press((Button.L, Button.R), duration_seconds=3.0)
            self._mark_current_status(status)
            self._set_preview_state(status, f"Controller state: {state}", [])

        try:
            self.controller.connect(
                cancel_cb=lambda: self.control.refresh() in {"stop", "restart", "reset"},
                status_cb=_on_status,
            )
        except ControllerConnectCancelled:
            requested = self._consume_pending_command()
            if requested == "restart":
                self._mark_current_status("stopped", last_outcome="connect_cancelled")
                self._set_preview_state("stopped", "Controller connection cancelled for restart.", [])
                raise RestartRequested from None
            if requested == "reset":
                self._mark_current_status("stopped", last_outcome="connect_cancelled")
                self._set_preview_state("stopped", "Controller connection cancelled for reset.", [])
                raise ResetRequested from None
            self._mark_current_status("stopped", last_outcome="connect_cancelled")
            self._set_preview_state("stopped", "Controller connection cancelled.", [])
            raise StopRequested from None

        self._controller_connected = True
        self._mark_current_status("paired")
        self._set_preview_state("paired", "Controller connected.", [])
        return True

    def run_service(self, attempts: int = 0) -> LoopOutcome:
        print("Service is ready. Current mode: stopped.")
        pending_command: str | None = None
        while True:
            command = pending_command or self._wait_for_command()
            pending_command = None
            if command == "stop":
                self._disconnect_controller("Stop requested while idle.")
                continue
            if command == "pair":
                self._pair_controller()
                continue

            try:
                runtime = self._load_selected_runtime()
            except Exception as exc:
                self._mark_current_status("stopped", last_outcome="sequence_load_failed")
                self._set_preview_state("error", str(exc), [])
                print(f"Could not load the selected sequence: {exc}")
                continue

            try:
                self.connect()
            except StopRequested:
                self._disconnect_controller("Stop requested while connecting. Controller disconnected.")
                print("Stop requested while connecting the controller. Automation is now idle.")
                continue
            except RestartRequested:
                print("Restart requested while connecting the controller. Starting a fresh loop.")
                pending_command = "restart"
                continue
            except ResetRequested:
                print("Reset requested while connecting the controller. Forcing a game reset next.")
                pending_command = "reset"
                continue
            self._latest_outcome_frame_path = None
            self._latest_state_roi_path = None
            self._entered_initial_state_once = False
            self._current_loop_decision_scores[runtime.sequence_id] = {}
            if command == "restart":
                self._previous_loop_decision_scores[runtime.sequence_id] = {}
            stats = self.stats.start_new_loop(runtime.sequence_id)
            self._last_checkpoint_monotonic = time.monotonic()
            skip_startup_recovery = False
            if command == "reset":
                self._force_game_reset()
                skip_startup_recovery = True

            try:
                outcome = self.run_once(
                    runtime,
                    max_loops=attempts,
                    skip_startup_recovery=skip_startup_recovery,
                )
            except StopRequested:
                snapshot = self.stats.finish_loop(runtime.sequence_id, "stopped", status="stopped")
                self._disconnect_controller("Stop requested. Controller disconnected.")
                self._print_timers(runtime.sequence_id, snapshot)
                print("Stop requested. Automation is now idle.")
                continue
            except RestartRequested:
                snapshot = self.stats.finish_loop(
                    runtime.sequence_id,
                    "restart_requested",
                    status="stopped",
                )
                self._print_timers(runtime.sequence_id, snapshot)
                print("Restart requested. Starting a fresh loop.")
                pending_command = "restart"
                continue
            except ResetRequested as exc:
                snapshot = self.stats.finish_loop(
                    runtime.sequence_id,
                    "reset_requested",
                    status="stopped",
                )
                if not exc.timers_printed:
                    self._print_timers(runtime.sequence_id, snapshot)
                print("Reset requested. Forcing a game reset and starting a fresh loop.")
                pending_command = "reset"
                continue

            self._checkpoint_stats(force=True)
            snapshot = self.stats.finish_loop(runtime.sequence_id, outcome.status, status="stopped")
            self._disconnect_controller("Loop finished. Controller disconnected.")
            self._print_timers(runtime.sequence_id, snapshot)
            print(f"Outcome: {outcome.status} - {outcome.detail}")

    def run_once(
        self,
        runtime: SequenceRuntime,
        *,
        max_loops: int = 0,
        skip_startup_recovery: bool = False,
    ) -> LoopOutcome:
        self._current_sequence_id = runtime.sequence_id
        self._abort_if_control_requested()
        self._checkpoint_stats()
        start_match = None
        if not skip_startup_recovery:
            start_match = self._wait_for_recovery_match(runtime, step="Startup recovery scan")
            if start_match is None:
                return self._fail_recovery(
                    runtime,
                    state=None,
                    detail="Startup recovery scan failed before entering the sequence.",
                    outcome_name="startup_recovery_failed",
                    saved_state_name="startup-recovery",
                    notify=False,
                )
        if start_match is None:
            start_match = self._synthetic_state_match(runtime, runtime.definition.initial_state)

        return self._run_from_state(runtime, start_match, max_loops=max_loops)

    def _run_from_state(
        self,
        runtime: SequenceRuntime,
        entry_match: StateMatch,
        *,
        max_loops: int,
    ) -> LoopOutcome:
        current_match = entry_match
        while True:
            state = runtime.definition.states[current_match.state_name]
            limit_outcome = self._maybe_count_loop(runtime, state, max_loops=max_loops)
            if limit_outcome is not None:
                return limit_outcome

            result = self._execute_state(runtime, state, current_match)
            if isinstance(result, LoopOutcome):
                return result
            current_match = result.match

    def _maybe_count_loop(
        self,
        runtime: SequenceRuntime,
        state: StateSpec,
        *,
        max_loops: int,
    ) -> LoopOutcome | None:
        if state.name != runtime.definition.initial_state:
            return None

        if not self._entered_initial_state_once:
            self._entered_initial_state_once = True
            return None

        self._promote_loop_decision_scores(runtime.sequence_id)
        snapshot = self.stats.record_retry(runtime.sequence_id)
        print(f'Completed loop {snapshot.loop_counter} for sequence "{runtime.sequence_id}".')
        if max_loops and snapshot.loop_counter >= max_loops:
            return LoopOutcome(
                "max_attempts",
                f"Stopped after {snapshot.loop_counter} completed loops on sequence {runtime.sequence_id}.",
            )
        return None

    def _execute_state(
        self,
        runtime: SequenceRuntime,
        state: StateSpec,
        entry_match: StateMatch,
    ) -> LoopOutcome | StateTransition:
        self._on_state_reached(runtime, state, entry_match)

        if not state.next_states:
            return self._complete_terminal_state(runtime, state, entry_match)

        if state.action is not None and state.action.frequency_hz <= 0:
            print(f'Running single action for "{state.name}".')
            self._perform_action(state.action)
            self._checkpoint_stats()

        if state.reset_loop:
            frame = self.capture.get_frame()
            saved = self._save_target_failed_roi(frame, runtime.sequence_id, state, entry_match)
            pre_snapshot = self.stats.snapshot(runtime.sequence_id)
            self._promote_loop_decision_scores(runtime.sequence_id)
            snapshot = self.stats.record_retry(runtime.sequence_id)
            print(
                f'State "{state.name}" reset loop — loop {snapshot.loop_counter}.'
                + (f" Saved ROI to {saved}." if saved else "")
            )
            self._print_timers(runtime.sequence_id, pre_snapshot)
            raise ResetRequested(timers_printed=True)

        if state.decision_mode == "loop_baseline_step":
            return self._run_loop_baseline_step(runtime, state)

        immediate_transition = self._immediate_transition(runtime, state)
        if immediate_transition is not None:
            return immediate_transition

        next_action_at: float | None = None
        if state.action is not None and state.action.frequency_hz > 0:
            next_action_at = time.monotonic()

        deadline = None
        if state.timeout_ms > 0:
            deadline = time.monotonic() + (state.timeout_ms / 1000.0)

        matched_since = {
            next_state: None
            for next_state in state.next_states
            if runtime.detector_for(next_state) is not None
        }
        score_buffers: dict[str, deque] = {
            ns: deque(maxlen=runtime.definition.states[ns].scene.score_window)
            for ns in state.next_states
            if runtime.detector_for(ns) is not None
            and runtime.definition.states[ns].scene is not None
            and runtime.definition.states[ns].scene.score_window > 1
        }
        last_best_match: StateMatch | None = None
        state_entry_time = time.monotonic()

        while True:
            self._abort_if_control_requested()
            now = time.monotonic()
            if deadline is not None and now >= deadline:
                score_detail = ""
                if last_best_match is not None:
                    score_detail = f" Last candidate: {self._describe_match_decision(last_best_match)}."
                if state.timeout_next_state is not None:
                    tns = state.timeout_next_state
                    print(f'Timeout in state "{state.name}": no {state.next_states} detected, transitioning to "{tns}".{score_detail}')
                    transition_match = self._synthetic_state_match(runtime, tns)
                    if (
                        transition_match.result is None
                        and last_best_match is not None
                        and last_best_match.result is not None
                    ):
                        transition_match = StateMatch(
                            state_name=tns,
                            detector=last_best_match.detector,
                            result=last_best_match.result,
                            decision_score=last_best_match.decision_score,
                            static_threshold=last_best_match.static_threshold,
                            dynamic_threshold=last_best_match.dynamic_threshold,
                            effective_threshold=last_best_match.effective_threshold,
                            decision_reason=last_best_match.decision_reason,
                        )
                    self._print_target_decision(
                        state_name=state.name,
                        next_state=tns,
                        criterion="timeout_next_state",
                        detail=self._describe_match_decision(last_best_match),
                    )
                    return StateTransition(
                        next_state=tns,
                        match=transition_match,
                        detail="criterion=timeout_next_state",
                    )
                return self._handle_timeout(runtime, state, f'Timeout in state "{state.name}".{score_detail}')

            frame = self.capture.get_frame()
            self._frame_buffer.append(frame)
            if state.decision_mode == "best_score":
                transition, best_match = self._find_best_score_transition(
                    runtime,
                    state,
                    frame,
                    now,
                    matched_since,
                )
            else:
                transition, best_match = self._find_next_transition(
                    runtime,
                    state,
                    frame,
                    now,
                    matched_since,
                    score_buffers,
                )
            if transition is not None:
                detect_ms = int((time.monotonic() - state_entry_time) * 1000)
                matched_state = runtime.definition.states[transition.next_state]
                th = _effective_threshold(transition.match)
                if th is None:
                    th = matched_state.scene.threshold if matched_state.scene else 0.0
                score = _match_score(transition.match)
                timeout_part = f", timeout={state.timeout_ms}ms" if state.timeout_ms > 0 else ""
                extra = f", {transition.detail}" if transition.detail else ""
                held_part = f", held={transition.held_ms}ms" if transition.held_ms > 0 else ""
                print(f'Detected "{transition.next_state}": score={score:.4f}/th={th:.4f}{held_part}{extra}, in={detect_ms}ms{timeout_part}.')
                if transition.next_state.startswith("target_") or state.name.startswith("target_"):
                    self._print_target_decision(
                        state_name=state.name,
                        next_state=transition.next_state,
                        criterion="detector_match",
                        detail=self._describe_match_decision(transition.match),
                    )
                return transition

            if best_match is not None:
                last_best_match = best_match
                detail = f'Waiting for one of: {", ".join(state.next_states)}'
                if state.decision_mode == "best_score":
                    detail = (
                        f'Comparing outcomes: {", ".join(state.next_states)} '
                        f'(margin={state.decision_margin:.4f})'
                    )
                self._set_preview_detector(
                    step=f'State "{state.name}"',
                    detector=best_match.detector,
                    result=best_match.result,
                    detail=detail,
                )

            if next_action_at is not None and now >= next_action_at and state.action is not None:
                scheduled_at = next_action_at
                interval = state.action.interval_seconds or 0.0
                self._perform_action(state.action)
                self._checkpoint_stats()
                next_action_at = scheduled_at + interval

            sleep_for = self.config.match_poll_interval
            if next_action_at is not None:
                sleep_for = min(sleep_for, max(0.0, next_action_at - time.monotonic()))
            if deadline is not None:
                sleep_for = min(sleep_for, max(0.0, deadline - time.monotonic()))
            if sleep_for > 0:
                time.sleep(sleep_for)

    def _immediate_transition(
        self,
        runtime: SequenceRuntime,
        state: StateSpec,
    ) -> StateTransition | None:
        if not state.next_states:
            return None
        if state.action is not None and state.action.frequency_hz > 0:
            return None
        first_next = state.next_states[0]
        if runtime.definition.states[first_next].scene is not None:
            return None

        print(f'Immediately transitioning from "{state.name}" to "{first_next}".')
        return StateTransition(
            next_state=first_next,
            match=self._synthetic_state_match(runtime, first_next),
        )

    def _run_loop_baseline_step(
        self,
        runtime: SequenceRuntime,
        state: StateSpec,
    ) -> StateTransition:
        next_state_name = state.next_states[0]
        next_state = runtime.definition.states[next_state_name]
        detector = runtime.detector_for(next_state_name)
        if detector is None or next_state.scene is None or state.timeout_next_state is None:
            raise SequenceConfigError(
                f'State "{state.name}" loop_baseline_step is missing detector configuration.'
            )

        threshold = next_state.scene.threshold
        score_window = max(1, next_state.scene.score_window)
        buf = deque(maxlen=score_window) if score_window > 1 else None
        observed_scores: list[float] = []
        last_candidate: StateMatch | None = None
        deadline = time.monotonic() + (state.timeout_ms / 1000.0)
        state_entry_time = time.monotonic()

        while True:
            self._abort_if_control_requested()
            now = time.monotonic()
            if now >= deadline:
                break

            frame = self.capture.get_frame()
            self._frame_buffer.append(frame)
            result = detector.match(frame)
            candidate = StateMatch(
                next_state_name,
                detector,
                result,
                decision_score=result.score,
                static_threshold=threshold,
                effective_threshold=threshold,
            )
            if buf is not None:
                buf.append(result.score)
                if len(buf) == buf.maxlen:
                    candidate.decision_score = _trimmed_mean(buf)
                    observed_scores.append(candidate.decision_score)
                else:
                    candidate.decision_reason = f"window_not_ready {len(buf)}/{buf.maxlen}"
            else:
                observed_scores.append(result.score)
            last_candidate = candidate

            preview_detail = f'Observing "{next_state_name}" for loop decision: score={_match_score(candidate):.4f}.'
            self._set_preview_detector(
                step=f'State "{state.name}"',
                detector=detector,
                result=result,
                detail=preview_detail,
            )

            sleep_for = min(self.config.match_poll_interval, max(0.0, deadline - time.monotonic()))
            if sleep_for > 0:
                time.sleep(sleep_for)

        if last_candidate is None:
            return StateTransition(
                next_state=state.timeout_next_state,
                match=self._synthetic_state_match(runtime, state.timeout_next_state),
                detail="criterion=loop_baseline_step_empty",
            )

        loop_score = float(median(observed_scores)) if observed_scores else _match_score(last_candidate)
        baseline, trend, predicted_failed = self._predict_failed_loop_score(runtime.sequence_id, state)
        ok_step = state.decision_ok_step
        detect_ms = int((time.monotonic() - state_entry_time) * 1000)

        decision_match = StateMatch(
            state_name=next_state_name,
            detector=last_candidate.detector,
            result=last_candidate.result,
            decision_score=loop_score,
            static_threshold=threshold,
            dynamic_threshold=predicted_failed,
            effective_threshold=(predicted_failed + ok_step) if predicted_failed is not None else threshold,
        )

        if predicted_failed is None:
            bootstrap_threshold = threshold + ok_step
            if loop_score <= bootstrap_threshold:
                winner_name = next_state_name
                reason = "bootstrap_failed_static_plus_ok_step"
            else:
                winner_name = state.timeout_next_state
                reason = "bootstrap_ok_above_static_plus_ok_step"
            detail = (
                f"score={loop_score:.4f}, static_th={threshold:.4f}, "
                f"bootstrap_th={bootstrap_threshold:.4f}, ok_step={ok_step:.4f}, "
                f"criterion={reason}, observed={len(observed_scores)}"
            )
        else:
            jump = loop_score - predicted_failed
            if jump >= ok_step:
                winner_name = state.timeout_next_state
                reason = "ok_step_detected"
            else:
                winner_name = next_state_name
                reason = "failed_baseline_regime"
            detail = (
                f"score={loop_score:.4f}, predicted_failed={predicted_failed:.4f}, "
                f"baseline={baseline:.4f}, trend={trend:.4f}, jump={jump:+.4f}, "
                f"ok_step={ok_step:.4f}, criterion={reason}, observed={len(observed_scores)}"
            )

        decision_match.decision_reason = reason

        if winner_name == next_state_name:
            self._remember_failed_loop_score(runtime.sequence_id, state.name, loop_score, state.decision_history_window)
            transition_match = decision_match
        else:
            transition_match = StateMatch(
                state_name=winner_name,
                detector=decision_match.detector,
                result=decision_match.result,
                decision_score=decision_match.decision_score,
                static_threshold=decision_match.static_threshold,
                dynamic_threshold=decision_match.dynamic_threshold,
                effective_threshold=decision_match.effective_threshold,
                decision_reason=decision_match.decision_reason,
            )

        self._print_target_decision(
            state_name=state.name,
            next_state=winner_name,
            criterion="loop_baseline_step",
            detail=detail,
        )
        return StateTransition(
            next_state=winner_name,
            match=transition_match,
            detail=f"criterion=loop_baseline_step, observe_ms={detect_ms}, {detail}",
            held_ms=0,
        )

    def _predict_failed_loop_score(
        self,
        sequence_id: str,
        state: StateSpec,
    ) -> tuple[float | None, float, float | None]:
        histories = self._failed_loop_score_history.setdefault(sequence_id, {})
        history_deque = histories.get(state.name)
        if history_deque is None or not history_deque:
            persisted = self.stats.failed_loop_score_history(sequence_id, state.name)
            if persisted:
                history_deque = deque(
                    persisted[-state.decision_history_window:],
                    maxlen=max(1, state.decision_history_window),
                )
                histories[state.name] = history_deque

        history = list(history_deque or ())
        if not history:
            return None, 0.0, None

        history_window = max(1, state.decision_history_window)
        history = history[-history_window:]
        baseline = float(median(history))

        trend_window = max(1, state.decision_trend_window)
        if len(history) < 2:
            trend = 0.0
        else:
            deltas = [curr - prev for prev, curr in zip(history[:-1], history[1:])]
            deltas = deltas[-trend_window:]
            trend = float(median(deltas)) if deltas else 0.0

        return baseline, trend, baseline + trend

    def _find_next_transition(
        self,
        runtime: SequenceRuntime,
        state: StateSpec,
        frame,
        now: float,
        matched_since: dict[str, float | None],
        score_buffers: dict[str, deque] | None = None,
    ) -> tuple[StateTransition | None, StateMatch | None]:
        best_match: StateMatch | None = None
        preferred_preview_match: StateMatch | None = None
        for next_state_name in state.next_states:
            detector = runtime.detector_for(next_state_name)
            if detector is None:
                continue

            next_state = runtime.definition.states[next_state_name]
            result = detector.match(frame)
            candidate = StateMatch(next_state_name, detector, result, decision_score=result.score)
            if preferred_preview_match is None:
                preferred_preview_match = candidate

            threshold = next_state.scene.threshold if next_state.scene else 0.0
            candidate.static_threshold = threshold
            buf = score_buffers.get(next_state_name) if score_buffers else None
            if buf is not None:
                buf.append(result.score)
                current_mean = _trimmed_mean(buf)
                candidate.decision_score = current_mean
                is_matched = len(buf) == buf.maxlen and current_mean < threshold
                candidate.effective_threshold = threshold
                if len(buf) < buf.maxlen:
                    candidate.decision_reason = f"window_not_ready {len(buf)}/{buf.maxlen}"
                elif candidate.decision_score is not None and candidate.decision_score >= threshold:
                    candidate.decision_reason = "score_above_threshold"
                else:
                    candidate.decision_reason = "score_passed_threshold"
            else:
                is_matched = result.matched
                candidate.effective_threshold = threshold
                candidate.decision_reason = "score_passed_threshold" if is_matched else "score_above_threshold"

            if best_match is None or _match_score(candidate) < _match_score(best_match):
                best_match = candidate

            if is_matched:
                started = matched_since[next_state_name]
                if started is None:
                    matched_since[next_state_name] = now
                    started = now
                held_ms = int(max(0.0, now - started) * 1000)
                required_hold_ms = max(0, next_state.scene.hold_ms if next_state.scene else 0)
                if held_ms >= required_hold_ms:
                    score_used = _match_score(candidate)
                    candidate.decision_reason = "accepted"
                    self._set_preview_detector(
                        step=f'State "{state.name}"',
                        detector=detector,
                        result=result,
                        detail=f'Matched "{next_state_name}": score={score_used:.4f}/th={threshold:.4f}.',
                    )
                    return StateTransition(next_state=next_state_name, match=candidate, held_ms=held_ms), candidate
                candidate.decision_reason = f"hold_pending {held_ms}/{required_hold_ms}ms"
            else:
                matched_since[next_state_name] = None

        return None, preferred_preview_match or best_match

    def _find_best_score_transition(
        self,
        runtime: SequenceRuntime,
        state: StateSpec,
        frame,
        now: float,
        matched_since: dict[str, float | None],
    ) -> tuple[StateTransition | None, StateMatch | None]:
        candidates: list[StateMatch] = []
        for next_state_name in state.next_states:
            detector = runtime.detector_for(next_state_name)
            if detector is None:
                continue
            result = detector.match(frame)
            candidates.append(StateMatch(next_state_name, detector, result, decision_score=result.score))

        if not candidates:
            return None, None

        candidates.sort(key=_match_score)
        preview_match = candidates[0]
        matched_candidates = [
            candidate
            for candidate in candidates
            if candidate.result is not None and candidate.result.matched
        ]
        if not matched_candidates:
            for candidate in candidates:
                matched_since[candidate.state_name] = None
            return None, preview_match

        best_match = matched_candidates[0]
        runner_up = matched_candidates[1] if len(matched_candidates) > 1 else None
        best_result = best_match.result
        if best_result is None:
            return None, preview_match

        runner_up_score = _match_score(runner_up)
        best_score = _match_score(best_match)
        score_margin = runner_up_score - best_score
        winner_name = best_match.state_name

        if runner_up is not None and score_margin < state.decision_margin:
            matched_since[winner_name] = None
            for candidate in matched_candidates[1:]:
                matched_since[candidate.state_name] = None
            return None, best_match

        started = matched_since[winner_name]
        if started is None:
            matched_since[winner_name] = now
            started = now
        held_ms = int(max(0.0, now - started) * 1000)
        next_state = runtime.definition.states[winner_name]
        required_hold_ms = max(0, next_state.scene.hold_ms if next_state.scene else 0)
        if held_ms >= required_hold_ms:
            winner_threshold = next_state.scene.threshold if next_state.scene else 0.0
            margin_detail = f"margin={score_margin:.4f}" if runner_up is not None else None
            self._set_preview_detector(
                step=f'State "{state.name}"',
                detector=best_match.detector,
                result=best_result,
                detail=f'Decided "{winner_name}": score={best_score:.4f}/th={winner_threshold:.4f}.',
            )
            return (
                StateTransition(next_state=winner_name, match=best_match, detail=margin_detail, held_ms=held_ms),
                best_match,
            )

        for candidate in candidates:
            if candidate.state_name == winner_name:
                continue
            matched_since[candidate.state_name] = None

        return None, best_match

    def _complete_terminal_state(
        self,
        runtime: SequenceRuntime,
        state: StateSpec,
        match: StateMatch,
    ) -> LoopOutcome:
        frame = self.capture.get_frame()
        detail = f'Reached terminal state "{state.name}" in sequence "{runtime.sequence_id}".'
        directory = self._save_target_ok(frame, runtime.sequence_id, state, match)
        detail = f"{detail} Saved to {directory}."

        if state.notification == "mail":
            self._notify(
                f'Sequence "{runtime.sequence_id}" reached "{state.name}"',
                self._notification_body(detail),
            )
        return LoopOutcome("completed", detail)

    def _wait_for_command(self) -> str:
        sequence_id = self._ensure_selected_sequence()
        self._current_sequence_id = sequence_id
        self.stats.mark_status(sequence_id, "stopped")
        while True:
            command = self._consume_control_command()
            if command == "stop":
                print("Stop command received.")
                return command
            if command == "pair":
                print("Pair command received.")
                return command
            if command == "restart":
                print("Restart command received.")
                return command
            if command == "reset":
                print("Reset command received.")
                return command
            time.sleep(self.config.control_poll_interval)

    def _consume_control_command(self) -> str:
        command = self.control.refresh()
        if command in {"pair", "restart", "reset", "stop"}:
            self.control.set_command("noop")
            return command
        return "noop"

    def _consume_pending_command(self) -> str:
        command = self.control.refresh()
        if command in {"restart", "reset", "stop"}:
            self.control.set_command("noop")
            return command
        return "noop"

    def _wait_for_state_match(
        self,
        runtime: SequenceRuntime,
        state_names: tuple[str, ...],
        *,
        timeout_ms: int,
        step: str,
        zero_timeout_is_infinite: bool,
        hold_ms_override: int | None = None,
    ) -> StateMatch | None:
        detectable_states = [
            state_name
            for state_name in state_names
            if runtime.detector_for(state_name) is not None
        ]
        if not detectable_states:
            return None

        matched_since = {state_name: None for state_name in detectable_states}
        score_buffers: dict[str, deque] = {
            state_name: deque(maxlen=runtime.definition.states[state_name].scene.score_window)
            for state_name in detectable_states
            if runtime.definition.states[state_name].scene is not None
            and runtime.definition.states[state_name].scene.score_window > 1
        }
        single_pass = timeout_ms <= 0 and not zero_timeout_is_infinite
        scan_start = time.monotonic()
        deadline = None if timeout_ms <= 0 else time.monotonic() + (timeout_ms / 1000.0)
        scan_index = 0
        while True:
            self._abort_if_control_requested()
            now = time.monotonic()
            if deadline is not None and now >= deadline:
                return None

            scan_index += 1
            pass_started = time.monotonic()
            frame = self.capture.get_frame()
            boxes: list[OverlayBox] = []
            best_match: StateMatch | None = None
            scan_entries: list[tuple[str, MatchResult, float, int, int]] = []
            for state_name in detectable_states:
                state = runtime.definition.states[state_name]
                detector = runtime.detector_for(state_name)
                if detector is None:
                    continue
                result = detector.match(frame)
                candidate = StateMatch(state_name, detector, result, decision_score=result.score)
                boxes.extend(self._boxes_for_state(state_name, state.scene, result))

                buf = score_buffers.get(state_name)
                if buf is not None:
                    buf.append(result.score)
                    threshold = state.scene.threshold if state.scene is not None else 0.0
                    current_mean = _trimmed_mean(buf)
                    candidate.decision_score = current_mean
                    is_matched = len(buf) == buf.maxlen and current_mean < threshold
                else:
                    is_matched = result.matched

                if best_match is None or _match_score(candidate) < _match_score(best_match):
                    best_match = candidate

                if is_matched:
                    started = matched_since[state_name]
                    if started is None:
                        matched_since[state_name] = now
                        started = now
                    held_ms = int(max(0.0, now - started) * 1000)
                    configured_hold_ms = state.scene.hold_ms if state.scene is not None else 0
                    effective_hold_ms = hold_ms_override if hold_ms_override is not None else configured_hold_ms
                    threshold = state.scene.threshold if state.scene is not None else 0.0
                    scan_entries.append((state_name, result, threshold, held_ms, effective_hold_ms))
                    if held_ms >= effective_hold_ms:
                        detect_ms = int((now - scan_start) * 1000)
                        timeout_part = f", timeout={timeout_ms}ms" if timeout_ms > 0 else ""
                        score_used = _match_score(candidate)
                        self._set_preview_detector(
                            step=step,
                            detector=detector,
                            result=result,
                            detail=f'Matched "{state_name}": score={score_used:.4f}/th={threshold:.4f}.',
                        )
                        held_part = f", held={held_ms}ms" if held_ms > 0 else ""
                        print(f'Detected "{state_name}": score={score_used:.4f}/th={threshold:.4f}{held_part}, in={detect_ms}ms{timeout_part} ({step}).')
                        return candidate
                else:
                    matched_since[state_name] = None
                    configured_hold_ms = state.scene.hold_ms if state.scene is not None else 0
                    effective_hold_ms = hold_ms_override if hold_ms_override is not None else configured_hold_ms
                    threshold = state.scene.threshold if state.scene is not None else 0.0
                    scan_entries.append((state_name, result, threshold, 0, effective_hold_ms))
                    matched_since[state_name] = None

            pass_elapsed_ms = int(max(0.0, time.monotonic() - pass_started) * 1000)
            scan_detail = self._format_scan_entries(scan_entries)
            if self._should_log_scan_entries(step):
                remaining_ms = None
                if deadline is not None:
                    remaining_ms = max(0, int((deadline - time.monotonic()) * 1000))
                print(
                    f'{step} pass {scan_index} took {pass_elapsed_ms} ms'
                    + ("" if remaining_ms is None else f", remaining={remaining_ms} ms")
                    + f": {scan_detail}"
                )
            self._set_preview_boxes(
                step=step,
                detail=scan_detail,
                boxes=boxes,
            )
            if single_pass:
                return None
            sleep_for = self.config.match_poll_interval
            if deadline is not None:
                sleep_for = min(sleep_for, max(0.0, deadline - time.monotonic()))
            if sleep_for > 0:
                time.sleep(sleep_for)

    def _synthetic_state_match(self, runtime: SequenceRuntime, state_name: str) -> StateMatch:
        detector = runtime.detector_for(state_name)
        result = None
        if detector is not None:
            result = detector.match(self.capture.get_frame())
        return StateMatch(
            state_name=state_name,
            detector=detector,
            result=result,
            decision_score=result.score if result is not None else None,
        )

    def _handle_timeout(
        self,
        runtime: SequenceRuntime,
        state: StateSpec | None,
        detail: str,
    ) -> LoopOutcome | StateTransition:
        self._timeout_reset_count += 1
        print(f"{detail} Attempting recovery...")
        recovery_match = self._wait_for_recovery_match(runtime, step="Recovery")
        if recovery_match is not None:
            return StateTransition(next_state=recovery_match.state_name, match=recovery_match)

        return self._fail_recovery(
            runtime,
            state=state,
            detail=f"{detail} Recovery failed.",
            outcome_name="recovery_failed",
            saved_state_name="timeout" if state is None else state.name,
            notify=True,
        )

    def _fail_recovery(
        self,
        runtime: SequenceRuntime,
        *,
        state: StateSpec | None,
        detail: str,
        outcome_name: str,
        saved_state_name: str,
        notify: bool,
    ) -> LoopOutcome:
        frame = self.capture.get_frame()
        saved = self._save_failed_recovery(frame, runtime.sequence_id)
        outcome_detail = f"{detail} Saved frame to {saved}."
        if notify:
            self._notify(
                f'Sequence "{runtime.sequence_id}" {outcome_name.replace("_", " ")}',
                self._notification_body(outcome_detail),
            )
        return LoopOutcome(outcome_name, outcome_detail)

    def _wait_for_recovery_match(
        self,
        runtime: SequenceRuntime,
        *,
        step: str,
    ) -> StateMatch | None:
        self.controller.release_all()
        timeout_ms = runtime.definition.recovery.timeout_ms
        recovery_states = runtime.definition.recovery.states
        detectable_states = tuple(
            state_name
            for state_name in recovery_states
            if runtime.definition.states[state_name].scene is not None
        )
        print(
            f"{step}: recovery timeout={timeout_ms} ms, "
            f"states={list(detectable_states)}"
        )
        return self._wait_for_state_match(
            runtime,
            detectable_states,
            timeout_ms=timeout_ms,
            step=step,
            zero_timeout_is_infinite=False,
            hold_ms_override=0,
        )

    def _should_log_scan_entries(self, step: str) -> bool:
        normalized = step.lower()
        return "recovery" in normalized

    def _format_scan_entries(
        self,
        scan_entries: list[tuple[str, MatchResult, float, int, int]],
    ) -> str:
        if not scan_entries:
            return "No detector scores."
        sorted_entries = sorted(scan_entries, key=lambda entry: entry[1].score)
        parts: list[str] = []
        for state_name, result, threshold, held_ms, hold_ms in sorted_entries:
            status = "matched" if result.matched else "searching"
            hold_part = f", hold={held_ms}/{hold_ms}ms" if hold_ms > 0 else ""
            offset_part = f", offset=({result.offset_x},{result.offset_y})"
            parts.append(
                f"{state_name}={result.score:.4f}/th={threshold:.4f} "
                f"({status}{hold_part}{offset_part})"
            )
        return "; ".join(parts)

    def _perform_action(self, action: ActionSpec) -> None:
        self._note_button_press(action.buttons)
        self.controller.press(
            *action.buttons,
            down=action.down_seconds,
            up=action.up_seconds,
        )

    def _force_game_reset(self) -> None:
        buttons = (Button.A, Button.B, Button.X, Button.Y)
        detail = "Sending forced game reset buttons (A+B+X+Y)."
        self._set_preview_state("resetting", detail, [])
        print(detail)
        self._note_button_press(buttons, duration_seconds=2.0)
        self.controller.press(*buttons, down=0.4, up=1.0)

    def _on_state_reached(
        self,
        runtime: SequenceRuntime,
        state: StateSpec,
        match: StateMatch,
    ) -> None:
        self._record_loop_decision_score(runtime.sequence_id, state.name, match)
        detail = f'Entered state "{state.name}" in sequence "{runtime.sequence_id}".'
        self._set_preview_detector(
            step=f'State "{state.name}"',
            detector=match.detector,
            result=match.result,
            detail=detail,
        )
        print(detail)

    def _record_loop_decision_score(self, sequence_id: str, state_name: str, match: StateMatch) -> None:
        score = _match_score(match)
        if score == float("inf"):
            return
        self._current_loop_decision_scores.setdefault(sequence_id, {})[state_name] = score

    def _remember_failed_loop_score(
        self,
        sequence_id: str,
        state_name: str,
        score: float,
        maxlen: int,
    ) -> None:
        histories = self._failed_loop_score_history.setdefault(sequence_id, {})
        history = histories.get(state_name)
        if history is None or history.maxlen != maxlen:
            existing = list(history) if history is not None else []
            history = deque(existing[-maxlen:], maxlen=maxlen)
            histories[state_name] = history
        history.append(score)
        self.stats.set_failed_loop_score_history(sequence_id, state_name, list(history))

    def _promote_loop_decision_scores(self, sequence_id: str) -> None:
        current = self._current_loop_decision_scores.get(sequence_id, {})
        if current:
            self._previous_loop_decision_scores[sequence_id] = dict(current)
        self._current_loop_decision_scores[sequence_id] = {}

    def _previous_loop_decision_score(self, sequence_id: str, state_name: str) -> float | None:
        return self._previous_loop_decision_scores.get(sequence_id, {}).get(state_name)

    def _describe_match_decision(self, match: StateMatch | None) -> str:
        if match is None:
            return "no candidate"
        score = _match_score(match)
        effective_th = _effective_threshold(match)
        threshold_part = ""
        if effective_th is not None:
            threshold_part = f"/eff_th={effective_th:.4f}"
            if match.static_threshold is not None:
                threshold_part += f" (static={match.static_threshold:.4f}"
                if match.dynamic_threshold is not None:
                    threshold_part += f", dyn={match.dynamic_threshold:.4f}"
                threshold_part += ")"
        reason_part = f", reason={match.decision_reason}" if match.decision_reason else ""
        return f'{match.state_name}={score:.4f}{threshold_part}{reason_part}'

    def _print_target_decision(
        self,
        *,
        state_name: str,
        next_state: str,
        criterion: str,
        detail: str,
    ) -> None:
        print(f'=== TARGET DECISION: {state_name} -> {next_state} | criterion={criterion} | {detail} ===')

    def _pair_controller(self) -> None:
        if self._controller_connected:
            self._disconnect_controller("Pair requested while already connected. Controller disconnected.")
            self._mark_current_status("stopped")
            self._set_preview_state("stopped", "Controller disconnected. Waiting for restart.", [])
            print("Controller was already connected and is now disconnected. Waiting for restart.")
            return

        restore_reconnect = None
        if hasattr(self.controller, "set_reconnect"):
            restore_reconnect = True
            self.controller.set_reconnect(False)

        try:
            self.connect()
        except StopRequested:
            self._mark_current_status("stopped", last_outcome="pair_cancelled")
            self._disconnect_controller("Pairing cancelled. Controller disconnected.")
            print("Pairing cancelled. Automation is now idle.")
            return
        except RestartRequested:
            self._mark_current_status("stopped", last_outcome="pair_cancelled")
            self._disconnect_controller("Pairing interrupted by restart. Controller disconnected.")
            self.control.set_command("restart")
            print("Pairing interrupted by restart request. Starting a fresh loop next.")
            return
        except ResetRequested:
            self._mark_current_status("stopped", last_outcome="pair_cancelled")
            self._disconnect_controller("Pairing interrupted by reset. Controller disconnected.")
            self.control.set_command("reset")
            print("Pairing interrupted by reset request. Forcing a game reset next.")
            return
        finally:
            if restore_reconnect is not None:
                self.controller.set_reconnect(restore_reconnect)

        self._mark_current_status("stopped")
        self._disconnect_controller("Pairing completed. Controller disconnected.")
        self._set_preview_state("stopped", "Controller paired and disconnected. Waiting for restart.", [])
        print("Controller paired successfully and disconnected. Future restarts will reconnect as needed.")

    def preview_overlay_lines(self) -> list[str]:
        sequence_id = self._current_sequence_id or self._ensure_selected_sequence()
        snapshot = self.stats.snapshot(sequence_id)
        with self._preview_lock:
            step = self._preview_step
        return [
            f"sequence: {sequence_id}",
            f"mode: {snapshot.status}",
            f"total: {_format_duration(snapshot.total_elapsed_seconds)}",
            f"loop: {_format_duration(snapshot.loop_elapsed_seconds)}",
            f"count: {snapshot.loop_counter}",
            f"timeouts: {self._timeout_reset_count}",
            f"step: {step}",
        ]

    def preview_overlay_state(self) -> OverlayState:
        now = time.monotonic()
        with self._preview_lock:
            boxes = list(self._preview_boxes)
            button_lines = self._preview_button_lines(now)
        return OverlayState(
            lines=self.preview_overlay_lines(),
            boxes=boxes,
            bottom_right_lines=button_lines,
        )

    def _abort_if_control_requested(self) -> None:
        command = self._consume_pending_command()
        if command == "stop":
            raise StopRequested
        if command == "restart":
            raise RestartRequested
        if command == "reset":
            raise ResetRequested

    def _checkpoint_stats(self, force: bool = False) -> None:
        if self._current_sequence_id is None:
            return
        now = time.monotonic()
        if not force and now - self._last_checkpoint_monotonic < self.config.stats_checkpoint_interval:
            return
        self.stats.checkpoint_running(self._current_sequence_id)
        self._last_checkpoint_monotonic = now

    def _mark_current_status(self, status: str, last_outcome: str | None = None) -> None:
        if self._current_sequence_id is None:
            self._current_sequence_id = self._ensure_selected_sequence()
        self.stats.mark_status(self._current_sequence_id, status, last_outcome=last_outcome)

    def _disconnect_controller(self, detail: str | None = None) -> None:
        if not self._controller_connected:
            return
        try:
            self.controller.release_all()
        except Exception:
            pass
        try:
            self.controller.close()
        except Exception as exc:
            print(f"Controller disconnect warning: {exc}")
        finally:
            self._controller_connected = False
        if detail:
            print(detail)

    def _set_preview_state(
        self,
        step: str,
        detail: str | None,
        boxes: list[OverlayBox],
    ) -> None:
        with self._preview_lock:
            self._preview_step = step
            self._preview_detail = detail
            self._preview_boxes = list(boxes)

    def _note_button_press(
        self,
        buttons: tuple[Button, ...],
        *,
        duration_seconds: float = 1.5,
    ) -> None:
        labels = tuple(button.value for button in buttons)
        with self._preview_lock:
            self._preview_buttons = labels
            self._preview_buttons_until_monotonic = time.monotonic() + max(0.1, duration_seconds)

    def _preview_button_lines(self, now: float) -> list[str]:
        if not self._preview_buttons or now > self._preview_buttons_until_monotonic:
            return []
        return [" + ".join(self._preview_buttons)]

    def _set_preview_boxes(
        self,
        *,
        step: str,
        detail: str | None,
        boxes: list[OverlayBox],
    ) -> None:
        self._set_preview_state(step, detail, boxes)

    def _set_preview_detector(
        self,
        *,
        step: str,
        detector,
        result: MatchResult | None = None,
        detail: str | None = None,
    ) -> None:
        boxes: list[OverlayBox] = []
        roi = getattr(detector, "roi", None) if detector is not None else None
        if roi is not None:
            focus_roi = roi
            if result is not None and isinstance(detector, StaticImageDetector):
                focus_roi = Roi(
                    x=roi.x + result.offset_x,
                    y=roi.y + result.offset_y,
                    width=roi.width,
                    height=roi.height,
                )
            boxes = [
                self._box_for_roi(
                    self._detector_name(detector),
                    focus_roi,
                    matched=bool(result and result.matched),
                )
            ]

        if detail is None and detector is not None and result is not None:
            detail = self._format_detector_result(detector, result)
        self._set_preview_state(step, detail, boxes)

    def _boxes_for_state(self, state_name: str, scene, result: MatchResult | None) -> list[OverlayBox]:
        if scene is None or scene.roi is None:
            return []
        focus_roi = Roi(
            x=scene.roi.x,
            y=scene.roi.y,
            width=scene.roi.width,
            height=scene.roi.height,
        )
        matched = False
        if result is not None:
            focus_roi = Roi(
                x=scene.roi.x + result.offset_x,
                y=scene.roi.y + result.offset_y,
                width=scene.roi.width,
                height=scene.roi.height,
            )
            matched = result.matched
        return [self._box_for_roi(state_name, focus_roi, matched=matched)]

    def _box_for_roi(self, label: str, roi: Roi, *, matched: bool) -> OverlayBox:
        if matched:
            outline = (64, 220, 120, 255)
            fill = (64, 220, 120, 56)
        else:
            outline = (255, 215, 0, 255)
            fill = (255, 215, 0, 48)
        return OverlayBox(
            x=roi.x,
            y=roi.y,
            width=roi.width,
            height=roi.height,
            label=label,
            outline=outline,
            fill=fill,
        )

    def _detector_name(self, detector) -> str:
        return getattr(detector, "name", detector.__class__.__name__.replace("Detector", "").lower())

    def _format_detector_result(self, detector, result: MatchResult) -> str:
        if result.detail:
            return f'{self._detector_name(detector)}: score={result.score:.4f} ({result.detail})'
        return f'{self._detector_name(detector)}: score={result.score:.4f}'

    def _notification_body(self, detail: str) -> str:
        sequence_id = self._current_sequence_id or self._ensure_selected_sequence()
        snapshot = self.stats.snapshot(sequence_id)
        with self._preview_lock:
            step = self._preview_step
            preview_detail = self._preview_detail
        lines = [
            detail,
            "",
            f"sequence: {sequence_id}",
            f"mode: {snapshot.status}",
            f"total: {_format_duration(snapshot.total_elapsed_seconds)}",
            f"loop: {_format_duration(snapshot.loop_elapsed_seconds)}",
            f"count: {snapshot.loop_counter}",
            f"last: {snapshot.last_outcome or '-'}",
            f"step: {step}",
            f"timeouts: {self._timeout_reset_count}",
        ]
        if preview_detail:
            lines.append(f"debug: {preview_detail}")
        return "\n".join(lines)

    def _notify(self, title: str, detail: str) -> None:
        if self.notify_cb is None:
            return
        attachments = [
            path
            for path in (self._latest_outcome_frame_path, self._latest_state_roi_path)
            if path is not None and path.exists()
        ]
        try:
            self.notify_cb(title, detail, attachments)
        except Exception as exc:
            print(f"Notification failed: {exc}")

    def _save_target_failed_roi(
        self,
        frame,
        sequence_id: str,
        state: StateSpec,
        match: StateMatch,
    ) -> Path | None:
        roi = self._resolve_match_roi(state, detector=match.detector, result=match.result)
        if roi is None:
            return None
        cropped = roi.crop(frame)
        directory = self.config.debug_dir / sequence_id / "target_failed"
        directory.mkdir(parents=True, exist_ok=True)
        score_used = _match_score(match)
        path = directory / f"{time.strftime('%Y%m%d-%H%M%S')}-score-{score_used:.4f}.jpg"
        path.write_bytes(encode_rgb_frame(cropped, quality=95))
        self._latest_state_roi_path = path
        return path

    def _save_failed_recovery(self, frame, sequence_id: str) -> Path:
        directory = self.config.debug_dir / sequence_id / "failed_recovery"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{time.strftime('%Y%m%d-%H%M%S')}.jpg"
        path.write_bytes(encode_rgb_frame(frame, quality=95))
        self._latest_outcome_frame_path = path
        return path

    def _save_target_ok(
        self,
        frame,
        sequence_id: str,
        state: StateSpec,
        match: StateMatch,
    ) -> Path:
        directory = self.config.debug_dir / sequence_id / "target_ok"
        directory.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d-%H%M%S")
        for i, buf_frame in enumerate(self._frame_buffer):
            buf_path = directory / f"{ts}-before-{i:02d}.jpg"
            buf_path.write_bytes(encode_rgb_frame(buf_frame, quality=95))
        full_path = directory / f"{ts}-full.jpg"
        full_path.write_bytes(encode_rgb_frame(frame, quality=95))
        self._latest_outcome_frame_path = full_path
        roi = self._resolve_match_roi(state, detector=match.detector, result=match.result)
        if roi is not None:
            roi_path = directory / f"{ts}-roi.jpg"
            roi_path.write_bytes(encode_rgb_frame(roi.crop(frame), quality=95))
            self._latest_state_roi_path = roi_path
        return directory

    def _resolve_match_roi(
        self,
        state: StateSpec,
        *,
        detector,
        result: MatchResult | None,
    ) -> Roi | None:
        if result is None:
            return None
        if state.scene is not None:
            return Roi(
                x=state.scene.roi.x + result.offset_x,
                y=state.scene.roi.y + result.offset_y,
                width=state.scene.roi.width,
                height=state.scene.roi.height,
            )

        detector_roi = getattr(detector, "roi", None) if detector is not None else None
        if detector_roi is None:
            return None
        return Roi(
            x=detector_roi.x + result.offset_x,
            y=detector_roi.y + result.offset_y,
            width=detector_roi.width,
            height=detector_roi.height,
        )

    def _load_selected_runtime(self) -> SequenceRuntime:
        self.control.refresh()
        definitions = load_sequences(self.config.sequences_dir)
        if not definitions:
            raise RuntimeError(f"No sequence JSON files were found in {self.config.sequences_dir}.")

        selected_sequence = self.control.selected_sequence
        if selected_sequence not in definitions:
            selected_sequence = self.config.default_sequence or next(iter(definitions))
            self.control.set_selected_sequence(selected_sequence)

        definition = definitions[selected_sequence]
        self._current_sequence_id = selected_sequence
        try:
            return build_runtime(definition)
        except SequenceConfigError as exc:
            raise RuntimeError(
                f'Sequence "{selected_sequence}" is invalid: {exc}'
            ) from exc

    def _ensure_selected_sequence(self) -> str:
        self.control.refresh()
        definitions = load_sequences(self.config.sequences_dir)
        if not definitions:
            raise RuntimeError(f"No sequence JSON files were found in {self.config.sequences_dir}.")

        selected_sequence = self.control.selected_sequence
        if selected_sequence in definitions:
            return selected_sequence

        selected_sequence = self.config.default_sequence or next(iter(definitions))
        self.control.set_selected_sequence(selected_sequence)
        return selected_sequence

    def _print_timers(self, sequence_id: str, stats: LoopStatsSnapshot) -> None:
        print(
            f"Timers ({sequence_id}): "
            f"total={_format_duration(stats.total_elapsed_seconds)} "
            f"loop={_format_duration(stats.loop_elapsed_seconds)} "
            f"count={stats.loop_counter}"
        )


def _format_duration(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"
