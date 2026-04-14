from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Callable
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from control import Button, ControllerBackend, ControllerConnectCancelled
from vision import CameraCapture, encode_rgb_frame
from vision.detector import (
    BlackScreenDetector,
    InvariantColorDetector,
    MatchResult,
    Roi,
    StaticImageDetector,
)
from vision.stream import OverlayBox, OverlayState

NotifyCallback = Callable[[str, str, list[Path]], None]


@dataclass(slots=True)
class LoopOutcome:
    status: str
    detail: str


@dataclass(slots=True)
class RecoveryStageMatch:
    name: str
    detector: object
    result: MatchResult
    detail: str


class StopRequested(Exception):
    pass


class RestartRequested(Exception):
    pass


class UnexpectedSceneMatched(Exception):
    def __init__(self, scene_name: str, score: float) -> None:
        self.scene_name = scene_name
        self.score = score
        super().__init__(f'Unexpected scene "{scene_name}" matched (score={score:.4f}).')


@dataclass(slots=True)
class LoopStatsSnapshot:
    loop_counter: int
    total_elapsed_seconds: float
    loop_elapsed_seconds: float
    status: str
    last_outcome: str | None


@dataclass(slots=True)
class PersistentLoopStats:
    path: Path
    loop_counter: int = 0
    total_elapsed_seconds: float = 0.0
    current_loop_elapsed_seconds_accum: float = 0.0
    active_loop_started_at: str | None = None
    status: str = "stopped"
    last_outcome: str | None = None
    updated_at: str | None = None

    @classmethod
    def load(cls, path: Path) -> PersistentLoopStats:
        if not path.exists():
            return cls(path=path)

        data = json.loads(path.read_text())
        return cls(
            path=path,
            loop_counter=int(data.get("loop_counter", 0)),
            total_elapsed_seconds=float(data.get("total_elapsed_seconds", 0.0)),
            current_loop_elapsed_seconds_accum=float(data.get("current_loop_elapsed_seconds", 0.0)),
            active_loop_started_at=data.get("active_loop_started_at"),
            status=data.get("status", "stopped"),
            last_outcome=data.get("last_outcome"),
            updated_at=data.get("updated_at"),
        )

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.updated_at = _utcnow().isoformat()
        self.path.write_text(
            json.dumps(
                {
                    "loop_counter": self.loop_counter,
                    "total_elapsed_seconds": self.total_elapsed_seconds,
                    "current_loop_elapsed_seconds": self.current_loop_elapsed_seconds_accum,
                    "active_loop_started_at": self.active_loop_started_at,
                    "status": self.status,
                    "last_outcome": self.last_outcome,
                    "updated_at": self.updated_at,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        _chmod_if_possible(self.path, 0o644)

    def snapshot(self) -> LoopStatsSnapshot:
        now = _utcnow()
        return LoopStatsSnapshot(
            loop_counter=self.loop_counter,
            total_elapsed_seconds=self.total_elapsed_seconds + self.current_loop_total_seconds(now),
            loop_elapsed_seconds=self.current_loop_elapsed_seconds(now),
            status=self.status,
            last_outcome=self.last_outcome,
        )

    def start_new_loop(self) -> LoopStatsSnapshot:
        if self.current_loop_total_seconds() > 0:
            self.total_elapsed_seconds += self.current_loop_total_seconds()
            self.current_loop_elapsed_seconds_accum = 0.0
            self.active_loop_started_at = None

        self.loop_counter += 1
        self.current_loop_elapsed_seconds_accum = 0.0
        self.active_loop_started_at = _utcnow().isoformat()
        self.status = "running"
        self.save()
        return self.snapshot()

    def checkpoint_running(self) -> LoopStatsSnapshot:
        if self.active_loop_started_at is not None:
            self.current_loop_elapsed_seconds_accum += self._active_segment_seconds()
            self.active_loop_started_at = _utcnow().isoformat()
            self.status = "running"
            self.save()
        return self.snapshot()

    def finish_loop(self, outcome: str, *, status: str = "stopped") -> LoopStatsSnapshot:
        self.total_elapsed_seconds += self.current_loop_total_seconds()
        self.current_loop_elapsed_seconds_accum = 0.0
        self.active_loop_started_at = None
        self.status = status
        self.last_outcome = outcome
        self.save()
        return self.snapshot()

    def mark_status(self, status: str, last_outcome: str | None = None) -> LoopStatsSnapshot:
        self.status = status
        if last_outcome is not None:
            self.last_outcome = last_outcome
        self.save()
        return self.snapshot()

    def current_loop_elapsed_seconds(self, now: datetime | None = None) -> float:
        return self.current_loop_total_seconds(now)

    def current_loop_total_seconds(self, now: datetime | None = None) -> float:
        return self.current_loop_elapsed_seconds_accum + self._active_segment_seconds(now)

    def normalize_on_startup(self) -> LoopStatsSnapshot:
        if self.active_loop_started_at is not None and self.updated_at is not None:
            updated_at = datetime.fromisoformat(self.updated_at)
            started_at = datetime.fromisoformat(self.active_loop_started_at)
            if updated_at >= started_at:
                self.current_loop_elapsed_seconds_accum += (updated_at - started_at).total_seconds()
        self.active_loop_started_at = None
        self.status = "stopped"
        self.save()
        return self.snapshot()

    def _active_segment_seconds(self, now: datetime | None = None) -> float:
        if self.active_loop_started_at is None:
            return 0.0
        if now is None:
            now = _utcnow()
        started_at = datetime.fromisoformat(self.active_loop_started_at)
        return max(0.0, (now - started_at).total_seconds())


@dataclass(slots=True)
class PersistentLoopControl:
    path: Path
    command: str = "noop"
    updated_at: str | None = None

    @classmethod
    def load(cls, path: Path) -> PersistentLoopControl:
        if not path.exists():
            return cls(path=path)
        data = json.loads(path.read_text())
        return cls(
            path=path,
            command=data.get("command", "noop"),
            updated_at=data.get("updated_at"),
        )

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.updated_at = _utcnow().isoformat()
        self.path.write_text(
            json.dumps(
                {
                    "command": self.command,
                    "updated_at": self.updated_at,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        _chmod_if_possible(self.path, 0o666)

    def set_command(self, command: str) -> None:
        self.command = command
        self.save()

    def refresh(self) -> str:
        if self.path.exists():
            data = json.loads(self.path.read_text())
            self.command = data.get("command", "noop")
            self.updated_at = data.get("updated_at")
        return self.command


@dataclass(slots=True)
class CameraLoopConfig:
    start_detector: StaticImageDetector
    press_start_detector: StaticImageDetector
    select_save_detector: StaticImageDetector
    previously_detector: StaticImageDetector
    ready_detector: StaticImageDetector
    target_failed_detector: StaticImageDetector
    black_screen_detector: BlackScreenDetector
    target_ok_detector: InvariantColorDetector | None = None
    target_fail_example_detector: InvariantColorDetector | None = None
    startup_mash_down: float = 0.05
    startup_mash_up: float = 0.15
    startup_continue_poll_gap: float = 0.20
    press_interval: float = 1.0
    settle_time: float = 0.35
    poll_interval: float = 0.25
    match_poll_interval: float = 0.05
    step_timeout: float = 45.0
    start_scene_timeout: float = 6.0
    blackscreen_timeout: float = 25.0
    outcome_timeout: float = 20.0
    outcome_settle_delay: float = 0.7
    target_failed_hold: float = 0.8
    target_ok_hold: float = 1.0
    target_ok_score_margin: float = 0.04
    success_candidate_hold: float = 3.5
    success_candidate_fail_margin: float = 0.03
    success_confirm_checks: int = 3
    success_confirm_interval: float = 0.4
    recent_failed_max: int = 20
    recent_failed_similarity_threshold: float = 0.06
    recent_failed_stride: int = 4
    recent_failed_store_margin: float = 0.02
    timeout_recovery_limit: int = 1
    debug_dir: Path = Path("debug/camera/outcomes")
    failed_roi_dir: Path = Path("debug/camera/failed_rois")
    stats_file: Path = Path("debug/camera/loop_stats.json")
    control_file: Path = Path("debug/camera/loop_control.json")
    restart_combo_hold: float = 0.15
    restart_combo_release: float = 1.5
    post_connect_restart_delay: float = 1.5
    control_poll_interval: float = 0.5
    stats_checkpoint_interval: float = 5.0


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
        self.stats = PersistentLoopStats.load(config.stats_file)
        self.control = PersistentLoopControl.load(config.control_file)
        self._last_checkpoint_monotonic = time.monotonic()
        self._controller_connected = False
        self._preview_lock = threading.Lock()
        self._preview_step = "idle"
        self._preview_detail: str | None = None
        self._preview_boxes: list[OverlayBox] = []
        self._latest_outcome_frame_path: Path | None = None
        self._latest_failed_roi_path: Path | None = None
        self._latest_candidate_roi_path: Path | None = None
        self._timeout_reset_count = 0
        self._recent_failed_rois: deque[np.ndarray] = deque(maxlen=self.config.recent_failed_max)

    def connect(self) -> bool:
        if self._controller_connected:
            return False
        print("Connecting controller...")
        self.stats.mark_status("connecting")
        self._set_preview_state("connecting", "Initializing controller connection...", [])

        def _on_status(state: str) -> None:
            print(f"Controller state: {state}")
            normalized = str(state).strip().lower().replace(" ", "_")
            status = "pairing" if normalized in {"reconnecting", "pairing"} else "connecting"
            if normalized == "connected":
                status = "paired"
            self.stats.mark_status(status)
            self._set_preview_state(status, f"Controller state: {state}", [])

        try:
            self.controller.connect(
                cancel_cb=lambda: self.control.refresh() in {"stop", "restart"},
                status_cb=_on_status,
            )
        except ControllerConnectCancelled:
            self.stats.mark_status("stopped", last_outcome="connect_cancelled")
            self._set_preview_state("stopped", "Controller connection cancelled.", [])
            raise StopRequested
        self._controller_connected = True
        self.stats.mark_status("paired")
        self._set_preview_state("paired", "Controller connected.", [])
        return True

    def initialize(self) -> LoopStatsSnapshot:
        self.control.set_command("noop")
        stats = self.stats.normalize_on_startup()
        return stats

    def run_service(self, attempts: int = 0) -> LoopOutcome:
        attempt = 0
        print("Service is ready. Current mode: stopped.")
        while True:
            command = self._wait_for_command()
            if command == "stop":
                continue
            if command == "pair":
                self._pair_controller()
                continue

            attempt = 0
            while True:
                attempt += 1
                connected_now = self.connect()
                startup_recovery_match = None
                if attempt == 1:
                    recovery_wait = self.config.post_connect_restart_delay if connected_now else 0.0
                    startup_recovery_match = self._wait_for_startup_recovery_match(recovery_wait)
                if startup_recovery_match is None:
                    self._trigger_restart_loop()
                else:
                    self._set_preview_detector(
                        step="Startup recovery pending",
                        detector=startup_recovery_match.detector,
                        result=startup_recovery_match.result,
                        detail=f"Matched {startup_recovery_match.detail}; skipping restart combo.",
                    )
                    print(
                        f'Startup recovery found "{startup_recovery_match.name}" '
                        f"(score={startup_recovery_match.result.score:.4f}); skipping restart combo."
                    )
                stats = self.stats.start_new_loop()
                self._last_checkpoint_monotonic = time.monotonic()
                print(f"\n=== Attempt {attempt} / Loop {stats.loop_counter} ===")
                self._print_timers(stats)
                try:
                    outcome = self.run_once(
                        startup_recovery_match=startup_recovery_match if attempt == 1 else None,
                    )
                except StopRequested:
                    snapshot = self.stats.finish_loop("stopped", status="stopped")
                    self._print_timers(snapshot)
                    print("Stop requested. Automation is now idle.")
                    self._notify("Loop stopped", "Stop requested while the loop was running.")
                    break
                except RestartRequested:
                    snapshot = self.stats.finish_loop("restart_requested", status="stopped")
                    self._print_timers(snapshot)
                    print("Restart requested. Starting a fresh loop.")
                    self.control.set_command("restart")
                    continue

                self._checkpoint_stats(force=True)
                print(f"Outcome: {outcome.status} - {outcome.detail}")

                if outcome.status == "retry":
                    if "Timed out" in outcome.detail:
                        self._timeout_reset_count += 1
                    snapshot = self.stats.finish_loop("retry", status="stopped")
                    self._print_timers(snapshot)
                    if attempts and attempt >= attempts:
                        self.stats.mark_status("stopped", last_outcome="max_attempts")
                        final_outcome = LoopOutcome("max_attempts", f"Stopped after {attempt} retry attempts.")
                        self._notify(self._notification_title(final_outcome), final_outcome.detail)
                        return final_outcome
                    if self._consume_control_command() == "stop":
                        print("Stop requested after retry. Automation is now idle.")
                        self._notify("Loop stopped", "Stop requested after an automatic retry.")
                        break
                    continue

                final_status = "stopped"
                if outcome.status == "success_candidate":
                    snapshot = self.stats.finish_loop(outcome.status, status=final_status)
                else:
                    snapshot = self.stats.finish_loop(outcome.status, status=final_status)
                self._print_timers(snapshot)
                print("Automation returned to stopped mode.")
                self._notify(self._notification_title(outcome), outcome.detail)
                break

        return LoopOutcome("stopped", "Service stopped.")

    def run_once(
        self,
        recovery_attempts_left: int | None = None,
        *,
        startup_recovery_match: RecoveryStageMatch | None = None,
    ) -> LoopOutcome:
        if recovery_attempts_left is None:
            recovery_attempts_left = max(0, int(self.config.timeout_recovery_limit))
        self._abort_if_control_requested()
        self._checkpoint_stats()

        if startup_recovery_match is not None:
            self._set_preview_detector(
                step="Startup recovery",
                detector=startup_recovery_match.detector,
                result=startup_recovery_match.result,
                detail=f"Matched {startup_recovery_match.detail}",
            )
            print(
                f'Startup recovery matched "{startup_recovery_match.name}" '
                f"(score={startup_recovery_match.result.score:.4f})."
            )
            return self._resume_from_recovery_stage(
                startup_recovery_match.name,
                recovery_attempts_left,
                detail=f"Startup recovery matched {startup_recovery_match.detail}.",
            )

        entry_match = self._wait_for_start_scene_match(
            timeout=self.config.start_scene_timeout,
            label="Checking starting scene",
        )
        if entry_match is None:
            frame = self.capture.get_frame()
            saved = self._save_outcome_frame(frame)
            return LoopOutcome(
                "retry",
                f"Could not confirm a known stage after restart. Saved frame to {saved}.",
            )

        if entry_match.name == "continue":
            print(f'Start scene confirmed as {entry_match.detail}.')
            return self._run_from_continue(recovery_attempts_left)

        print(f'Start scene confirmed as {entry_match.detail}.')
        return self._run_from_start_scene(recovery_attempts_left)

    def _find_startup_recovery_match(self) -> RecoveryStageMatch | None:
        match = self._scan_recovery_stage(self.capture.get_frame())
        if match is None:
            return None
        if match.name == "game_launch":
            return None
        return match

    def _wait_for_startup_recovery_match(self, timeout: float) -> RecoveryStageMatch | None:
        timeout = max(0.0, float(timeout))
        if timeout <= 0:
            return self._find_startup_recovery_match()

        print(
            "Controller connected. Watching for a resumable stage for "
            f"{timeout:.1f}s before restart combo..."
        )
        deadline = time.monotonic() + timeout
        while True:
            self._abort_if_control_requested()
            match = self._find_startup_recovery_match()
            if match is not None:
                return match
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(self.config.poll_interval, remaining))
        print("No resumable stage detected; using restart combo.")
        return None

    def _run_from_start_scene(self, recovery_attempts_left: int) -> LoopOutcome:
        stage_name = self._mash_until_any_match(
            button=Button.A,
            candidates=[
                ("continue", self.config.select_save_detector),
                ("ready", self.config.ready_detector),
                ("target_failed", self.config.target_failed_detector),
                ("target_ok", self.config.target_ok_detector),
            ],
            timeout=self.config.step_timeout,
            label='Mashing A until "continue" appears',
            primary_name="continue",
            post_press_poll_gap=self.config.startup_continue_poll_gap,
        )
        if stage_name is None:
            return self._handle_timeout(
                'Timed out while mashing A for "continue".',
                recovery_attempts_left,
            )
        if stage_name == "continue":
            return self._run_from_continue(recovery_attempts_left)
        return self._resume_from_recovery_stage(
            stage_name,
            recovery_attempts_left,
            detail=f'Opener advanced to "{stage_name}" while waiting for "continue".',
        )

    def _run_from_continue(self, recovery_attempts_left: int) -> LoopOutcome:
        self._abort_if_control_requested()
        self._set_preview_detector(
            step='Matched "continue"',
            detector=self.config.select_save_detector,
            detail='Pressing A once, then immediately mashing B until "ready" appears.',
        )
        self._press(
            Button.A,
            down=self.config.startup_mash_down,
            up=self.config.startup_mash_up,
        )
        self._checkpoint_stats()
        stage_name = self._mash_until_any_match(
            button=Button.B,
            candidates=[
                ("ready", self.config.ready_detector),
            ],
            timeout=self.config.step_timeout,
            label='Mashing B after "continue" until "ready" appears',
            primary_name="ready",
            press_first=True,
        )
        if stage_name is None:
            return self._handle_timeout(
                'Timed out while mashing B after "continue" for "ready".',
                recovery_attempts_left,
            )
        return self._run_from_black_screen(recovery_attempts_left)

    def _run_from_previously(self, recovery_attempts_left: int) -> LoopOutcome:
        if not self._mash_until_missing(
            button=Button.B,
            detector=self.config.previously_detector,
            timeout=self.config.step_timeout,
            label='Mashing B until "previously" disappears',
        ):
            return self._handle_timeout(
                'Timed out while mashing B to dismiss "previously".',
                recovery_attempts_left,
            )

        return self._run_from_ready(recovery_attempts_left)

    def _run_from_ready(self, recovery_attempts_left: int) -> LoopOutcome:
        if not self._wait_until_match(
            detector=self.config.ready_detector,
            timeout=self.config.step_timeout,
            label='Waiting for the "ready" scene after "previously"',
        ):
            return self._handle_timeout(
                'Timed out waiting for the "ready" scene after "previously".',
                recovery_attempts_left,
            )

        return self._run_from_black_screen(recovery_attempts_left)

    def _run_from_black_screen(self, recovery_attempts_left: int) -> LoopOutcome:
        if not self._mash_until_missing(
            button=Button.A,
            detector=self.config.ready_detector,
            timeout=self.config.step_timeout,
            label='Mashing A until "ready" disappears',
        ):
            return self._handle_timeout(
                'Timed out while mashing A to dismiss "ready".',
                recovery_attempts_left,
            )
        return self._wait_for_outcome(recovery_attempts_left)

    def _wait_for_command(self) -> str:
        if self._controller_connected:
            self.stats.mark_status("paired")
        else:
            self.stats.mark_status("stopped")
        while True:
            command = self._consume_control_command()
            if command == "pair":
                print("Pair command received.")
                return command
            if command == "restart":
                print("Restart command received.")
                return command
            time.sleep(self.config.control_poll_interval)

    def _consume_control_command(self) -> str:
        command = self.control.refresh()
        if command in {"pair", "restart", "stop"}:
            self.control.set_command("noop")
            return command
        return "noop"

    def _wait_for_start_scene_match(
        self,
        *,
        timeout: float,
        label: str,
    ) -> RecoveryStageMatch | None:
        self._abort_if_control_requested()
        self._set_preview_boxes(
            step=label,
            detail='Scanning launch, press start, and continue ROIs.',
            boxes=[
                self._box_for_roi("game_launch", self.config.start_detector.roi, matched=False),
                self._box_for_roi("press_start", self.config.press_start_detector.roi, matched=False),
                self._box_for_roi("continue", self.config.select_save_detector.roi, matched=False),
            ],
        )
        deadline = time.monotonic() + max(0.0, float(timeout))
        best_match: RecoveryStageMatch | None = None
        while time.monotonic() < deadline:
            self._abort_if_control_requested()
            frame = self.capture.get_frame()
            match, results = self._scan_start_scene(frame)
            if match is not None:
                self._set_preview_detector(
                    step=label,
                    detector=match.detector,
                    result=match.result,
                    detail=f"Matched {match.detail}",
                )
                print(f'Known stage matched: "{match.name}" (score={match.result.score:.4f}).')
                return match

            start_result = results["game_launch"]
            press_start_result = results["press_start"]
            select_result = results["continue"]
            detail = (
                "No known start scene matched yet. "
                f"launch={start_result.score:.4f} "
                f"press_start={press_start_result.score:.4f} "
                f'continue={select_result.score:.4f}'
            )
            self._set_preview_boxes(
                step=label,
                detail=detail,
                boxes=[
                    self._box_for_roi("game_launch", self.config.start_detector.roi, matched=False),
                    self._box_for_roi("press_start", self.config.press_start_detector.roi, matched=False),
                    self._box_for_roi("continue", self.config.select_save_detector.roi, matched=False),
                ],
            )
            if best_match is None or start_result.score < best_match.result.score:
                best_match = RecoveryStageMatch(
                    name="game_launch",
                    detector=self.config.start_detector,
                    result=start_result,
                    detail=f'game_launch (score={start_result.score:.4f})',
                )
            time.sleep(self.config.match_poll_interval)

        if best_match is not None:
            print(
                "Current frame does not look like a known start scene: "
                f"closest was {best_match.detail}"
            )
        else:
            print("Current frame does not look like a known start scene.")
        return None

    def _scan_start_scene(self, frame) -> tuple[RecoveryStageMatch | None, dict[str, MatchResult]]:
        checks = [
            ("continue", self.config.select_save_detector),
            ("press_start", self.config.press_start_detector),
            ("game_launch", self.config.start_detector),
        ]
        results: dict[str, MatchResult] = {}
        for name, detector in checks:
            result = detector.match(frame)
            results[name] = result
            if result.matched:
                return (
                    RecoveryStageMatch(
                        name=name,
                        detector=detector,
                        result=result,
                        detail=f'{name} (score={result.score:.4f})',
                    ),
                    results,
                )
        return None, results

    def _press_until_select_save(self) -> bool:
        return self._press_until_match(
            button=Button.A,
            detector=self.config.select_save_detector,
            timeout=self.config.step_timeout,
            label='Pressing A until "continue" appears',
            abort_detectors=[
                self.config.previously_detector,
                self.config.ready_detector,
            ],
        )

    def _mash_until_any_match(
        self,
        *,
        button: Button,
        candidates: list[tuple[str, object | None]],
        timeout: float,
        label: str,
        primary_name: str | None = None,
        press_first: bool = False,
        down: float | None = None,
        up: float | None = None,
        post_press_poll_gap: float = 0.0,
    ) -> str | None:
        print(label)
        deadline = time.monotonic() + timeout
        press_down = self.config.startup_mash_down if down is None else down
        press_up = self.config.startup_mash_up if up is None else up
        if press_first:
            self._press(
                button,
                down=press_down,
                up=press_up,
            )
            self._checkpoint_stats()
            matched_name = self._wait_for_any_match_until(
                candidates=candidates,
                label=label,
                primary_name=primary_name,
                deadline=min(deadline, time.monotonic() + max(0.0, post_press_poll_gap)),
            )
            if matched_name is not None:
                return matched_name
        while time.monotonic() < deadline:
            self._abort_if_control_requested()
            matched_name = self._check_any_match(
                candidates=candidates,
                label=label,
                primary_name=primary_name,
            )
            if matched_name is not None:
                return matched_name

            self._press(
                button,
                down=press_down,
                up=press_up,
            )
            self._checkpoint_stats()
            matched_name = self._wait_for_any_match_until(
                candidates=candidates,
                label=label,
                primary_name=primary_name,
                deadline=min(deadline, time.monotonic() + max(0.0, post_press_poll_gap)),
            )
            if matched_name is not None:
                return matched_name
        return None

    def _check_any_match(
        self,
        *,
        candidates: list[tuple[str, object | None]],
        label: str,
        primary_name: str | None = None,
    ) -> str | None:
        frame = self.capture.get_frame()
        best_detector: object | None = None
        best_result: MatchResult | None = None
        primary_detector: object | None = None
        primary_result: MatchResult | None = None
        target_ok, target_fail_example = self._match_target_examples(frame)

        for name, detector in candidates:
            active_detector = detector
            result: MatchResult | None = None
            matched = False

            if name == "target_ok":
                if self.config.target_ok_detector is None or target_ok is None:
                    continue
                active_detector = self.config.target_ok_detector
                result = target_ok
                matched = self._is_confident_target_ok(target_ok, target_fail_example)
            elif detector is not None:
                result = detector.match(frame)
                matched = result.matched
            else:
                continue

            if primary_name is not None and name == primary_name:
                primary_detector = active_detector
                primary_result = result

            print(f"  {self._format_detector_result(active_detector, result)}")
            if matched:
                if primary_name is None or name == primary_name:
                    self._set_preview_detector(step=label, detector=active_detector, result=result)
                elif primary_detector is not None and primary_result is not None:
                    self._set_preview_detector(step=label, detector=primary_detector, result=primary_result)
                print(f'  matched "{name}"')
                return name
            if best_result is None or result.score < best_result.score:
                best_detector = active_detector
                best_result = result

        if primary_detector is not None and primary_result is not None:
            self._set_preview_detector(step=label, detector=primary_detector, result=primary_result)
        elif best_detector is not None and best_result is not None:
            self._set_preview_detector(step=label, detector=best_detector, result=best_result)
        return None

    def _wait_for_any_match_until(
        self,
        *,
        candidates: list[tuple[str, object | None]],
        label: str,
        primary_name: str | None = None,
        deadline: float,
    ) -> str | None:
        while time.monotonic() < deadline:
            self._abort_if_control_requested()
            matched_name = self._check_any_match(
                candidates=candidates,
                label=label,
                primary_name=primary_name,
            )
            if matched_name is not None:
                print(f'  matched between presses: "{matched_name}"')
                return matched_name
            time.sleep(min(self.config.match_poll_interval, max(0.0, deadline - time.monotonic())))
        return None

    def _mash_until_missing(
        self,
        *,
        button: Button,
        detector: StaticImageDetector,
        timeout: float,
        label: str,
    ) -> bool:
        print(label)
        deadline = time.monotonic() + timeout
        seen_once = False
        while time.monotonic() < deadline:
            self._abort_if_control_requested()
            frame = self.capture.get_frame()
            result = detector.match(frame)
            self._set_preview_detector(
                step=label,
                detector=detector,
                result=result,
                detail="Waiting for the prompt to disappear." if seen_once else None,
            )
            print(f"  {self._format_detector_result(detector, result)}")
            if result.matched:
                seen_once = True
            if seen_once and not result.matched:
                self._set_preview_detector(
                    step=label,
                    detector=detector,
                    result=result,
                    detail="Prompt disappeared.",
                )
                print("  no longer visible")
                return True

            self._press(
                button,
                down=self.config.startup_mash_down,
                up=self.config.startup_mash_up,
            )
            self._checkpoint_stats()
        return False

    def _press_until_match(
        self,
        *,
        button: Button,
        detector,
        timeout: float,
        label: str,
        abort_detectors: list[StaticImageDetector] | None = None,
    ) -> bool:
        print(label)
        deadline = time.monotonic() + timeout
        try:
            while time.monotonic() < deadline:
                self._abort_if_control_requested()
                frame = self.capture.get_frame()
                self._raise_on_unexpected_scene(
                    frame=frame,
                    expected_detector=detector,
                    abort_detectors=abort_detectors,
                    step=label,
                )
                result = detector.match(frame)
                self._set_preview_detector(step=label, detector=detector, result=result)
                print(f"  {self._format_detector_result(detector, result)}")
                if result.matched:
                    print("  matched")
                    return True

                self._press(button)
                time.sleep(self.config.settle_time)
                self._checkpoint_stats()
                if self._wait_until_next_press_for_match(
                    detector=detector,
                    label=label,
                    deadline=deadline,
                    abort_detectors=abort_detectors,
                ):
                    return True
            return False
        finally:
            pass

    def _press_until_missing(
        self,
        *,
        button: Button,
        detector: StaticImageDetector,
        timeout: float,
        label: str,
    ) -> bool:
        print(label)
        deadline = time.monotonic() + timeout
        seen_once = False
        try:
            while time.monotonic() < deadline:
                self._abort_if_control_requested()
                frame = self.capture.get_frame()
                result = detector.match(frame)
                self._set_preview_detector(
                    step=label,
                    detector=detector,
                    result=result,
                    detail="Waiting for the prompt to disappear." if seen_once else None,
                )
                print(f"  {self._format_detector_result(detector, result)}")
                if result.matched:
                    seen_once = True
                if seen_once and not result.matched:
                    self._set_preview_detector(
                        step=label,
                        detector=detector,
                        result=result,
                        detail="Prompt disappeared.",
                    )
                    print("  no longer visible")
                    return True

                self._press(button)
                time.sleep(self.config.settle_time)
                self._sleep_until_next_press()
            return False
        finally:
            pass

    def _wait_until_match(
        self,
        *,
        detector,
        timeout: float,
        label: str,
    ) -> bool:
        print(label)
        deadline = time.monotonic() + timeout
        try:
            while time.monotonic() < deadline:
                self._abort_if_control_requested()
                frame = self.capture.get_frame()
                result = detector.match(frame)
                self._set_preview_detector(step=label, detector=detector, result=result)
                print(f"  {self._format_detector_result(detector, result)}")
                if result.matched:
                    print("  matched")
                    return True
                time.sleep(self.config.poll_interval)
                self._checkpoint_stats()
            return False
        finally:
            pass

    def _wait_until_next_press_for_match(
        self,
        *,
        detector,
        label: str,
        deadline: float,
        abort_detectors: list[StaticImageDetector] | None = None,
    ) -> bool:
        end = min(deadline, time.monotonic() + max(0.0, self.config.press_interval - self.config.settle_time))
        while time.monotonic() < end:
            self._abort_if_control_requested()
            frame = self.capture.get_frame()
            self._raise_on_unexpected_scene(
                frame=frame,
                expected_detector=detector,
                abort_detectors=abort_detectors,
                step=label,
            )
            result = detector.match(frame)
            self._set_preview_detector(step=label, detector=detector, result=result)
            if result.matched:
                print(f"  matched between presses: {self._format_detector_result(detector, result)}")
                return True
            time.sleep(min(self.config.match_poll_interval, max(0.0, end - time.monotonic())))
        return False

    def _raise_on_unexpected_scene(
        self,
        *,
        frame,
        expected_detector,
        abort_detectors: list[StaticImageDetector] | None,
        step: str,
    ) -> None:
        if not abort_detectors:
            return

        expected_name = self._detector_name(expected_detector)
        for abort_detector in abort_detectors:
            result = abort_detector.match(frame)
            if not result.matched:
                continue
            scene_name = self._detector_name(abort_detector)
            self._set_preview_detector(
                step=f'Unexpected scene while waiting for "{expected_name}"',
                detector=abort_detector,
                result=result,
                detail=f'Matched "{scene_name}" during step: {step}',
            )
            print(f'  unexpected scene "{scene_name}" matched while waiting for "{expected_name}"')
            raise UnexpectedSceneMatched(scene_name, result.score)

    def _wait_for_outcome(self, recovery_attempts_left: int) -> LoopOutcome:
        print('Waiting for outcome after "ready" disappeared...')
        if self.config.outcome_settle_delay > 0:
            print(f"Settling outcome scene for {self.config.outcome_settle_delay:.1f}s...")
            time.sleep(self.config.outcome_settle_delay)
        deadline = time.monotonic() + self.config.outcome_timeout
        first_failed_at: float | None = None
        first_non_black_at: float | None = None
        first_target_ok_at: float | None = None
        safe_fail_score = self.config.target_failed_detector.threshold + self.config.success_candidate_fail_margin

        while time.monotonic() < deadline:
            self._abort_if_control_requested()
            now = time.monotonic()
            frame = self.capture.get_frame()
            ready = self.config.ready_detector.match(frame)
            failed = self.config.target_failed_detector.match(frame)
            target_ok, target_fail_example = self._match_target_examples(frame)

            if failed.matched:
                if first_failed_at is None:
                    first_failed_at = now
                failed_for = now - first_failed_at
                self._set_preview_detector(
                    step="Outcome: target failed candidate",
                    detector=self.config.target_failed_detector,
                    result=failed,
                    detail=f"Matched for {failed_for:.1f}s (score={failed.score:.4f})",
                )
                print(f"  target failed candidate: score={failed.score:.4f}, visible_for={failed_for:.1f}s")
                if failed_for >= self.config.target_failed_hold:
                    self._set_preview_detector(
                        step="Outcome: target failed",
                        detector=self.config.target_failed_detector,
                        result=failed,
                        detail=f"Matched for {failed_for:.1f}s; confirming retry.",
                    )
                    self._remember_failed_roi(frame, failed)
                    self._save_failed_roi(frame, failed)
                    return LoopOutcome(
                        "retry",
                        f"Target failed image matched for {failed_for:.1f}s (score={failed.score:.4f}).",
                    )
            else:
                first_failed_at = None

            self._set_preview_target_roi(
                step="Waiting for outcome",
                detail=self._outcome_debug_detail(
                    ready=ready,
                    failed=failed,
                    target_ok=target_ok,
                    target_fail_example=target_fail_example,
                    visible_for=0.0 if first_non_black_at is None else now - first_non_black_at,
                    ok_for=0.0 if first_target_ok_at is None else now - first_target_ok_at,
                ),
            )

            if ready.matched:
                first_non_black_at = None
                first_target_ok_at = None
                print(
                    f'  still on "ready": ready={ready.score:.4f}, '
                    f"failed={failed.score:.4f}"
                )
                self._press(
                    Button.A,
                    down=self.config.startup_mash_down,
                    up=self.config.startup_mash_up,
                )
                self._checkpoint_stats()
                continue

            blocked_by_fail = failed.score < safe_fail_score or self._looks_like_recent_fail(frame, failed)
            confident_target_ok = self._is_confident_target_ok(target_ok, target_fail_example)

            if blocked_by_fail:
                first_non_black_at = None
                first_target_ok_at = None
            else:
                if first_non_black_at is None:
                    first_non_black_at = now
                if confident_target_ok:
                    if first_target_ok_at is None:
                        first_target_ok_at = now
                else:
                    first_target_ok_at = None

            visible_for = 0.0 if first_non_black_at is None else now - first_non_black_at
            ok_for = 0.0 if first_target_ok_at is None else now - first_target_ok_at
            print(
                "  visible scene: "
                f"ready={ready.score:.4f}, "
                f"failed={failed.score:.4f}, "
                f"visible_for={visible_for:.1f}s"
                f"{self._target_example_debug_suffix(target_ok, target_fail_example, ok_for)}"
            )

            if self.config.target_ok_detector is not None:
                if confident_target_ok and ok_for >= self.config.target_ok_hold:
                    self._set_preview_detector(
                        step="Outcome: target ok candidate",
                        detector=self.config.target_ok_detector,
                        result=target_ok,
                        detail=(
                            f"Matched for {ok_for:.1f}s; "
                            f"{self._target_example_detail(target_ok, target_fail_example)}"
                        ),
                    )
                    if self._confirm_success_candidate():
                        self._save_candidate_roi(frame, failed)
                        saved = self._save_outcome_frame(frame)
                        return LoopOutcome(
                            "success_candidate",
                            f"Target ok matched for {ok_for:.1f}s. Saved frame to {saved}.",
                        )
                    first_non_black_at = None
                    first_target_ok_at = None
            elif visible_for >= self.config.success_candidate_hold:
                self._set_preview_target_roi(
                    step="Outcome: success candidate",
                    detail=f"Visible for {visible_for:.1f}s with no failure image.",
                )
                if self._confirm_success_candidate():
                    self._save_candidate_roi(frame, failed)
                    saved = self._save_outcome_frame(frame)
                    return LoopOutcome(
                        "success_candidate",
                        f"No failure image detected after transition. Saved frame to {saved}.",
                    )
                first_non_black_at = None

            time.sleep(self.config.poll_interval)
            self._checkpoint_stats()

        return self._handle_timeout(
            "Timed out during outcome detection.",
            recovery_attempts_left,
        )

    def _handle_timeout(self, detail: str, recovery_attempts_left: int) -> LoopOutcome:
        if recovery_attempts_left <= 0:
            frame = self.capture.get_frame()
            self._save_candidate_roi(frame, None)
            saved = self._save_outcome_frame(frame)
            return LoopOutcome(
                "retry",
                f"{detail} Recovery budget exhausted; resetting loop. Saved frame to {saved}.",
            )

        print(f"{detail} Attempting timeout recovery scan...")
        recovery = self._attempt_timeout_recovery(
            detail=detail,
            recovery_attempts_left=recovery_attempts_left - 1,
        )
        if recovery is not None:
            return recovery

        frame = self.capture.get_frame()
        self._save_candidate_roi(frame, None)
        saved = self._save_outcome_frame(frame)
        return LoopOutcome(
            "retry",
            f"{detail} Recovery could not identify a known stage; resetting loop. Saved frame to {saved}.",
        )

    def _attempt_timeout_recovery(
        self,
        *,
        detail: str,
        recovery_attempts_left: int,
    ) -> LoopOutcome | None:
        self._abort_if_control_requested()
        frame = self.capture.get_frame()
        match = self._scan_recovery_stage(frame)
        if match is None:
            print("  recovery scan did not match a known stage")
            return None

        if match.detector is not None:
            self._set_preview_detector(
                step="Timeout recovery",
                detector=match.detector,
                result=match.result,
                detail=f"Matched {match.detail}",
            )

        print(
            f'  recovery matched "{match.name}" '
            f"(score={match.result.score:.4f})"
        )
        return self._resume_from_recovery_stage(
            match.name,
            recovery_attempts_left,
            detail=f"{detail} Recovery matched {match.detail}.",
        )

    def _scan_recovery_stage(self, frame) -> RecoveryStageMatch | None:
        failed = self.config.target_failed_detector.match(frame)
        if failed.matched:
            return RecoveryStageMatch(
                name="target_failed",
                detector=self.config.target_failed_detector,
                result=failed,
                detail=f'target_failed (score={failed.score:.4f})',
            )

        target_ok, target_fail_example = self._match_target_examples(frame)
        if self._is_confident_target_ok(target_ok, target_fail_example) and target_ok is not None:
            return RecoveryStageMatch(
                name="target_ok",
                detector=self.config.target_ok_detector,
                result=target_ok,
                detail=(
                    "target_ok "
                    f"(ok={target_ok.score:.4f} "
                    f"fail_example={'n/a' if target_fail_example is None else f'{target_fail_example.score:.4f}'})"
                ),
            )

        for name, detector in (
            ("ready", self.config.ready_detector),
            ("previously", self.config.previously_detector),
            ("continue", self.config.select_save_detector),
            ("press_start", self.config.press_start_detector),
        ):
            result = detector.match(frame)
            if result.matched:
                return RecoveryStageMatch(
                    name=name,
                    detector=detector,
                    result=result,
                    detail=f'{name} (score={result.score:.4f})',
                )

        return None

    def _resume_from_recovery_stage(
        self,
        stage_name: str,
        recovery_attempts_left: int,
        *,
        detail: str,
    ) -> LoopOutcome:
        print(detail)
        if stage_name == "target_failed":
            frame = self.capture.get_frame()
            failed = self.config.target_failed_detector.match(frame)
            if failed.matched:
                self._remember_failed_roi(frame, failed)
                self._save_failed_roi(frame, failed)
            return LoopOutcome(
                "retry",
                f"{detail} Confirmed target failed; resetting loop.",
            )

        if stage_name == "target_ok":
            frame = self.capture.get_frame()
            self._save_candidate_roi(frame, None)
            saved = self._save_outcome_frame(frame)
            return LoopOutcome(
                "success_candidate",
                f"{detail} Recovered on target_ok; saved frame to {saved}.",
            )

        if stage_name == "ready":
            return self._run_from_black_screen(recovery_attempts_left)

        if stage_name == "previously":
            return self._run_from_previously(recovery_attempts_left)

        if stage_name == "continue":
            return self._run_from_continue(recovery_attempts_left)

        if stage_name == "press_start":
            return self._run_from_start_scene(recovery_attempts_left)

        frame = self.capture.get_frame()
        self._save_candidate_roi(frame, None)
        saved = self._save_outcome_frame(frame)
        return LoopOutcome(
            "retry",
            f"{detail} Recovery stage \"{stage_name}\" is not resumable; resetting loop. Saved frame to {saved}.",
        )

    def _confirm_success_candidate(self) -> bool:
        checks = max(1, int(self.config.success_confirm_checks))
        interval = max(0.0, float(self.config.success_confirm_interval))
        safe_fail_score = self.config.target_failed_detector.threshold + self.config.success_candidate_fail_margin
        for idx in range(checks):
            self._abort_if_control_requested()
            frame = self.capture.get_frame()
            ready = self.config.ready_detector.match(frame)
            failed = self.config.target_failed_detector.match(frame)
            target_ok, target_fail_example = self._match_target_examples(frame)
            detail = (
                f"Confirm {idx + 1}/{checks}: "
                f"{self._outcome_debug_detail(ready=ready, failed=failed, target_ok=target_ok, target_fail_example=target_fail_example)}"
            )
            if self.config.target_ok_detector is not None and target_ok is not None:
                self._set_preview_detector(
                    step="Outcome: confirming success",
                    detector=self.config.target_ok_detector,
                    result=target_ok,
                    detail=detail,
                )
            else:
                self._set_preview_target_roi(
                    step="Outcome: confirming success",
                    detail=detail,
                )
            if ready.matched:
                return False
            if failed.score < safe_fail_score:
                return False
            if self._looks_like_recent_fail(frame, failed):
                return False
            if self.config.target_ok_detector is not None and not self._is_confident_target_ok(
                target_ok,
                target_fail_example,
            ):
                return False
            if interval > 0 and idx < checks - 1:
                time.sleep(interval)
        return True

    def _match_target_examples(self, frame) -> tuple[MatchResult | None, MatchResult | None]:
        target_ok = None
        if self.config.target_ok_detector is not None:
            target_ok = self.config.target_ok_detector.match(frame)

        target_fail_example = None
        if self.config.target_fail_example_detector is not None:
            target_fail_example = self.config.target_fail_example_detector.match(frame)

        return target_ok, target_fail_example

    def _is_confident_target_ok(
        self,
        target_ok: MatchResult | None,
        target_fail_example: MatchResult | None,
    ) -> bool:
        if target_ok is None or not target_ok.matched:
            return False
        if target_fail_example is None:
            return True
        return target_ok.score + self.config.target_ok_score_margin < target_fail_example.score

    def _target_example_detail(
        self,
        target_ok: MatchResult | None,
        target_fail_example: MatchResult | None,
    ) -> str:
        details: list[str] = []
        if target_ok is not None:
            details.append(f"ok={target_ok.score:.4f}")
        if target_fail_example is not None:
            details.append(f"fail_example={target_fail_example.score:.4f}")
        return " ".join(details) if details else "target_ok=disabled"

    def _target_example_debug_suffix(
        self,
        target_ok: MatchResult | None,
        target_fail_example: MatchResult | None,
        ok_for: float,
    ) -> str:
        if target_ok is None and target_fail_example is None:
            return ""
        confident = self._is_confident_target_ok(target_ok, target_fail_example)
        return (
            f", {self._target_example_detail(target_ok, target_fail_example)}"
            f", ok_for={ok_for:.1f}s"
            f", confident_ok={'yes' if confident else 'no'}"
        )

    def _outcome_debug_detail(
        self,
        *,
        ready: MatchResult,
        failed: MatchResult,
        target_ok: MatchResult | None,
        target_fail_example: MatchResult | None,
        visible_for: float | None = None,
        ok_for: float | None = None,
    ) -> str:
        parts = [f"ready={ready.score:.4f}", f"failed={failed.score:.4f}"]
        target_detail = self._target_example_detail(target_ok, target_fail_example)
        if target_detail != "target_ok=disabled":
            parts.append(target_detail)
            parts.append(
                "confident_ok="
                + ("yes" if self._is_confident_target_ok(target_ok, target_fail_example) else "no")
            )
        if visible_for is not None:
            parts.append(f"visible={visible_for:.1f}s")
        if ok_for is not None and target_ok is not None:
            parts.append(f"ok_for={ok_for:.1f}s")
        return " ".join(parts)

    def _save_outcome_frame(self, frame) -> Path:
        self.config.debug_dir.mkdir(parents=True, exist_ok=True)
        path = self.config.debug_dir / f"outcome-{time.strftime('%Y%m%d-%H%M%S')}.jpg"
        self.capture.save_frame(path)
        self._latest_outcome_frame_path = path
        return path

    def _save_failed_roi(self, frame, result: MatchResult) -> Path:
        roi = self.config.target_failed_detector.roi
        crop_roi = Roi(
            x=roi.x + result.offset_x,
            y=roi.y + result.offset_y,
            width=roi.width,
            height=roi.height,
        )
        cropped = crop_roi.crop(frame)
        self.config.failed_roi_dir.mkdir(parents=True, exist_ok=True)
        path = self.config.failed_roi_dir / (
            f"target-failed-{time.strftime('%Y%m%d-%H%M%S')}-score-{result.score:.4f}.jpg"
        )
        path.write_bytes(encode_rgb_frame(cropped, quality=95))
        self._latest_failed_roi_path = path
        print(f"  saved failed ROI to {path}")
        return path

    def _remember_failed_roi(self, frame, result: MatchResult) -> None:
        store_cutoff = self.config.target_failed_detector.threshold + self.config.recent_failed_store_margin
        if result.score > store_cutoff:
            return
        roi = self._aligned_target_roi(result)
        signature = self._roi_signature(frame, roi)
        self._recent_failed_rois.append(signature)

    def _looks_like_recent_fail(self, frame, result: MatchResult) -> bool:
        if not self._recent_failed_rois:
            return False
        roi = self._aligned_target_roi(result)
        signature = self._roi_signature(frame, roi)
        for prior in self._recent_failed_rois:
            if self._roi_distance(signature, prior) <= self.config.recent_failed_similarity_threshold:
                return True
        return False

    def _aligned_target_roi(self, result: MatchResult) -> Roi:
        roi = self.config.target_failed_detector.roi
        return Roi(
            x=roi.x + result.offset_x,
            y=roi.y + result.offset_y,
            width=roi.width,
            height=roi.height,
        )

    def _roi_signature(self, frame, roi: Roi) -> np.ndarray:
        cropped = roi.crop(frame)
        stride = max(1, int(self.config.recent_failed_stride))
        small = cropped[::stride, ::stride]
        grayscale = (
            0.2126 * small[:, :, 0] + 0.7152 * small[:, :, 1] + 0.0722 * small[:, :, 2]
        )
        return (grayscale / 255.0).astype(np.float32, copy=False)

    def _roi_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        if a.shape != b.shape:
            h = min(a.shape[0], b.shape[0])
            w = min(a.shape[1], b.shape[1])
            a = a[:h, :w]
            b = b[:h, :w]
        return float(np.abs(a - b).mean())

    def _save_candidate_roi(self, frame, result: MatchResult | None) -> Path:
        roi = self.config.target_failed_detector.roi
        offset_x = result.offset_x if result is not None else 0
        offset_y = result.offset_y if result is not None else 0
        crop_roi = Roi(
            x=roi.x + offset_x,
            y=roi.y + offset_y,
            width=roi.width,
            height=roi.height,
        )
        cropped = crop_roi.crop(frame)
        self.config.failed_roi_dir.mkdir(parents=True, exist_ok=True)
        path = self.config.failed_roi_dir / (
            f"candidate-roi-{time.strftime('%Y%m%d-%H%M%S')}.jpg"
        )
        path.write_bytes(encode_rgb_frame(cropped, quality=95))
        self._latest_candidate_roi_path = path
        print(f"  saved candidate ROI to {path}")
        return path

    def _press(self, button: Button, *, down: float = 0.1, up: float = 0.1) -> None:
        self.controller.press(button, down=down, up=up)

    def _press_combo(self, *buttons: Button, down: float, up: float) -> None:
        self.controller.press(*buttons, down=down, up=up)

    def _sleep_until_next_press(self) -> None:
        delay = max(0.0, self.config.press_interval - self.config.settle_time)
        if delay > 0:
            time.sleep(delay)

    def _trigger_restart_loop(self) -> None:
        print("Triggering the next loop with A+B+X+Y...")
        self._press_combo(
            Button.A,
            Button.B,
            Button.X,
            Button.Y,
            down=self.config.restart_combo_hold,
            up=self.config.restart_combo_release,
        )

    def _print_timers(self, stats: LoopStatsSnapshot) -> None:
        print(
            "Timers: "
            f"total={_format_duration(stats.total_elapsed_seconds)} "
            f"loop={_format_duration(stats.loop_elapsed_seconds)} "
            f"count={stats.loop_counter}"
        )

    def _pair_controller(self) -> None:
        if self._controller_connected:
            self.stats.mark_status("paired")
            self._set_preview_state("paired", "Controller already connected. Waiting for restart.", [])
            print("Controller is already connected. Waiting for restart.")
            return

        restore_reconnect = None
        if hasattr(self.controller, "set_reconnect"):
            restore_reconnect = True
            self.controller.set_reconnect(False)

        try:
            self.connect()
        except StopRequested:
            self.stats.mark_status("stopped", last_outcome="pair_cancelled")
            print("Pairing cancelled. Automation is now idle.")
            return
        finally:
            if restore_reconnect is not None:
                self.controller.set_reconnect(restore_reconnect)

        self.stats.mark_status("paired")
        self._set_preview_state("paired", "Controller connected. Waiting for restart.", [])
        print("Controller connected in pairing mode. Future restarts will reuse the connection.")

    def preview_overlay_lines(self) -> list[str]:
        snapshot = self.stats.snapshot()
        with self._preview_lock:
            step = self._preview_step
        lines = [
            f"mode: {snapshot.status}",
            f"total: {_format_duration(snapshot.total_elapsed_seconds)}",
            f"loop: {_format_duration(snapshot.loop_elapsed_seconds)}",
            f"count: {snapshot.loop_counter}",
            f"timeouts: {self._timeout_reset_count}",
            f"step: {step}",
        ]
        return lines

    def preview_overlay_state(self) -> OverlayState:
        with self._preview_lock:
            boxes = list(self._preview_boxes)
        return OverlayState(lines=self.preview_overlay_lines(), boxes=boxes)

    def _abort_if_control_requested(self) -> None:
        command = self.control.refresh()
        if command == "stop":
            self.control.set_command("noop")
            raise StopRequested
        if command == "restart":
            self.control.set_command("noop")
            raise RestartRequested

    def _checkpoint_stats(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_checkpoint_monotonic < self.config.stats_checkpoint_interval:
            return
        self.stats.checkpoint_running()
        self._last_checkpoint_monotonic = now

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
        detector_name = self._detector_name(detector)
        if isinstance(detector, StaticImageDetector):
            roi = detector.roi
        else:
            roi = detector.roi
        if roi is None:
            boxes: list[OverlayBox] = []
        else:
            focus_roi = roi
            if (
                result is not None
                and isinstance(detector, StaticImageDetector)
                and detector_name != "continue"
            ):
                focus_roi = Roi(
                    x=roi.x + result.offset_x,
                    y=roi.y + result.offset_y,
                    width=roi.width,
                    height=roi.height,
                )
            boxes = [self._box_for_roi(detector_name, focus_roi, matched=(result.matched if result else False))]

        if detail is None and result is not None:
            detail = self._format_detector_result(detector, result)
        self._set_preview_state(step, detail, boxes)

    def _set_preview_target_roi(
        self,
        *,
        step: str,
        detail: str | None,
        matched: bool = False,
    ) -> None:
        self._set_preview_boxes(
            step=step,
            detail=detail,
            boxes=[self._box_for_roi("target", self.config.target_failed_detector.roi, matched=matched)],
        )

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
        detector_name = self._detector_name(detector)
        if result.detail:
            return f"{detector_name}: score={result.score:.4f} ({result.detail})"
        return f"{detector_name}: score={result.score:.4f}"

    def _notification_title(self, outcome: LoopOutcome) -> str:
        if outcome.status == "success_candidate":
            return "Successful target candidate"
        return "Loop stopped"

    def _notify(self, title: str, detail: str) -> None:
        if self.notify_cb is None:
            return
        try:
            self.notify_cb(title, self._notification_body(detail), self._notification_attachments())
        except Exception as exc:
            print(f"Notification failed: {exc}")

    def _notification_body(self, detail: str) -> str:
        snapshot = self.stats.snapshot()
        with self._preview_lock:
            step = self._preview_step
            preview_detail = self._preview_detail
        lines = [
            detail,
            "",
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

    def _notification_attachments(self) -> list[Path]:
        attachments: list[Path] = []
        for path in (
            self._latest_outcome_frame_path,
            self._latest_candidate_roi_path,
            self._latest_failed_roi_path,
        ):
            if path is not None and path.exists() and path not in attachments:
                attachments.append(path)
        return attachments



def _utcnow() -> datetime:
    return datetime.now(UTC)


def _chmod_if_possible(path: Path, mode: int) -> None:
    try:
        os.chmod(path, mode)
    except PermissionError:
        return
    except OSError:
        return


def _format_duration(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"
