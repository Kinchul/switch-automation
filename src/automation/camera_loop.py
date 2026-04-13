from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from control import Button, ControllerBackend, ControllerConnectCancelled
from vision import CameraCapture, encode_rgb_frame
from vision.detector import BlackScreenDetector, MatchResult, Roi, StaticImageDetector
from vision.stream import OverlayBox, OverlayState

NotifyCallback = Callable[[str, str, list[Path]], None]


@dataclass(slots=True)
class LoopOutcome:
    status: str
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
    press_interval: float = 1.0
    settle_time: float = 0.35
    poll_interval: float = 0.25
    match_poll_interval: float = 0.05
    step_timeout: float = 45.0
    blackscreen_timeout: float = 25.0
    outcome_timeout: float = 20.0
    outcome_settle_delay: float = 0.7
    target_failed_hold: float = 0.8
    success_candidate_hold: float = 3.5
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

    def connect(self) -> bool:
        if self._controller_connected:
            return False
        print("Connecting controller...")
        try:
            self.controller.connect(
                cancel_cb=lambda: self.control.refresh() in {"stop", "restart"},
                status_cb=lambda state: print(f"Controller state: {state}"),
            )
        except ControllerConnectCancelled:
            raise StopRequested
        self._controller_connected = True
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
                if connected_now and self.config.post_connect_restart_delay > 0:
                    print(
                        "Controller connected. Waiting "
                        f"{self.config.post_connect_restart_delay:.1f}s before restart combo..."
                    )
                    time.sleep(self.config.post_connect_restart_delay)
                self._trigger_restart_loop()
                stats = self.stats.start_new_loop()
                self._last_checkpoint_monotonic = time.monotonic()
                print(f"\n=== Attempt {attempt} / Loop {stats.loop_counter} ===")
                self._print_timers(stats)
                try:
                    outcome = self.run_once()
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

    def run_once(self) -> LoopOutcome:
        self._abort_if_control_requested()
        self._checkpoint_stats()

        if not self._ensure_starting_scene():
            return LoopOutcome("error", "Could not confirm the starting scene.")

        try:
            if not self._press_until_select_save():
                return LoopOutcome("retry", 'Timed out before reaching "continue".')
        except UnexpectedSceneMatched as exc:
            return LoopOutcome(
                "error",
                f'Reached "{exc.scene_name}" before "continue" (score={exc.score:.4f}); '
                "stopped to avoid advancing the game blindly.",
            )

        self._abort_if_control_requested()
        self._set_preview_detector(
            step='Matched "continue"',
            detector=self.config.select_save_detector,
            detail='Pressing A once to skip the screen.',
        )
        self._press(Button.A)
        print('Pressed A once to skip the "continue" screen.')

        if not self._press_until_missing(
            button=Button.B,
            detector=self.config.previously_detector,
            timeout=self.config.step_timeout,
            label='Pressing B until "previously" disappears',
        ):
            return LoopOutcome("retry", 'Timed out while trying to dismiss the "previously" screen.')

        if not self._wait_until_match(
            detector=self.config.ready_detector,
            timeout=self.config.step_timeout,
            label='Waiting for the "ready" scene after "previously"',
        ):
            return LoopOutcome("retry", 'Timed out waiting for the "ready" scene after "previously".')

        if not self._press_until_match(
            button=Button.A,
            detector=self.config.black_screen_detector,
            timeout=self.config.blackscreen_timeout,
            label='Pressing A until the fight black screen appears from "ready"',
        ):
            return LoopOutcome("retry", "Timed out waiting for the fight transition black screen.")

        return self._wait_for_outcome()

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

    def _ensure_starting_scene(self) -> bool:
        self._abort_if_control_requested()
        self._set_preview_boxes(
            step="Checking starting scene",
            detail='Comparing the launch, press start, and "continue" ROIs.',
            boxes=[
                self._box_for_roi("game_launch", self.config.start_detector.roi, matched=False),
                self._box_for_roi("press_start", self.config.press_start_detector.roi, matched=False),
                self._box_for_roi("continue", self.config.select_save_detector.roi, matched=False),
            ],
        )
        frame = self.capture.get_frame()
        start_result = self.config.start_detector.match(frame)
        press_start_result = self.config.press_start_detector.match(frame)
        select_result = self.config.select_save_detector.match(frame)

        if start_result.matched:
            self._set_preview_detector(
                step='Start scene: "game_launch"',
                detector=self.config.start_detector,
                result=start_result,
            )
            print(
                f'Start scene confirmed with "{self.config.start_detector.name}" '
                f"(score={start_result.score:.4f})."
            )
            return True

        if press_start_result.matched:
            self._set_preview_detector(
                step='Start scene: "press_start"',
                detector=self.config.press_start_detector,
                result=press_start_result,
            )
            print(
                f'Start scene confirmed with "{self.config.press_start_detector.name}" '
                f"(score={press_start_result.score:.4f})."
            )
            return True

        if select_result.matched:
            self._set_preview_detector(
                step='Start scene: "continue"',
                detector=self.config.select_save_detector,
                result=select_result,
            )
            print(
                'Already on "continue"; continuing from there '
                f"(score={select_result.score:.4f})."
            )
            return True

        self._set_preview_boxes(
            step="Unknown starting scene",
            detail=(
                "No known start detector matched. "
                f"launch={start_result.score:.4f} press_start={press_start_result.score:.4f} "
                f'continue={select_result.score:.4f}'
            ),
            boxes=[
                self._box_for_roi("game_launch", self.config.start_detector.roi, matched=False),
                self._box_for_roi("press_start", self.config.press_start_detector.roi, matched=False),
                self._box_for_roi("continue", self.config.select_save_detector.roi, matched=False),
            ],
        )
        print(
            "Current frame does not look like a known starting scene: "
            f"launch={start_result.score:.4f}, press_start={press_start_result.score:.4f}, "
            f'continue={select_result.score:.4f}'
        )
        return False

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

    def _wait_for_outcome(self) -> LoopOutcome:
        print("Waiting for outcome after the black screen...")
        if self.config.outcome_settle_delay > 0:
            print(f"Settling outcome scene for {self.config.outcome_settle_delay:.1f}s...")
            time.sleep(self.config.outcome_settle_delay)
        deadline = time.monotonic() + self.config.outcome_timeout
        first_failed_at: float | None = None
        first_non_black_at: float | None = None

        try:
            while time.monotonic() < deadline:
                self._abort_if_control_requested()
                frame = self.capture.get_frame()
                failed = self.config.target_failed_detector.match(frame)
                if failed.matched:
                    if first_failed_at is None:
                        first_failed_at = time.monotonic()
                    failed_for = time.monotonic() - first_failed_at
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
                        self._save_failed_roi(frame, failed)
                        return LoopOutcome(
                            "retry",
                            f"Target failed image matched for {failed_for:.1f}s (score={failed.score:.4f}).",
                        )
                else:
                    first_failed_at = None

                black = self.config.black_screen_detector.match(frame)
                self._set_preview_detector(
                    step="Waiting for outcome",
                    detector=self.config.black_screen_detector,
                    result=black,
                    detail=(
                        f"black={black.score:.4f} failed={failed.score:.4f}"
                        if black.matched
                        else f"visible scene for {0.0 if first_non_black_at is None else time.monotonic() - first_non_black_at:.1f}s"
                    ),
                )
                if black.matched:
                    first_non_black_at = None
                    print(
                        f"  still on black transition: black={black.score:.4f}, "
                        f"failed={failed.score:.4f}"
                    )
                else:
                    if first_non_black_at is None:
                        first_non_black_at = time.monotonic()
                    visible_for = time.monotonic() - first_non_black_at
                    print(
                        f"  visible scene: black={black.score:.4f}, "
                        f"failed={failed.score:.4f}, visible_for={visible_for:.1f}s"
                    )
                    if visible_for >= self.config.success_candidate_hold:
                        self._set_preview_detector(
                            step="Outcome: success candidate",
                            detector=self.config.black_screen_detector,
                            result=black,
                            detail=f"Visible for {visible_for:.1f}s with no failure image.",
                        )
                        self._save_candidate_roi(frame, failed)
                        saved = self._save_outcome_frame(frame)
                        return LoopOutcome(
                            "success_candidate",
                            f"No failure image detected after transition. Saved frame to {saved}.",
                        )

                time.sleep(self.config.poll_interval)
                self._checkpoint_stats()

            self._save_candidate_roi(self.capture.get_frame(), None)
            saved = self._save_outcome_frame(self.capture.get_frame())
            return LoopOutcome(
                "timeout",
                f"Outcome phase timed out without a failure match. Saved frame to {saved}.",
            )
        finally:
            pass

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

    def _press(self, button: Button) -> None:
        self.controller.press(button)

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
            f"step: {step}",
            f"timeouts: {self._timeout_reset_count}",
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
            if result is not None and isinstance(detector, StaticImageDetector):
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
