from __future__ import annotations

import argparse
import email.message
import sys
from pathlib import Path
import os
import fcntl
import signal
import smtplib
import ssl
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from automation.camera_loop import CameraLoopConfig, CameraLoopRunner, PersistentLoopControl, PersistentLoopStats
from control import NxbtBackend
from vision import open_capture
from vision.detector import BlackScreenDetector, InvariantColorDetector, Roi, StaticImageDetector
from vision.stream import MjpegPreviewServer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the camera-guided Switch loop using saved reference images."
    )
    parser.add_argument(
        "--action",
        choices=["run", "pair", "restart", "stop", "status"],
        default="run",
        help="`run` starts the foreground service, `pair` connects the controller and waits idle, `restart` triggers a new loop, `stop` idles the service, `status` prints the current control/stats files.",
    )
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--warmup", type=float, default=2.0)
    parser.add_argument(
        "--pairing-menu",
        action="store_true",
        help='Deprecated: pairing mode is always used now. Keep the Switch on "Change Grip/Order".',
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=None,
        help="Optional BlueZ adapter path to use (e.g. /org/bluez/hci0).",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=ROOT / "images",
        help="Directory containing the reference images 0-5.",
    )
    parser.add_argument(
        "--search-margin",
        type=int,
        default=24,
        help="How many pixels detectors may search around the reference ROI.",
    )
    parser.add_argument(
        "--attempts",
        type=int,
        default=0,
        help="Maximum attempts before stopping. Use 0 for unlimited retries.",
    )
    parser.add_argument("--press-interval", type=float, default=1.0)
    parser.add_argument("--settle-time", type=float, default=0.35)
    parser.add_argument("--poll-interval", type=float, default=0.25)
    parser.add_argument(
        "--match-poll-interval",
        type=float,
        default=0.05,
        help="How often to re-check the detector between button presses.",
    )
    parser.add_argument(
        "--target-failed-threshold",
        type=float,
        default=0.07,
        help="Maximum normalized difference for the target-failed image match. Lower is stricter.",
    )
    parser.add_argument("--step-timeout", type=float, default=45.0)
    parser.add_argument("--blackscreen-timeout", type=float, default=25.0)
    parser.add_argument("--outcome-timeout", type=float, default=20.0)
    parser.add_argument(
        "--outcome-settle-delay",
        type=float,
        default=0.7,
        help="How long to wait after the black-screen transition before validating the target scene.",
    )
    parser.add_argument(
        "--post-connect-restart-delay",
        type=float,
        default=1.5,
        help="How long to wait after a fresh controller connect before sending the restart combo.",
    )
    parser.add_argument(
        "--target-failed-hold",
        type=float,
        default=0.8,
        help="How long the target-failed image must stay matched before we confirm a retry.",
    )
    parser.add_argument(
        "--target-ok-threshold",
        type=float,
        default=0.16,
        help="Maximum invariant-match score for the target_ok example. Lower is stricter.",
    )
    parser.add_argument(
        "--target-ok-hold",
        type=float,
        default=0.8,
        help="How long target_ok must stay confidently matched before we stop the loop.",
    )
    parser.add_argument(
        "--target-ok-score-margin",
        type=float,
        default=0.03,
        help="How much better target_ok must score than target_fail before we accept it.",
    )
    parser.add_argument(
        "--success-candidate-hold",
        type=float,
        default=3.5,
        help="How long a non-black, non-failure scene must persist before we stop as a success candidate. Higher is safer.",
    )
    parser.add_argument(
        "--success-candidate-fail-margin",
        type=float,
        default=0.03,
        help="Extra margin above the fail threshold required before we treat a scene as safe for success-candidate timing.",
    )
    parser.add_argument(
        "--success-confirm-checks",
        type=int,
        default=3,
        help="How many extra confirmation samples to take before declaring a success candidate.",
    )
    parser.add_argument(
        "--success-confirm-interval",
        type=float,
        default=0.4,
        help="Delay between success confirmation samples.",
    )
    parser.add_argument(
        "--recent-failed-store-margin",
        type=float,
        default=0.02,
        help="Extra margin above the fail threshold required before storing a failed ROI for similarity checks.",
    )
    parser.add_argument(
        "--timeout-recovery-limit",
        type=int,
        default=1,
        help="How many times a timed-out step may scan known ROIs and resume before forcing a reset.",
    )
    parser.add_argument(
        "--debug-dir",
        type=Path,
        default=ROOT / "debug" / "camera" / "outcomes",
    )
    parser.add_argument(
        "--failed-roi-dir",
        type=Path,
        default=ROOT / "debug" / "camera" / "failed_rois",
        help="Directory where matched target-failed ROI crops are saved.",
    )
    parser.add_argument(
        "--stats-file",
        type=Path,
        default=ROOT / "debug" / "camera" / "loop_stats.json",
        help="JSON file used to persist total time, current loop time, and loop count.",
    )
    parser.add_argument(
        "--control-file",
        type=Path,
        default=ROOT / "debug" / "camera" / "loop_control.json",
        help="JSON file used to send restart/stop commands to the running service.",
    )
    parser.add_argument(
        "--service-lock",
        type=Path,
        default=ROOT / "debug" / "camera" / "service.lock",
        help="Lock file to ensure only one service instance owns the camera/Bluetooth.",
    )
    parser.add_argument(
        "--feed-port",
        type=int,
        default=8080,
        help="HTTP port for the always-on MJPEG preview feed.",
    )
    parser.add_argument(
        "--feed-fps",
        type=float,
        default=5.0,
        help="Preview feed frame rate while the service is running.",
    )
    parser.add_argument(
        "--notify-email-to",
        type=str,
        default=os.getenv("SWITCH_NOTIFY_EMAIL_TO"),
        help="Destination email address for stop/success notifications.",
    )
    parser.add_argument(
        "--notify-email-from",
        type=str,
        default=os.getenv("SWITCH_NOTIFY_EMAIL_FROM"),
        help="From address used for stop/success notifications.",
    )
    parser.add_argument(
        "--smtp-host",
        type=str,
        default=os.getenv("SWITCH_SMTP_HOST"),
        help="SMTP host for email notifications.",
    )
    parser.add_argument(
        "--smtp-port",
        type=int,
        default=int(os.getenv("SWITCH_SMTP_PORT", "587")),
        help="SMTP port for email notifications.",
    )
    parser.add_argument(
        "--smtp-user",
        type=str,
        default=os.getenv("SWITCH_SMTP_USER"),
        help="SMTP username for email notifications.",
    )
    parser.add_argument(
        "--smtp-password-env",
        type=str,
        default="SWITCH_SMTP_PASSWORD",
        help="Environment variable name containing the SMTP password.",
    )
    parser.add_argument(
        "--smtp-ssl",
        action="store_true",
        default=_env_flag("SWITCH_SMTP_SSL"),
        help="Use SMTP over implicit SSL instead of STARTTLS.",
    )
    parser.add_argument(
        "--smtp-no-starttls",
        action="store_true",
        help="Disable STARTTLS for plain SMTP connections.",
    )
    return parser


def build_config(args: argparse.Namespace) -> CameraLoopConfig:
    images_dir = args.images_dir
    target_roi = Roi(x=1040, y=186, width=381, height=337)
    target_ok_path = images_dir / "target_ok.png"
    target_fail_path = images_dir / "target_fail.png"

    return CameraLoopConfig(
        start_detector=StaticImageDetector(
            name="game_launch",
            image_path=images_dir / "0_game_launch.png",
            roi=Roi(x=610, y=373, width=669, height=334),
            threshold=0.12,
            search_margin=args.search_margin,
            stride=6,
            search_step=4,
        ),
        press_start_detector=StaticImageDetector(
            name="press_start",
            image_path=images_dir / "1_press_start.png",
            roi=Roi(x=419, y=169, width=763, height=295),
            threshold=0.12,
            search_margin=args.search_margin,
            stride=6,
            search_step=4,
        ),
        select_save_detector=StaticImageDetector(
            name="continue",
            image_path=images_dir / "2_select_save.png",
            roi=Roi(x=479, y=198, width=302, height=72),
            threshold=0.08,
            search_margin=args.search_margin,
            stride=2,
            search_step=2,
        ),
        previously_detector=StaticImageDetector(
            name="previously",
            image_path=images_dir / "3_previously.png",
            roi=Roi(x=384, y=151, width=650, height=85),
            threshold=0.08,
            search_margin=args.search_margin,
            stride=2,
            search_step=2,
        ),
        ready_detector=StaticImageDetector(
            name="ready",
            image_path=images_dir / "4_ready.png",
            roi=Roi(x=838, y=384, width=200, height=233),
            threshold=0.16,
            search_margin=args.search_margin,
            stride=8,
            search_step=4,
        ),
        target_failed_detector=StaticImageDetector(
            name="target_failed",
            image_path=images_dir / "6_target_failed.png",
            roi=target_roi,
            threshold=args.target_failed_threshold,
            search_margin=args.search_margin,
            stride=4,
            search_step=2,
        ),
        black_screen_detector=BlackScreenDetector(
            roi=Roi(x=320, y=180, width=1280, height=720),
            max_mean_luma=18.0,
            max_luma_stddev=14.0,
        ),
        target_ok_detector=(
            InvariantColorDetector(
                name="target_ok",
                image_path=target_ok_path,
                roi=target_roi,
                threshold=args.target_ok_threshold,
                search_margin=0,
                stride=4,
                search_step=2,
                feature_size=72,
                mask_chroma_threshold=0.08,
                trim_fraction=0.10,
            )
            if target_ok_path.exists()
            else None
        ),
        target_fail_example_detector=(
            InvariantColorDetector(
                name="target_fail_example",
                image_path=target_fail_path,
                roi=target_roi,
                threshold=args.target_ok_threshold,
                search_margin=0,
                stride=4,
                search_step=2,
                feature_size=72,
                mask_chroma_threshold=0.08,
                trim_fraction=0.10,
            )
            if target_fail_path.exists()
            else None
        ),
        press_interval=args.press_interval,
        settle_time=args.settle_time,
        poll_interval=args.poll_interval,
        match_poll_interval=args.match_poll_interval,
        step_timeout=args.step_timeout,
        blackscreen_timeout=args.blackscreen_timeout,
        outcome_timeout=args.outcome_timeout,
        outcome_settle_delay=args.outcome_settle_delay,
        post_connect_restart_delay=args.post_connect_restart_delay,
        target_failed_hold=args.target_failed_hold,
        target_ok_hold=args.target_ok_hold,
        target_ok_score_margin=args.target_ok_score_margin,
        success_candidate_hold=args.success_candidate_hold,
        success_candidate_fail_margin=args.success_candidate_fail_margin,
        success_confirm_checks=args.success_confirm_checks,
        success_confirm_interval=args.success_confirm_interval,
        recent_failed_store_margin=args.recent_failed_store_margin,
        timeout_recovery_limit=args.timeout_recovery_limit,
        debug_dir=args.debug_dir,
        failed_roi_dir=args.failed_roi_dir,
        stats_file=args.stats_file,
        control_file=args.control_file,
    )


def _format_duration(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _env_flag(name: str) -> bool:
    value = os.getenv(name)
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _guess_ip_addresses() -> list[str]:
    import subprocess

    try:
        result = subprocess.run(
            ["hostname", "-I"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return []
    return [part for part in result.stdout.split() if part]


def _handle_control_action(args: argparse.Namespace) -> int:
    control = PersistentLoopControl.load(args.control_file)
    stats = PersistentLoopStats.load(args.stats_file)

    if args.action == "pair":
        control.set_command("pair")
        print(f"Requested pairing via {args.control_file}")
        return 0

    if args.action == "restart":
        control.set_command("restart")
        print(f"Requested restart via {args.control_file}")
        return 0

    if args.action == "stop":
        control.set_command("stop")
        print(f"Requested stop via {args.control_file}")
        return 0

    if args.action == "status":
        snapshot = stats.snapshot()
        command = control.refresh()
        print(f"command={command}")
        print(f"status={snapshot.status}")
        print(f"loop_counter={snapshot.loop_counter}")
        print(f"total_elapsed={_format_duration(snapshot.total_elapsed_seconds)}")
        print(f"loop_elapsed={_format_duration(snapshot.loop_elapsed_seconds)}")
        print(f"last_outcome={snapshot.last_outcome}")
        print(f"stats_file={args.stats_file}")
        print(f"control_file={args.control_file}")
        return 0

    raise RuntimeError(f"Unsupported action: {args.action}")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.action != "run":
        return _handle_control_action(args)

    lock_handle = _acquire_service_lock(args.service_lock)
    if lock_handle is None:
        return 2
    _install_signal_handlers()

    capture = open_capture(
        camera_index=args.camera_index,
        width=args.width,
        height=args.height,
        fps=args.fps,
        warmup=args.warmup,
    )
    controller = NxbtBackend(
        reconnect=True,
        adapter_path=args.adapter_path,
    )
    notifier = _build_email_notifier(args)
    runner = CameraLoopRunner(
        controller=controller,
        capture=capture,
        config=build_config(args),
        notify_cb=notifier,
    )
    preview = _start_preview_server(
        capture=capture,
        port=args.feed_port,
        fps=args.feed_fps,
        overlay_state_fn=runner.preview_overlay_state,
    )

    try:
        stats = runner.initialize()
        print(
            "Started in stopped mode: "
            f"total={_format_duration(stats.total_elapsed_seconds)} "
            f"loop={_format_duration(stats.loop_elapsed_seconds)} "
            f"count={stats.loop_counter}"
        )
        print("Camera is active. Controller connection will happen only when pairing or a loop is requested.")
        print("Use `--action pair` first if you want to pair before starting the loop.")
        print(f"Preview feed: http://127.0.0.1:{preview.port}/stream.mjpg")
        for address in _guess_ip_addresses():
            print(f"Preview feed: http://{address}:{preview.port}/stream.mjpg")
        print(
            f"Use `./.venv/bin/python scripts/run_camera_loop.py --action pair`, "
            f"`--action restart`, or `--action stop` from another terminal."
        )
        outcome = runner.run_service(attempts=args.attempts)
        print(f"Final outcome: {outcome.status} - {outcome.detail}")
        return 0
    finally:
        try:
            lock_handle.close()
        except Exception:
            pass
        try:
            args.service_lock.unlink(missing_ok=True)
        except Exception:
            pass
        controller.close()
        preview.close()
        capture.close()


def _build_email_notifier(args: argparse.Namespace):
    if not args.notify_email_to or not args.notify_email_from or not args.smtp_host:
        return None

    recipients = [part.strip() for part in args.notify_email_to.split(",") if part.strip()]
    if not recipients:
        return None

    password = os.getenv(args.smtp_password_env) if args.smtp_password_env else None
    use_starttls = not args.smtp_ssl and not args.smtp_no_starttls

    def _notify(subject: str, body: str, attachments: list[Path]) -> None:
        message = email.message.EmailMessage()
        message["Subject"] = f"[switch-automation] {subject}"
        message["From"] = args.notify_email_from
        message["To"] = ", ".join(recipients)
        message.set_content(body)
        for attachment in attachments:
            if not attachment.exists():
                continue
            data = attachment.read_bytes()
            message.add_attachment(
                data,
                maintype="image",
                subtype="jpeg",
                filename=attachment.name,
            )

        if args.smtp_ssl:
            with smtplib.SMTP_SSL(args.smtp_host, args.smtp_port, context=ssl.create_default_context()) as smtp:
                if args.smtp_user:
                    smtp.login(args.smtp_user, password or "")
                smtp.send_message(message)
            return

        with smtplib.SMTP(args.smtp_host, args.smtp_port) as smtp:
            smtp.ehlo()
            if use_starttls:
                smtp.starttls(context=ssl.create_default_context())
                smtp.ehlo()
            if args.smtp_user:
                smtp.login(args.smtp_user, password or "")
            smtp.send_message(message)

    return _notify


def _start_preview_server(
    *,
    capture,
    port: int,
    fps: float,
    overlay_state_fn,
):
    replaced_holder = False
    while True:
        try:
            return MjpegPreviewServer(
                capture=capture,
                port=port,
                fps=fps,
                overlay_state_fn=overlay_state_fn,
            ).start()
        except OSError as exc:
            if getattr(exc, "errno", None) == 98:
                holder_pid = _pid_listening_on_port(port)
                if (
                    not replaced_holder
                    and holder_pid is not None
                    and holder_pid != os.getpid()
                    and _pid_matches_camera_service(holder_pid)
                ):
                    replaced_holder = True
                    print(f"Preview port {port} is held by stale run_camera_loop process (pid={holder_pid}). Replacing it...")
                    _terminate_process(holder_pid)
                    time.sleep(0.2)
                    continue

                holder_detail = ""
                if holder_pid is not None:
                    holder_command = _command_for_pid(holder_pid)
                    if holder_command:
                        holder_detail = f" Holder pid={holder_pid}: {holder_command}"
                    else:
                        holder_detail = f" Holder pid={holder_pid}."
                raise OSError(
                    f"Preview port {port} is already in use. Refusing to fall back to another port."
                    f"{holder_detail}"
                ) from exc
            raise


def _acquire_service_lock(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    takeover_attempted = False
    while True:
        handle = lock_path.open("a+")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            holder = ""
            pid = None
            try:
                handle.seek(0)
                holder = handle.read().strip()
                pid = _parse_lock_pid(holder)
            except Exception:
                holder = ""
            finally:
                handle.close()

            if (
                not takeover_attempted
                and pid is not None
                and _pid_matches_camera_service(pid)
            ):
                takeover_attempted = True
                print(f"Replacing existing run_camera_loop service (pid={pid})...")
                _terminate_process(pid)
                continue

            print("Another run_camera_loop service is already active.")
            if holder:
                print(f"Lock file info: {holder}")
            return None

        handle.seek(0)
        handle.truncate(0)
        handle.write(f"pid={os.getpid()} started_at={time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        handle.flush()
        os.chmod(lock_path, 0o666)
        return handle


def _install_signal_handlers() -> None:
    def _handle_signal(signum, _frame):
        raise SystemExit(128 + signum)

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, _handle_signal)


def _parse_lock_pid(holder: str) -> int | None:
    for part in holder.split():
        if part.startswith("pid="):
            value = part.split("=", 1)[1]
            if value.isdigit():
                return int(value)
    return None


def _pid_matches_camera_service(pid: int) -> bool:
    command = _command_for_pid(pid)
    if command is None:
        return False
    return _command_looks_like_camera_service(command)


def _command_for_pid(pid: int) -> str | None:
    try:
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "args="],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip()


def _command_looks_like_camera_service(command: str) -> bool:
    return (
        command != ""
        and "python" in command
        and "scripts/run_camera_loop.py" in command
        and (
            "--action run" in command
            or "--action" not in command
        )
    )


def _pid_listening_on_port(port: int) -> int | None:
    try:
        result = subprocess.run(
            ["ss", "-ltnp"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None

    needle = f":{port}"
    for line in result.stdout.splitlines():
        if needle not in line:
            continue
        marker = "pid="
        if marker not in line:
            continue
        pid_text = line.split(marker, 1)[1].split(",", 1)[0].split(")", 1)[0].strip()
        if pid_text.isdigit():
            return int(pid_text)
    return None


def _terminate_process(pid: int) -> None:
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except Exception:
        return

    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        if not Path(f"/proc/{pid}").exists():
            return
        time.sleep(0.1)

    try:
        os.kill(pid, signal.SIGKILL)
    except Exception:
        pass


if __name__ == "__main__":
    raise SystemExit(main())
