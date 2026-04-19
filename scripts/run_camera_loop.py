from __future__ import annotations

import argparse
import email.message
import os
import signal
import smtplib
import ssl
import subprocess
import sys
import time
from pathlib import Path

try:
    import fcntl
except ModuleNotFoundError:
    fcntl = None


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from automation.persistence import PersistentLoopControl, PersistentLoopStatsStore
from automation.sequence import load_sequences


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the sequence-driven camera-guided Switch loop."
    )
    parser.add_argument(
        "--action",
        choices=[
            "run",
            "pair",
            "restart",
            "reset",
            "stop",
            "status",
            "select-sequence",
            "select_sequence",
            "reset-stats",
            "reset_stats",
            "list-sequences",
            "list_sequences",
        ],
        default="run",
        help=(
            "`run` starts the foreground service, `pair` connects the controller and waits idle, "
            "`restart` triggers a new loop, `reset` forces a game reset then starts a new loop, "
            "`stop` idles the service, `status` prints control and stats, `select-sequence` "
            "persists the sequence to use on the next restart, "
            "`reset-stats` clears all per-sequence stats, and `list-sequences` lists the available JSON files."
        ),
    )
    parser.add_argument("--sequence", type=str, default=None, help="Sequence id for select-sequence.")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--warmup", type=float, default=2.0)
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=None,
        help="Optional BlueZ adapter path to use (e.g. /org/bluez/hci0).",
    )
    parser.add_argument(
        "--attempts",
        type=int,
        default=0,
        help="Maximum retry attempts before stopping. Use 0 for unlimited retries.",
    )
    parser.add_argument(
        "--sequences-dir",
        type=Path,
        default=ROOT / "sequences",
        help="Directory containing sequence JSON files.",
    )
    parser.add_argument(
        "--default-sequence",
        type=str,
        default="sulfura",
        help="Sequence id to use when the control file has no valid selection.",
    )
    parser.add_argument(
        "--match-poll-interval",
        type=float,
        default=0.05,
        help="How often the sequence engine re-checks detectors between actions.",
    )
    parser.add_argument(
        "--stats-file",
        type=Path,
        default=ROOT / "debug" / "camera" / "loop_stats.json",
        help="JSON file used to persist per-sequence time and retry counters.",
    )
    parser.add_argument(
        "--control-file",
        type=Path,
        default=ROOT / "debug" / "camera" / "loop_control.json",
        help="JSON file used to send commands and store the selected sequence.",
    )
    parser.add_argument(
        "--debug-dir",
        type=Path,
        default=ROOT / "debug" / "camera",
        help="Root directory under which per-sequence image subdirectories are created.",
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
        help="Destination email address for state notifications.",
    )
    parser.add_argument(
        "--notify-email-from",
        type=str,
        default=os.getenv("SWITCH_NOTIFY_EMAIL_FROM"),
        help="From address used for state notifications.",
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
    from automation.camera_loop import CameraLoopConfig

    return CameraLoopConfig(
        sequences_dir=args.sequences_dir,
        default_sequence=args.default_sequence,
        debug_dir=args.debug_dir,
        stats_file=args.stats_file,
        control_file=args.control_file,
        match_poll_interval=args.match_poll_interval,
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
    stats = PersistentLoopStatsStore.load(args.stats_file)
    sequences = load_sequences(args.sequences_dir)

    normalized_action = args.action.replace("_", "-")
    if normalized_action == "pair":
        control.set_command("pair")
        print(f"Requested pairing via {args.control_file}")
        return 0

    if normalized_action == "restart":
        control.set_command("restart")
        print(f"Requested restart via {args.control_file}")
        return 0

    if normalized_action == "reset":
        control.set_command("reset")
        print(f"Requested reset via {args.control_file}")
        return 0

    if normalized_action == "stop":
        control.set_command("stop")
        print(f"Requested stop via {args.control_file}")
        return 0

    if normalized_action == "select-sequence":
        if not args.sequence:
            raise RuntimeError("--sequence is required with --action select-sequence.")
        if args.sequence not in sequences:
            available = ", ".join(sorted(sequences)) or "(none)"
            raise RuntimeError(
                f'Sequence "{args.sequence}" was not found in {args.sequences_dir}. Available: {available}'
            )
        control.set_selected_sequence(args.sequence)
        print(f'Selected sequence "{args.sequence}" via {args.control_file}')
        return 0

    if normalized_action == "reset-stats":
        stats.reset()
        print(f"Reset all sequence statistics in {args.stats_file}")
        return 0

    if normalized_action == "list-sequences":
        if not sequences:
            print("No sequences found.")
            return 0
        for sequence_id, definition in sequences.items():
            marker = " *" if control.selected_sequence == sequence_id else ""
            if definition.name:
                print(f"{sequence_id}{marker} - {definition.name}")
            else:
                print(f"{sequence_id}{marker}")
        return 0

    if normalized_action == "status":
        command = control.refresh()
        print(f"command={command}")
        print(f"selected_sequence={control.selected_sequence}")
        print(f"control_file={args.control_file}")
        print(f"stats_file={args.stats_file}")
        if sequences:
            print("available_sequences=" + ",".join(sorted(sequences)))
        else:
            print("available_sequences=")
        for sequence_id, snapshot in stats.snapshots().items():
            print(
                "sequence="
                f"{sequence_id} "
                f"status={snapshot.status} "
                f"loop_counter={snapshot.loop_counter} "
                f"total_elapsed={_format_duration(snapshot.total_elapsed_seconds)} "
                f"loop_elapsed={_format_duration(snapshot.loop_elapsed_seconds)} "
                f"last_outcome={snapshot.last_outcome}"
            )
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

    from automation.camera_loop import CameraLoopRunner
    from control import NxbtBackend
    from vision import open_capture

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
        control = PersistentLoopControl.load(args.control_file)
        print(
            "Started in stopped mode: "
            f"selected_sequence={control.selected_sequence} "
            f"total={_format_duration(stats.total_elapsed_seconds)} "
            f"loop={_format_duration(stats.loop_elapsed_seconds)} "
            f"count={stats.loop_counter}"
        )
        print("Camera is active. Controller connection will happen only when pairing or a loop is requested.")
        print(f"Preview feed: http://127.0.0.1:{preview.port}/stream.mjpg")
        for address in _guess_ip_addresses():
            print(f"Preview feed: http://{address}:{preview.port}/stream.mjpg")
        print(
            "Use "
            "`./.venv/bin/python scripts/run_camera_loop.py --action pair`, "
            "`--action restart`, `--action reset`, `--action stop`, "
            "`--action select-sequence --sequence YOUR_SEQUENCE`, or "
            "`--action reset-stats` from another terminal."
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
    from vision.stream import MjpegPreviewServer

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
    if fcntl is None:
        handle = lock_path.open("w")
        handle.write(f"pid={os.getpid()} started_at={time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        handle.flush()
        return handle

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
