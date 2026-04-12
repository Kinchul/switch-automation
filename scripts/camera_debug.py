from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vision import open_capture


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preview, stream, and collect reference frames from the Raspberry Pi CSI camera."
    )
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--camera-index", type=int, default=0)
    common.add_argument("--width", type=int, default=1920)
    common.add_argument("--height", type=int, default=1080)
    common.add_argument("--fps", type=int, default=20)
    common.add_argument("--warmup", type=float, default=2.0)

    subparsers = parser.add_subparsers(dest="command", required=True)

    snapshot = subparsers.add_parser(
        "snapshot",
        parents=[common],
        help="Save a single JPEG frame.",
    )
    snapshot.add_argument(
        "--output",
        type=Path,
        default=ROOT / "debug" / "camera" / "snapshot.jpg",
        help="JPEG path to write.",
    )
    snapshot.set_defaults(handler=_handle_snapshot)

    sample = subparsers.add_parser(
        "sample",
        parents=[common],
        help="Save a series of JPEG frames.",
    )
    sample.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "debug" / "camera" / "samples",
        help="Directory to write sample frames into.",
    )
    sample.add_argument("--count", type=int, default=10, help="How many frames to capture.")
    sample.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Seconds to wait between captures.",
    )
    sample.set_defaults(handler=_handle_sample)

    stream = subparsers.add_parser(
        "stream",
        parents=[common],
        help="Start an H.264 TCP feed that VLC can open remotely.",
    )
    stream.add_argument("--port", type=int, default=8888)
    stream.add_argument(
        "--exit-on-disconnect",
        action="store_true",
        help="Stop after the current VLC client disconnects instead of restarting the feed.",
    )
    stream.set_defaults(handler=_handle_stream)
    return parser


def _timestamped_name(index: int) -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{index:03d}.jpg"


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


def _capture(args: argparse.Namespace):
    return open_capture(
        camera_index=args.camera_index,
        width=args.width,
        height=args.height,
        fps=args.fps,
        warmup=args.warmup,
    )


def _handle_snapshot(args: argparse.Namespace) -> int:
    capture = _capture(args)
    try:
        output = capture.save_frame(args.output)
    finally:
        capture.close()

    print(f"Saved camera snapshot to {output}")
    return 0


def _handle_sample(args: argparse.Namespace) -> int:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    capture = _capture(args)
    try:
        for index in range(args.count):
            output = output_dir / _timestamped_name(index)
            capture.save_frame(output)
            print(f"[{index + 1}/{args.count}] saved {output}")
            if index + 1 < args.count:
                time.sleep(args.interval)
    finally:
        capture.close()

    return 0


def _handle_stream(args: argparse.Namespace) -> int:
    command = [
        "rpicam-vid",
        "-t",
        "0",
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--framerate",
        str(args.fps),
        "--inline",
        "--listen",
        "-o",
        f"tcp://0.0.0.0:{args.port}",
    ]

    print("Starting camera stream...")
    for address in _guess_ip_addresses():
        print(f"  VLC URL: tcp/h264://{address}:{args.port}")
    if args.exit_on_disconnect:
        print("The feed will stop after the current VLC client disconnects.")
    else:
        print("The feed will restart automatically when VLC disconnects. Press Ctrl-C to stop.")

    while True:
        try:
            result = subprocess.run(command, check=False)
        except KeyboardInterrupt:
            return 130

        if result.returncode == 0:
            return 0

        if args.exit_on_disconnect:
            print(f"Camera stream exited with code {result.returncode}.")
            return result.returncode

        print(
            f"Camera stream client disconnected or exited with code {result.returncode}; "
            "waiting for the next connection..."
        )
        time.sleep(1.0)
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
