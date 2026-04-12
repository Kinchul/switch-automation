from __future__ import annotations

import argparse

from automation import HuntRunner
from control import Button, NxbtBackend


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="switch-automation")
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="Validate the local control backend import.")
    doctor.set_defaults(handler=_handle_doctor)

    connect = subparsers.add_parser("connect", help="Connect a controller backend to the Switch.")
    connect.set_defaults(handler=_handle_connect)

    press = subparsers.add_parser("press", help="Press one or more buttons.")
    press.add_argument("buttons", nargs="+", choices=[button.value for button in Button])
    press.add_argument("--down", type=float, default=0.1)
    press.add_argument("--up", type=float, default=0.1)
    press.set_defaults(handler=_handle_press)

    macro = subparsers.add_parser("macro", help="Run an NXBT macro string.")
    macro.add_argument("commands", help="Macro commands, e.g. 'A 0.1s\\n0.1s'")
    macro.set_defaults(handler=_handle_macro)
    return parser


def _make_runner() -> HuntRunner:
    return HuntRunner(controller=NxbtBackend())


def _handle_doctor(_args: argparse.Namespace) -> int:
    _make_runner()
    print("NXBT backend import looks available.")
    return 0


def _handle_connect(_args: argparse.Namespace) -> int:
    runner = _make_runner()
    runner.connect()
    print("Controller connected.")
    return 0


def _handle_press(args: argparse.Namespace) -> int:
    runner = _make_runner()
    runner.connect()
    buttons = [Button(button_name) for button_name in args.buttons]
    runner.controller.press(*buttons, down=args.down, up=args.up)
    print(f"Pressed: {', '.join(args.buttons)}")
    return 0


def _handle_macro(args: argparse.Namespace) -> int:
    runner = _make_runner()
    runner.connect()
    runner.run_macro(args.commands)
    print("Macro sent.")
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
