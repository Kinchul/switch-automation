from __future__ import annotations

import argparse
import curses
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "nxbt"))

import nxbt  # type: ignore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Control a Nintendo Switch from the terminal with NXBT."
    )
    parser.add_argument(
        "--pairing-menu",
        action="store_true",
        help='connect through "Change Grip/Order" instead of reconnect mode',
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="enable verbose NXBT logging",
    )
    return parser


@dataclass(frozen=True)
class Binding:
    label: str
    buttons: tuple[object, ...]


KEY_BINDINGS = {
    "w": Binding("DPAD_UP", (nxbt.Buttons.DPAD_UP,)),
    "s": Binding("DPAD_DOWN", (nxbt.Buttons.DPAD_DOWN,)),
    "d": Binding("DPAD_RIGHT", (nxbt.Buttons.DPAD_RIGHT,)),
    "f": Binding("DPAD_LEFT", (nxbt.Buttons.DPAD_LEFT,)),
    "x": Binding("X", (nxbt.Buttons.X,)),
    "y": Binding("Y", (nxbt.Buttons.Y,)),
    "b": Binding("B", (nxbt.Buttons.B,)),
    "a": Binding("A", (nxbt.Buttons.A,)),
    "q": Binding("L", (nxbt.Buttons.L,)),
    "e": Binding("R", (nxbt.Buttons.R,)),
    "z": Binding("ZL", (nxbt.Buttons.ZL,)),
    "c": Binding("ZR", (nxbt.Buttons.ZR,)),
    "\r": Binding("PLUS", (nxbt.Buttons.PLUS,)),
    "\n": Binding("PLUS", (nxbt.Buttons.PLUS,)),
    "\x7f": Binding("MINUS", (nxbt.Buttons.MINUS,)),
    "h": Binding("HOME", (nxbt.Buttons.HOME,)),
    "v": Binding("CAPTURE", (nxbt.Buttons.CAPTURE,)),
    "p": Binding("L+R", (nxbt.Buttons.L, nxbt.Buttons.R)),
}

SPECIAL_KEY_BINDINGS = {
    curses.KEY_UP: Binding("DPAD_UP", (nxbt.Buttons.DPAD_UP,)),
    curses.KEY_DOWN: Binding("DPAD_DOWN", (nxbt.Buttons.DPAD_DOWN,)),
    curses.KEY_LEFT: Binding("DPAD_LEFT", (nxbt.Buttons.DPAD_LEFT,)),
    curses.KEY_RIGHT: Binding("DPAD_RIGHT", (nxbt.Buttons.DPAD_RIGHT,)),
    curses.KEY_ENTER: Binding("PLUS", (nxbt.Buttons.PLUS,)),
    curses.KEY_BACKSPACE: Binding("MINUS", (nxbt.Buttons.MINUS,)),
}


HELP_TEXT = """
Keyboard controls
  arrows      D-pad
  w s d f     D-pad fallback
  a b x y     A B X Y
  q / e       L / R
  z / c       ZL / ZR
  Enter       PLUS
  Backspace   MINUS
  h           HOME
  v           CAPTURE
  p           L+R join pulse
  combos      press multiple mapped keys quickly together
  ?           show this help
  . or Ctrl-C quit
""".strip()


class KeyboardController:
    def __init__(self, pairing_menu: bool = False, debug: bool = False):
        self._pairing_menu = pairing_menu
        self._service = nxbt.Nxbt(debug=debug)
        self._controller_index: int | None = None
        self._shutdown = False

    def connect(self) -> None:
        adapters = self._service.get_available_adapters()
        if not adapters:
            raise RuntimeError("No Bluetooth adapters available.")

        adapter = adapters[0]
        print(f"Using adapter: {adapter}")

        try:
            nxbt.clean_sdp_records()
            print("Cleaned BlueZ SDP records.")
        except Exception as exc:
            print(f"Warning: could not clean SDP records: {exc}")

        try:
            subprocess.run(
                ["btmgmt", "--index", "0", "io-cap", "0x03"],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as exc:
            print(f"Warning: could not set IO capability: {exc}")

        reconnect_addresses = None
        if not self._pairing_menu:
            reconnect_addresses = self._service.get_switch_addresses() or None

        if reconnect_addresses:
            print("Reconnecting to a previously paired Switch...")
        else:
            print('Waiting for the Switch on "Change Grip/Order"...')

        self._controller_index = self._service.create_controller(
            nxbt.PRO_CONTROLLER,
            adapter,
            reconnect_address=reconnect_addresses,
        )

        last_state = None
        while True:
            state = dict(self._service.state[self._controller_index])
            state_name = state.get("state")
            if state_name != last_state:
                print(f"State: {state_name}")
                last_state = state_name

            if state_name == "crashed":
                raise RuntimeError(state.get("errors") or "controller crashed")

            if state_name == "connected":
                break

            time.sleep(0.25)

        if not reconnect_addresses:
            self.join_pairing_menu()

    def press(self, binding: Binding) -> None:
        self._require_connection()
        self.press_buttons(binding.label, binding.buttons)

    def press_buttons(self, label: str, buttons: tuple[object, ...] | list[object]) -> None:
        self._require_connection()
        self._service.press_buttons(
            self._controller_index,
            list(buttons),
            down=0.08,
            up=0.12,
        )
        console_line(f"Sent: {label}")

    def join_pairing_menu(self) -> None:
        self._require_connection()
        console_line("Sending L+R to join the pairing screen...")
        for _ in range(2):
            self._service.press_buttons(
                self._controller_index,
                [nxbt.Buttons.L, nxbt.Buttons.R],
                down=0.1,
                up=0.2,
            )
            time.sleep(0.25)
        console_line("Join pulse sent.")

    def shutdown(self) -> None:
        if self._shutdown:
            return
        self._shutdown = True

        if self._controller_index is None:
            try:
                self._service.close()
            except Exception:
                pass
            return
        try:
            self._service.remove_controller(self._controller_index)
        except Exception:
            pass
        self._controller_index = None
        try:
            self._service.close()
        except Exception:
            pass

    def _require_connection(self) -> None:
        if self._controller_index is None:
            raise RuntimeError("Controller is not connected.")


def console_line(text: str = "") -> None:
    sys.stdout.write(f"\r{text}\r\n")
    sys.stdout.flush()


def read_key_burst(stdscr, timeout: float = 0.1, combo_window: float = 0.05) -> list[int]:
    stdscr.timeout(max(1, int(timeout * 1000)))
    first_key = stdscr.getch()
    if first_key == -1:
        return []

    if first_key in SPECIAL_KEY_BINDINGS:
        return [first_key]

    keys: list[int] = [first_key]
    deadline = time.monotonic() + combo_window

    while time.monotonic() < deadline:
        stdscr.timeout(10)
        next_key = stdscr.getch()
        if next_key == -1:
            continue
        if next_key in SPECIAL_KEY_BINDINGS:
            break
        keys.append(next_key)
        deadline = time.monotonic() + combo_window

    return keys


def resolve_binding(key: int) -> Binding | None:
    binding = SPECIAL_KEY_BINDINGS.get(key)
    if binding is not None:
        return binding
    if not (0 <= key <= 255):
        return None
    char = chr(key)
    lowered = char.lower()
    return KEY_BINDINGS.get(lowered) or KEY_BINDINGS.get(char)


def run_loop(stdscr, controller: KeyboardController, stop_requested_ref) -> None:
    stdscr.keypad(True)
    curses.noecho()
    curses.cbreak()
    try:
        curses.curs_set(0)
    except curses.error:
        pass

    while not stop_requested_ref["stop"]:
        try:
            keys = read_key_burst(stdscr)
        except curses.error:
            continue

        if not keys:
            continue
        if any(key in (3, ord(".")) for key in keys):
            break
        if ord("?") in keys:
            console_line()
            for line in HELP_TEXT.splitlines():
                console_line(line)
            console_line()
            continue

        labels = []
        buttons = []
        for key in keys:
            binding = resolve_binding(key)
            if binding is None:
                continue
            labels.append(binding.label)
            for button in binding.buttons:
                if button not in buttons:
                    buttons.append(button)

        if not buttons:
            continue
        controller.press_buttons(" + ".join(labels), buttons)


def main() -> int:
    args = build_parser().parse_args()
    controller = KeyboardController(
        pairing_menu=args.pairing_menu,
        debug=args.debug,
    )
    stop_requested = {"stop": False}

    def shutdown(*_args):
        stop_requested["stop"] = True

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    try:
        controller.connect()
        print()
        print(HELP_TEXT)
        print()

        curses.wrapper(run_loop, controller, stop_requested)
    finally:
        controller.shutdown()
        console_line("Exiting keyboard controller.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
