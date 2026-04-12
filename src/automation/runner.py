from __future__ import annotations

from dataclasses import dataclass

from control import Button, ControllerBackend


@dataclass(slots=True)
class HuntRunner:
    controller: ControllerBackend

    def connect(self) -> None:
        self.controller.connect()

    def press_a(self) -> None:
        self.controller.press(Button.A)

    def press_home(self) -> None:
        self.controller.press(Button.HOME)

    def run_macro(self, commands: str) -> None:
        self.controller.macro(commands)
