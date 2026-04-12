from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

from .backend import Button, ControllerBackend


def _import_nxbt():
    try:
        import nxbt  # type: ignore
        return nxbt
    except ModuleNotFoundError:
        repo_root = Path(__file__).resolve().parents[3]
        local_checkout = repo_root / "tools" / "nxbt"
        if local_checkout.exists():
            sys.path.insert(0, str(local_checkout))
            import nxbt  # type: ignore
            return nxbt
        raise RuntimeError(
            "NXBT is not installed. Run `python -m pip install -e ./tools/nxbt` "
            "or install nxbt into the current environment."
        )


class NxbtBackend(ControllerBackend):
    def __init__(self, controller_type: str = "PRO_CONTROLLER", reconnect: bool = True):
        self._nxbt_mod = _import_nxbt()
        self._controller_type_name = controller_type
        self._reconnect = reconnect
        self._service = None
        self._controller_index = None

    def connect(self) -> None:
        if self._service is None:
            self._service = self._nxbt_mod.Nxbt()

        controller_type = getattr(self._nxbt_mod, self._controller_type_name)
        kwargs = {}
        if self._reconnect:
            switch_addresses = self._service.get_switch_addresses()
            if switch_addresses:
                kwargs["reconnect_address"] = switch_addresses

        self._controller_index = self._service.create_controller(controller_type, **kwargs)
        self._service.wait_for_connection(self._controller_index)

    def press(self, *buttons: Button, down: float = 0.1, up: float = 0.1) -> None:
        self._require_connection()
        mapped_buttons = [getattr(self._nxbt_mod.Buttons, button.value) for button in buttons]
        self._service.press_buttons(self._controller_index, mapped_buttons, down=down, up=up)

    def macro(self, commands: str, block: bool = True):
        self._require_connection()
        return self._service.macro(self._controller_index, commands, block=block)

    def press_sequence(self, sequence: Iterable[tuple[Button, float, float]]) -> None:
        for button, down, up in sequence:
            self.press(button, down=down, up=up)

    def _require_connection(self) -> None:
        if self._service is None or self._controller_index is None:
            raise RuntimeError("Controller is not connected. Call connect() first.")
