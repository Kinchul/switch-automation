from __future__ import annotations

import sys
import time
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Iterable

from .backend import Button, ControllerBackend, ControllerConnectCancelled


def _import_nxbt():
    try:
        import nxbt  # type: ignore
        return nxbt
    except ModuleNotFoundError:
        repo_root = Path(__file__).resolve().parents[2]
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
    def __init__(
        self,
        controller_type: str = "PRO_CONTROLLER",
        reconnect: bool = True,
        adapter_path: str | None = None,
    ):
        self._nxbt_mod = _import_nxbt()
        self._controller_type_name = controller_type
        self._reconnect = reconnect
        self._adapter_path = adapter_path
        self._service = None
        self._controller_index = None

    def connect(
        self,
        *,
        cancel_cb: Callable[[], bool] | None = None,
        status_cb: Callable[[str], None] | None = None,
    ) -> None:
        if self._service is None:
            self._service = self._nxbt_mod.Nxbt()

        if self._controller_index is not None:
            state_name = self._controller_state_name()
            if state_name == "connected":
                return
            self._cleanup_controller()

        controller_type = getattr(self._nxbt_mod, self._controller_type_name)
        adapter_path = self._pick_adapter_path(status_cb)
        kwargs = {}
        reconnect_addresses = None
        if self._reconnect:
            reconnect_addresses = self._service.get_switch_addresses() or None
            if reconnect_addresses:
                kwargs["reconnect_address"] = reconnect_addresses

        if status_cb is not None:
            if reconnect_addresses:
                status_cb("Reconnect mode enabled. Reusing the stored Switch pairing.")
            else:
                status_cb('Pairing mode enabled. Keep the Switch on "Change Grip/Order".')

        try:
            self._nxbt_mod.clean_sdp_records()
            if status_cb is not None:
                status_cb("Cleaned BlueZ SDP records.")
        except Exception as exc:
            if status_cb is not None:
                status_cb(f"Warning: could not clean SDP records: {exc}")

        try:
            subprocess.run(
                ["btmgmt", "--index", "0", "io-cap", "0x03"],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if status_cb is not None:
                status_cb("Set IO capability to 0x03.")
        except Exception as exc:
            if status_cb is not None:
                status_cb(f"Warning: could not set IO capability: {exc}")

        if status_cb is not None:
            status_cb(
                "Creating controller: "
                f"type={self._controller_type_name} adapter={adapter_path} "
                f"reconnect={bool(reconnect_addresses)}"
            )
        self._controller_index = self._service.create_controller(
            controller_type,
            adapter_path=adapter_path,
            **kwargs,
        )
        if status_cb is not None:
            status_cb(f"Controller index: {self._controller_index}")
        last_state = None
        try:
            while True:
                if cancel_cb is not None and cancel_cb():
                    raise ControllerConnectCancelled("Controller connect cancelled")

                state_name = self._controller_state_name()
                if state_name != last_state:
                    last_state = state_name
                    if status_cb is not None:
                        status_cb(state_name)

                if state_name == "connected":
                    if not reconnect_addresses:
                        self._join_pairing_menu(status_cb)
                    return

                if state_name == "crashed":
                    errors = self._controller_errors()
                    if not errors:
                        errors = (
                            "NXBT controller process crashed without a traceback. "
                            "A transient BlueZ/DBus timeout during pairing is a likely cause."
                        )
                    raise OSError("The watched controller has crashed", errors)

                time.sleep(0.25)
        except ControllerConnectCancelled:
            self._cleanup_controller()
            raise

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

    def close(self) -> None:
        self._cleanup_controller()
        if self._service is None:
            return
        service = self._service
        self._service = None
        try:
            service.close()
        except Exception:
            pass

    def set_reconnect(self, reconnect: bool) -> None:
        self._reconnect = reconnect

    def _require_connection(self) -> None:
        if self._service is None or self._controller_index is None:
            raise RuntimeError("Controller is not connected. Call connect() first.")

    def _controller_state_name(self) -> str | None:
        if self._service is None or self._controller_index is None:
            return None
        try:
            return self._service.state[self._controller_index].get("state")
        except Exception:
            return None

    def _controller_errors(self):
        if self._service is None or self._controller_index is None:
            return None
        try:
            return self._service.state[self._controller_index].get("errors")
        except Exception:
            return None

    def _join_pairing_menu(self, status_cb: Callable[[str], None] | None) -> None:
        if self._service is None or self._controller_index is None:
            return
        try:
            if status_cb is not None:
                status_cb("Sending repeated L+R to join the pairing screen...")
            # The Switch can show the "controller synced" toast before the
            # newly paired controller is actually admitted to the player list.
            # A stronger post-connect join burst is more reliable here.
            for _ in range(6):
                self._service.press_buttons(
                    self._controller_index,
                    [self._nxbt_mod.Buttons.L, self._nxbt_mod.Buttons.R],
                    down=0.12,
                    up=0.35,
                )
                time.sleep(0.35)
            if status_cb is not None:
                status_cb("Join burst sent.")
        except Exception as exc:
            if status_cb is not None:
                status_cb(f"Warning: could not send L+R join pulse: {exc}")

    def _pick_adapter_path(self, status_cb: Callable[[str], None] | None) -> str:
        if self._service is None:
            raise RuntimeError("NXBT service is not initialized.")

        adapters = list(self._service.get_available_adapters())
        if status_cb is not None:
            if adapters:
                status_cb(f"Available adapters: {', '.join(adapters)}")
            else:
                status_cb("No Bluetooth adapters reported by BlueZ.")

        adapter_path = self._adapter_path or (adapters[0] if adapters else None)
        if adapter_path is None:
            raise RuntimeError("No Bluetooth adapters available to create a controller.")
        return adapter_path

    def _cleanup_controller(self) -> None:
        if self._service is None or self._controller_index is None:
            self._controller_index = None
            return
        controller_index = self._controller_index
        self._controller_index = None
        try:
            self._service.remove_controller(controller_index)
        except Exception:
            pass
