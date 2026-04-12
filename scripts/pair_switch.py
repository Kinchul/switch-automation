from __future__ import annotations

import signal
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "nxbt"))

import nxbt  # type: ignore


def main() -> int:
    service = nxbt.Nxbt(debug=True)
    controller_index = None
    auto_join_sent = False

    def shutdown(*_args):
        if controller_index is not None:
            try:
                service.remove_controller(controller_index)
            except Exception:
                pass
        raise SystemExit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    adapters = service.get_available_adapters()
    if not adapters:
        print("No Bluetooth adapters available.")
        return 1

    print(f"Using adapter: {adapters[0]}")
    print('Create controller and wait for the Switch on "Change Grip/Order"...')

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
        print("Set IO capability to NoInputNoOutput.")
    except Exception as exc:
        print(f"Warning: could not set IO capability: {exc}")

    controller_index = service.create_controller(nxbt.PRO_CONTROLLER, adapters[0])

    last_state = None
    while True:
        state = dict(service.state[controller_index])
        state_name = state.get("state")

        if state_name != last_state:
            print(f"State: {state_name}")
            last_state = state_name

        if state_name == "crashed":
            print("Controller crashed:")
            print(state.get("errors"))
            return 1

        if state_name == "connected":
            if not auto_join_sent:
                print("Controller connected. Sending L+R to join the pairing screen...")
                for _ in range(3):
                    service.press_buttons(
                        controller_index,
                        [nxbt.Buttons.L, nxbt.Buttons.R],
                        down=0.1,
                        up=0.2,
                    )
                    time.sleep(0.3)
                auto_join_sent = True
                print("Join input sent. Keeping session alive...")
            while True:
                time.sleep(1)

        time.sleep(0.5)


if __name__ == "__main__":
    raise SystemExit(main())
