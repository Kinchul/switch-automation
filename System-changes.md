# System Changes

This file documents persistent system-level changes made while getting `switch-automation` working on this Raspberry Pi, why they matter, and how to revert them if they interfere with normal Raspberry Pi usage.

Repo-local paths in this document are written relative to the repository root. True system paths such as `/etc/...` and `/var/...` remain absolute because they refer to machine-level configuration outside this repo.

## 1. BlueZ systemd override: controller-only `bluetoothd`

Changed file:
- `/etc/systemd/system/bluetooth.service.d/override.conf`

Current contents:

```ini
[Service]
ExecStart=
ExecStart=/usr/libexec/bluetooth/bluetoothd --compat --noplugin=*
```

Why this was changed:
- Early controller-emulation attempts hit `Address already in use` on the HID L2CAP ports.
- This is the classic BlueZ `input` plugin conflict with user-space HID emulation.
- Later, `NXBT`-based troubleshooting showed the Pi was still advertising a large set of unrelated BlueZ services.
- The stricter `--compat --noplugin=*` mode is closer to the controller-only setup used by newer Switch emulation projects.

Possible side effects:
- This is more invasive than only disabling `input`.
- Bluetooth keyboard/mouse, audio profiles, and other normal BlueZ plugin-driven features may stop working while this override is active.

How to revert:

```bash
sudo rm /etc/systemd/system/bluetooth.service.d/override.conf
sudo systemctl daemon-reload
sudo systemctl restart bluetooth.service
```

## 2. BlueZ policy changes in `/etc/bluetooth/main.conf`

Changed file:
- `/etc/bluetooth/main.conf`

Backup created:
- `/etc/bluetooth/main.conf.bak-20260412-144604`

Changes applied:
- `AlwaysPairable = true`
- `JustWorksRepairing = always`

Why this was changed:
- The Switch reached Secure Simple Pairing, but `bluetoothd` kept issuing `User Confirmation Negative Reply`.
- The Pi also had a stored Switch pairing record, so BlueZ's stricter Just Works re-pairing policy was likely interfering.

Possible side effects:
- Makes BlueZ more permissive about pairing and Just Works re-pairing.
- This can be less desirable for a normal general-purpose Bluetooth setup.

How to revert using the backup:

```bash
sudo cp /etc/bluetooth/main.conf.bak-20260412-144604 /etc/bluetooth/main.conf
sudo systemctl restart bluetooth.service
```

How to revert manually:
- Change `AlwaysPairable = true` back to `#AlwaysPairable = false`
- Change `JustWorksRepairing = always` back to `#JustWorksRepairing = never`
- Restart Bluetooth:

```bash
sudo systemctl restart bluetooth.service
```

## 3. Pi-side stored Switch pairing record was backed up and removed once

Original Pi-side Switch record path:
- `/var/lib/bluetooth/DC:A6:32:AE:06:C9/7C:BB:8A:AF:60:A0`

Why this was changed:
- The Pi already had a stored link key for the Switch.
- To test a truly fresh pair from the Pi side, that record was backed up and the active record directory was removed before restarting Bluetooth.

Current state:
- A new pairing record has since been recreated by BlueZ after successful pairing.

Possible side effects:
- Removing the record forced re-pairing from the Pi side.

How to inspect the current and backup records:

```bash
sudo ls -la /var/lib/bluetooth/DC:A6:32:AE:06:C9
```

How to revert to the older backup record:
- Stop any running `joycontrol` or Bluetooth tests first.
- Move the current `7C:BB:8A:AF:60:A0` directory aside.
- Rename the backup directory back to `7C:BB:8A:AF:60:A0`.
- Restart Bluetooth.

Generic example:

```bash
sudo mv /var/lib/bluetooth/DC:A6:32:AE:06:C9/7C:BB:8A:AF:60:A0 /var/lib/bluetooth/DC:A6:32:AE:06:C9/7C:BB:8A:AF:60:A0.new
sudo mv /var/lib/bluetooth/DC:A6:32:AE:06:C9/7C:BB:8A:AF:60:A0.bak-<timestamp> /var/lib/bluetooth/DC:A6:32:AE:06:C9/7C:BB:8A:AF:60:A0
sudo systemctl restart bluetooth.service
```

## 4. Workspace-local Python environment settings

Changed files in this workspace only:
- `.vscode/settings.json`
- `.venv/pyvenv.cfg`

Why this was changed:
- VS Code and the shell were using system Python/pip instead of the workspace venv.
- The local Bluetooth control stack also needs access to the system `dbus` package from the venv.

Changes:
- VS Code interpreter pinned to `.venv/bin/python`
- Terminal environment activation enabled
- `.venv/pyvenv.cfg` changed to `include-system-site-packages = true`

Possible side effects:
- The workspace venv can now import system Python packages.
- This is usually fine for this project, but it makes the venv less isolated.

How to revert:

```bash
rm .vscode/settings.json
```

Edit `.venv/pyvenv.cfg` and change:

```ini
include-system-site-packages = true
```

back to:

```ini
include-system-site-packages = false
```

## 5. Installed `libbluetooth-dev` to build a Bluetooth utility

Installed package:
- `libbluetooth-dev`

Why this was changed:
- The Pi 4's built-in Cypress Bluetooth controller may need its public BD_ADDR changed to a Nintendo-range address for newer Switch firmware compatibility.
- Building the `bdaddr` utility required BlueZ development headers.

Possible side effects:
- Minimal during normal use; this is a development package and does not change runtime Bluetooth behavior by itself.

How to revert:

```bash
sudo apt-get remove -y libbluetooth-dev
```

## 6. Bluetooth public MAC address (BD_ADDR) was changed

Original address:
- `DC:A6:32:AE:06:C9`

Current address:
- `94:58:CB:AE:06:C9`

Why this was changed:
- Newer Switch firmware can be picky about the adapter identity used for controller pairing.
- Community reports for newer Switch firmware suggest that some adapters, especially Raspberry Pi internal Bluetooth, only work reliably when the controller's public Bluetooth address uses a Nintendo OUI range.

Possible side effects:
- Other devices will see the Pi under a different Bluetooth hardware address.
- Existing Bluetooth pairings with other devices may stop matching and may need to be re-paired.
- This is a more invasive change than the others because it alters the adapter's public identity.

How to verify the current address:

```bash
sudo ./tools/bdaddr/bdaddr -i hci0
```

How to revert to the original address:

```bash
sudo ./tools/bdaddr/bdaddr -i hci0 -r DC:A6:32:AE:06:C9
sudo hciconfig hci0 reset
```

## Notes

- This file is only for persistent machine-level changes, not normal code changes inside this repo.
- Temporary runtime actions like restarting `bluetooth.service`, running `rfkill unblock bluetooth`, or starting test processes are not persistent config changes by themselves.
