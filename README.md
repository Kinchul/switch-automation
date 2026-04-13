# Switch-Automation

Goal: Automate a Nintendo Switch from a Raspberry Pi by:

- sending controller input through a modern Bluetooth backend
- reading game state from camera frames
- running hunt logic in one automation loop

## Project layout

- `src/control/`
  Bluetooth controller backends. The default path is an `NXBT` wrapper.
- `src/automation/`
  High-level hunt loops, macros, and decision logic.
- `src/vision/`
  CSI camera capture plus future state-detection helpers.
- `scripts/camera_debug.py`
  Camera helper for snapshots, sample collection, and a VLC-viewable TCP preview feed.
- `scripts/roi_picker.py`
  Small GUI helper to draw an ROI on a saved snapshot and print `{x, y, width, height}`.
- `scripts/run_camera_loop.py`
  First camera-guided automation loop using saved reference images plus a black-screen detector.
- `scripts/keyboard_control.py`
  Working terminal controller built on `NXBT` and `curses`, with arrow-key support and clean shutdown handling.
- `scripts/pair_switch.py`
  Pairing helper for first-time setup. It automatically sends `L` + `R` after the controller appears on the Switch pairing screen.
- `tools/nxbt/`
  Vendored local copy of `NXBT`, kept in-tree for now with the local Bluetooth compatibility fixes used by this project.
- `tools/bdaddr/`
  Vendored local `bdaddr` utility source used to inspect or restore the Pi Bluetooth MAC address when needed. This stays as a normal directory in the repo, not a submodule.
- `System-changes.md`
  Notes for the machine-level Bluetooth and environment changes made on the Raspberry Pi outside normal project code.

## Quick start

Create the local package in editable mode:

```bash
source .venv/bin/activate
python -m pip install -e .
```

If you want the control backend available from the local checkout:

```bash
python -m pip install -e ./tools/nxbt
```

Run the CLI help:

```bash
switch-automation --help
```

For an immediate manual control test from the terminal:

```bash
sudo ./.venv/bin/python scripts/keyboard_control.py
```

If you want to force first-time pairing through `Change Grip/Order`:

```bash
sudo ./.venv/bin/python scripts/keyboard_control.py --pairing-menu
```

To run the dedicated pairing helper:

```bash
sudo ./.venv/bin/python scripts/pair_switch.py
```

To save a camera snapshot from the CSI camera:

```bash
./.venv/bin/python scripts/camera_debug.py snapshot --output debug/camera/snapshot.jpg
```

To collect a small batch of sample frames:

```bash
./.venv/bin/python scripts/camera_debug.py sample --output-dir debug/camera/samples --count 20 --interval 1.0
```

To start a VLC-viewable TCP preview feed:

```bash
./.venv/bin/python scripts/camera_debug.py stream --width 1920 --height 1080 --fps 20
```

Then open `tcp/h264://PI_IP:8888` in VLC from another machine.

To draw an ROI on a saved snapshot:

```bash
./.venv/bin/python scripts/roi_picker.py debug/camera/snapshot.jpg
```

Drag a box over the important part of the image. The script prints ROI coordinates as JSON.

To run the first camera-guided loop:

```bash
sudo ./.venv/bin/python scripts/run_camera_loop.py --width 1920 --height 1080 --fps 20
```

To run it as a boot-time `systemd` service instead:

```bash
sudo install -m 644 systemd/switch-camera-loop.service /etc/systemd/system/switch-camera-loop.service
sudo install -m 644 systemd/switch-camera-loop.env.example /etc/default/switch-camera-loop
sudo systemctl daemon-reload
sudo systemctl enable --now switch-camera-loop.service
```

Adjust `/etc/default/switch-camera-loop` for your repo path, mail settings, or runner flags.

Useful service commands:

```bash
sudo systemctl status switch-camera-loop.service
sudo journalctl -u switch-camera-loop.service -f
sudo systemctl restart switch-camera-loop.service
sudo systemctl stop switch-camera-loop.service
```

This runner starts in stopped mode and immediately starts the camera preview feed. The Bluetooth controller is only connected when a loop is triggered.

Preview feed while the service is running:

```text
http://PI_IP:8080/stream.mjpg
```

You can open that URL in VLC or a browser. `restart` and `stop` only affect the control logic; the feed keeps running until the foreground service exits.

Trigger a new loop:

```bash
sudo ./.venv/bin/python scripts/run_camera_loop.py --action restart
```

If the controller is not connected yet, use `--action pair` while the Switch is on "Change Grip/Order". Once paired, future restarts reuse that active connection.

```bash
sudo ./.venv/bin/python scripts/run_camera_loop.py --action run --width 1920 --height 1080 --fps 20
```

After pairing, return the game to the state where `A+B+X+Y` should trigger the next loop, then use `--action restart`.

Stop the service and leave it idle:

```bash
sudo ./.venv/bin/python scripts/run_camera_loop.py --action stop
```

Inspect the current command/timers/counter:

```bash
./.venv/bin/python scripts/run_camera_loop.py --action status
```

## Local PC workflow

From a Windows PC, you can sync the repo to the Pi without committing every tweak.

Set the defaults once inside [`scripts/pi_sync.ps1`](</c:/Users/vkind/Documents/GitHub/switch-automation/scripts/pi_sync.ps1>) by editing:

```powershell
$DefaultHost = "pi@YOUR_PI_HOSTNAME_OR_IP"
$DefaultRemoteDir = "switch-automation"
```

Sync the current checkout to the Pi with `scp`:

```powershell
.\scripts\pi_sync.ps1
```

You can still override either value for a one-off run:

```powershell
.\scripts\pi_sync.ps1 -PiHost pi@192.168.1.50 -RemoteDir switch-automation
```

Use `-DryRun` if you want to preview the file list without uploading:

```powershell
.\scripts\pi_sync.ps1 -DryRun
```

The sync script uses Windows OpenSSH `scp`, so no WSL or extra install is required. It uploads the filtered repo contents, but it does not delete files that already exist on the Pi and were removed locally.

The runner currently:
- confirms the start scene from the launch or press-start screens
- presses `A` until the saved `select save` reference appears
- presses `A` once to skip that screen
- presses `B` until the saved `previously` reference disappears
- waits for the saved `ready` reference after `previously` disappears
- presses `A` until a black-screen transition is detected
- waits for either the saved `target failed` image or a non-black success candidate
- if the target failed image appears, sends `A+B+X+Y` together to trigger the next loop

If no failure image appears after the black transition, the script stops and saves the outcome frame under `debug/camera/outcomes/`.

The runner also persists loop stats in `debug/camera/loop_stats.json`:
- total elapsed time across loops
- current active loop elapsed time
- loop counter

Those stats survive software restarts so the timers and count resume cleanly. The control command file lives at `debug/camera/loop_control.json`.

## Near-term plan

1. Use `NXBT` for reliable button presses and macros.
2. Read frames from the CSI camera mounted in front of the Switch display.
3. Tune the static-image thresholds and replace the current success-candidate fallback with a real success detector.
