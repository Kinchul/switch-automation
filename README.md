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

## Near-term plan

1. Use `NXBT` for reliable button presses and macros.
2. Read frames from the CSI camera mounted in front of the Switch display.
3. Add simple state detection before building more game-specific logic.
