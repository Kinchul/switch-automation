# shiny-hunter

`shiny-hunter` is the new home for this project.

The repo is now centered on one goal: automate a Nintendo Switch from a Raspberry Pi by:

- sending controller input through a modern Bluetooth backend
- reading game state from captured video frames
- running hunt logic in one automation loop

## Project layout

- `src/shiny_hunter/control/`
  Bluetooth controller backends. The default path is an `NXBT` wrapper.
- `src/shiny_hunter/automation/`
  High-level hunt loops, macros, and decision logic.
- `src/shiny_hunter/vision/`
  Frame capture and future OCR/template-matching helpers.
- `tools/nxbt/`
  Vendored local copy of `NXBT`, including the small compatibility fixes used by this project.
- `tools/bdaddr/`
  Local utility source used to inspect or restore the Pi Bluetooth MAC address when needed.

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
shiny-hunter --help
```

For an immediate manual control test from the terminal:

```bash
sudo ./.venv/bin/python scripts/keyboard_control.py
```

If you want to force first-time pairing through `Change Grip/Order`:

```bash
sudo ./.venv/bin/python scripts/keyboard_control.py --pairing-menu
```

## Near-term plan

1. Use `NXBT` for reliable button presses and macros.
2. Read frames from an HDMI capture device attached to the Raspberry Pi.
3. Add simple state detection before building more game-specific logic.
