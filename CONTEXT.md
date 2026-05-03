# Switch Automation — Project Context

Quick brief for resuming work on this repo. Read once, then dive in.

## What this is

Nintendo Switch automation running on a Raspberry Pi 4. The Pi:

1. Captures the Switch's video output via a camera, detects in-game states with template matching against reference ROIs.
2. Emulates a Pro Controller over Bluetooth (BlueZ + nxbt) to drive sequences of inputs.
3. Loops gameplay (e.g. Pokémon shiny hunting) and notifies on outcome.

User: Valentin (Swiss, French-speaking, senior dev). Prefers terse, concrete answers and metric/SI units. Working OS: Windows 11 with the repo at `c:\Users\vkind\Documents\GitHub\switch-automation`. Pi target: Pi 4, Raspberry Pi OS Bookworm, Python 3.13.

## Repository layout

```
src/
  cli.py
  config.py
  automation/
    camera_loop.py        # CameraLoopRunner — main state machine, sequence execution, HUD state
    persistence.py        # PersistentLoopControl, PersistentLoopStatsStore (JSON files in debug/)
    sequence.py           # Sequence JSON loader, runtime, action specs
    runner.py
  control/
    backend.py            # ControllerBackend ABC, Button enum
    nxbt_backend.py       # NxbtBackend (uses tools/nxbt submodule)
  vision/
    capture.py            # CameraCapture orchestrator (auto-selects a source)
    detector.py           # StaticImageDetector, MatchResult, Roi
    sources/              # Camera backends (priority order: usb → csi → black)
      base.py             # CameraSource ABC
      usb.py              # UsbCameraSource — STUB for HDMI grabber, raises so we fall through
      csi.py              # CsiCameraSource — Picamera2 (Raspberry Pi CSI camera)
      black.py            # BlackFrameSource — final fallback (numpy zeros)
    hud/
      overlay.py          # OverlayState, OverlayBox, draw_overlay (+ wobble banner for "NO IMAGE")
    output/
      base.py             # FrameSink ABC (target_fps, target_size)
      pipeline.py         # OutputPipeline — single capture/render loop, fan-out to sinks
      mjpeg.py            # MjpegSink (HTTP MJPEG server on port 8080)
      local_display.py    # FramebufferSink (direct mmap → /dev/fb0, RGB565)
scripts/
  run_camera_loop.py      # Service entrypoint (action=run|pair|stop|restart|reset|status|...)
  camera_debug.py         # Snapshot/preview tool
  pi_sync_bg.ps1          # Windows file-watcher → scp to Pi (excludes .venv, .git, debug/, ...)
  setup_switch_bluetooth.sh / finalize_switch_bluetooth.sh
tools/
  nxbt/                   # git submodule — controller emulation
  bdaddr/                 # git submodule — BT MAC tooling
systemd/
  switch-camera-loop.service        # main service (runs as root)
  switch-camera-loop.env.example    # template for /etc/default/switch-camera-loop
  switch-bt-setup.service / switch-bt-finalize.service
sequences/                # JSON sequences (sulfura, electhor, ...)
images/                   # Reference ROI images keyed by sequence/state
debug/                    # Runtime data: loop_stats.json, loop_control.json, captured outcomes
System-changes.md         # Persistent BlueZ tweaks made on the Pi
```

## Architecture invariants

**Camera capture.** `CameraCapture.start()` tries sources in `preferred_sources=("usb", "csi", "black")`. Each source's `start()` may raise; we catch, log, and try the next. Whichever wins gets a daemon thread reading frames into a one-slot buffer (`_latest_frame`). `CameraCapture.source_name` exposes the active source's name — used by the HUD to draw the "NO IMAGE" banner when source is `black`.

**Output pipeline.** `OutputPipeline` runs at `max(sink.target_fps)`. Each tick:

1. `capture.get_frame()` once.
2. `_render_per_size`: for each distinct `sink.target_size`, letterbox the capture to that size, then `draw_overlay` once. Sinks sharing a size share the rendered ndarray.
3. Fan out to each sink's `consume(frame)`. Sinks self-pace if they want lower fps (e.g. `MjpegSink` skips encoding when `1/target_fps` hasn't elapsed).

Default sinks:

- `MjpegSink(port=8080, target_fps=5, target_size=None)` — full-resolution stream.
- `FramebufferSink(fbdev=/dev/fb0, target_fps=30, target_size=(480,320))` — 16bpp RGB565 mmap. Auto-detects panel geometry from `/sys/class/graphics/fbN/`. Disables itself cleanly if the device is missing or wrong bpp.

`MjpegSink.start()` is **idempotent** — the runner pre-binds the port (for EADDRINUSE-takeover logic), then the pipeline calls `start()` again. The early-return guard prevents a double-bind.

**HUD overlay.** Font size: `max(14, min(36, short_side // 30))`. Banner (e.g. "NO IMAGE"): `max(28, min(120, short_side // 8))`, wobbles via sin/cos of `time.monotonic()`. `OverlayState.banner` is set by `CameraLoopRunner.preview_overlay_state()` based on `capture.source_name`.

**Sequence runner.** `CameraLoopRunner.run_service()` is the long-running loop. It reads commands from `debug/camera/loop_control.json` (set by other invocations of the script), connects/disconnects the controller as needed, runs the selected sequence, and persists stats per sequence to `debug/camera/loop_stats.json`.

## Service runtime on the Pi

- Unit: `systemd/switch-camera-loop.service` → installed at `/etc/systemd/system/switch-camera-loop.service`.
- Runs as **root** (needed for raw HID L2CAP and `/dev/fb0` access).
- Reads optional env from `/etc/default/switch-camera-loop` (template: `systemd/switch-camera-loop.env.example`).
- Entrypoint: `./.venv/bin/python scripts/run_camera_loop.py $SWITCH_CAMERA_LOOP_ARGS`.
- Logs: `journalctl -u switch-camera-loop.service -f`.

## Pi-specific gotchas (already resolved, don't re-debug)

1. **Python venv.** Must be created with `--system-site-packages` so apt-installed `python3-picamera2` and `python3-dbus` are visible. `picamera2` _cannot_ come from pip — it binds to system libcamera.
2. **nxbt deps.** Install nxbt with `--no-deps`; let `dbus-python` come from apt (`python3-dbus`). The pinned `dbus-python==1.2.16` in `tools/nxbt/setup.py` won't build on Python 3.13.
3. **SDL on Bookworm.** Doesn't ship the `fbcon` driver and `kmsdrm` triggers `EGL_BAD_ACCESS` on this hardware. We bypass SDL entirely — `FramebufferSink` writes RGB565 directly via mmap.
4. **JT3.5TR display overlay.** Use `dtoverlay=piscreen2r,rotate=270` in `/boot/firmware/config.txt`. The `tft35a` overlay isn't shipped on current Raspberry Pi OS; `piscreen` panel works but conflicts with a separate `ads7846` line. `piscreen2r` handles both display + touch.
5. **Framebuffer device.** With no HDMI attached, the SPI panel takes `/dev/fb0` (not `/dev/fb1`). Default `--local-display-fbdev` is `/dev/fb0`.
6. **Touch.** `/dev/input/event4`, ADS7846. Not yet wired — `evdev` is already installed (nxbt dep). User wants this later.
7. **BlueZ.** Several persistent overrides in `/etc/systemd/system/bluetooth.service.d/override.conf` and `/etc/bluetooth/main.conf`. See `System-changes.md`.

## Dependency setup (current)

`pyproject.toml`:

- `dependencies = ["numpy>=1.24", "Pillow>=10.0"]`
- `[fast-jpeg]` → simplejpeg
- `[controller]` → empty marker (nxbt installed manually because PEP 508 forbids relative `file://`)
- `[pi]` → fast-jpeg + controller

Install on the Pi:

```bash
sudo apt install -y python3-picamera2 python3-dbus libdbus-1-dev libdbus-glib-1-dev pkg-config
git submodule update --init --recursive
python3 -m venv .venv --system-site-packages
./.venv/bin/pip install -U pip
./.venv/bin/pip install -e ".[pi]"
./.venv/bin/pip install Flask Flask-SocketIO eventlet blessed pynput psutil cryptography Jinja2 itsdangerous Werkzeug
./.venv/bin/pip install -e ./tools/nxbt --no-deps
./.venv/bin/python -c "import numpy, PIL, simplejpeg, nxbt, picamera2, dbus; print('ok')"
```

`pygame` is no longer a dependency (we used to render through SDL; now it's direct framebuffer).

## Known stubs / TODOs

- **`UsbCameraSource`** ([src/vision/sources/usb.py](src/vision/sources/usb.py)) — empty stub. User has an HDMI grabber arriving "future". When wired up, it should `start()` cleanly so it preempts CSI.
- **Touch input.** `/dev/input/event4` is open but unused. Wire it up to send controller commands when user asks.
- **Stream JPEG quality** is hardcoded to 80 in `MjpegSink`. No CLI flag yet — add one if user wants higher fps without more CPU.

## How to communicate

- Terse. Direct answers. No trailing summaries.
- Reference files with markdown links: `[overlay.py:43](src/vision/hud/overlay.py#L43)`.
- Don't add comments to code unless the _why_ is non-obvious. Don't write planning docs unless asked.
- For exploratory questions, give a 2-3 sentence recommendation with the main tradeoff. Don't implement until confirmed.
- For risky actions (force push, rm -rf, drops), confirm first.
- Use SI/metric units by default (org instruction).
