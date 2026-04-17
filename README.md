# Switch-Automation

Automate a Nintendo Switch from a Raspberry Pi by separating:

- `control`: Bluetooth button input
- `vision`: camera capture plus static-image matching
- `sequence`: JSON-defined automation flows

The active sequence files live in `sequences/`. The default one is [sulfura.json](</c:/Users/vkind/Documents/GitHub/switch-automation/sequences/sulfura.json>).

## Project Layout

- `src/control/`
  Controller backends. The default backend is `NXBT`.
- `src/vision/`
  Camera capture, static-image detector, and preview streaming.
- `src/automation/sequence.py`
  Sequence JSON loading plus defaults merging.
- `src/automation/persistence.py`
  Persisted selected sequence plus per-sequence stats.
- `src/automation/camera_loop.py`
  Generic sequence runner.
- `sequences/`
  Automation JSON files.
- `scripts/run_camera_loop.py`
  Foreground service and control actions.

## Service Commands

Start the service:

```bash
./.venv/bin/python scripts/run_camera_loop.py --action run
```

Pair the controller:

```bash
./.venv/bin/python scripts/run_camera_loop.py --action pair
```

Trigger the selected sequence:

```bash
./.venv/bin/python scripts/run_camera_loop.py --action restart
```

Force a game reset and then start the selected sequence without the startup recovery scan:

```bash
./.venv/bin/python scripts/run_camera_loop.py --action reset
```

Stop the service:

```bash
./.venv/bin/python scripts/run_camera_loop.py --action stop
```

Show status:

```bash
./.venv/bin/python scripts/run_camera_loop.py --action status
```

List sequences:

```bash
./.venv/bin/python scripts/run_camera_loop.py --action list-sequences
```

Select a sequence:

```bash
./.venv/bin/python scripts/run_camera_loop.py --action select-sequence --sequence sulfura
```

Reset all per-sequence stats:

```bash
./.venv/bin/python scripts/run_camera_loop.py --action reset-stats
```

## Sequence Format

Sequence id:

- The filename is the sequence id.
- `sulfura.json` becomes `sulfura`.

Top-level fields:

- `recovery`: object. States the runner can detect during timeout recovery.
- `defaults`: object. Shared values applied to every state unless the state overrides them.
- `states`: object. Ordered state map. The first state is the normal start state.

`recovery` fields:

- `states`: string array. Ordered list of states scanned when a timeout triggers recovery.
- `timeout_ms`: integer. How long timeout recovery scanning lasts. `0` means one immediate scan.

`defaults` fields:

- `timeout_ms`: integer. Default timeout for each state.
- `scene`: object. Default scene tuning for all states with a scene.
- `action`: object. Default button timing for all actions.

`defaults.scene` fields:

- `threshold`: number. Optional default match threshold.
- `search_margin`: integer.
- `stride`: integer.
- `search_step`: integer.
- `hold_ms`: integer. How long a match must remain visible before the state is accepted.

`defaults.action` fields:

- `frequency_hz`: number. `0` means one press, `> 0` means repeated presses.
- `down_ms`: integer.
- `up_ms`: integer.

State fields:

- `scene`: object or empty. If omitted, the state is procedural and runs immediately when entered.
- `next_states`: string or string array. Ordered next states.
- `action`: object or empty. Button action to perform in that state.
- `timeout_ms`: integer. Overrides `defaults.timeout_ms`.
- `decision_mode`: string. Optional. Accepted values: `""` and `"best_score"`.
- `decision_margin`: number. Optional minimum score gap required by `decision_mode: "best_score"`.
- `notification`: string. Accepted values: `""` and `"mail"`.

Scene fields:

- `image_path`: string. Relative to the JSON file.
  Full-screen screenshots are preferred. Cropped template images also work; the detector anchors the whole template at the configured `roi.x` and `roi.y`.
- `roi`: object with `x`, `y`, `width`, `height`.
- `threshold`: number. Optional if supplied by `defaults.scene`.
- `search_margin`: integer.
- `stride`: integer.
- `search_step`: integer.
- `hold_ms`: integer.

Action fields:

- `buttons`: string array.
- `frequency_hz`: number.
- `down_ms`: integer.
- `up_ms`: integer.

Accepted button values:

- `A`, `B`, `X`, `Y`, `L`, `R`, `ZL`, `ZR`, `PLUS`, `MINUS`, `HOME`, `CAPTURE`, `DPAD_UP`, `DPAD_DOWN`, `DPAD_LEFT`, `DPAD_RIGHT`

## Notes

- The first state in `states` is the normal loop start.
- `recovery.states` is only used after a timeout.
- A state with no `scene` is a procedural step. This is used in the default sequence to split "press A once, then mash B" into two states without extra schema fields.
- Terminal states are states with no `next_states`. The runner stops there.
- Per-sequence stats are stored in `debug/camera/loop_stats.json`.
- The selected sequence is stored in `debug/camera/loop_control.json`.
