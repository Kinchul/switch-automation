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

- `threshold`: number. Optional default match threshold. If omitted here, every state must define its own.
- `search_margin`: integer. Pixel radius around the ROI within which the detector slides the template. Higher values tolerate more positional drift but are slower. Default: `24`.
- `stride`: integer. Pixel pooling factor applied to both the reference crop and the search area before comparison. `4` means each `4x4` block is averaged into one comparison sample. Higher values are faster but less precise. Default: `4`.
- `search_step`: integer. Step size in original pixels between candidate positions inside the search area. Must be ≥ `stride`; the detector converts it to downsampled steps internally. Larger values skip more positions (faster, coarser). Default: `2`.
- `hold_ms`: integer. How long in milliseconds a match must remain continuously visible before the state is accepted. `0` means a single matching frame is enough. Default: `0`.
- `score_window`: integer. Number of consecutive frames whose scores are averaged (trimmed mean, dropping the top 25 %) before comparing to the threshold. `1` means no averaging. Helps reject single-frame noise. Default: `1`.
- `luma_weight`: number. Weight given to brightness differences in the detector score. Default: `0.7`.
- `chroma_weight`: number. Weight given to color differences in the detector score. Default: `0.3`.

`defaults.action` fields:

- `frequency_hz`: number. `0` means the button is pressed once when the state is entered. `> 0` means repeated presses at that frequency for the duration of the state.
- `down_ms`: integer. How long in milliseconds the button is held down per press. Default: `100`.
- `up_ms`: integer. How long in milliseconds to wait after releasing the button before the next press. Default: `100`.

State fields:

- `scene`: object or empty. If omitted, the state is procedural and transitions to its first `next_states` entry immediately on entry.
- `next_states`: string or string array. Ordered list of candidate next states. The runner detects whichever one matches first, except in `decision_mode: "best_score"`.
- `action`: object or empty. Button action to perform while waiting in this state. Merges with `defaults.action`.
- `timeout_ms`: integer. How long in milliseconds to wait for a match before triggering recovery or `timeout_next_state`. `0` means no timeout. Overrides `defaults.timeout_ms`.
- `timeout_next_state`: string. If set, a timeout transitions silently to this state instead of triggering global recovery. Requires `timeout_ms > 0`.
- `decision_mode`: string. `""` (default): transition to the first matching next state. `"best_score"`: wait until the timeout, then pick the next state whose detector has the lowest score, provided the gap to second place exceeds `decision_margin`. `"loop_baseline_step"`: observe the single detectable `next_states` candidate for the whole timeout window, compute one loop-level score, compare it against the learned failed baseline from recent failed loops, and choose between that failed state and `timeout_next_state`. This mode uses the candidate state's detector settings from its `scene` block, especially `threshold`, `score_window`, `search_margin`, `stride`, and `search_step`.
- `decision_margin`: number. Minimum score gap between first and second place required by `decision_mode: "best_score"`. Default: `0.0`.
- `decision_history_window`: integer. Only used by `decision_mode: "loop_baseline_step"`. Number of recent failed loop scores to keep for the learned failed baseline. Default: `9`.
- `decision_trend_window`: integer. Only used by `decision_mode: "loop_baseline_step"`. Number of recent failed loop-to-loop deltas used to estimate slow drift. Default: `5`.
- `decision_ok_step`: number. Only used by `decision_mode: "loop_baseline_step"`. Minimum absolute jump above the predicted failed score required to classify the loop as `timeout_next_state`. During bootstrap, when no failed baseline has been learned yet, this same value is added to the static threshold before allowing `timeout_next_state`. Default: `0.0`.
- `reset_loop`: boolean. If `true`, reaching this state increments the loop counter, saves a ROI snapshot, and restarts the sequence from the initial state. Used on failure states. Default: `false`.
- `notification`: string. `""` (default): no notification. `"mail"`: sends an email notification when this state is reached. Used on terminal success states.

Scene fields:

- `image_path`: string. Path to the reference image, relative to the JSON file.
  Full-screen screenshots are preferred. Cropped template images also work; the detector anchors the whole template at the configured `roi.x` and `roi.y`.
- `roi`: object with `x`, `y`, `width`, `height`. Defines the screen region where the template is expected. The detector searches within `search_margin` pixels of this position. Coordinates are in original frame pixels, before any downsampling.
- `threshold`: number. Maximum allowed MAE score (mean absolute error per pixel, normalised to `[0, 1]`) for a frame to be considered a match. Lower = stricter. A score of `0.0` means the template matches the frame pixel-perfectly; `1.0` means maximum difference. Typical useful values are between `0.03` and `0.10`. Optional if supplied by `defaults.scene`.
- `search_margin`: integer. Pixel radius around the ROI within which the detector slides the template looking for the best match position. `24` means the template is tested at every candidate position within ±24 px of the configured ROI. Increase this if the element can shift around on screen; decrease it to reduce false positives on nearby similar graphics.
- `stride`: integer. Pixel downsampling factor applied to both the reference crop and the live frame before comparison. `4` means only every 4th pixel (in each axis) is compared, making the detector 16× faster than `stride: 1` at the cost of precision. Use lower values (`2`) for small or low-contrast ROIs where fine detail matters; use higher values (`6`, `8`) for large ROIs where speed matters more.
- `search_step`: integer. Step size in original pixels between candidate positions inside the search area. `2` means the template is tried every 2 px across the search area (before downsampling). Must be ≥ `stride`. Larger values skip positions and are faster but may miss the best alignment by a few pixels, slightly raising the score.
- `hold_ms`: integer. How long in milliseconds the match must remain continuously visible before the state is accepted. `0` accepts on the first matching frame. Useful to avoid false positives caused by transition animations or brief visual glitches — set it to the minimum time the target screen is guaranteed to be stable.
- `score_window`: integer. Number of consecutive frames whose scores are averaged (trimmed mean, dropping the top 25 %) before comparing to the threshold. `1` means no averaging. Use `3`–`5` to filter out single-frame noise from compression artefacts or lighting spikes without adding much latency.
  In `decision_mode: "loop_baseline_step"`, this same field stabilizes the per-loop candidate score before the loop-level baseline comparison.

- `luma_weight`: number. Weight given to brightness differences in the pooled `YCbCr` detector score. Increase this for static UI/text scenes.
- `chroma_weight`: number. Weight given to color differences in the pooled `YCbCr` detector score. Increase this for color-change checks such as shiny detection.

Action fields:

- `buttons`: string array. One or more buttons to press simultaneously.
- `frequency_hz`: number. `0` means one press on state entry. `> 0` means repeated presses at that frequency for the duration of the state.
- `down_ms`: integer. How long in milliseconds the button is held down per press.
- `up_ms`: integer. How long in milliseconds to wait after releasing before the next press.

Accepted button values:

- `A`, `B`, `X`, `Y`, `L`, `R`, `ZL`, `ZR`, `PLUS`, `MINUS`, `HOME`, `CAPTURE`, `DPAD_UP`, `DPAD_DOWN`, `DPAD_LEFT`, `DPAD_RIGHT`

## Notes

- The first state in `states` is the normal loop start.
- `recovery.states` is only used after a timeout.
- A state with no `scene` is a procedural step. This is used in the default sequence to split "press A once, then mash B" into two states without extra schema fields.
- Terminal states are states with no `next_states`. The runner stops there.
- Per-sequence stats are stored in `debug/camera/loop_stats.json`.
- The selected sequence is stored in `debug/camera/loop_control.json`.
