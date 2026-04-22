from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from control import Button


class SequenceConfigError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class Roi:
    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class ActionDefaults:
    frequency_hz: float = 0.0
    down_ms: int = 100
    up_ms: int = 100


@dataclass(frozen=True, slots=True)
class ActionSpec:
    buttons: tuple[Button, ...]
    frequency_hz: float
    down_ms: int
    up_ms: int

    @property
    def down_seconds(self) -> float:
        return max(0.0, self.down_ms / 1000.0)

    @property
    def up_seconds(self) -> float:
        return max(0.0, self.up_ms / 1000.0)

    @property
    def interval_seconds(self) -> float | None:
        if self.frequency_hz <= 0:
            return None
        return 1.0 / self.frequency_hz


@dataclass(frozen=True, slots=True)
class SceneDefaults:
    threshold: float | None = None
    search_margin: int = 24
    stride: int = 4
    search_step: int = 2
    hold_ms: int = 0
    score_window: int = 1
    luma_weight: float = 0.7
    chroma_weight: float = 0.3


@dataclass(frozen=True, slots=True)
class SceneSpec:
    image_path: Path
    roi: Roi
    threshold: float
    search_margin: int
    stride: int
    search_step: int
    hold_ms: int
    score_window: int
    luma_weight: float
    chroma_weight: float

    def build_detector(self, name: str):
        from vision.detector import Roi as DetectorRoi, StaticImageDetector

        return StaticImageDetector(
            name=name,
            image_path=self.image_path,
            roi=DetectorRoi(
                x=self.roi.x,
                y=self.roi.y,
                width=self.roi.width,
                height=self.roi.height,
            ),
            threshold=self.threshold,
            search_margin=self.search_margin,
            stride=self.stride,
            search_step=self.search_step,
            luma_weight=self.luma_weight,
            chroma_weight=self.chroma_weight,
        )


@dataclass(frozen=True, slots=True)
class StateSpec:
    name: str
    scene: SceneSpec | None
    next_states: tuple[str, ...]
    action: ActionSpec | None = None
    timeout_ms: int = 0
    timeout_next_state: str | None = None
    decision_mode: str = ""
    decision_margin: float = 0.0
    decision_history_window: int = 9
    decision_trend_window: int = 5
    decision_ok_step: float = 0.0
    reset_loop: bool = False
    notification: str = ""


@dataclass(frozen=True, slots=True)
class RecoverySpec:
    states: tuple[str, ...]
    timeout_ms: int


@dataclass(frozen=True, slots=True)
class SequenceDefaults:
    timeout_ms: int = 0
    scene: SceneDefaults = SceneDefaults()
    action: ActionDefaults = ActionDefaults()


@dataclass(frozen=True, slots=True)
class SequenceDefinition:
    sequence_id: str
    name: str | None
    recovery: RecoverySpec
    defaults: SequenceDefaults
    initial_state: str
    states: dict[str, StateSpec]
    source_path: Path


@dataclass(slots=True)
class SequenceRuntime:
    definition: SequenceDefinition
    detectors: dict[str, object]

    @property
    def sequence_id(self) -> str:
        return self.definition.sequence_id

    def detector_for(self, state_name: str):
        return self.detectors.get(state_name)


def load_sequences(directory: Path) -> dict[str, SequenceDefinition]:
    if not directory.exists():
        return {}

    definitions: dict[str, SequenceDefinition] = {}
    for path in sorted(directory.glob("*.json")):
        definition = load_sequence(path)
        if definition.sequence_id in definitions:
            raise SequenceConfigError(
                f'Duplicate sequence id "{definition.sequence_id}" in {path}.'
            )
        definitions[definition.sequence_id] = definition
    return definitions


def load_sequence(path: Path) -> SequenceDefinition:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SequenceConfigError(f"Sequence file {path} must contain a JSON object.")

    sequence_id = path.stem
    name = str(data["name"]) if "name" in data and data["name"] not in (None, "") else None
    defaults = _parse_defaults(data.get("defaults"))

    recovery_raw = data.get("recovery") or {}
    if not isinstance(recovery_raw, dict):
        raise SequenceConfigError(f"Sequence {path} recovery must be an object.")
    recovery = RecoverySpec(
        states=tuple(_parse_state_list(recovery_raw.get("states"), "recovery.states")),
        timeout_ms=_as_int(recovery_raw.get("timeout_ms", 0), "recovery.timeout_ms"),
    )

    states_raw = data.get("states")
    if not isinstance(states_raw, dict) or not states_raw:
        raise SequenceConfigError(f"Sequence {path} must define a non-empty states object.")

    states: dict[str, StateSpec] = {}
    for state_name, raw_state in states_raw.items():
        if not isinstance(raw_state, dict):
            raise SequenceConfigError(f'State "{state_name}" in {path} must be an object.')
        states[state_name] = _parse_state(path, state_name, raw_state, defaults)

    initial_state = next(iter(states))
    definition = SequenceDefinition(
        sequence_id=sequence_id,
        name=name,
        recovery=recovery,
        defaults=defaults,
        initial_state=initial_state,
        states=states,
        source_path=path,
    )
    _validate_sequence(definition)
    return definition


def build_runtime(definition: SequenceDefinition) -> SequenceRuntime:
    detectors: dict[str, object] = {}
    for state_name, state in definition.states.items():
        if state.scene is None:
            continue
        try:
            detectors[state_name] = state.scene.build_detector(state_name)
        except Exception as exc:
            raise SequenceConfigError(
                f'Could not build detector for state "{state_name}" from {state.scene.image_path}: {exc}'
            ) from exc
    return SequenceRuntime(definition=definition, detectors=detectors)


def _validate_sequence(definition: SequenceDefinition) -> None:
    state_names = set(definition.states)

    if definition.states[definition.initial_state].scene is None:
        raise SequenceConfigError(
            f'Sequence "{definition.sequence_id}" initial state "{definition.initial_state}" must define a scene.'
        )

    for state_name in definition.recovery.states:
        if state_name not in state_names:
            raise SequenceConfigError(
                f'Sequence "{definition.sequence_id}" recovery state "{state_name}" is not defined.'
            )
        if definition.states[state_name].scene is None:
            raise SequenceConfigError(
                f'Recovery state "{state_name}" must define a scene.'
            )

    for state in definition.states.values():
        for next_state in state.next_states:
            if next_state not in state_names:
                raise SequenceConfigError(
                    f'State "{state.name}" points to missing next_states entry "{next_state}".'
                )
        if state.notification not in {"", "mail"}:
            raise SequenceConfigError(
                f'State "{state.name}" uses unsupported notification "{state.notification}".'
            )
        if state.decision_mode not in {"", "best_score", "loop_baseline_step"}:
            raise SequenceConfigError(
                f'State "{state.name}" uses unsupported decision_mode "{state.decision_mode}".'
            )
        if state.decision_margin < 0:
            raise SequenceConfigError(
                f'State "{state.name}" decision_margin must be non-negative.'
            )
        if state.decision_history_window <= 0:
            raise SequenceConfigError(
                f'State "{state.name}" decision_history_window must be positive.'
            )
        if state.decision_trend_window <= 0:
            raise SequenceConfigError(
                f'State "{state.name}" decision_trend_window must be positive.'
            )
        if state.decision_ok_step < 0:
            raise SequenceConfigError(
                f'State "{state.name}" decision_ok_step must be non-negative.'
            )
        if state.decision_mode == "best_score":
            if len(state.next_states) < 2:
                raise SequenceConfigError(
                    f'State "{state.name}" decision_mode requires at least two next_states.'
                )
            for next_state in state.next_states:
                if definition.states[next_state].scene is None:
                    raise SequenceConfigError(
                        f'State "{state.name}" decision_mode requires detectable next_states; "{next_state}" has no scene.'
                    )
        if state.decision_mode == "loop_baseline_step":
            if len(state.next_states) != 1:
                raise SequenceConfigError(
                    f'State "{state.name}" loop_baseline_step requires exactly one detectable next_state.'
                )
            next_state = state.next_states[0]
            if definition.states[next_state].scene is None:
                raise SequenceConfigError(
                    f'State "{state.name}" loop_baseline_step requires detectable next_state "{next_state}".'
                )
            if state.timeout_next_state is None:
                raise SequenceConfigError(
                    f'State "{state.name}" loop_baseline_step requires timeout_next_state.'
                )
            if state.timeout_ms <= 0:
                raise SequenceConfigError(
                    f'State "{state.name}" loop_baseline_step requires timeout_ms > 0.'
                )
        if state.timeout_next_state is not None:
            if state.timeout_next_state not in state_names:
                raise SequenceConfigError(
                    f'State "{state.name}" timeout_next_state "{state.timeout_next_state}" is not defined.'
                )
            if state.timeout_ms <= 0:
                raise SequenceConfigError(
                    f'State "{state.name}" with timeout_next_state requires timeout_ms > 0.'
                )
        if state.reset_loop and not state.next_states:
            raise SequenceConfigError(
                f'State "{state.name}" reset_loop requires at least one next_state.'
            )

        procedural_targets = [
            next_state
            for next_state in state.next_states
            if definition.states[next_state].scene is None
        ]
        if len(procedural_targets) > 1:
            raise SequenceConfigError(
                f'State "{state.name}" may point to at most one procedural next_states entry.'
            )
        if procedural_targets and procedural_targets[0] != state.next_states[0]:
            raise SequenceConfigError(
                f'State "{state.name}" procedural next_states entry must be listed first.'
            )


def _parse_defaults(raw_defaults: Any) -> SequenceDefaults:
    if raw_defaults in (None, "", {}):
        return SequenceDefaults()
    if not isinstance(raw_defaults, dict):
        raise SequenceConfigError("defaults must be an object.")

    scene_raw = raw_defaults.get("scene") or {}
    if not isinstance(scene_raw, dict):
        raise SequenceConfigError("defaults.scene must be an object.")

    action_raw = raw_defaults.get("action") or {}
    if not isinstance(action_raw, dict):
        raise SequenceConfigError("defaults.action must be an object.")

    luma_weight = _as_float(scene_raw.get("luma_weight", 0.7), "defaults.scene.luma_weight")
    chroma_weight = _as_float(scene_raw.get("chroma_weight", 0.3), "defaults.scene.chroma_weight")
    if luma_weight < 0 or chroma_weight < 0:
        raise SequenceConfigError("defaults.scene weights must be non-negative.")
    if luma_weight + chroma_weight <= 0:
        raise SequenceConfigError("defaults.scene weights must sum to a positive value.")

    return SequenceDefaults(
        timeout_ms=_as_int(raw_defaults.get("timeout_ms", 0), "defaults.timeout_ms"),
        scene=SceneDefaults(
            threshold=(
                None
                if scene_raw.get("threshold") is None
                else float(scene_raw.get("threshold"))
            ),
            search_margin=_as_int(scene_raw.get("search_margin", 24), "defaults.scene.search_margin"),
            stride=_as_int(scene_raw.get("stride", 4), "defaults.scene.stride"),
            search_step=_as_int(scene_raw.get("search_step", 2), "defaults.scene.search_step"),
            hold_ms=_as_int(scene_raw.get("hold_ms", 0), "defaults.scene.hold_ms"),
            score_window=_as_int(scene_raw.get("score_window", 1), "defaults.scene.score_window"),
            luma_weight=luma_weight,
            chroma_weight=chroma_weight,
        ),
        action=ActionDefaults(
            frequency_hz=float(action_raw.get("frequency_hz", 0)),
            down_ms=_as_int(action_raw.get("down_ms", 100), "defaults.action.down_ms"),
            up_ms=_as_int(action_raw.get("up_ms", 100), "defaults.action.up_ms"),
        ),
    )


def _parse_state(
    path: Path,
    state_name: str,
    raw_state: dict[str, Any],
    defaults: SequenceDefaults,
) -> StateSpec:
    scene_raw = raw_state.get("scene")
    scene = None
    if scene_raw not in (None, "", {}):
        if not isinstance(scene_raw, dict):
            raise SequenceConfigError(f'State "{state_name}" in {path} scene must be an object.')
        scene = _parse_scene(path, state_name, scene_raw, defaults.scene)

    next_states_value = raw_state.get("next_states")
    if next_states_value is None:
        next_states_value = raw_state.get("next_state")
    next_states = tuple(_parse_state_list(next_states_value, f"{state_name}.next_states"))
    timeout_next_state_raw = raw_state.get("timeout_next_state")
    return StateSpec(
        name=state_name,
        scene=scene,
        next_states=next_states,
        action=_parse_action(raw_state.get("action"), f"{state_name}.action", defaults.action),
        timeout_ms=_as_int(raw_state.get("timeout_ms", defaults.timeout_ms), f"{state_name}.timeout_ms"),
        timeout_next_state=str(timeout_next_state_raw) if timeout_next_state_raw not in (None, "") else None,
        decision_mode=str(raw_state.get("decision_mode") or ""),
        decision_margin=float(raw_state.get("decision_margin", 0.0) or 0.0),
        decision_history_window=_as_int(
            raw_state.get("decision_history_window", 9),
            f"{state_name}.decision_history_window",
        ),
        decision_trend_window=_as_int(
            raw_state.get("decision_trend_window", 5),
            f"{state_name}.decision_trend_window",
        ),
        decision_ok_step=float(raw_state.get("decision_ok_step", 0.0) or 0.0),
        reset_loop=bool(raw_state.get("reset_loop", False)),
        notification=str(raw_state.get("notification") or ""),
    )


def _parse_scene(
    path: Path,
    state_name: str,
    raw_scene: dict[str, Any],
    defaults: SceneDefaults,
) -> SceneSpec:
    detector_type = str(raw_scene.get("type") or "static_image")
    if detector_type != "static_image":
        raise SequenceConfigError(
            f'State "{state_name}" uses unsupported scene.type "{detector_type}".'
        )

    image_value = _coalesce(raw_scene, "image_path", "img_path", "img path")
    if image_value in (None, ""):
        raise SequenceConfigError(f'State "{state_name}" is missing scene.image_path.')
    roi_raw = raw_scene.get("roi")
    if roi_raw in (None, ""):
        raise SequenceConfigError(f'State "{state_name}" is missing scene.roi.')

    threshold_value = raw_scene.get("threshold", defaults.threshold)
    if threshold_value is None:
        raise SequenceConfigError(f'State "{state_name}" is missing scene.threshold.')
    luma_weight = _as_float(
        raw_scene.get("luma_weight", defaults.luma_weight),
        f"{state_name}.scene.luma_weight",
    )
    chroma_weight = _as_float(
        raw_scene.get("chroma_weight", defaults.chroma_weight),
        f"{state_name}.scene.chroma_weight",
    )
    if luma_weight < 0 or chroma_weight < 0:
        raise SequenceConfigError(f'State "{state_name}" scene weights must be non-negative.')
    if luma_weight + chroma_weight <= 0:
        raise SequenceConfigError(f'State "{state_name}" scene weights must sum to a positive value.')

    return SceneSpec(
        image_path=(path.parent / str(image_value)).resolve(),
        roi=_parse_roi(roi_raw, f"{state_name}.scene.roi"),
        threshold=float(threshold_value),
        search_margin=_as_int(
            raw_scene.get("search_margin", defaults.search_margin),
            f"{state_name}.scene.search_margin",
        ),
        stride=_as_int(raw_scene.get("stride", defaults.stride), f"{state_name}.scene.stride"),
        search_step=_as_int(
            raw_scene.get("search_step", defaults.search_step),
            f"{state_name}.scene.search_step",
        ),
        hold_ms=_as_int(raw_scene.get("hold_ms", defaults.hold_ms), f"{state_name}.scene.hold_ms"),
        score_window=_as_int(raw_scene.get("score_window", defaults.score_window), f"{state_name}.scene.score_window"),
        luma_weight=luma_weight,
        chroma_weight=chroma_weight,
    )


def _parse_action(raw_action: Any, label: str, defaults: ActionDefaults) -> ActionSpec | None:
    if raw_action in (None, "", {}):
        return None
    if not isinstance(raw_action, dict):
        raise SequenceConfigError(f"{label} must be an object.")

    buttons_raw = raw_action.get("buttons") or []
    if not isinstance(buttons_raw, list):
        raise SequenceConfigError(f"{label}.buttons must be a list.")

    buttons: list[Button] = []
    for button_name in buttons_raw:
        try:
            buttons.append(Button(str(button_name)))
        except ValueError as exc:
            raise SequenceConfigError(f'{label}.buttons contains unsupported value "{button_name}".') from exc

    if not buttons:
        return None

    return ActionSpec(
        buttons=tuple(buttons),
        frequency_hz=float(raw_action.get("frequency_hz", defaults.frequency_hz)),
        down_ms=_as_int(raw_action.get("down_ms", defaults.down_ms), f"{label}.down_ms"),
        up_ms=_as_int(raw_action.get("up_ms", defaults.up_ms), f"{label}.up_ms"),
    )


def _parse_roi(raw_roi: Any, label: str) -> Roi:
    if not isinstance(raw_roi, dict):
        raise SequenceConfigError(f"{label} must be an object.")
    return Roi(
        x=_as_int(raw_roi.get("x"), f"{label}.x"),
        y=_as_int(raw_roi.get("y"), f"{label}.y"),
        width=_as_int(raw_roi.get("width"), f"{label}.width"),
        height=_as_int(raw_roi.get("height"), f"{label}.height"),
    )


def _parse_state_list(value: Any, label: str) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value]
    raise SequenceConfigError(f"{label} must be a string or list of strings.")


def _coalesce(data: dict[str, Any], *keys: str):
    for key in keys:
        if key in data:
            return data[key]
    return None


def _as_int(value: Any, label: str) -> int:
    if value is None:
        raise SequenceConfigError(f"{label} is required.")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise SequenceConfigError(f"{label} must be an integer.") from exc


def _as_float(value: Any, label: str) -> float:
    if value is None:
        raise SequenceConfigError(f"{label} is required.")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise SequenceConfigError(f"{label} must be a number.") from exc
