from .runner import HuntRunner
from .persistence import LoopStatsSnapshot, PersistentLoopControl, PersistentLoopStatsStore
from .sequence import SequenceConfigError, SequenceDefinition, build_runtime, load_sequence, load_sequences

__all__ = [
    "CameraLoopConfig",
    "CameraLoopRunner",
    "HuntRunner",
    "LoopOutcome",
    "LoopStatsSnapshot",
    "PersistentLoopControl",
    "PersistentLoopStatsStore",
    "SequenceConfigError",
    "SequenceDefinition",
    "build_runtime",
    "load_sequence",
    "load_sequences",
]


def __getattr__(name: str):
    if name in {"CameraLoopConfig", "CameraLoopRunner", "LoopOutcome"}:
        from .camera_loop import CameraLoopConfig, CameraLoopRunner, LoopOutcome

        mapping = {
            "CameraLoopConfig": CameraLoopConfig,
            "CameraLoopRunner": CameraLoopRunner,
            "LoopOutcome": LoopOutcome,
        }
        return mapping[name]
    raise AttributeError(name)
