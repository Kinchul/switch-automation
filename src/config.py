from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class ControlConfig:
    controller_type: str = "PRO_CONTROLLER"
    reconnect: bool = True


@dataclass(slots=True)
class CaptureConfig:
    device: str = "/dev/video0"
    width: int = 1280
    height: int = 720
    fps: int = 30


@dataclass(slots=True)
class ProjectConfig:
    root: Path
    control: ControlConfig = field(default_factory=ControlConfig)
    capture: CaptureConfig = field(default_factory=CaptureConfig)
