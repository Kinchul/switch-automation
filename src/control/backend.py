from collections.abc import Callable
from abc import ABC, abstractmethod
from enum import Enum


class Button(str, Enum):
    A = "A"
    B = "B"
    X = "X"
    Y = "Y"
    L = "L"
    R = "R"
    ZL = "ZL"
    ZR = "ZR"
    PLUS = "PLUS"
    MINUS = "MINUS"
    HOME = "HOME"
    CAPTURE = "CAPTURE"
    DPAD_UP = "DPAD_UP"
    DPAD_DOWN = "DPAD_DOWN"
    DPAD_LEFT = "DPAD_LEFT"
    DPAD_RIGHT = "DPAD_RIGHT"


class ControllerBackend(ABC):
    @abstractmethod
    def connect(
        self,
        *,
        cancel_cb: Callable[[], bool] | None = None,
        status_cb: Callable[[str], None] | None = None,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def press(self, *buttons: Button, down: float = 0.1, up: float = 0.1) -> None:
        raise NotImplementedError

    @abstractmethod
    def macro(self, commands: str, block: bool = True):
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError


class ControllerConnectCancelled(RuntimeError):
    pass
