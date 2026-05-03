from __future__ import annotations

import json
from pathlib import Path
from time import sleep
from typing import Any

from .base import CameraSource, to_rgb_frame


class CsiCameraSource(CameraSource):
    """Raspberry Pi CSI camera via Picamera2."""

    name = "csi"

    def __init__(
        self,
        *,
        camera_index: int,
        width: int,
        height: int,
        fps: int,
        warmup: float,
        lock_auto_controls: bool,
        controls_path: Path | None,
    ) -> None:
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps
        self.warmup = warmup
        self.lock_auto_controls = lock_auto_controls
        self.controls_path = controls_path
        self._camera: Any | None = None

    def start(self) -> None:
        try:
            from picamera2 import Picamera2  # type: ignore
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Picamera2 is not installed. Install `python3-picamera2` on the Raspberry Pi."
            ) from exc

        camera_info = Picamera2.global_camera_info()
        if not camera_info:
            raise RuntimeError(
                "No Picamera2 cameras were detected. Confirm `rpicam-hello --list-cameras` "
                "shows the sensor and that the current environment can access `/dev/media*`."
            )
        if not (0 <= self.camera_index < len(camera_info)):
            raise RuntimeError(
                f"Camera index {self.camera_index} is out of range. "
                f"Detected cameras: {len(camera_info)}."
            )

        frame_duration_us = max(1, int(1_000_000 / self.fps))
        camera = Picamera2(self.camera_index)
        config = camera.create_video_configuration(
            main={"size": (self.width, self.height), "format": "RGB888"},
            controls={"FrameDurationLimits": (frame_duration_us, frame_duration_us)},
            buffer_count=4,
            queue=False,
        )
        camera.configure(config)
        camera.start()

        controls_loaded = self._apply_saved_controls(camera)
        if self.warmup > 0:
            sleep(self.warmup)
        if self.lock_auto_controls and not controls_loaded:
            self._lock_stable_controls(camera)

        self._camera = camera

    def read_frame(self) -> Any:
        camera = self._camera
        if camera is None:
            raise RuntimeError("CSI camera is not started.")
        return to_rgb_frame(camera.capture_array("main"))

    def close(self) -> None:
        camera = self._camera
        self._camera = None
        if camera is None:
            return
        try:
            camera.stop()
        finally:
            camera.close()

    def _apply_saved_controls(self, camera: Any) -> bool:
        if self.controls_path is None or not self.controls_path.exists():
            return False

        try:
            data = json.loads(self.controls_path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"Camera controls file ignored: could not read {self.controls_path} ({exc}).")
            return False
        if not isinstance(data, dict):
            print(f"Camera controls file ignored: {self.controls_path} did not contain a JSON object.")
            return False

        controls = getattr(camera, "camera_controls", {}) or {}
        locked_controls: dict[str, Any] = {}
        if "AeEnable" in controls:
            locked_controls["AeEnable"] = False
        if "AwbEnable" in controls:
            locked_controls["AwbEnable"] = False

        exposure_time = data.get("ExposureTime")
        if exposure_time is not None and "ExposureTime" in controls:
            locked_controls["ExposureTime"] = int(exposure_time)

        analogue_gain = data.get("AnalogueGain")
        if analogue_gain is not None and "AnalogueGain" in controls:
            locked_controls["AnalogueGain"] = float(analogue_gain)

        colour_gains = data.get("ColourGains")
        if (
            isinstance(colour_gains, list)
            and len(colour_gains) == 2
            and "ColourGains" in controls
        ):
            locked_controls["ColourGains"] = (
                float(colour_gains[0]),
                float(colour_gains[1]),
            )

        if not locked_controls:
            print(f"Camera controls file ignored: no applicable controls found in {self.controls_path}.")
            return False

        try:
            camera.set_controls(locked_controls)
            sleep(0.2)
        except Exception as exc:
            print(f"Camera controls file ignored: could not apply {self.controls_path} ({exc}).")
            return False

        detail_parts: list[str] = []
        if "ExposureTime" in locked_controls:
            detail_parts.append(f"ExposureTime={locked_controls['ExposureTime']}")
        if "AnalogueGain" in locked_controls:
            detail_parts.append(f"AnalogueGain={locked_controls['AnalogueGain']:.4f}")
        if "ColourGains" in locked_controls:
            cg = locked_controls["ColourGains"]
            detail_parts.append(f"ColourGains=({cg[0]:.4f}, {cg[1]:.4f})")
        print(
            f"Applied saved camera controls from {self.controls_path}."
            + (f" {' '.join(detail_parts)}" if detail_parts else "")
        )
        return True

    def _lock_stable_controls(self, camera: Any) -> None:
        try:
            controls = getattr(camera, "camera_controls", {}) or {}
            metadata = camera.capture_metadata()
        except Exception as exc:
            print(f"Camera auto-lock skipped: could not read metadata ({exc}).")
            return

        locked_controls: dict[str, Any] = {}
        if "AeEnable" in controls:
            locked_controls["AeEnable"] = False

        exposure_time = metadata.get("ExposureTime")
        if exposure_time is not None and "ExposureTime" in controls:
            locked_controls["ExposureTime"] = int(exposure_time)

        analogue_gain = metadata.get("AnalogueGain")
        if analogue_gain is not None and "AnalogueGain" in controls:
            locked_controls["AnalogueGain"] = float(analogue_gain)

        if "AwbEnable" in controls:
            locked_controls["AwbEnable"] = False

        colour_gains = metadata.get("ColourGains")
        if (
            isinstance(colour_gains, (tuple, list))
            and len(colour_gains) == 2
            and "ColourGains" in controls
        ):
            locked_controls["ColourGains"] = (
                float(colour_gains[0]),
                float(colour_gains[1]),
            )

        if not locked_controls:
            print("Camera auto-lock skipped: no matching AE/AWB controls were available.")
            return

        try:
            camera.set_controls(locked_controls)
            sleep(0.2)
        except Exception as exc:
            print(f"Camera auto-lock failed: {exc}")
            return

        detail_parts: list[str] = []
        if "ExposureTime" in locked_controls:
            detail_parts.append(f"ExposureTime={locked_controls['ExposureTime']}")
        if "AnalogueGain" in locked_controls:
            detail_parts.append(f"AnalogueGain={locked_controls['AnalogueGain']:.4f}")
        if "ColourGains" in locked_controls:
            cg = locked_controls["ColourGains"]
            detail_parts.append(f"ColourGains=({cg[0]:.4f}, {cg[1]:.4f})")
        print(
            "Camera auto controls locked."
            + (f" {' '.join(detail_parts)}" if detail_parts else "")
        )
        if self.controls_path is not None:
            self._save_controls_profile(locked_controls)

    def _save_controls_profile(self, locked_controls: dict[str, Any]) -> None:
        if self.controls_path is None:
            return

        payload: dict[str, Any] = {}
        if "ExposureTime" in locked_controls:
            payload["ExposureTime"] = int(locked_controls["ExposureTime"])
        if "AnalogueGain" in locked_controls:
            payload["AnalogueGain"] = float(locked_controls["AnalogueGain"])
        if "ColourGains" in locked_controls:
            colour_gains = locked_controls["ColourGains"]
            payload["ColourGains"] = [float(colour_gains[0]), float(colour_gains[1])]

        if not payload:
            return

        payload["camera_index"] = self.camera_index
        payload["width"] = self.width
        payload["height"] = self.height
        payload["fps"] = self.fps
        self.controls_path.parent.mkdir(parents=True, exist_ok=True)
        self.controls_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Saved camera controls to {self.controls_path}.")
