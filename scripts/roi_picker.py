from __future__ import annotations

import argparse
import json
import tempfile
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


@dataclass(slots=True)
class Roi:
    x: int
    y: int
    width: int
    height: int

    def to_dict(self) -> dict[str, int]:
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
        }


def build_scene_dict(
    roi: Roi,
    *,
    image_path: str | None,
    threshold: float | None,
) -> dict[str, object]:
    scene: dict[str, object] = {
        "roi": roi.to_dict(),
    }
    if image_path:
        scene["image_path"] = image_path
    if threshold is not None:
        scene["threshold"] = threshold
    return {"scene": scene}


class RoiPicker:
    def __init__(
        self,
        image_path: Path,
        max_width: int,
        max_height: int,
        *,
        output_format: str,
        scene_image_path: str | None,
        scene_threshold: float | None,
    ):
        self.image_path = image_path
        self.max_width = max_width
        self.max_height = max_height
        self.output_format = output_format
        self.scene_image_path = scene_image_path
        self.scene_threshold = scene_threshold

        source = Image.open(image_path).convert("RGB")
        scale = min(max_width / source.width, max_height / source.height, 1.0)
        self.scale = scale
        self.display_size = (
            max(1, int(round(source.width * scale))),
            max(1, int(round(source.height * scale))),
        )
        self.display_image = source.resize(self.display_size, Image.Resampling.LANCZOS)

        self._drag_start: tuple[int, int] | None = None
        self._rect_id: int | None = None
        self._temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        self._temp_file.close()
        self.display_image.save(self._temp_file.name, format="PNG")

        self.root = tk.Tk()
        self.root.title(f"ROI Picker - {image_path.name}")

        self.status_var = tk.StringVar(
            value=(
                f"Image: {source.width}x{source.height}  Display: {self.display_size[0]}x{self.display_size[1]}  "
                "Drag to select an ROI."
            )
        )
        self.roi_var = tk.StringVar(value="ROI: not selected")

        self.photo = tk.PhotoImage(file=self._temp_file.name)
        self.canvas = tk.Canvas(
            self.root,
            width=self.display_size[0],
            height=self.display_size[1],
            highlightthickness=0,
        )
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo)
        self.canvas.pack()

        info = tk.Frame(self.root)
        info.pack(fill="x", padx=8, pady=8)

        tk.Label(info, textvariable=self.status_var, anchor="w", justify="left").pack(fill="x")
        tk.Label(info, textvariable=self.roi_var, anchor="w", justify="left").pack(fill="x", pady=(4, 0))

        buttons = tk.Frame(self.root)
        buttons.pack(fill="x", padx=8, pady=(0, 8))
        tk.Button(buttons, text="Clear", command=self.clear_selection).pack(side="left")
        tk.Button(buttons, text="Quit", command=self.root.destroy).pack(side="right")

        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

        self.root.protocol("WM_DELETE_WINDOW", self.root.destroy)

    def run(self) -> None:
        try:
            self.root.mainloop()
        finally:
            try:
                Path(self._temp_file.name).unlink(missing_ok=True)
            except Exception:
                pass

    def clear_selection(self) -> None:
        if self._rect_id is not None:
            self.canvas.delete(self._rect_id)
            self._rect_id = None
        self._drag_start = None
        self.roi_var.set("ROI: not selected")

    def on_press(self, event) -> None:
        self._drag_start = (self._clamp_x(event.x), self._clamp_y(event.y))
        if self._rect_id is not None:
            self.canvas.delete(self._rect_id)
        self._rect_id = self.canvas.create_rectangle(
            event.x,
            event.y,
            event.x,
            event.y,
            outline="#00ff66",
            width=2,
        )

    def on_drag(self, event) -> None:
        if self._drag_start is None or self._rect_id is None:
            return
        x0, y0 = self._drag_start
        x1 = self._clamp_x(event.x)
        y1 = self._clamp_y(event.y)
        self.canvas.coords(self._rect_id, x0, y0, x1, y1)
        roi = self._display_roi_to_source(x0, y0, x1, y1)
        self.roi_var.set(f"ROI: {roi.to_dict()}")

    def on_release(self, event) -> None:
        if self._drag_start is None:
            return
        x0, y0 = self._drag_start
        x1 = self._clamp_x(event.x)
        y1 = self._clamp_y(event.y)
        roi = self._display_roi_to_source(x0, y0, x1, y1)
        if roi.width <= 0 or roi.height <= 0:
            self.clear_selection()
            return
        self.roi_var.set(f"ROI: {roi.to_dict()}")
        self._print_output(roi)

    def _print_output(self, roi: Roi) -> None:
        if self.output_format in {"roi", "both"}:
            print(json.dumps(roi.to_dict(), indent=2))
        if self.output_format in {"scene", "both"}:
            print(
                json.dumps(
                    build_scene_dict(
                        roi,
                        image_path=self.scene_image_path,
                        threshold=self.scene_threshold,
                    ),
                    indent=2,
                )
            )

    def _display_roi_to_source(self, x0: int, y0: int, x1: int, y1: int) -> Roi:
        left = min(x0, x1)
        top = min(y0, y1)
        right = max(x0, x1)
        bottom = max(y0, y1)

        source_left = int(round(left / self.scale))
        source_top = int(round(top / self.scale))
        source_right = int(round(right / self.scale))
        source_bottom = int(round(bottom / self.scale))
        return Roi(
            x=source_left,
            y=source_top,
            width=max(0, source_right - source_left),
            height=max(0, source_bottom - source_top),
        )

    def _clamp_x(self, value: int) -> int:
        return min(max(0, value), self.display_size[0] - 1)

    def _clamp_y(self, value: int) -> int:
        return min(max(0, value), self.display_size[1] - 1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Draw an ROI on a saved camera snapshot and print ROI JSON or a paste-ready scene block."
    )
    parser.add_argument("image", type=Path, help="Path to a saved snapshot image.")
    parser.add_argument(
        "--max-width",
        type=int,
        default=1600,
        help="Maximum width of the displayed image window.",
    )
    parser.add_argument(
        "--max-height",
        type=int,
        default=900,
        help="Maximum height of the displayed image window.",
    )
    parser.add_argument(
        "--output-format",
        choices=["roi", "scene", "both"],
        default="roi",
        help="Choose whether to print just the ROI, a scene object, or both.",
    )
    parser.add_argument(
        "--image-path",
        type=str,
        default=None,
        help="image_path value to include when printing scene JSON.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional threshold value to include when printing scene JSON.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    picker = RoiPicker(
        args.image,
        max_width=args.max_width,
        max_height=args.max_height,
        output_format=args.output_format,
        scene_image_path=args.image_path,
        scene_threshold=args.threshold,
    )
    picker.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
