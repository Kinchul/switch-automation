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
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        viewport_margin = 120
        self.viewport_size = (
            min(self.display_size[0], max(200, screen_width - viewport_margin)),
            min(self.display_size[1], max(200, screen_height - viewport_margin)),
        )

        self.status_var = tk.StringVar(
            value=(
                f"Image: {source.width}x{source.height}  Display: {self.display_size[0]}x{self.display_size[1]}  "
                "Drag to select an ROI. Use the mouse wheel or scrollbars to move around."
            )
        )
        self.roi_var = tk.StringVar(value="ROI: not selected")

        self.photo = tk.PhotoImage(file=self._temp_file.name)
        canvas_frame = tk.Frame(self.root)
        canvas_frame.pack(fill="both", expand=True, padx=8, pady=(8, 0))

        self.canvas = tk.Canvas(
            canvas_frame,
            width=self.viewport_size[0],
            height=self.viewport_size[1],
            highlightthickness=0,
        )
        h_scroll = tk.Scrollbar(canvas_frame, orient="horizontal", command=self.canvas.xview)
        v_scroll = tk.Scrollbar(canvas_frame, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(
            xscrollcommand=h_scroll.set,
            yscrollcommand=v_scroll.set,
            scrollregion=(0, 0, self.display_size[0], self.display_size[1]),
        )
        self.canvas.grid(row=0, column=0, sticky="nsew")
        v_scroll.grid(row=0, column=1, sticky="ns")
        h_scroll.grid(row=1, column=0, sticky="ew")
        canvas_frame.grid_rowconfigure(0, weight=1)
        canvas_frame.grid_columnconfigure(0, weight=1)

        self.canvas.create_image(0, 0, anchor="nw", image=self.photo)

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
        self.canvas.bind("<MouseWheel>", self.on_mousewheel)
        self.canvas.bind("<Shift-MouseWheel>", self.on_shift_mousewheel)

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
        canvas_x = self._canvas_x(event.x)
        canvas_y = self._canvas_y(event.y)
        self._drag_start = (canvas_x, canvas_y)
        if self._rect_id is not None:
            self.canvas.delete(self._rect_id)
        self._rect_id = self.canvas.create_rectangle(
            canvas_x,
            canvas_y,
            canvas_x,
            canvas_y,
            outline="#00ff66",
            width=2,
        )

    def on_drag(self, event) -> None:
        if self._drag_start is None or self._rect_id is None:
            return
        x0, y0 = self._drag_start
        x1 = self._canvas_x(event.x)
        y1 = self._canvas_y(event.y)
        self.canvas.coords(self._rect_id, x0, y0, x1, y1)
        roi = self._display_roi_to_source(x0, y0, x1, y1)
        self.roi_var.set(f"ROI: {roi.to_dict()}")

    def on_release(self, event) -> None:
        if self._drag_start is None:
            return
        x0, y0 = self._drag_start
        x1 = self._canvas_x(event.x)
        y1 = self._canvas_y(event.y)
        roi = self._display_roi_to_source(x0, y0, x1, y1)
        if roi.width <= 0 or roi.height <= 0:
            self.clear_selection()
            return
        self.roi_var.set(f"ROI: {roi.to_dict()}")
        self._print_output(roi)
        self._drag_start = None

    def on_mousewheel(self, event) -> None:
        self.canvas.yview_scroll(self._wheel_units(event.delta), "units")

    def on_shift_mousewheel(self, event) -> None:
        self.canvas.xview_scroll(self._wheel_units(event.delta), "units")

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

    def _canvas_x(self, value: int) -> int:
        return self._clamp_x(int(round(self.canvas.canvasx(value))))

    def _canvas_y(self, value: int) -> int:
        return self._clamp_y(int(round(self.canvas.canvasy(value))))

    @staticmethod
    def _wheel_units(delta: int) -> int:
        if delta == 0:
            return 0
        step = max(1, abs(delta) // 120)
        return -step if delta > 0 else step


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
