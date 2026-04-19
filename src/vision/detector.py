from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(frozen=True, slots=True)
class Roi:
    x: int
    y: int
    width: int
    height: int

    def crop(self, frame: np.ndarray) -> np.ndarray:
        return frame[self.y : self.y + self.height, self.x : self.x + self.width]


@dataclass(frozen=True, slots=True)
class MatchResult:
    matched: bool
    score: float
    offset_x: int = 0
    offset_y: int = 0
    detail: str | None = None


def load_image_rgb(path: str | Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def _downsample(frame: np.ndarray, stride: int) -> np.ndarray:
    return frame[::stride, ::stride]


def _prepare_reference_crop(image_path: str | Path, roi: Roi, stride: int) -> tuple[Roi, np.ndarray]:
    image = load_image_rgb(image_path)
    image_height, image_width = image.shape[:2]
    if roi.x < 0 or roi.y < 0 or roi.width <= 0 or roi.height <= 0:
        raise ValueError(
            f"Invalid ROI for {image_path}: x={roi.x} y={roi.y} width={roi.width} height={roi.height}."
        )

    if roi.x + roi.width <= image_width and roi.y + roi.height <= image_height:
        normalized_roi = roi
        crop = roi.crop(image)
    else:
        # Cropped template: keep the configured screen position, but use the
        # whole image as the reference and derive the candidate size from it.
        normalized_roi = Roi(
            x=roi.x,
            y=roi.y,
            width=image_width,
            height=image_height,
        )
        crop = image

    reference = _downsample(crop, stride).astype(np.int16, copy=False)
    if reference.size == 0:
        raise ValueError(
            f"ROI for {image_path} produced an empty reference crop after downsampling. "
            f"stride={stride}, roi=({roi.x},{roi.y},{roi.width},{roi.height})."
        )
    return normalized_roi, reference


class StaticImageDetector:
    def __init__(
        self,
        *,
        name: str,
        image_path: str | Path,
        roi: Roi,
        threshold: float,
        search_margin: int = 24,
        stride: int = 4,
        search_step: int = 2,
    ) -> None:
        self.name = name
        self.image_path = Path(image_path)
        self.threshold = threshold
        self.search_margin = search_margin
        self.stride = stride
        self.search_step = search_step
        self.roi, self._reference = _prepare_reference_crop(self.image_path, roi, self.stride)
        # Number of downsampled positions to advance per search step
        self._step = max(1, search_step // stride)

    def match(self, frame: np.ndarray) -> MatchResult:
        frame_h, frame_w = frame.shape[:2]

        y1 = max(0, self.roi.y - self.search_margin)
        y2 = min(frame_h, self.roi.y + self.roi.height + self.search_margin)
        x1 = max(0, self.roi.x - self.search_margin)
        x2 = min(frame_w, self.roi.x + self.roi.width + self.search_margin)

        # Crop once, downsample in place — O(1) numpy views until astype
        search = frame[y1:y2, x1:x2][::self.stride, ::self.stride].astype(np.int16)
        ref = self._reference  # (H_r, W_r, C) int16, pre-downsampled
        H_r, W_r, C = ref.shape
        H_s, W_s = search.shape[:2]

        if H_s < H_r or W_s < W_r:
            return MatchResult(matched=False, score=1.0)

        step = self._step
        n_y = H_s - H_r + 1
        n_x = W_s - W_r + 1
        # ceil division: number of sampled positions along each axis
        n_y_s = (n_y + step - 1) // step
        n_x_s = (n_x + step - 1) // step

        # Strided view: windows[i, j] == search[i*step : i*step+H_r, j*step : j*step+W_r, :]
        # Baking step into the outer strides avoids materialising unused positions.
        sy, sx, sc = search.strides
        windows = np.lib.stride_tricks.as_strided(
            search,
            shape=(n_y_s, n_x_s, H_r, W_r, C),
            strides=(sy * step, sx * step, sy, sx, sc),
            writeable=False,
        )

        # Vectorised MAE/255 for all sampled positions at once
        scores = np.abs(windows - ref).mean(axis=(2, 3, 4)) / 255.0  # (n_y_s, n_x_s)

        best_flat = int(scores.argmin())
        best_y, best_x = divmod(best_flat, n_x_s)
        best_score = float(scores[best_y, best_x])

        # Map back to pixel offsets relative to the nominal ROI position
        effective_stride = self.stride * step
        offset_y = (y1 + best_y * effective_stride) - self.roi.y
        offset_x = (x1 + best_x * effective_stride) - self.roi.x

        return MatchResult(
            matched=best_score <= self.threshold,
            score=best_score,
            offset_x=offset_x,
            offset_y=offset_y,
        )
