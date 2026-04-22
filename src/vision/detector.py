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
    if stride <= 1:
        return frame.astype(np.float32, copy=False)

    height, width = frame.shape[:2]
    pad_h = (-height) % stride
    pad_w = (-width) % stride
    if pad_h or pad_w:
        frame = np.pad(frame, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")

    return frame.reshape(
        frame.shape[0] // stride,
        stride,
        frame.shape[1] // stride,
        stride,
        frame.shape[2],
    ).mean(axis=(1, 3), dtype=np.float32)


def _rgb_to_ycbcr(frame: np.ndarray) -> np.ndarray:
    rgb = frame.astype(np.float32, copy=False)
    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]
    y = (0.299 * r) + (0.587 * g) + (0.114 * b)
    cb = 128.0 - (0.168736 * r) - (0.331264 * g) + (0.5 * b)
    cr = 128.0 + (0.5 * r) - (0.418688 * g) - (0.081312 * b)
    return np.stack((y, cb, cr), axis=-1)


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

    reference = _rgb_to_ycbcr(_downsample(crop, stride))
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
        luma_weight: float = 0.7,
        chroma_weight: float = 0.3,
    ) -> None:
        self.name = name
        self.image_path = Path(image_path)
        self.threshold = threshold
        self.search_margin = search_margin
        self.stride = stride
        self.search_step = search_step
        total_weight = luma_weight + chroma_weight
        if total_weight <= 0:
            raise ValueError("luma_weight + chroma_weight must be positive.")
        self.luma_weight = luma_weight / total_weight
        self.chroma_weight = chroma_weight / total_weight
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
        search = _rgb_to_ycbcr(_downsample(frame[y1:y2, x1:x2], self.stride))
        ref = self._reference  # (H_r, W_r, C) float32, pre-downsampled
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

        channel_mae = np.abs(windows - ref).mean(axis=(2, 3))
        scores = (
            (channel_mae[..., 0] * self.luma_weight)
            + (channel_mae[..., 1:].mean(axis=2) * self.chroma_weight)
        ) / 255.0

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
