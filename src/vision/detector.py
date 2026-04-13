from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

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


def load_image_rgb(path: str | Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def _downsample(frame: np.ndarray, stride: int) -> np.ndarray:
    return frame[::stride, ::stride]


def _prepare_reference_crop(image_path: str | Path, roi: Roi, stride: int) -> np.ndarray:
    image = load_image_rgb(image_path)
    crop = roi.crop(image)
    return _downsample(crop, stride).astype(np.int16, copy=False)


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
        self.roi = roi
        self.threshold = threshold
        self.search_margin = search_margin
        self.stride = stride
        self.search_step = search_step
        self._reference = _prepare_reference_crop(self.image_path, self.roi, self.stride)

    def match(self, frame: np.ndarray) -> MatchResult:
        frame_height, frame_width = frame.shape[:2]
        best_score = float("inf")
        best_offset = (0, 0)

        for offset_y in range(-self.search_margin, self.search_margin + 1, self.search_step):
            y = self.roi.y + offset_y
            if y < 0 or y + self.roi.height > frame_height:
                continue
            for offset_x in range(-self.search_margin, self.search_margin + 1, self.search_step):
                x = self.roi.x + offset_x
                if x < 0 or x + self.roi.width > frame_width:
                    continue

                candidate = frame[y : y + self.roi.height, x : x + self.roi.width]
                candidate = _downsample(candidate, self.stride).astype(np.int16, copy=False)
                score = float(np.abs(candidate - self._reference).mean() / 255.0)
                if score < best_score:
                    best_score = score
                    best_offset = (offset_x, offset_y)

        return MatchResult(
            matched=best_score <= self.threshold,
            score=best_score,
            offset_x=best_offset[0],
            offset_y=best_offset[1],
        )


class BlackScreenDetector:
    def __init__(
        self,
        *,
        roi: Roi | None = None,
        max_mean_luma: float = 18.0,
        max_luma_stddev: float = 12.0,
    ) -> None:
        self.roi = roi
        self.max_mean_luma = max_mean_luma
        self.max_luma_stddev = max_luma_stddev

    def match(self, frame: np.ndarray) -> MatchResult:
        if self.roi is not None:
            frame = self.roi.crop(frame)
        grayscale = (
            0.2126 * frame[:, :, 0] + 0.7152 * frame[:, :, 1] + 0.0722 * frame[:, :, 2]
        )
        mean_luma = float(grayscale.mean())
        stddev = float(grayscale.std())
        matched = mean_luma <= self.max_mean_luma and stddev <= self.max_luma_stddev
        score = max(mean_luma / max(self.max_mean_luma, 1.0), stddev / max(self.max_luma_stddev, 1.0))
        return MatchResult(matched=matched, score=score)
