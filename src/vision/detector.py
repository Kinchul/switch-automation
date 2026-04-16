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
                if candidate.shape != self._reference.shape:
                    raise RuntimeError(
                        f'Detector "{self.name}" reference shape {self._reference.shape} '
                        f"does not match candidate shape {candidate.shape}. "
                        f"Reference image: {self.image_path}"
                    )
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
