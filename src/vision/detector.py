from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

_RESAMPLING = Image.Resampling if hasattr(Image, "Resampling") else Image


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


@dataclass(frozen=True, slots=True)
class _InvariantSignature:
    chroma_rg: np.ndarray
    mask: np.ndarray
    luma: np.ndarray


def load_image_rgb(path: str | Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def _downsample(frame: np.ndarray, stride: int) -> np.ndarray:
    return frame[::stride, ::stride]


def _rgb_to_float(image: np.ndarray) -> np.ndarray:
    return image.astype(np.float32, copy=False) / 255.0


def _rgb_luma(image: np.ndarray) -> np.ndarray:
    return (
        0.2126 * image[:, :, 0] + 0.7152 * image[:, :, 1] + 0.0722 * image[:, :, 2]
    )


def _rgb_chroma(image: np.ndarray) -> np.ndarray:
    return image.max(axis=2) - image.min(axis=2)


def _focus_on_colorful_region(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    min_padding: int = 2,
    padding_ratio: float = 0.08,
) -> np.ndarray:
    if not np.any(mask):
        return image

    ys, xs = np.nonzero(mask)
    top = int(ys.min())
    bottom = int(ys.max()) + 1
    left = int(xs.min())
    right = int(xs.max()) + 1

    pad_y = max(min_padding, int(round((bottom - top) * padding_ratio)))
    pad_x = max(min_padding, int(round((right - left) * padding_ratio)))

    top = max(0, top - pad_y)
    bottom = min(image.shape[0], bottom + pad_y)
    left = max(0, left - pad_x)
    right = min(image.shape[1], right + pad_x)
    return image[top:bottom, left:right]


def _resize_rgb(image: np.ndarray, size: int) -> np.ndarray:
    clipped = np.clip(image, 0.0, 1.0)
    as_uint8 = (clipped * 255.0).astype(np.uint8)
    resized = Image.fromarray(as_uint8, mode="RGB").resize(
        (size, size),
        resample=_RESAMPLING.BILINEAR,
    )
    return np.asarray(resized, dtype=np.float32) / 255.0


def _trimmed_mean(values: np.ndarray, trim_fraction: float) -> float:
    flat = np.sort(values.reshape(-1))
    if flat.size == 0:
        return 1.0
    keep = max(1, int(round(flat.size * (1.0 - trim_fraction))))
    return float(flat[:keep].mean())


def _build_invariant_signature(
    image: np.ndarray,
    *,
    stride: int,
    feature_size: int,
    mask_chroma_threshold: float,
) -> _InvariantSignature:
    sampled = _downsample(image, stride)
    rgb = _rgb_to_float(sampled)
    chroma = _rgb_chroma(rgb)
    value = rgb.max(axis=2)
    focus_mask = np.logical_and(chroma >= mask_chroma_threshold, value >= 0.12)
    focused = _focus_on_colorful_region(rgb, focus_mask)
    resized = _resize_rgb(focused, feature_size)

    resized_chroma = _rgb_chroma(resized)
    resized_value = resized.max(axis=2)
    mask = np.logical_and(
        resized_chroma >= mask_chroma_threshold * 0.75,
        resized_value >= 0.10,
    ).astype(np.float32, copy=False)
    if float(mask.mean()) < 0.02:
        mask = np.ones_like(mask, dtype=np.float32)

    denom = np.maximum(resized.sum(axis=2, keepdims=True), 1e-4)
    chroma_rg = (resized[:, :, :2] / denom).astype(np.float32, copy=False)

    luma = _rgb_luma(resized)
    luma_mean = float(luma.mean())
    luma_std = float(luma.std())
    normalized_luma = ((luma - luma_mean) / max(luma_std, 1e-4)).astype(np.float32, copy=False)

    return _InvariantSignature(
        chroma_rg=chroma_rg,
        mask=mask,
        luma=normalized_luma,
    )


def _invariant_signature_distance(
    candidate: _InvariantSignature,
    reference: _InvariantSignature,
    *,
    trim_fraction: float,
) -> tuple[float, str]:
    active = np.maximum(candidate.mask, reference.mask) > 0.05

    color_error = np.linalg.norm(candidate.chroma_rg - reference.chroma_rg, axis=2) / np.sqrt(2.0)
    shape_error = np.abs(candidate.mask - reference.mask)
    luma_error = np.abs(candidate.luma - reference.luma) / 4.0
    combined_error = 0.60 * color_error + 0.25 * shape_error + 0.15 * luma_error

    if np.any(active):
        score = _trimmed_mean(combined_error[active], trim_fraction)
        color_score = float(color_error[active].mean())
        shape_score = float(shape_error[active].mean())
        luma_score = float(luma_error[active].mean())
    else:
        score = _trimmed_mean(combined_error, trim_fraction)
        color_score = float(color_error.mean())
        shape_score = float(shape_error.mean())
        luma_score = float(luma_error.mean())

    detail = (
        f"color={color_score:.4f} "
        f"shape={shape_score:.4f} "
        f"luma={luma_score:.4f}"
    )
    return score, detail


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


class InvariantColorDetector:
    def __init__(
        self,
        *,
        name: str,
        image_path: str | Path,
        roi: Roi,
        threshold: float,
        search_margin: int = 0,
        stride: int = 4,
        search_step: int = 2,
        feature_size: int = 64,
        mask_chroma_threshold: float = 0.08,
        trim_fraction: float = 0.12,
    ) -> None:
        self.name = name
        self.image_path = Path(image_path)
        self.roi = roi
        self.threshold = threshold
        self.search_margin = search_margin
        self.stride = stride
        self.search_step = search_step
        self.feature_size = feature_size
        self.mask_chroma_threshold = mask_chroma_threshold
        self.trim_fraction = trim_fraction
        self._reference = _build_invariant_signature(
            load_image_rgb(self.image_path),
            stride=self.stride,
            feature_size=self.feature_size,
            mask_chroma_threshold=self.mask_chroma_threshold,
        )

    def match(self, frame: np.ndarray) -> MatchResult:
        frame_height, frame_width = frame.shape[:2]
        best_score = float("inf")
        best_offset = (0, 0)
        best_detail: str | None = None

        for offset_y in range(-self.search_margin, self.search_margin + 1, self.search_step):
            y = self.roi.y + offset_y
            if y < 0 or y + self.roi.height > frame_height:
                continue
            for offset_x in range(-self.search_margin, self.search_margin + 1, self.search_step):
                x = self.roi.x + offset_x
                if x < 0 or x + self.roi.width > frame_width:
                    continue

                candidate = frame[y : y + self.roi.height, x : x + self.roi.width]
                signature = _build_invariant_signature(
                    candidate,
                    stride=self.stride,
                    feature_size=self.feature_size,
                    mask_chroma_threshold=self.mask_chroma_threshold,
                )
                score, detail = _invariant_signature_distance(
                    signature,
                    self._reference,
                    trim_fraction=self.trim_fraction,
                )
                if score < best_score:
                    best_score = score
                    best_offset = (offset_x, offset_y)
                    best_detail = detail

        return MatchResult(
            matched=best_score <= self.threshold,
            score=best_score,
            offset_x=best_offset[0],
            offset_y=best_offset[1],
            detail=best_detail,
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
        return MatchResult(
            matched=matched,
            score=score,
            detail=f"mean={mean_luma:.1f} std={stddev:.1f}",
        )
