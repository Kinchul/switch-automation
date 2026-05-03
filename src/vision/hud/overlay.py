from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class OverlayBox:
    x: int
    y: int
    width: int
    height: int
    label: str | None = None
    outline: tuple[int, int, int, int] = (255, 215, 0, 255)
    fill: tuple[int, int, int, int] | None = (255, 215, 0, 48)


@dataclass(slots=True)
class OverlayState:
    lines: list[str] = field(default_factory=list)
    top_left_lines: list[str] = field(default_factory=list)
    bottom_left_lines: list[str] = field(default_factory=list)
    top_right_lines: list[str] = field(default_factory=list)
    boxes: list[OverlayBox] = field(default_factory=list)
    bottom_right_lines: list[str] = field(default_factory=list)


def draw_overlay(frame, overlay: OverlayState):
    if (
        not overlay.lines
        and not overlay.top_left_lines
        and not overlay.bottom_left_lines
        and not overlay.top_right_lines
        and not overlay.boxes
        and not overlay.bottom_right_lines
    ):
        return frame

    from PIL import Image, ImageDraw
    import numpy as np

    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image, "RGBA")
    font_size = max(24, frame.shape[1] // 55)
    font = _load_overlay_font(font_size)
    small_font = _load_overlay_font(max(18, font_size - 4))

    if overlay.boxes:
        _draw_boxes(draw, overlay.boxes, font=small_font, frame_width=frame.shape[1], frame_height=frame.shape[0])

    for lines, anchor in (
        (overlay.lines, "bottom_left"),
        (overlay.top_left_lines, "top_left"),
        (overlay.bottom_left_lines, "bottom_left"),
        (overlay.top_right_lines, "top_right"),
        (overlay.bottom_right_lines, "bottom_right"),
    ):
        if lines:
            _draw_corner_lines(
                draw,
                lines,
                font=font,
                frame_width=frame.shape[1],
                frame_height=frame.shape[0],
                anchor=anchor,
            )

    return np.ascontiguousarray(image)


def _draw_corner_lines(draw, lines: list[str], *, font, frame_width: int, frame_height: int, anchor: str) -> None:
    if not lines:
        return

    font_size = getattr(font, "size", 24)
    padding_x = max(14, font_size // 2)
    padding_y = max(10, font_size // 3)
    line_gap = max(6, font_size // 5)

    line_boxes = [draw.textbbox((0, 0), line, font=font) for line in lines]
    text_width = max((box[2] - box[0]) for box in line_boxes)
    text_height = sum((box[3] - box[1]) for box in line_boxes) + line_gap * (len(lines) - 1)
    box_width = text_width + padding_x * 2
    box_height = text_height + padding_y * 2

    if anchor == "bottom_right":
        box = (
            max(12, frame_width - box_width - 12),
            max(12, frame_height - box_height - 12),
            frame_width - 12,
            frame_height - 12,
        )
    elif anchor == "bottom_left":
        box = (
            12,
            max(12, frame_height - box_height - 12),
            12 + box_width,
            frame_height - 12,
        )
    elif anchor == "top_right":
        box = (
            max(12, frame_width - box_width - 12),
            12,
            frame_width - 12,
            12 + box_height,
        )
    else:
        box = (12, 12, 12 + box_width, 12 + box_height)

    draw.rounded_rectangle(box, radius=10, fill=(0, 0, 0, 160), outline=(255, 255, 255, 64))

    y = box[1] + padding_y
    for line, bbox in zip(lines, line_boxes, strict=False):
        draw.text((box[0] + padding_x, y), line, font=font, fill=(255, 255, 255, 255))
        y += (bbox[3] - bbox[1]) + line_gap


def _draw_boxes(draw, boxes: list[OverlayBox], *, font, frame_width: int, frame_height: int) -> None:
    for box in boxes:
        left = max(0, min(frame_width - 1, box.x))
        top = max(0, min(frame_height - 1, box.y))
        right = max(left + 1, min(frame_width, box.x + box.width))
        bottom = max(top + 1, min(frame_height, box.y + box.height))

        if box.fill is not None:
            draw.rectangle((left, top, right, bottom), fill=box.fill)
        draw.rectangle((left, top, right, bottom), outline=box.outline, width=4)

        if not box.label:
            continue

        label_bbox = draw.textbbox((0, 0), box.label, font=font)
        label_width = label_bbox[2] - label_bbox[0]
        label_height = label_bbox[3] - label_bbox[1]
        label_left = left
        label_top = max(0, top - label_height - 12)
        label_right = min(frame_width, label_left + label_width + 16)
        label_bottom = min(frame_height, label_top + label_height + 10)
        draw.rounded_rectangle(
            (label_left, label_top, label_right, label_bottom),
            radius=8,
            fill=(0, 0, 0, 170),
            outline=box.outline,
        )
        draw.text((label_left + 8, label_top + 4), box.label, font=font, fill=(255, 255, 255, 255))


def _load_overlay_font(size: int):
    from PIL import ImageFont

    font_candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in font_candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()
