from __future__ import annotations

import mmap
import os
import sys
from dataclasses import dataclass, field
from typing import Any

from .base import FrameSink


@dataclass(slots=True)
class FramebufferSink(FrameSink):
    """Render frames directly to a Linux framebuffer device (e.g. JT3.5TR on /dev/fb0).

    The sink writes raw RGB565 little-endian pixels to a memory-mapped fbdev,
    bypassing SDL/pygame entirely. The framebuffer's geometry, line stride and
    pixel format are auto-detected via the FBIOGET_VSCREENINFO / FBIOGET_FSCREENINFO
    ioctls. Currently only the common 16 bpp RGB565 layout is supported, which
    matches every fbtft-driven SPI panel I've seen.

    Failures during ``start`` (device missing, wrong pixel format, permission
    denied) are reported but do not raise — the pipeline simply skips this sink.
    """

    name: str = "framebuffer"
    target_fps: float = 30.0
    target_size: tuple[int, int] | None = (480, 320)
    fbdev: str = "/dev/fb0"
    # Logical render resolution. If the panel's actual geometry differs we
    # honour the panel and the configured width/height become irrelevant.
    width: int = 480
    height: int = 320
    _fb: Any = field(init=False, default=None, repr=False)
    _mmap: Any = field(init=False, default=None, repr=False)
    _line_length: int = field(init=False, default=0, repr=False)
    _bytes_per_pixel: int = field(init=False, default=0, repr=False)
    _disabled_reason: str | None = field(init=False, default=None, repr=False)

    def start(self) -> None:
        if not os.path.exists(self.fbdev):
            self._disabled_reason = f"{self.fbdev} not present"
            print(f"Local display disabled: {self._disabled_reason}", file=sys.stderr)
            return

        try:
            geometry = _read_fb_geometry(self.fbdev)
        except Exception as exc:
            self._disabled_reason = f"could not query {self.fbdev}: {exc}"
            print(f"Local display disabled: {self._disabled_reason}", file=sys.stderr)
            return

        if geometry.bits_per_pixel != 16:
            self._disabled_reason = (
                f"unsupported framebuffer depth {geometry.bits_per_pixel} bpp on {self.fbdev}; "
                "only 16-bit RGB565 is implemented"
            )
            print(f"Local display disabled: {self._disabled_reason}", file=sys.stderr)
            return

        try:
            fd = os.open(self.fbdev, os.O_RDWR)
            buffer = mmap.mmap(
                fd,
                geometry.smem_len,
                mmap.MAP_SHARED,
                mmap.PROT_WRITE | mmap.PROT_READ,
            )
        except Exception as exc:
            self._disabled_reason = f"could not mmap {self.fbdev}: {exc}"
            print(f"Local display disabled: {self._disabled_reason}", file=sys.stderr)
            return

        self._fb = fd
        self._mmap = buffer
        self.width = geometry.xres
        self.height = geometry.yres
        self.target_size = (geometry.xres, geometry.yres)
        self._line_length = geometry.line_length
        self._bytes_per_pixel = geometry.bits_per_pixel // 8
        print(
            f"Local display active on {self.fbdev}: "
            f"{self.width}x{self.height} {geometry.bits_per_pixel}bpp, "
            f"stride={self._line_length} bytes."
        )

    def consume(self, frame: Any) -> None:
        if self._mmap is None:
            return
        try:
            payload = _rgb_to_rgb565_bytes(frame, self._line_length)
            self._mmap.seek(0)
            self._mmap.write(payload)
        except Exception as exc:
            print(f"Local display render error: {exc}", file=sys.stderr)

    def close(self) -> None:
        if self._mmap is not None:
            try:
                self._mmap.close()
            except Exception:
                pass
            self._mmap = None
        if self._fb is not None:
            try:
                os.close(self._fb)
            except Exception:
                pass
            self._fb = None

@dataclass(slots=True)
class _FbGeometry:
    xres: int
    yres: int
    bits_per_pixel: int
    line_length: int
    smem_len: int


def _read_fb_geometry(fbdev: str) -> _FbGeometry:
    """Return panel geometry by parsing /sys/class/graphics/<fb>/.

    Avoids ioctl/struct fcntl wrangling — fbtft (and every mainline fbdev) exposes
    these attributes via sysfs in a stable text format.
    """
    name = os.path.basename(fbdev)
    sys_root = f"/sys/class/graphics/{name}"
    if not os.path.isdir(sys_root):
        raise RuntimeError(f"{sys_root} not present")

    virtual_size = _read_text(f"{sys_root}/virtual_size")  # "WIDTH,HEIGHT"
    bits_per_pixel = int(_read_text(f"{sys_root}/bits_per_pixel"))
    line_length = int(_read_text(f"{sys_root}/stride"))
    smem_len_path = f"{sys_root}/smem_len"
    if os.path.exists(smem_len_path):
        smem_len = int(_read_text(smem_len_path))
    else:
        # Fall back to file size of the device; fbtft reports the right thing.
        smem_len = line_length * int(virtual_size.split(",")[1])

    width_str, height_str = virtual_size.split(",", 1)
    return _FbGeometry(
        xres=int(width_str),
        yres=int(height_str),
        bits_per_pixel=bits_per_pixel,
        line_length=line_length,
        smem_len=smem_len,
    )


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read().strip()


def _rgb_to_rgb565_bytes(frame, line_length: int) -> bytes:
    """Convert a (H, W, 3) uint8 RGB ndarray to packed RGB565 little-endian bytes.

    If the framebuffer's stride is wider than ``W * 2`` (line padding), the
    returned buffer pads each row to match.
    """
    import numpy as np  # type: ignore

    height, width, _ = frame.shape
    r = frame[:, :, 0].astype(np.uint16)
    g = frame[:, :, 1].astype(np.uint16)
    b = frame[:, :, 2].astype(np.uint16)
    rgb565 = ((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3)

    row_bytes = width * 2
    if line_length == row_bytes:
        return rgb565.astype("<u2").tobytes()

    padded = np.zeros((height, line_length // 2), dtype=np.uint16)
    padded[:, :width] = rgb565
    return padded.astype("<u2").tobytes()
