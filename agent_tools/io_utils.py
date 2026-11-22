"""Image I/O helpers shared by all agent tools.

cv2.imread / cv2.imwrite fail silently on Windows paths that contain
non-ASCII characters (e.g. Chinese folder names), so we go through
numpy + imdecode / imencode instead.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def read_image(path: str | Path) -> np.ndarray:
    """Read an image at full bit depth as BGR (3 channels).

    - 8-bit and 16-bit images keep their depth.
    - Grayscale is promoted to BGR so downstream code sees one layout.
    - EXIF orientation is applied by OpenCV's decoder (flags below do not
      include IMREAD_IGNORE_ORIENTATION), so phone JPEGs come out upright.
    - RAW files (.NEF/.CR3/.ARW...) are demosaiced with rawpy into 16-bit BGR
      (camera white balance, sRGB gamma, camera orientation applied).
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    if is_raw(path):
        return _read_raw(path)
    buf = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
    if img is None:
        raise ValueError(f"OpenCV could not decode: {path}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


RAW_EXTS = {".nef", ".nrw", ".cr2", ".cr3", ".arw", ".raf", ".rw2", ".orf", ".dng", ".pef", ".srw"}


def is_raw(path: str | Path) -> bool:
    return Path(path).suffix.lower() in RAW_EXTS


def _read_raw(path: Path) -> np.ndarray:
    import rawpy  # optional dependency, only needed for RAW input

    # rawpy/LibRaw opens files through the C runtime, which can choke on non-ASCII
    # Windows paths; reading the bytes ourselves avoids that.
    with open(path, "rb") as fh, rawpy.imread(fh) as raw:
        rgb = raw.postprocess(use_camera_wb=True, output_bps=16)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def write_image(path: str | Path, img: np.ndarray, jpeg_quality: int = 95) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower()
    params: list[int] = []
    if ext in (".jpg", ".jpeg"):
        if img.dtype != np.uint8:
            raise ValueError("JPEG can only store 8-bit images; use .png or .tif for 16-bit.")
        if img.ndim == 3 and img.shape[2] == 4:
            raise ValueError("JPEG has no alpha channel; use .png or .tif.")
        params = [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)]
    ok, enc = cv2.imencode(ext, img, params)
    if not ok:
        raise ValueError(f"OpenCV could not encode {ext}")
    enc.tofile(str(path))
    return path


def to_gray_u8(img: np.ndarray) -> np.ndarray:
    """Convert any BGR/BGRA image of any depth to 8-bit grayscale (for analysis only)."""
    if img.dtype == np.uint16:
        img = (img / 257.0).astype(np.uint8)
    elif img.dtype in (np.float32, np.float64):
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    if img.ndim == 3:
        code = cv2.COLOR_BGRA2GRAY if img.shape[2] == 4 else cv2.COLOR_BGR2GRAY
        img = cv2.cvtColor(img, code)
    return img


def to_bgr_u8(img: np.ndarray) -> np.ndarray:
    """8-bit BGR copy, used for debug overlays."""
    if img.dtype == np.uint16:
        img = (img / 257.0).astype(np.uint8)
    elif img.dtype in (np.float32, np.float64):
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img.copy()
