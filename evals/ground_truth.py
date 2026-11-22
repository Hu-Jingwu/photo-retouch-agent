"""Recover the rotation you applied in Lightroom by aligning RAW -> exported JPEG.

SIFT features + RANSAC similarity transform (rotation + uniform scale + shift).
The rotation part is the ground-truth leveling angle, CCW-positive, same
convention as apply_rotation. Crop shows up only as scale/shift and is ignored.
Caveat: if Upright / perspective transforms were used in Lightroom, a similarity
fit is only approximate; low inlier counts flag that.
"""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from agent_tools.io_utils import read_image, to_gray_u8


def _small_gray(path: str | Path, side: int = 1600) -> tuple[np.ndarray, float, tuple[int, int]]:
    g = to_gray_u8(read_image(path))
    full_hw = g.shape[:2]
    s = side / max(g.shape)
    g = cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return cv2.equalizeHist(g), s, full_hw   # RAW render and graded export differ a lot in tone


def lightroom_rotation(raw_path: str | Path, export_path: str | Path) -> dict:
    """Rotation AND crop you applied in Lightroom.

    crop_rect is your crop expressed in the RAW frame after leveling by your
    angle (same coordinates apply_rotation + apply_crop use), full resolution.
    """
    (a, sa, (H, W)), (b, sb, (He, We)) = _small_gray(raw_path), _small_gray(export_path)
    sift = cv2.SIFT_create(4000)
    k1, d1 = sift.detectAndCompute(a, None)
    k2, d2 = sift.detectAndCompute(b, None)
    pairs = cv2.BFMatcher().knnMatch(d1, d2, k=2)
    good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < 0.75 * n.distance]
    if len(good) < 20:
        raise RuntimeError(f"only {len(good)} feature matches; cannot align")
    p1 = np.float32([k1[m.queryIdx].pt for m in good])
    p2 = np.float32([k2[m.trainIdx].pt for m in good])
    M, inl = cv2.estimateAffinePartial2D(p1, p2, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if M is None:
        raise RuntimeError("RANSAC alignment failed")
    angle = math.degrees(math.atan2(-M[1, 0], M[0, 0]))
    # export corners -> small export coords -> small RAW coords -> full RAW coords
    corners = np.float32([[0, 0], [We, 0], [We, He], [0, He]]) * sb
    inv = cv2.invertAffineTransform(M)
    raw_pts = (corners @ inv[:, :2].T + inv[:, 2]) / sa
    # into the frame leveled by your angle (rotation about the RAW centre, like apply_rotation)
    R = cv2.getRotationMatrix2D((W / 2.0, H / 2.0), angle, 1.0)
    lev = raw_pts @ R[:, :2].T + R[:, 2]
    x0, y0 = lev.min(0)
    x1, y1 = lev.max(0)
    return {
        "crop_rect": {"x": int(round(x0)), "y": int(round(y0)),
                      "w": int(round(x1 - x0)), "h": int(round(y1 - y0))},
        "crop_aspect": round((x1 - x0) / (y1 - y0), 3),
        "angle_deg": round(angle, 3),
        "scale": round(math.hypot(M[0, 0], M[1, 0]), 4),
        "matches": len(good),
        "inliers": int(inl.sum()),
        "reliable": int(inl.sum()) >= 50,
    }
