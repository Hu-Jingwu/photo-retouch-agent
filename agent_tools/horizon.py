"""Horizon leveling tools.

Division of labour (from the architecture board):
    CV estimates tilt; the model decides.

So nothing here decides *whether* to level. `estimate_tilt` measures and
reports several independent candidates plus a debug overlay the multimodal
orchestrator can look at. The orchestrator then calls `apply_rotation` with
the angle it chose (or skips leveling: intentional Dutch angle, sloped
terrain, no reliable lines). `check_level` is the deterministic
"residual tilt" check used by the evaluator after rendering.

Angle convention used everywhere in this module
-----------------------------------------------
    tilt_deg       How much the scene appears rotated on screen,
                   counter-clockwise positive. A horizon that rises to the
                   right has tilt > 0.
    correction_deg Rotation to apply to fix it = -tilt_deg.
                   Counter-clockwise positive, same as cv2.getRotationMatrix2D.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from .io_utils import is_raw, read_image, to_bgr_u8, to_gray_u8, write_image


# --------------------------------------------------------------------------- #
# Line detection
# --------------------------------------------------------------------------- #
@dataclass
class _Segments:
    xy: np.ndarray        # (N, 4) x1, y1, x2, y2 in working-image pixels
    length: np.ndarray    # (N,)
    angle: np.ndarray     # (N,) on-screen angle in degrees, CCW positive, (-90, 90]


def _detect_segments(gray: np.ndarray, min_len: float) -> tuple[_Segments, str]:
    """LSD first (sub-pixel accurate); HoughLinesP fallback."""
    xy = None
    method = "lsd"
    try:
        lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
        lines = lsd.detect(gray)[0]
        if lines is not None and len(lines):
            xy = lines.reshape(-1, 4).astype(np.float64)
    except (cv2.error, AttributeError):
        xy = None
    if xy is None or len(xy) == 0:
        method = "hough"
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 720, threshold=60,
                                minLineLength=int(min_len), maxLineGap=int(min_len * 0.2))
        xy = (lines.reshape(-1, 4).astype(np.float64) if lines is not None
              else np.zeros((0, 4)))

    dx = xy[:, 2] - xy[:, 0]
    dy = xy[:, 3] - xy[:, 1]
    length = np.hypot(dx, dy)
    keep = length >= min_len
    xy, dx, dy, length = xy[keep], dx[keep], dy[keep], length[keep]
    # Image y points down; flip it so angles are CCW-positive as seen on screen.
    angle = np.degrees(np.arctan2(-dy, dx))
    angle = (angle + 90.0) % 180.0 - 90.0          # -> [-90, 90)
    angle[angle == -90.0] = 90.0                    # -> (-90, 90]
    return _Segments(xy, length, angle), method


# --------------------------------------------------------------------------- #
# Robust statistics
# --------------------------------------------------------------------------- #
def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    v, w = values[order], weights[order]
    cdf = np.cumsum(w)
    return float(v[np.searchsorted(cdf, 0.5 * cdf[-1])])


def _robust_perspective_fit(roll, pos, L, thresh=1.0, max_slope=30.0, max_n=150):
    """Fit roll = a + b * pos with a RANSAC over line pairs, then weighted LSQ on inliers.

    Why: with a pitched camera, verticals converge (keystone) and their apparent
    lean grows linearly with horizontal distance from the image centre; the
    line through the centre stays vertical, so the intercept `a` is the true
    camera roll. Same for horizontals vs. vertical position under yaw.
    Includes zero-slope hypotheses, so a scene without perspective still fits.
    """
    order = np.argsort(-L)[:max_n]
    r, p, w = roll[order], pos[order], L[order]
    best, best_score = (float(_weighted_median(r, w)), 0.0), -1.0
    hyps = [(ri, 0.0) for ri in r]
    n = len(r)
    for i in range(n):
        for j in range(i + 1, n):
            dp = p[j] - p[i]
            if abs(dp) < 0.05:
                continue
            b = (r[j] - r[i]) / dp
            if abs(b) <= max_slope:
                hyps.append((r[i] - b * p[i], b))
    for a, b in hyps:
        score = w[np.abs(r - (a + b * p)) <= thresh].sum()
        if score > best_score:
            best, best_score = (a, b), score
    a, b = best
    inl = np.abs(roll - (a + b * pos)) <= thresh
    for _ in range(3):                                   # refine on inliers
        if inl.sum() < 2:
            break
        sw = np.sqrt(L[inl])
        A = np.stack([np.ones(inl.sum()), pos[inl]], 1) * sw[:, None]
        a, b = np.linalg.lstsq(A, roll[inl] * sw, rcond=None)[0]
        inl = np.abs(roll - (a + b * pos)) <= thresh
    return float(a), float(b), inl


def _candidate(roll: np.ndarray, length: np.ndarray, pos: np.ndarray, norm_len: float) -> dict | None:
    """Summarise one family of lines into a tilt candidate.

    pos: each segment's position across the axis where perspective acts
         (x for verticals, y for horizontals), normalised to [-0.5, 0.5].
    """
    if len(roll) == 0:
        return None
    if len(roll) >= 4 and np.ptp(pos) >= 0.25:
        tilt, slope, inlier = _robust_perspective_fit(roll, pos, length)
        method = "perspective_fit"
    else:
        tilt = _weighted_median(roll, length)
        mad0 = _weighted_median(np.abs(roll - tilt), length)
        inlier = np.abs(roll - tilt) <= max(3.0 * mad0, 0.5)
        tilt, slope, method = _weighted_median(roll[inlier], length[inlier]), 0.0, "median"
    resid = np.abs(roll - (tilt + slope * pos))[inlier]
    mad = _weighted_median(resid, length[inlier]) if inlier.any() else 99.0
    coverage = float(length[inlier].sum() / norm_len)
    inlier_frac = float(length[inlier].sum() / length.sum())
    # Heuristic confidence in [0, 1]: enough line length, tight residuals, most lines agree.
    confidence = (1.0 - math.exp(-coverage / 0.75)) * math.exp(-mad / 1.0) * inlier_frac ** 0.5
    return {
        "tilt_deg": round(float(tilt), 3),
        "method": method,
        # degrees of extra lean from one image edge to the other; large = strong perspective
        "perspective_slope_deg": round(float(slope), 2),
        "spread_deg": round(float(mad), 3),
        "num_lines": int(inlier.sum()),
        "inlier_fraction": round(inlier_frac, 3),
        "coverage": round(coverage, 3),
        "confidence": round(confidence, 3),
    }


def _dominant_horizon_line(seg: _Segments, idx: np.ndarray, w: int, h: int) -> dict | None:
    """Find the longest set of collinear near-horizontal segments (sea / skyline horizon).

    Segments are binned by (angle, offset of the line through the image centre);
    the bin with the largest total length wins.
    """
    if len(idx) == 0:
        return None
    xy, ang, L = seg.xy[idx], seg.angle[idx], seg.length[idx]
    cx, cy = w / 2.0, h / 2.0
    mx = (xy[:, 0] + xy[:, 2]) / 2.0
    my = (xy[:, 1] + xy[:, 3]) / 2.0
    # y where each segment's infinite line crosses x = cx (image coords, y down)
    y_at_c = my + (mx - cx) * np.tan(np.radians(ang))
    a_bin = np.round(ang / 0.5).astype(int)
    r_bin = np.round(y_at_c / max(h * 0.015, 2.0)).astype(int)
    best_key, best_len = None, 0.0
    totals: dict[tuple[int, int], float] = {}
    for k_a, k_r, l in zip(a_bin, r_bin, L):
        # include neighbouring bins so a line split across a bin edge still counts
        for da in (-1, 0, 1):
            for dr in (-1, 0, 1):
                key = (k_a + da, k_r + dr)
                totals[key] = totals.get(key, 0.0) + l
    for key, tot in totals.items():
        if tot > best_len:
            best_key, best_len = key, tot
    assert best_key is not None
    members = (np.abs(a_bin - best_key[0]) <= 1) & (np.abs(r_bin - best_key[1]) <= 1)
    tilt = float(np.average(ang[members], weights=L[members]))
    y_c = float(np.average(y_at_c[members], weights=L[members]))
    width_coverage = float(L[members].sum() / w)
    return {
        "tilt_deg": round(tilt, 3),
        "y_at_center_frac": round(y_c / h, 3),   # 0 = top, 1 = bottom
        "width_coverage": round(width_coverage, 3),
        "num_segments": int(members.sum()),
        "confidence": round(1.0 - math.exp(-width_coverage / 0.4), 3),
        "_members": np.asarray(idx)[members],    # stripped before returning
    }


# --------------------------------------------------------------------------- #
# Tool 1: estimate_tilt
# --------------------------------------------------------------------------- #
def estimate_tilt(
    image_path: str,
    max_tilt_deg: float = 20.0,
    region: dict | None = None,
    work_size: int = 1024,
    overlay_path: str | None = None,
) -> dict:
    """Measure camera roll from straight lines. Returns candidates, never a decision.

    region: optional {"x","y","w","h"} in full-resolution pixels to analyse only
            part of the frame (e.g. the valid area after a rotation).
    overlay_path: where to save the debug overlay PNG; defaults next to the image.
    """
    img = read_image(image_path)
    full_h, full_w = img.shape[:2]
    if region:
        x, y, rw, rh = (int(region[k]) for k in ("x", "y", "w", "h"))
        img = img[y:y + rh, x:x + rw]
    gray = to_gray_u8(img)
    h0, w0 = gray.shape
    scale = work_size / max(h0, w0)
    if scale < 1.0:
        gray = cv2.resize(gray, (round(w0 * scale), round(h0 * scale)), interpolation=cv2.INTER_AREA)
    h, w = gray.shape
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    seg, method = _detect_segments(gray, min_len=0.04 * max(h, w))

    horiz = np.where(np.abs(seg.angle) <= max_tilt_deg)[0]
    vert = np.where(np.abs(np.abs(seg.angle) - 90.0) <= max_tilt_deg)[0]
    # A vertical edge rotated CCW by t has angle 90 + t  ->  t = angle -/+ 90
    v_roll = np.where(seg.angle[vert] > 0, seg.angle[vert] - 90.0, seg.angle[vert] + 90.0)

    # Segment positions relative to the FULL image's optical centre, not the crop's:
    # the keystone fit's intercept is only the camera roll at the principal point.
    s_eff = min(scale, 1.0)
    off_x, off_y = (int(region["x"]), int(region["y"])) if region else (0, 0)
    mid_x = (((seg.xy[:, 0] + seg.xy[:, 2]) / 2.0) / s_eff + off_x - full_w / 2.0) / full_w
    mid_y = (((seg.xy[:, 1] + seg.xy[:, 3]) / 2.0) / s_eff + off_y - full_h / 2.0) / full_h
    cand_h = _candidate(seg.angle[horiz], seg.length[horiz], mid_y[horiz], norm_len=w)
    cand_v = _candidate(v_roll, seg.length[vert], mid_x[vert], norm_len=h)
    cand_line = _dominant_horizon_line(seg, horiz, w, h)
    line_members = cand_line.pop("_members") if cand_line else np.array([], int)

    # ---- a suggestion, clearly labelled as such; the orchestrator decides ----
    flags: list[str] = []
    pool = []
    if cand_line and cand_line["width_coverage"] >= 0.5:
        # A long straight edge is only a horizon if the other horizontals agree with it.
        # In a perspective scene (building shot at an angle) it is usually a receding edge.
        agrees = cand_h is not None and (
            cand_h["spread_deg"] < 1.0
            and abs(cand_h["perspective_slope_deg"]) < 2.0
            and abs(cand_h["tilt_deg"] - cand_line["tilt_deg"]) < 1.0)
        if agrees:
            pool.append(("horizon_line", cand_line["tilt_deg"], cand_line["confidence"] * 1.2))
        else:
            flags.append("dominant_line_may_be_perspective")
            pool.append(("horizon_line", cand_line["tilt_deg"], cand_line["confidence"] * 0.2))
    # Perspective penalty. The keystone fit for verticals is principled (the vertical
    # through the image centre stays vertical under pitch), so it is penalised gently.
    # Horizontals under yaw converge to a side vanishing point; their fit is only
    # right if that point sits at the image centre's height, so they are penalised hard.
    if cand_h:
        pool.append(("horizontal_lines", cand_h["tilt_deg"],
                     cand_h["confidence"] * math.exp(-abs(cand_h["perspective_slope_deg"]) / 5.0)))
    if cand_v:
        pool.append(("vertical_lines", cand_v["tilt_deg"],
                     cand_v["confidence"] * math.exp(-abs(cand_v["perspective_slope_deg"]) / 15.0)))
    if any(c and abs(c["perspective_slope_deg"]) > 2.0 for c in (cand_h, cand_v)):
        flags.append("strong_perspective")
    pool = [p for p in pool if p[2] > 0.05]

    if not pool:
        flags.append("no_reliable_lines")
        suggested_tilt, basis, overall_conf = 0.0, "none", 0.0
    else:
        basis, suggested_tilt, overall_conf = max(pool, key=lambda p: p[2])
        overall_conf = min(overall_conf, 1.0)
    if cand_h and cand_v and abs(cand_h["tilt_deg"] - cand_v["tilt_deg"]) > 1.0:
        flags.append("horizontal_vertical_disagree")
    if any(abs(c["tilt_deg"]) > max_tilt_deg * 0.9 for c in (cand_h, cand_v) if c):
        flags.append("near_search_limit")
    if abs(suggested_tilt) < 0.2 and overall_conf > 0.3:
        flags.append("already_level")

    # ---- debug overlay for the multimodal model ----
    if overlay_path is None:
        p = Path(image_path)
        overlay_path = str(p.with_name(p.stem + "_tilt_overlay.png"))
    _draw_overlay(img, scale, seg, horiz, vert, line_members, overlay_path)

    return {
        "image_size": {"w": full_w, "h": full_h},
        "analysed_region": region,
        "detector": method,
        "convention": "tilt_deg: CCW-positive apparent rotation. correction_deg = -tilt_deg, "
                      "pass it to apply_rotation as angle_deg.",
        "candidates": {
            "horizon_line": cand_line,       # single long collinear edge (sea, skyline)
            "horizontal_lines": cand_h,      # all near-horizontal structure
            "vertical_lines": cand_v,        # buildings, poles, door frames
        },
        "suggestion": {
            "basis": basis,
            "tilt_deg": round(float(suggested_tilt), 3),
            "correction_deg": round(-float(suggested_tilt), 3),
            "confidence": round(float(overall_conf), 3),
        },
        "flags": flags,
        "overlay_path": overlay_path,
    }


def _draw_overlay(img, scale, seg, horiz, vert, line_members, out_path):
    vis = to_bgr_u8(img)
    if scale < 1.0:
        vis = cv2.resize(vis, (round(vis.shape[1] * scale), round(vis.shape[0] * scale)),
                         interpolation=cv2.INTER_AREA)
    h, w = vis.shape[:2]
    # faint level reference grid
    for frac in (0.25, 0.5, 0.75):
        y = int(h * frac)
        cv2.line(vis, (0, y), (w, y), (200, 200, 200), 1, cv2.LINE_AA)
    t = max(1, round(max(h, w) / 600))
    for i in horiz:
        x1, y1, x2, y2 = seg.xy[i].astype(int)
        cv2.line(vis, (x1, y1), (x2, y2), (0, 220, 0), t, cv2.LINE_AA)        # green
    for i in vert:
        x1, y1, x2, y2 = seg.xy[i].astype(int)
        cv2.line(vis, (x1, y1), (x2, y2), (255, 140, 0), t, cv2.LINE_AA)      # blue
    for i in line_members:
        x1, y1, x2, y2 = seg.xy[i].astype(int)
        cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255), t * 3, cv2.LINE_AA)    # red
    write_image(out_path, vis)


# --------------------------------------------------------------------------- #
# Tool 2: apply_rotation
# --------------------------------------------------------------------------- #
def largest_rotated_rect(w: float, h: float, angle_deg: float) -> tuple[float, float]:
    """Largest axis-aligned rectangle inside a w x h image rotated by angle_deg."""
    if w <= 0 or h <= 0:
        return 0.0, 0.0
    a = math.radians(angle_deg)
    sin_a, cos_a = abs(math.sin(a)), abs(math.cos(a))
    long_side, short_side = (w, h) if w >= h else (h, w)
    if short_side <= 2.0 * sin_a * cos_a * long_side or abs(sin_a - cos_a) < 1e-10:
        x = 0.5 * short_side
        wr, hr = (x / sin_a, x / cos_a) if w >= h else (x / cos_a, x / sin_a)
    else:
        cos_2a = cos_a * cos_a - sin_a * sin_a
        wr, hr = (w * cos_a - h * sin_a) / cos_2a, (h * cos_a - w * sin_a) / cos_2a
    return wr, hr


def apply_rotation(
    image_path: str,
    angle_deg: float,
    output_path: str | None = None,
    crop: Literal["none", "max_rect"] = "none",
    border: Literal["transparent", "reflect", "black"] = "transparent",
) -> dict:
    """Rotate the full-resolution image by angle_deg (CCW positive) about its centre.

    crop="none" keeps the original canvas and returns `valid_rect`, so the
    composition/auto-crop tool can choose the final crop inside it (the board
    says: crop after leveling, on the leveled frame).
    crop="max_rect" crops to the largest clean rectangle immediately.
    """
    if abs(angle_deg) > 45:
        raise ValueError("angle_deg must be within +/-45; larger values are orientation, not leveling.")
    img = read_image(image_path)
    h, w = img.shape[:2]

    if border == "transparent" and crop == "none":
        alpha_max = 65535 if img.dtype == np.uint16 else 255
        alpha = np.full((h, w, 1), alpha_max, dtype=img.dtype)
        img = np.concatenate([img, alpha], axis=2)
    border_mode = {"transparent": cv2.BORDER_CONSTANT, "black": cv2.BORDER_CONSTANT,
                   "reflect": cv2.BORDER_REFLECT_101}[border]

    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle_deg, 1.0)
    out = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LANCZOS4,
                         borderMode=border_mode, borderValue=(0, 0, 0, 0))

    wr, hr = largest_rotated_rect(w, h, angle_deg)
    # shave 1 px for interpolation fringe at the rotated edge
    rw, rh = max(int(math.floor(wr)) - 2, 1), max(int(math.floor(hr)) - 2, 1)
    rx, ry = (w - rw) // 2, (h - rh) // 2
    valid_rect = {"x": rx, "y": ry, "w": rw, "h": rh}

    if crop == "max_rect":
        out = out[ry:ry + rh, rx:rx + rw]

    src = Path(image_path)
    if output_path is None:
        has_alpha = out.ndim == 3 and out.shape[2] == 4
        if is_raw(src):
            ext = ".tif"            # RAW can't be written back; keep 16-bit in TIFF
        elif has_alpha and src.suffix.lower() in (".jpg", ".jpeg"):
            ext = ".png"
        else:
            ext = src.suffix
        output_path = str(src.with_name(f"{src.stem}_leveled{ext}"))
    write_image(output_path, out)

    return {
        "output_path": output_path,
        "angle_deg": angle_deg,
        "output_size": {"w": int(out.shape[1]), "h": int(out.shape[0])},
        "valid_rect": valid_rect if crop == "none" else {"x": 0, "y": 0, "w": rw, "h": rh},
        "kept_area_fraction": round(rw * rh / (w * h), 4),
        "cropped": crop == "max_rect",
        "note": "EXIF/ICC metadata is not carried over by OpenCV; copy it with exiftool if needed.",
    }


# --------------------------------------------------------------------------- #
# Tool 3: check_level  (deterministic evaluator check: residual tilt)
# --------------------------------------------------------------------------- #
def check_level(image_path: str, tolerance_deg: float = 0.3,
                region: dict | None = None, min_confidence: float = 0.3) -> dict:
    """Pass/fail on residual tilt after editing. 'inconclusive' when lines are too weak."""
    est = estimate_tilt(image_path, max_tilt_deg=10.0, region=region)
    s = est["suggestion"]
    if s["confidence"] < min_confidence:
        status = "inconclusive"
    else:
        status = "pass" if abs(s["tilt_deg"]) <= tolerance_deg else "fail"
    return {
        "status": status,
        "residual_tilt_deg": s["tilt_deg"],
        "confidence": s["confidence"],
        "basis": s["basis"],
        "tolerance_deg": tolerance_deg,
        "flags": est["flags"],
        "overlay_path": est["overlay_path"],
    }
