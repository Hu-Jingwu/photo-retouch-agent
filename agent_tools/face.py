"""Face-part retouch: the tools of the face sub-agent.

Modelled on the 人像美化 panel of PixCake (see docs/pixcake_face_panel.md): blemishes,
dark circles, eye bags, shine, nasolabial folds, skin smoothing/tone, face shape (slim,
small face, V-face, cheekbones, jaw, chin), eyes (size, brighten, iris, whites), nose
(size, wings) and teeth. Everything is driven by the 478 MediaPipe face landmarks.

Design rules
- Every edit is local to a landmark-defined region with soft edges inside the face,
  so nothing spills onto the background (no halo).
- Shape edits are small smooth warps; their size is capped and measured
  (identity guard: max shift <= 6% of face width, eyes <= +20% area, nose <= 15%).
- Colour edits never change someone's skin tone: whitening/luminous are small lifts;
  eye whites and teeth are capped well below paper-white.
"""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from . import color as C
from . import people as P
from .io_utils import write_image

# ---- landmark groups (MediaPipe face mesh, 478 points incl. iris) -------------------
R_EYE, L_EYE = P.RIGHT_EYE, P.LEFT_EYE                   # subject's right eye = image left
R_LOWER_LID = [33, 7, 163, 144, 145, 153, 154, 155, 133]
L_LOWER_LID = [263, 249, 390, 373, 374, 380, 381, 382, 362]
R_IRIS, L_IRIS = [469, 470, 471, 472], [474, 475, 476, 477]
R_IRIS_C, L_IRIS_C = 468, 473
INNER_LIPS = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]
NOSE_TIP, NOSE_WING_R, NOSE_WING_L = 1, 129, 358
MOUTH_R, MOUTH_L = 61, 291
CHEEK_R, CHEEK_L = 205, 425
CHEEKBONE_R, CHEEKBONE_L = 234, 454
JAW_R, JAW_L = [172, 136, 150, 149, 176], [397, 365, 379, 378, 400]
CHIN = [152, 148, 377, 176, 400]

# ---- warp strengths ----------------------------------------------------------------------
# Strength of each shape edit at slider 100. Calibrated so that PixCake's 写真 slider values
# reproduce the shape change PixCake's own export actually makes (measured from landmarks
# on DSC_6390: eyes x1.07 area, nose x0.974, mouth x0.994, jaw x0.982 width) - see
# evals/calibrate_pixcake.py. Uncalibrated ones keep conservative hand-set values.
GAIN = {"face_slim": 0.038, "jaw": 0.032, "eye_size": 0.098, "nose_size": 0.045, "nose_wings": 0.045,
        "mouth_size": 0.01}   # calibrated 2026-11-22 on DSC_6390 (was 0.07/0.06/0.22/0.18/0.18/0.12: ~2-4x too strong)

# ---- parameters ------------------------------------------------------------------------
UNSIGNED = ("acne_spots", "dark_circles", "eye_bags", "oil_shine", "nasolabial", "smoothing",
            "texture_keep", "luminous", "whitening", "rosy", "tone_unify", "small_face", "jaw",
            "eye_brighten", "iris", "eye_whites", "nose_wings", "teeth_whiten", "lipstick", "contour")
SIGNED = ("chin_length", "nose_size", "mouth_size")
PAIRED = ("face_slim", "v_face", "cheekbone", "eye_size")      # number or [right, left] (subject's)
PAIRED_SIGNED = ("eye_size",)
ALL_KEYS = UNSIGNED + SIGNED + PAIRED

PRESETS = {
    # PixCake's own 女 defaults translated to our scale (see docs/pixcake_face_panel.md)
    "pixcake_female_default": {"acne_spots": 100, "smoothing": 60, "texture_keep": 40, "luminous": 20,
                               "tone_unify": 40, "face_slim": 25, "small_face": 20, "eye_size": 25,
                               "eye_brighten": 40, "iris": 60, "eye_whites": 30, "teeth_whiten": 40},
    # subtle, no reshaping: safe for everyone
    "natural": {"acne_spots": 80, "dark_circles": 40, "oil_shine": 40, "smoothing": 35, "texture_keep": 60,
                "tone_unify": 30, "eye_brighten": 25, "eye_whites": 20, "teeth_whiten": 25},
    "male_default": {"acne_spots": 80, "dark_circles": 40, "oil_shine": 50, "smoothing": 25, "texture_keep": 70,
                     "tone_unify": 25, "eye_brighten": 20, "eye_whites": 20, "teeth_whiten": 25},
}

# PixCake's 写真 one-click preset (人像 part), read from the app on DSC_6390, 2026-11-22.
# Its own slider values, mapped to our parameters (see docs/pixcake_face_panel.md).
PIXCAKE_XIEZHEN = {
    "acne_spots": 100, "oil_shine": 30, "dark_circles": 75, "nasolabial": 50,
    "smoothing": 70, "luminous": 20, "whitening": 10, "tone_unify": 50,
    "face_slim": 20, "jaw": 40, "small_face": 20, "eye_size": 40,
    "nose_size": 35, "nose_wings": 30, "mouth_size": 30,
    "eye_brighten": 40, "iris": 70, "eye_whites": 30, "teeth_whiten": 40,
}


def _scaled(preset: dict, s: float, keep=("texture_keep",)) -> dict:
    out = {k: (v if k in keep else round(v * s, 1)) for k, v in preset.items()}
    out["texture_keep"] = 70        # scaling down must never make skin MORE plastic: keep pores
    return out


PRESETS["pixcake_xiezhen_100"] = _scaled(PIXCAKE_XIEZHEN, 1.0)     # the full 网红 look, for reference
PRESETS["pixcake_xiezhen_70"] = _scaled(PIXCAKE_XIEZHEN, 0.7)      # same direction at 70%: the user's taste


# lipstick shades as target Lab (L*, a*, b*); "natural" = the lips' own hue, a little richer
LIP_SHADES = {"natural": None, "rose": (52.0, 40.0, 14.0), "coral": (60.0, 42.0, 30.0),
              "red": (46.0, 55.0, 32.0), "berry": (40.0, 42.0, 6.0)}
OUTER_LIPS = P.LIPS
# contour (立体) landmarks: highlights on the T-zone and cheek tops, shadows under the cheekbones and jaw
NOSE_BRIDGE = [168, 6, 197, 195, 5]
FOREHEAD_C, CHIN_C = 151, 175
CHEEK_TOP_R, CHEEK_TOP_L = 50, 280
HOLLOW_R, HOLLOW_L = [123, 147, 187], [352, 376, 411]


def normalize(params: dict | None, preset: str | None = None) -> dict:
    out: dict = {k: 0.0 for k in UNSIGNED + SIGNED}
    out.update({k: [0.0, 0.0] for k in PAIRED})
    out["texture_keep"] = 50.0
    out["lip_shade"] = "natural"
    if preset and preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r}; use one of {list(PRESETS)}")
    src = dict(PRESETS[preset]) if preset else {}
    src.update(params or {})
    for k, v in src.items():
        if k == "lip_shade":
            if v not in LIP_SHADES:
                raise ValueError(f"lip_shade must be one of {list(LIP_SHADES)}")
            out[k] = v
            continue
        if k not in ALL_KEYS:
            raise ValueError(f"unknown face parameter {k!r}; valid: {list(ALL_KEYS)}")
        lo = -100 if (k in SIGNED or k in PAIRED_SIGNED) else 0
        if k in PAIRED:
            pair = v if isinstance(v, (list, tuple)) else [v, v]
            out[k] = [float(np.clip(float(x), lo, 100)) for x in pair[:2]]
        else:
            out[k] = float(np.clip(float(v), lo, 100))
    return out


# ---- geometry helpers ----------------------------------------------------------------
def _poly_mask(shape, pts, sigma):
    m = np.zeros(shape, np.float32)
    cv2.fillPoly(m, [np.round(pts).astype(np.int32)], 1.0)
    return cv2.GaussianBlur(m, (0, 0), max(0.5, sigma)) if sigma > 0 else m


def _disk_mask(shape, c, r, sigma):
    m = np.zeros(shape, np.float32)
    cv2.circle(m, (int(round(c[0])), int(round(c[1]))), max(1, int(round(r))), 1.0, -1)
    return cv2.GaussianBlur(m, (0, 0), max(0.5, sigma))


def _roi(shape, center, radius):
    h, w = shape
    x0, x1 = int(max(0, center[0] - radius)), int(min(w, center[0] + radius + 1))
    y0, y1 = int(max(0, center[1] - radius)), int(min(h, center[1] + radius + 1))
    return x0, y0, x1, y1


def _radial(dx, dy, center, radius, k):
    """Bulge (k > 0) / pinch (k < 0) around `center` within `radius`: d = (p - c) * k * (1 - (r/R)^2)^2."""
    x0, y0, x1, y1 = _roi(dx.shape, center, radius)
    if x1 <= x0 or y1 <= y0 or k == 0:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    ux, uy = xx - center[0], yy - center[1]
    t = np.clip(1 - (ux * ux + uy * uy) / (radius * radius), 0, 1) ** 2
    dx[y0:y1, x0:x1] += ux * k * t
    dy[y0:y1, x0:x1] += uy * k * t


def _push(dx, dy, points, vecs, sigma):
    """Move points by vecs with smooth Gaussian falloff (normalised so overlaps don't add up)."""
    h, w = dx.shape
    pts = np.asarray(points, np.float32)
    lo = pts.min(0) - 3 * sigma
    hi = pts.max(0) + 3 * sigma
    x0, y0 = int(max(0, lo[0])), int(max(0, lo[1]))
    x1, y1 = int(min(w, hi[0])), int(min(h, hi[1]))
    if x1 <= x0 or y1 <= y0:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    nx = np.zeros_like(xx)
    ny = np.zeros_like(xx)
    den = np.zeros_like(xx)
    for (px, py), (vx, vy) in zip(pts, vecs):
        g = np.exp(-((xx - px) ** 2 + (yy - py) ** 2) / (2 * sigma * sigma))
        nx += g * vx
        ny += g * vy
        den += g
    den = np.maximum(den, 1.0)
    dx[y0:y1, x0:x1] += nx / den
    dy[y0:y1, x0:x1] += ny / den


def _face_geometry(lm):
    fw = float(np.linalg.norm(lm[CHEEKBONE_R] - lm[CHEEKBONE_L]))
    fh = float(np.linalg.norm(lm[10] - lm[152]))
    ew_r = float(np.linalg.norm(lm[33] - lm[133]))
    ew_l = float(np.linalg.norm(lm[263] - lm[362]))
    return fw, fh, ew_r, ew_l


# ---- measurements ----------------------------------------------------------------------
def measure_face(rgb, lm, face_skin) -> dict:
    h, w = rgb.shape[:2]
    lab = C._lab(rgb)
    L = lab[..., 0]
    fw, fh, ew_r, ew_l = _face_geometry(lm)
    out = {"face_width_px": round(fw, 1), "face_aspect_h_w": round(fh / max(fw, 1), 3)}
    jaw_w = float(np.linalg.norm(lm[172] - lm[397]))
    out["jaw_to_cheek_ratio"] = round(jaw_w / max(fw, 1), 3)
    for side, eye, (a, b), ew in (("right", R_EYE, (159, 145), ew_r), ("left", L_EYE, (386, 374), ew_l)):
        out[f"eye_open_{side}"] = round(float(np.linalg.norm(lm[a] - lm[b])) / max(ew, 1), 3)
        out[f"eye_width_rel_{side}"] = round(ew / max(fw, 1), 3)
    out["nose_width_rel"] = round(float(np.linalg.norm(lm[NOSE_WING_R] - lm[NOSE_WING_L])) / max(fw, 1), 3)
    out["mouth_width_rel"] = round(float(np.linalg.norm(lm[MOUTH_R] - lm[MOUTH_L])) / max(fw, 1), 3)
    # skin: under-eye darkness vs cheeks, redness, shine
    sig = fw * 0.02
    under = np.zeros((h, w), np.float32)
    for lid, ew in ((R_LOWER_LID, ew_r), (L_LOWER_LID, ew_l)):
        under = np.maximum(under, _under_eye_mask((h, w), lm, lid, ew, 0))
    cheeks = np.maximum(_disk_mask((h, w), lm[CHEEK_R], fw * 0.08, 0.5), _disk_mask((h, w), lm[CHEEK_L], fw * 0.08, 0.5))
    if under.sum() > 10 and cheeks.sum() > 10:
        out["under_eye_darkness_L"] = round(float(L[cheeks > 0.5].mean() - L[under > 0.5].mean()), 1)
        out["cheek_redness_a"] = round(float(lab[..., 1][cheeks > 0.5].mean()), 1)
    sk = face_skin > 0.6
    if sk.sum() > 50:
        base = cv2.GaussianBlur(L, (0, 0), max(1.0, fw * 0.05))
        out["shine_pct"] = round(float(((L - base > 6) & (L > 60) & sk).sum() / sk.sum() * 100), 2)
        out["skin_L"] = round(float(np.median(L[sk])), 1)
    inner = _poly_mask((h, w), lm[INNER_LIPS], 0)
    if inner.sum() > 20:
        teeth = (inner > 0.5) & (L > 45) & (np.hypot(lab[..., 1], lab[..., 2]) < 35)
        out["teeth_visible"] = bool(teeth.sum() > 0.15 * inner.sum())
        if out["teeth_visible"]:
            out["teeth_L_b"] = [round(float(L[teeth].mean()), 1), round(float(lab[..., 2][teeth].mean()), 1)]
    out["symmetry_eye_size"] = round(min(ew_r, ew_l) / max(ew_r, ew_l, 1), 3)
    lips = _lip_mask((h, w), lm, fw) > 0.5
    if lips.sum() > 20:
        out["lip_chroma"] = round(float(np.hypot(lab[..., 1], lab[..., 2])[lips].mean()), 1)
    if sk.sum() > 50:     # how much light and shadow already shapes the face (flat light -> low)
        low = cv2.GaussianBlur(L, (0, 0), max(1.0, fw * 0.08))
        out["face_relief"] = round(float(low[sk].std()), 1)
    return out


def _under_eye_mask(shape, lm, lid, ew, sigma):
    pts = lm[lid].astype(np.float32)
    lower = pts + np.array([0, 0.42 * ew], np.float32)
    poly = np.concatenate([pts, lower[::-1]], 0)
    return _poly_mask(shape, poly, sigma)


def _lip_mask(shape, lm, fw):
    """Lips only: outer lip contour minus the mouth opening (teeth, tongue), soft-edged."""
    sig = max(0.5, fw * 0.006)
    return np.clip(_poly_mask(shape, lm[OUTER_LIPS], sig) - _poly_mask(shape, lm[INNER_LIPS], sig), 0, 1)


def _contour_masks(shape, lm, fw):
    """(highlight, shadow) masks for 立体, each normalised to a peak of 1."""
    th = max(1, int(fw * 0.04))
    hl = np.zeros(shape, np.float32)
    cv2.polylines(hl, [np.round(lm[NOSE_BRIDGE]).astype(np.int32)], False, 1.0, th)
    for idx, r in ((FOREHEAD_C, 0.09), (CHIN_C, 0.05), (CHEEK_TOP_R, 0.05), (CHEEK_TOP_L, 0.05)):
        cv2.circle(hl, tuple(int(v) for v in np.round(lm[idx])), max(1, int(fw * r)), 1.0, -1)
    sh = np.zeros(shape, np.float32)
    for line in (HOLLOW_R, HOLLOW_L):
        cv2.polylines(sh, [np.round(lm[line]).astype(np.int32)], False, 1.0, max(1, int(fw * 0.06)))
    cx = float(lm[NOSE_TIP, 0])
    jaw = lm[P.JAW].copy()
    jaw[:, 0] += np.sign(cx - jaw[:, 0]) * fw * 0.04          # just inside the jaw line
    cv2.polylines(sh, [np.round(jaw).astype(np.int32)], False, 1.0, max(1, int(fw * 0.05)))
    out = []
    for m in (hl, sh):
        m = cv2.GaussianBlur(m, (0, 0), max(1.0, fw * 0.05))
        out.append(m / max(float(m.max()), 1e-6))
    return out


# ---- rendering -------------------------------------------------------------------------
def render_faces(rgb, ana, params: dict, face_ids=None) -> tuple[np.ndarray, dict]:
    h, w = rgb.shape[:2]
    out = rgb.copy()
    dx = np.zeros((h, w), np.float32)
    dy = np.zeros((h, w), np.float32)
    info = {"faces_edited": [], "identity": []}
    m = ana["masks"]
    people = [(i, p) for i, p in enumerate(ana["people"]) if p["landmarks"] is not None
              and (face_ids is None or i in face_ids)]
    for i, p in people:
        lm = p["landmarks"]
        fw, fh, ew_r, ew_l = _face_geometry(lm)
        sig = max(1.0, fw * 0.025)
        face_region = np.clip(_poly_mask((h, w), lm[P.FACE_OVAL], sig) * 1.0, 0, 1)
        skin = face_region * m["face_skin"] * (1 - m["features"])
        lab = C._lab(out)

        # -- blemishes
        if params["acne_spots"] > 0:
            o2, n = P._blemishes(out, m["face_skin"] * face_region, m["features"], fw, [p])
            out = out + params["acne_spots"] / 100 * (o2 - out)
            info["blemishes_removed"] = info.get("blemishes_removed", 0) + n
            lab = C._lab(out)
        cheeks = np.maximum(_disk_mask((h, w), lm[CHEEK_R], fw * 0.09, sig), _disk_mask((h, w), lm[CHEEK_L], fw * 0.09, sig))
        ref = cheeks > 0.5
        if (params["dark_circles"] > 0 or params["eye_bags"] > 0) and ref.sum() > 10:
            tgt = lab[ref].mean(0)
            for lid, ew in ((R_LOWER_LID, ew_r), (L_LOWER_LID, ew_l)):
                um = _under_eye_mask((h, w), lm, lid, ew, ew * 0.12) * (1 - m["eyes"])
                if params["dark_circles"] > 0:
                    a = params["dark_circles"] / 100 * um
                    lab[..., 0] += 0.8 * a * np.maximum(tgt[0] - lab[..., 0], 0)
                    lab[..., 1] += 0.6 * a * (tgt[1] - lab[..., 1])
                    lab[..., 2] += 0.6 * a * (tgt[2] - lab[..., 2])
                if params["eye_bags"] > 0:
                    r = max(2, int(ew * 0.12))
                    sm = P.guided_filter(lab[..., 0] / 100, lab[..., 0] / 100, r, 0.0004) * 100
                    lab[..., 0] += params["eye_bags"] / 100 * um * (sm - lab[..., 0])
            out = np.clip(cv2.cvtColor(lab, cv2.COLOR_Lab2RGB), 0, 1)
            lab = C._lab(out)
        if params["nasolabial"] > 0:
            band = np.zeros((h, w), np.float32)
            for wing, corner, sgn in ((NOSE_WING_R, MOUTH_R, -1), (NOSE_WING_L, MOUTH_L, 1)):
                a_, b_ = lm[wing], lm[corner]
                off = np.array([sgn * 0.05 * fw, 0], np.float32)
                cv2.line(band, tuple(np.round(a_ + off).astype(int)), tuple(np.round(b_ + off).astype(int)),
                         1.0, max(1, int(fw * 0.05)))
            band = cv2.GaussianBlur(band, (0, 0), sig) * face_region
            Lb = cv2.GaussianBlur(lab[..., 0], (0, 0), max(1.0, fw * 0.05))
            lab[..., 0] += params["nasolabial"] / 100 * band * np.maximum(Lb - lab[..., 0], 0)
            out = np.clip(cv2.cvtColor(lab, cv2.COLOR_Lab2RGB), 0, 1)
            lab = C._lab(out)
        if params["oil_shine"] > 0:
            Lb = cv2.GaussianBlur(lab[..., 0], (0, 0), max(1.0, fw * 0.05))
            spec = np.clip((lab[..., 0] - Lb - 4) / 8, 0, 1) * (lab[..., 0] > 55)
            lab[..., 0] -= params["oil_shine"] / 100 * 0.7 * skin * spec * (lab[..., 0] - Lb)
            out = np.clip(cv2.cvtColor(lab, cv2.COLOR_Lab2RGB), 0, 1)

        # -- skin
        if params["smoothing"] > 0:
            r = max(2, int(fw * 0.045))
            low = np.stack([P.guided_filter(out[..., c], out[..., c], r, 0.0025) for c in range(3)], -1)
            fine = out - cv2.GaussianBlur(out, (0, 0), max(0.6, fw * 0.006))
            sm = low + params["texture_keep"] / 100 * 1.1 * fine
            out = out + params["smoothing"] / 100 * 0.85 * skin[..., None] * (sm - out)
        if params["tone_unify"] > 0:
            out = P._even_tone(out, skin, fw, params["tone_unify"])
        if params["rosy"] > 0:
            lab = C._lab(out)
            lab[..., 1] += cheeks * skin * params["rosy"] / 100 * 5
            out = np.clip(cv2.cvtColor(lab, cv2.COLOR_Lab2RGB), 0, 1)

        # -- makeup: lipstick moves the lips' colour toward a shade; lightness texture is kept
        if params["lipstick"] > 0:
            lips = _lip_mask((h, w), lm, fw)
            sel = lips > 0.5
            if sel.sum() > 20:
                lab = C._lab(out)
                Lm, am, bm = (float(lab[..., c][sel].mean()) for c in range(3))
                shade = LIP_SHADES[params["lip_shade"]]
                if shade is None:                                  # natural: own hue, richer
                    gain = min(1.35, 45.0 / max(np.hypot(am, bm), 1.0))
                    shade = (Lm, am * gain, bm * gain)
                a = 0.65 * params["lipstick"] / 100 * lips
                lab[..., 0] += a * 0.35 * (shade[0] - Lm)          # uniform shift: lip lines stay
                lab[..., 1] += a * (shade[1] - lab[..., 1])
                lab[..., 2] += a * (shade[2] - lab[..., 2])
                out = np.clip(cv2.cvtColor(lab, cv2.COLOR_Lab2RGB), 0, 1)

        # -- contour (立体): wide, low-frequency dodge & burn inside the face skin only
        if params["contour"] > 0:
            hl, sh = _contour_masks((h, w), lm, fw)
            where = face_region * m["face_skin"] * (1 - m["features"])
            ev = params["contour"] / 100 * (0.19 * hl - 0.17 * sh) * where
            out = np.clip(out * np.power(2.0, ev)[..., None], 0, 1)
            info["contour_max_ev"] = max(info.get("contour_max_ev", 0.0), float(np.abs(ev).max()))

        # -- eyes
        r_iris = [float(np.linalg.norm(lm[R_IRIS[0]] - lm[R_IRIS[2]])) / 2, float(np.linalg.norm(lm[L_IRIS[0]] - lm[L_IRIS[2]])) / 2]
        for eye, ic, ri, ew in ((R_EYE, R_IRIS_C, r_iris[0], ew_r), (L_EYE, L_IRIS_C, r_iris[1], ew_l)):
            em = _poly_mask((h, w), lm[eye], ew * 0.04)
            iris = _disk_mask((h, w), lm[ic], max(1.0, ri * 0.95), max(0.5, ri * 0.15)) * em
            sclera = np.clip(em - iris * 1.3, 0, 1)
            if params["eye_brighten"] > 0:
                out = P._brighten(out, em, 0.4 * params["eye_brighten"] / 100)
            lab = C._lab(out)
            if params["eye_whites"] > 0:
                a = params["eye_whites"] / 100 * sclera
                lab[..., 1] *= (1 - 0.6 * a)
                lab[..., 2] *= (1 - 0.6 * a)
                lab[..., 0] = np.minimum(lab[..., 0] + a * 6, np.maximum(lab[..., 0], 88))
            if params["iris"] > 0:
                Li = lab[..., 0]
                blur = cv2.GaussianBlur(Li, (0, 0), max(0.5, ri * 0.3))
                lab[..., 0] = Li + params["iris"] / 100 * 0.6 * iris * (Li - blur)
                lab[..., 1] *= (1 + 0.15 * params["iris"] / 100 * iris)
                lab[..., 2] *= (1 + 0.15 * params["iris"] / 100 * iris)
            out = np.clip(cv2.cvtColor(lab, cv2.COLOR_Lab2RGB), 0, 1)

        # -- teeth
        if params["teeth_whiten"] > 0:
            lab = C._lab(out)
            inner = _poly_mask((h, w), lm[INNER_LIPS], 0)
            teeth = inner * (lab[..., 0] > 40) * (np.hypot(lab[..., 1], lab[..., 2]) < 40)
            teeth = cv2.GaussianBlur(teeth.astype(np.float32), (0, 0), max(0.5, fw * 0.004))
            a = params["teeth_whiten"] / 100 * teeth
            lab[..., 2] -= a * 0.6 * np.maximum(lab[..., 2], 0)
            lab[..., 1] -= a * 0.3 * np.maximum(lab[..., 1], 0)
            lab[..., 0] = np.minimum(lab[..., 0] + a * 8, np.maximum(lab[..., 0], 90))
            out = np.clip(cv2.cvtColor(lab, cv2.COLOR_Lab2RGB), 0, 1)

        # -- shape (warps)
        cx = float(lm[NOSE_TIP, 0])
        fc = lm[P.FACE_OVAL].mean(0)
        dx0, dy0 = dx.copy(), dy.copy()
        fs = params["face_slim"]
        if any(fs):
            pts, vecs = [], []
            for idx in P.JAW:
                side = 0 if lm[idx, 0] < cx else 1          # image-left = subject's right
                pts.append(lm[idx])
                vecs.append(((cx - lm[idx, 0]) * GAIN["face_slim"] * fs[side] / 100, 0.0))
            _push(dx, dy, pts, vecs, fw * 0.12)
        vf = params["v_face"]
        if any(vf):
            pts, vecs = [], []
            for side, idxs in ((0, JAW_R), (1, JAW_L)):
                for idx in idxs:
                    pts.append(lm[idx])
                    vecs.append(((cx - lm[idx, 0]) * 0.10 * vf[side] / 100, -0.02 * fh * vf[side] / 100))
            _push(dx, dy, pts, vecs, fw * 0.10)
        cb = params["cheekbone"]
        if any(cb):
            for side, idx in ((0, CHEEKBONE_R), (1, CHEEKBONE_L)):
                _push(dx, dy, [lm[idx]], [((cx - lm[idx, 0]) * 0.05 * cb[side] / 100, 0.0)], fw * 0.12)
        if params["jaw"] > 0:
            pts = [lm[172], lm[397]]
            vecs = [((cx - lm[172, 0]) * GAIN["jaw"] * params["jaw"] / 100, 0.0),
                    ((cx - lm[397, 0]) * GAIN["jaw"] * params["jaw"] / 100, 0.0)]
            _push(dx, dy, pts, vecs, fw * 0.10)
        if params["chin_length"] != 0:
            _push(dx, dy, [lm[i] for i in CHIN], [(0.0, 0.05 * fh * params["chin_length"] / 100)] * len(CHIN), fw * 0.12)
        if params["small_face"] > 0:
            _radial(dx, dy, fc, fh * 0.75, -0.05 * params["small_face"] / 100)
        es = params["eye_size"]
        for side, ic, ew in ((0, R_IRIS_C, ew_r), (1, L_IRIS_C, ew_l)):
            if es[side]:
                _radial(dx, dy, lm[ic], ew * 0.95, GAIN["eye_size"] * es[side] / 100)
        if params["nose_size"] != 0:
            nc = (lm[NOSE_TIP] + lm[NOSE_WING_R] + lm[NOSE_WING_L]) / 3
            nwid = float(np.linalg.norm(lm[NOSE_WING_R] - lm[NOSE_WING_L]))
            _radial(dx, dy, nc, nwid * 1.2, -GAIN["nose_size"] * params["nose_size"] / 100)
        if params["mouth_size"] != 0:          # + = smaller mouth (PixCake 嘴巴大小 negative)
            mc = (lm[MOUTH_R] + lm[MOUTH_L] + lm[13] + lm[14]) / 4
            mwid = float(np.linalg.norm(lm[MOUTH_R] - lm[MOUTH_L]))
            _radial(dx, dy, mc, mwid * 0.75, -GAIN["mouth_size"] * params["mouth_size"] / 100)
        if params["nose_wings"] > 0:
            nwid = float(np.linalg.norm(lm[NOSE_WING_R] - lm[NOSE_WING_L]))
            ncx = (lm[NOSE_WING_R, 0] + lm[NOSE_WING_L, 0]) / 2
            pts = [lm[NOSE_WING_R], lm[NOSE_WING_L]]
            vecs = [((ncx - lm[NOSE_WING_R, 0]) * GAIN["nose_wings"] * params["nose_wings"] / 100, 0.0),
                    ((ncx - lm[NOSE_WING_L, 0]) * GAIN["nose_wings"] * params["nose_wings"] / 100, 0.0)]
            _push(dx, dy, pts, vecs, nwid * 0.25)

        # identity guard: how far did this face's own points move?
        fdx, fdy = dx - dx0, dy - dy0
        sample = np.round(lm[:468]).astype(int)
        sample[:, 0] = np.clip(sample[:, 0], 0, w - 1)
        sample[:, 1] = np.clip(sample[:, 1], 0, h - 1)
        sh = np.hypot(fdx[sample[:, 1], sample[:, 0]], fdy[sample[:, 1], sample[:, 0]])
        moved = lm[:468] + np.stack([fdx[sample[:, 1], sample[:, 0]], fdy[sample[:, 1], sample[:, 0]]], 1)

        def area(pts):
            return abs(cv2.contourArea(np.round(pts).astype(np.float32)))

        eye_ratio = max(area(moved[R_EYE]) / max(area(lm[R_EYE]), 1), area(moved[L_EYE]) / max(area(lm[L_EYE]), 1))
        nose_ratio = float(np.linalg.norm(moved[NOSE_WING_R] - moved[NOSE_WING_L])) / \
            max(float(np.linalg.norm(lm[NOSE_WING_R] - lm[NOSE_WING_L])), 1)
        mouth_ratio = float(np.linalg.norm(moved[MOUTH_R] - moved[MOUTH_L])) / \
            max(float(np.linalg.norm(lm[MOUTH_R] - lm[MOUTH_L])), 1)
        info["identity"].append({"face": i, "max_shift_frac": round(float(sh.max() / fw), 4),
                                 "eye_area_ratio": round(float(eye_ratio), 3),
                                 "nose_width_ratio": round(nose_ratio, 3),
                                 "mouth_width_ratio": round(mouth_ratio, 3)})
        info["faces_edited"].append(i)

    # luminous / whitening act on ALL visible skin (face and body together, as PixCake's 肤色透亮
    # does): lifting only the face would make it look lit separately from the body
    if people and (params["luminous"] > 0 or params["whitening"] > 0):
        all_skin = m["skin"]
        lab = C._lab(out)
        lift = (params["luminous"] * 4 + params["whitening"] * 6) / 100          # L* points, small
        lab[..., 0] += all_skin * lift * (1 - lab[..., 0] / 100)
        lab[..., 2] -= all_skin * params["whitening"] / 100 * 0.2 * np.maximum(lab[..., 2], 0)
        out = np.clip(cv2.cvtColor(lab, cv2.COLOR_Lab2RGB), 0, 1)

    out = P._apply_warp(out, dx, dy)
    info["_dx"], info["_dy"] = dx, dy            # exact warp field (used by calibration)
    bg = m["person"] < 0.2
    shift = np.hypot(dx, dy)
    info["background_max_shift_px_frac"] = float(shift[bg].max() / w) if bg.any() and shift.any() else 0.0
    return np.clip(out, 0, 1).astype(np.float32), info


def face_checks(before, after, ana, info) -> list[dict]:
    checks = []
    for c in P.retouch_checks(before, after, ana, info):
        if c["rule"] in ("no halo around people", "skin texture kept", "background not warped"):
            checks.append(c)
        elif c["rule"] == "face not lit separately":
            # the main editor's relight usually used most of the 0.8 EV allowance already,
            # so the face specialist may add only a little on top
            ok = c["value"] <= 0.35
            checks.append({**c, "status": "pass" if ok else "fail",
                           "detail": "face brightened beyond the body by THIS edit, EV (<= 0.35): "
                                     "use luminous (all skin) rather than face-only lifts"})

    def add(rule, ok, value, detail):
        checks.append({"rule": rule, "status": "pass" if ok else "fail", "value": value, "detail": detail})

    for idn in info.get("identity", []):
        f = idn["face"]
        add(f"face {f}: identity kept (small shape change)", idn["max_shift_frac"] <= 0.06, idn["max_shift_frac"],
            "max landmark shift / face width (<= 6%)")
        add(f"face {f}: eyes not over-enlarged", 0.8 <= idn["eye_area_ratio"] <= 1.2, idn["eye_area_ratio"],
            "eye area after/before (0.8..1.2)")
        add(f"face {f}: nose change small", 0.85 <= idn["nose_width_ratio"] <= 1.1, idn["nose_width_ratio"],
            "nose width after/before (0.85..1.1)")
        add(f"face {f}: mouth change small", 0.88 <= idn.get("mouth_width_ratio", 1) <= 1.1,
            idn.get("mouth_width_ratio", 1), "mouth width after/before (0.88..1.1)")
    lab = C._lab(after)
    for i, p in enumerate(ana["people"]):
        lm = p["landmarks"]
        if lm is None:
            continue
        inner = _poly_mask(after.shape[:2], lm[INNER_LIPS], 0) > 0.5
        teeth = inner & (lab[..., 0] > 45) & (np.hypot(lab[..., 1], lab[..., 2]) < 35)
        if teeth.sum() > 0.15 * max(inner.sum(), 1) and teeth.sum() > 20:
            tl = float(lab[..., 0][teeth].mean())
            add(f"face {i}: teeth natural (not paper-white)", tl <= 92, round(tl, 1), "teeth L* <= 92")
    for i, p in enumerate(ana["people"]):
        lm = p["landmarks"]
        if lm is None or i not in info.get("faces_edited", []):
            continue
        fw = _face_geometry(lm)[0]
        lab0 = C._lab(before)
        outer = _poly_mask(after.shape[:2], lm[OUTER_LIPS], 0) > 0.5
        lips = _lip_mask(after.shape[:2], lm, fw) > 0.5
        if lips.sum() > 20:
            chroma = float(np.hypot(lab[..., 1], lab[..., 2])[lips].mean())
            add(f"face {i}: lip colour natural", chroma <= 60, round(chroma, 1), "lip chroma C* after (<= 60)")
            k = max(3, int(fw * 0.03)) | 1
            near = cv2.dilate(outer.astype(np.uint8), np.ones((k, k), np.uint8)) > 0
            far = cv2.dilate(outer.astype(np.uint8), np.ones((3 * k, 3 * k), np.uint8)) > 0
            near_ring, far_ring = near & ~outer, far & ~near
            if near_ring.sum() > 20 and far_ring.sum() > 20:
                dab = np.hypot(lab[..., 1] - lab0[..., 1], lab[..., 2] - lab0[..., 2])
                spill = float(dab[near_ring].mean() - dab[far_ring].mean())
                add(f"face {i}: lipstick stays on the lips", spill <= 2.0, round(spill, 2),
                    "colour change just outside the lips minus a bit further out, Δab (<= 2)")
    if "contour_max_ev" in info:
        add("contour subtle", info["contour_max_ev"] <= 0.2, round(info["contour_max_ev"], 3),
            "largest dodge/burn of the 立体 contour, EV (<= 0.2)")
    return checks


# ---- tools ------------------------------------------------------------------------------
def _closeups(images, ana, labels, path, face_ids=None):
    tiles = []
    for i, p in enumerate(ana["people"]):
        if p["landmarks"] is None or (face_ids is not None and i not in face_ids):
            continue
        x, y, bw, bh = p["face"]["box"]
        s = int(max(bw, bh) * 1.25)
        cx, cy = int(x + bw / 2), int(y + bh / 2)
        h, w = images[0].shape[:2]
        x0, y0, x1, y1 = max(0, cx - s), max(0, cy - s), min(w, cx + s), min(h, cy + s)
        for im, lab_ in zip(images, labels):
            crop = im[y0:y1, x0:x1]
            k = 460 / crop.shape[0]
            t = cv2.resize((crop[..., ::-1] * 255).astype(np.uint8), (max(1, round(crop.shape[1] * k)), 460),
                           interpolation=cv2.INTER_CUBIC if k > 1 else cv2.INTER_AREA)
            tiles.append(C._label(t, f"face {i}: {lab_}"))
    if tiles:
        C._sheet(tiles, path, cols=max(1, len(images)) * (2 if len(tiles) > 4 else 1))
    return bool(tiles)


def inspect_faces(image_path: str, sheet_path: str | None = None) -> dict:
    rgb, ana = P._load(image_path)
    m = ana["masks"]
    faces = []
    vis = rgb.copy()
    for i, p in enumerate(ana["people"]):
        if p["landmarks"] is None:
            faces.append({"id": i, "landmarks": False})
            continue
        faces.append({"id": i, "face_height_frac": round(p["face"]["box"][3] / rgb.shape[0], 3),
                      "facing": {1: "image right", -1: "image left", 0: "camera"}[p["face"]["facing"]],
                      **measure_face(rgb, p["landmarks"], m["face_skin"])})
        lm = p["landmarks"]
        for grp, col in ((P.FACE_OVAL, (1, 1, 0)), (R_EYE, (0, 1, 1)), (L_EYE, (0, 1, 1)),
                         (INNER_LIPS, (1, 0, 1)), ([NOSE_TIP, NOSE_WING_R, NOSE_WING_L], (0, 1, 0))):
            cv2.polylines(vis, [np.round(lm[grp]).astype(np.int32)], True, col, 1)
    if sheet_path is None:
        p_ = Path(image_path)
        sheet_path = str(p_.with_name(p_.stem + "_faces.jpg"))
    _closeups([rgb, vis], ana, ["photo", "landmarks"], sheet_path)
    return {"num_faces": len(faces), "faces": faces, "sheet_path": sheet_path, "presets": PRESETS,
            "note": "eye_open = eye height/width; *_rel = width / face width; under_eye_darkness_L = cheek L* "
                    "minus under-eye L* (> 6 noticeable); shine_pct = % face skin with specular shine (> 3 shiny)."}


def preview_face_edit(image_path: str, params: dict | None = None, preset: str | None = None,
                      face_ids: list[int] | None = None, sheet_path: str | None = None) -> dict:
    rgb, ana = P._load(image_path)
    p = normalize(params, preset)
    out, info = render_faces(rgb, ana, p, face_ids)
    if sheet_path is None:
        p_ = Path(image_path)
        sheet_path = str(p_.with_name(p_.stem + "_face_preview.jpg"))
    _closeups([rgb, out], ana, ["before", "after"], sheet_path, face_ids)
    checks = face_checks(rgb, out, ana, info)
    return {"sheet_path": sheet_path, "effective": {k: v for k, v in p.items() if (any(v) if isinstance(v, list) else v)},
            "identity": info["identity"], "checks_failed": [c for c in checks if c["status"] == "fail"]}


def apply_face_edit(image_path: str, params: dict | None = None, preset: str | None = None,
                    face_ids: list[int] | None = None, output_path: str | None = None) -> dict:
    full, ana_full = P.load_full(image_path)
    p = normalize(params, preset)
    out, info = render_faces(full, ana_full, p, face_ids)
    src = Path(image_path)
    if output_path is None:
        output_path = str(src.with_name(f"{src.stem}_face.tif"))
    write_image(output_path, (np.clip(out, 0, 1)[..., ::-1] * 65535 + 0.5).astype(np.uint16))
    H, W = full.shape[:2]
    kp = min(1.0, 1600 / max(H, W))
    prev = cv2.resize(out, None, fx=kp, fy=kp, interpolation=cv2.INTER_AREA)
    preview_path = str(Path(output_path).with_name(Path(output_path).stem + "_preview.jpg"))
    C._save_rgb8(preview_path, prev)
    before = cv2.resize(full, (prev.shape[1], prev.shape[0]), interpolation=cv2.INTER_AREA)
    ana_prev = P._load(image_path, max(prev.shape[:2]))[1]
    checks = face_checks(before, prev, ana_prev, info)
    return {"output_path": output_path, "preview_path": preview_path, "identity": info["identity"],
            "effective": {k: v for k, v in p.items() if (any(v) if isinstance(v, list) else v)},
            "checks": checks}
