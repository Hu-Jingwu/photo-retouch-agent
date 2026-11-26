"""Depth-of-field check and background blur (背景虚化).

analyze_depth   judges the depth structure BEFORE any blur is added. In a photo, how soft an area
                is (defocus) grows with its distance from the focal plane, so the existing blur is a
                depth map for free: the subject is the focal plane, areas already softer than the
                subject are farther away. Defocus is measured at edges by re-blurring
                (Zhuo & Sim 2011, "Defocus map estimation from a single image"). It reports a verdict, a suggested strength and a coloured
                depth map for the model to look at.
apply_background_blur
                blurs each background area in proportion to its estimated depth with a lens-like
                disc kernel; the subject's colours are excluded from the blur (normalised
                convolution), so no halo; matching grain is put back so the background does not
                look plastic. Guards: subject untouched, no halo, background really softer.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from . import color as C
from . import people as P
from .io_utils import write_image

MAX_RADIUS_FRAC = 0.02      # disc radius at strength 100, as a share of the long side


# ---- depth proxy from defocus ---------------------------------------------------------------
SIGMA0 = 1.0            # re-blur used to measure the defocus at each edge
FAR_PX = 3.0            # extra defocus (px at the 1600-px work size) that counts as "far" (z = 1)


def defocus_map(rgb: np.ndarray, person: np.ndarray) -> dict:
    """Defocus radius at every edge (Zhuo & Sim 2011): blur the photo once more by SIGMA0 and compare
    the gradient strength at each edge before/after; an edge already blurred by sigma_d loses less:
    ratio = sqrt(sigma_d^2 + SIGMA0^2) / sigma_d. The sparse edge values are spread into a smooth map.
    Returns sigma_d (px), the subject's own sigma_d, and z = how much farther than the subject (0..1)."""
    h, w = rgb.shape[:2]
    side = max(h, w)
    gray = cv2.cvtColor(np.clip(rgb, 0, 1).astype(np.float32), cv2.COLOR_RGB2GRAY)
    g8 = (gray * 255).astype(np.uint8)
    edges = cv2.Canny(g8, 15, 40) > 0           # low thresholds: soft (far) edges must count too

    def mag(im):
        return np.hypot(cv2.Sobel(im, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(im, cv2.CV_32F, 0, 1, ksize=3))

    m1, m2 = mag(gray), mag(cv2.GaussianBlur(gray, (0, 0), SIGMA0))
    ok = edges & (m1 > 0.015)
    R = m1 / np.maximum(m2, 1e-6)
    sig = SIGMA0 / np.sqrt(np.maximum(R * R - 1, 1e-3))
    sig = np.clip(sig, 0, 15).astype(np.float32)
    # the subject's own outline is a sharp edge too: it must not count as "sharp background",
    # or the background right next to the subject looks in focus and keeps a sharp ring
    band = max(3, int(side * 0.015)) | 1
    outline = cv2.dilate((person > 0.5).astype(np.uint8), np.ones((band, band), np.uint8)) > 0
    known = (ok & ~(outline & (person < 0.6))).astype(np.float32)
    inner = cv2.erode((person > 0.6).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    subj = ok & inner
    sig_subj = float(np.percentile(sig[subj], 30)) if subj.sum() > 30 else float(np.percentile(sig[ok], 10))
    S = max(3.0, side * 0.03)
    smooth = cv2.GaussianBlur(sig * known, (0, 0), S) / np.maximum(cv2.GaussianBlur(known, (0, 0), S), 1e-4)
    excess = np.maximum(smooth - sig_subj, 0)
    z = np.clip(excess / FAR_PX, 0, 1).astype(np.float32)
    return {"z": z, "sigma": smooth, "known": known, "sigma_subject": sig_subj, "raw": sig, "edges": ok}


def _judge(d, person, ana):
    h, w = person.shape
    bg = person < 0.2
    bg_known = bg & (d["known"] > 0)
    bg_share = float(bg.mean())
    out = {"background_share": round(bg_share, 3), "subject_defocus_px": round(d["sigma_subject"], 2)}
    if bg_known.sum() < 100:
        out.update(verdict="no_background_detail", suggested_strength=[30, 50],
                   note="the background is plain (sky, wall, studio): blur barely shows; it only smooths it")
        return out
    extra = float(np.median(d["raw"][bg_known]) - d["sigma_subject"])
    pm = (person > 0.5).astype(np.uint8)
    k = max(3, int(0.04 * max(h, w)))
    near = (cv2.dilate(pm, np.ones((k, k), np.uint8)) > 0) & bg_known
    near_extra = float(np.median(d["raw"][near]) - d["sigma_subject"]) if near.sum() > 30 else extra
    out.update(background_extra_defocus_px=round(extra, 2), near_subject_extra_defocus_px=round(near_extra, 2))
    if extra > 1.5:
        out.update(verdict="already_shallow", suggested_strength=[0, 35],
                   note="the background is already much softer than the subject (shallow depth of field): add "
                        "little or nothing")
    elif extra < 0.5 and near_extra < 0.5:
        out.update(verdict="sharp_background", suggested_strength=[35, 60],
                   note="the background is as sharp as the subject. Either the whole scene is in focus (phone, small "
                        "aperture: blur fits) or the background is right behind the subject (a wall a step away: "
                        "strong blur looks fake, keep it <= 30 or skip). Judge from the photo.")
    else:
        out.update(verdict="mixed_depth", suggested_strength=[45, 75],
                   note="parts of the background are already softer (farther); the blur follows the depth map")
    faces = [p["face"]["box"][3] for p in ana["people"]]
    if len(faces) > 1 and max(faces) / max(min(faces), 1) > 1.6:
        out["note"] += "; people at different distances: everyone is kept sharp"
    if bg_share < 0.1:
        out["note"] += "; very little background in the frame, so the effect is small"
    return out


def _overlay(rgb, z, person, path, verdict):
    gray = cv2.cvtColor((np.clip(rgb, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    base = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cm = cv2.applyColorMap((z * 255).astype(np.uint8), cv2.COLORMAP_JET)
    bgm = (person < 0.5)[..., None]
    vis = np.where(bgm, (0.45 * base + 0.55 * cm).astype(np.uint8), base)
    cnts, _ = cv2.findContours((person > 0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, cnts, -1, (0, 255, 0), 2)
    txt = f"depth from defocus: blue = focal plane, red = far | {verdict}"
    cv2.rectangle(vis, (0, 0), (min(vis.shape[1], 11 * len(txt)), 30), (255, 255, 255), -1)
    cv2.putText(vis, txt, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)
    write_image(path, vis, jpeg_quality=88)


def analyze_depth(image_path: str, overlay_path: str | None = None) -> dict:
    rgb, ana = P._load(image_path)
    if not ana["people"]:
        return {"verdict": "no_subject", "suggested_strength": [0, 0],
                "note": "no person found: background blur needs a person to keep sharp"}
    person = ana["masks"]["person"]
    d = defocus_map(rgb, person)
    out = _judge(d, person, ana)
    if overlay_path is None:
        p_ = Path(image_path)
        overlay_path = str(p_.with_name(p_.stem + "_depth.jpg"))
    _overlay(rgb, d["z"], person, overlay_path, out["verdict"])
    out.update(people=len(ana["people"]), overlay_path=overlay_path)
    return out


# ---- blur ---------------------------------------------------------------------------------
def _disc(r: float) -> np.ndarray:
    r = max(1.0, r)
    n = int(np.ceil(r)) * 2 + 1
    yy, xx = np.mgrid[:n, :n] - n // 2
    k = np.clip(r + 0.5 - np.hypot(xx, yy), 0, 1).astype(np.float32)    # anti-aliased disc (lens bokeh)
    return k / k.sum()


def render_blur(rgb: np.ndarray, person: np.ndarray, z: np.ndarray, strength: float, base: float,
                seed: int = 0) -> tuple[np.ndarray, dict]:
    """Depth-scaled disc blur of everything except the person. Work is done at 1/ds resolution."""
    H, W = rgb.shape[:2]
    R = strength / 100 * MAX_RADIUS_FRAC * max(H, W)
    if R < 1:
        return rgb.copy(), {"max_radius_px": 0.0}
    ds = max(1, int(R // 12))                                          # keep the low-res disc <= ~12 px
    lw, lh = max(8, W // ds), max(8, H // ds)
    small = cv2.resize(rgb, (lw, lh), interpolation=cv2.INTER_AREA)
    a_small = cv2.resize(person, (lw, lh), interpolation=cv2.INTER_AREA)
    z_small = cv2.resize(z, (lw, lh), interpolation=cv2.INTER_LINEAR)
    wbg = np.clip(1 - a_small, 0, 1) ** 2                                # the subject never feeds the blur
    r_low = R / ds
    want = r_low * (base + (1 - base) * z_small)                         # per-pixel radius, low-res px
    radii = [r_low / 3, 2 * r_low / 3, r_low]
    layers = []
    for r in radii:
        k = _disc(r)
        num = cv2.filter2D(small * wbg[..., None], -1, k, borderType=cv2.BORDER_REFLECT)
        den = cv2.filter2D(wbg, -1, k, borderType=cv2.BORDER_REFLECT)
        layers.append(num / np.maximum(den, 1e-3)[..., None])
    # interpolate between radius 0 (the sharp photo) and the three blurred layers
    pos = np.clip(want / (r_low / 3), 0, 3)                              # 0..3 in layer units
    w0 = np.clip(1 - pos, 0, 1)                                          # share of the sharp photo
    mix = np.zeros_like(small)
    for i, L in enumerate(layers):
        wi = np.clip(1 - np.abs(pos - (i + 1)), 0, 1)
        if i == 0:
            wi = np.where(pos < 1, pos, wi)
        mix += wi[..., None] * L
    wsum = np.maximum(1 - w0, 1e-4)
    mix = mix / wsum[..., None]
    w0_full = cv2.resize(w0, (W, H), interpolation=cv2.INTER_LINEAR)[..., None]
    mix_full = cv2.resize(mix, (W, H), interpolation=cv2.INTER_LINEAR)
    bg = w0_full * rgb + (1 - w0_full) * mix_full
    # grain: blurring removes the sensor noise; put the same amount back so the background matches
    gray = cv2.cvtColor(rgb.astype(np.float32), cv2.COLOR_RGB2GRAY)
    resid = np.abs(gray - cv2.GaussianBlur(gray, (0, 0), 1.0))
    flat = (person < 0.2) & (cv2.GaussianBlur(resid, (0, 0), 3) < np.percentile(resid, 40) * 1.5)
    sigma = float(np.median(resid[flat]) * 1.25) if flat.sum() > 100 else 0.0
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, sigma, (H, W, 1)).astype(np.float32)
    bg = bg + noise * (1 - w0_full)
    # solid inside, soft only at the edge: a mask that is 0.97 inside the body would let 3% of the
    # blurred background show through the whole person
    a = np.clip((person - 0.1) / 0.8, 0, 1)[..., None]
    out = np.clip(a * rgb + (1 - a) * bg, 0, 1).astype(np.float32)
    return out, {"max_radius_px": round(R, 1), "grain_sigma": round(sigma, 4), "base": base}


def blur_checks(before, after, person) -> list[dict]:
    checks = []

    def add(rule, ok, value, detail):
        checks.append({"rule": rule, "status": "pass" if ok else "fail", "value": value, "detail": detail})

    Lb, La = C._lab(before)[..., 0], C._lab(after)[..., 0]
    core = person > 0.95
    if core.sum() > 100:
        d = float(np.abs(La - Lb)[core].mean())
        add("subject untouched", d <= 0.5, round(d, 3), "mean L* change inside the person (<= 0.5)")
    pm = (person > 0.5).astype(np.uint8)
    h, w = pm.shape
    r1 = max(3, int(0.015 * max(h, w)))
    d1 = cv2.dilate(pm, np.ones((r1, r1), np.uint8))
    d2 = cv2.dilate(pm, np.ones((3 * r1, 3 * r1), np.uint8))
    d3 = cv2.dilate(pm, np.ones((6 * r1, 6 * r1), np.uint8))
    near, far = (d1 > 0) & (pm == 0), (d3 > 0) & (d2 == 0)
    if near.sum() > 50 and far.sum() > 50:
        dL = La - Lb
        diff = abs(float(dL[near].mean()) - float(dL[far].mean()))
        add("no halo around the subject", diff <= 2.5, round(diff, 2),
            "L* change right outside the person minus further out (<= 2.5)")
    bg = person < 0.2
    if bg.sum() > 100:
        hp = lambda L: np.abs(L - cv2.GaussianBlur(L, (0, 0), 1.5))
        ratio = float(hp(La)[bg].mean() / max(hp(Lb)[bg].mean(), 1e-6))
        add("background softer", ratio <= 0.95, round(ratio, 2), "background fine detail after/before (<= 0.95)")
    return checks


def _sheet(before, after, person, path):
    h, w = before.shape[:2]
    ys, xs = np.nonzero(person > 0.5)
    tiles = []
    for im, lab in ((before, "before"), (after, "after")):
        t = cv2.resize((np.clip(im, 0, 1)[..., ::-1] * 255).astype(np.uint8), (800, int(800 * h / w)),
                       interpolation=cv2.INTER_AREA)
        tiles.append(C._label(t, lab))
    if len(ys):                                   # close-up of the subject's top outline (hair, shoulders)
        cy, cx = int(ys.min() + 0.06 * (ys.max() - ys.min())), int(np.median(xs[ys < ys.min() + 0.1 * (ys.max() - ys.min())]))
        s = int(0.2 * max(h, w))
        y0, x0 = max(0, cy - s // 2), max(0, cx - s // 2)
        for im, lab in ((before, "edge before"), (after, "edge after")):
            c = im[y0:y0 + s, x0:x0 + s]
            t = cv2.resize((np.clip(c, 0, 1)[..., ::-1] * 255).astype(np.uint8), (400, 400),
                           interpolation=cv2.INTER_CUBIC)
            tiles.append(C._label(t, lab))
    C._sheet(tiles, path, cols=2)


def apply_background_blur(image_path: str, strength: float = 60, output_path: str | None = None,
                          sheet_path: str | None = None) -> dict:
    strength = float(np.clip(strength, 0, 100))
    small, ana = P._load(image_path)
    if not ana["people"]:
        raise ValueError("no person found: background blur needs a person to keep sharp")
    person_s = ana["masks"]["person"]
    d = defocus_map(small, person_s)
    judge = _judge(d, person_s, ana)
    base = 0.6 if judge["verdict"] in ("sharp_background", "no_background_detail") else 0.35
    full, ana_full = P.load_full(image_path)
    H, W = full.shape[:2]
    z_full = cv2.resize(d["z"], (W, H), interpolation=cv2.INTER_LINEAR)
    out, info = render_blur(full, ana_full["masks"]["person"], z_full, strength, base)
    src = Path(image_path)
    if output_path is None:
        output_path = str(src.with_name(f"{src.stem}_blur.tif"))
    write_image(output_path, (out[..., ::-1] * 65535 + 0.5).astype(np.uint16))
    kp = min(1.0, 1600 / max(H, W))
    prev = cv2.resize(out, None, fx=kp, fy=kp, interpolation=cv2.INTER_AREA)
    preview_path = str(Path(output_path).with_name(Path(output_path).stem + "_preview.jpg"))
    C._save_rgb8(preview_path, prev)
    before = cv2.resize(full, (prev.shape[1], prev.shape[0]), interpolation=cv2.INTER_AREA)
    person_p = cv2.resize(ana_full["masks"]["person"], (prev.shape[1], prev.shape[0]), interpolation=cv2.INTER_AREA)
    checks = blur_checks(before, prev, person_p)
    if sheet_path is None:
        sheet_path = str(src.with_name(src.stem + "_blur_sheet.jpg"))
    _sheet(before, prev, person_p, sheet_path)
    return {"output_path": output_path, "preview_path": preview_path, "sheet_path": sheet_path,
            "strength": strength, "verdict": judge["verdict"], "info": info,
            "checks_failed": [c for c in checks if c["status"] == "fail"],
            "checks": checks}
