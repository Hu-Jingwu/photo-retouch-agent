"""Global colour & tone tools  (approach A + B).

A  A parametric, Lightroom-style develop pipeline. Every adjustment is a named
   slider, so edits are explainable, reversible and can later be exported as XMP.
   Order:  white balance -> exposure -> local highlights/shadows -> tone curve
           (blacks/whites, contrast, curve points) -> 8-colour HSL ->
           vibrance/saturation -> colour grading -> grain
B  A library of named looks ("recipes"), each just a set of slider values,
   applied on top of the base correction with a strength 0..1.

The model never has to guess what "+0.3 EV" looks like: preview_variants
renders a strip of values side by side so it can choose visually ("inference-
time puzzle", after the MonetGPT idea of learning operations from puzzles).
"""
from __future__ import annotations

import copy
import json
import math
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from .io_utils import read_image, write_image

COLORS = ("red", "orange", "yellow", "green", "aqua", "blue", "purple", "magenta")
HUE_CENTERS = (0.0, 30.0, 60.0, 120.0, 180.0, 240.0, 270.0, 300.0)   # HSV degrees
SCALARS = {"exposure": (-3.0, 3.0), "contrast": (-100, 100), "highlights": (-100, 100),
           "shadows": (-100, 100), "whites": (-100, 100), "blacks": (-100, 100),
           "temperature": (-100, 100), "tint": (-100, 100), "vibrance": (-100, 100),
           "saturation": (-100, 100), "grain": (0, 100)}
BANDS = ("shadows", "midtones", "highlights")
PREVIEW_SIDE = 1200

# --------------------------------------------------------------------------- #
# B: look recipes. Our own looks, written from general colour-grading practice;
# not copies of any camera maker's proprietary film simulations or recipes.
# --------------------------------------------------------------------------- #
RECIPES: dict[str, dict] = {
    "clean_bright": {
        "label": "清透明亮 Clean & bright",
        "description": "Open shadows, soft highlights, clear blues; everyday / travel / portraits.",
        "params": {"exposure": 0.15, "contrast": -10, "highlights": -25, "shadows": 25, "whites": 10,
                   "vibrance": 15, "temperature": -4,
                   "hsl": {"blue": {"saturation": 10}, "aqua": {"luminance": 10}}},
    },
    "vivid_positive": {
        "label": "鲜艳正片 Vivid slide film",
        "description": "Punchy contrast and rich greens/blues, like slide film; landscapes.",
        "params": {"contrast": 22, "vibrance": 20, "saturation": 8, "blacks": -8,
                   "hsl": {"green": {"hue": -10, "saturation": 15}, "blue": {"saturation": 20, "luminance": -10}},
                   "curve": [[0, 0], [64, 56], [192, 202], [255, 255]]},
    },
    "muted_documentary": {
        "label": "低饱和人文 Muted documentary",
        "description": "Low saturation, firm contrast, faded blacks, cool shadows; street / reportage.",
        "params": {"saturation": -22, "contrast": 15, "shadows": -10,
                   "hsl": {"blue": {"hue": -5, "saturation": -20}, "red": {"saturation": -10}},
                   "grading": {"shadows": {"hue": 205, "saturation": 12}, "highlights": {"hue": 45, "saturation": 8}},
                   "curve": [[0, 14], [128, 126], [255, 250]]},
    },
    "warm_portrait": {
        "label": "暖调人像 Warm portrait",
        "description": "Warm, soft, flattering skin; portraits and golden hour.",
        "params": {"temperature": 10, "tint": 3, "vibrance": 10, "highlights": -15, "shadows": 12,
                   "hsl": {"orange": {"saturation": -5, "luminance": 10}, "red": {"luminance": 5}},
                   "grading": {"highlights": {"hue": 40, "saturation": 10}}},
    },
    "airy_japanese": {
        "label": "日系空气感 Airy",
        "description": "Bright, low contrast, lifted blacks, cool-minty greens; casual / lifestyle.",
        "params": {"exposure": 0.3, "contrast": -25, "highlights": -30, "shadows": 30, "blacks": 20,
                   "saturation": -10, "temperature": -8, "tint": -3,
                   "hsl": {"green": {"hue": 15, "saturation": -20, "luminance": 10},
                           "blue": {"hue": -5, "saturation": -10}},
                   "curve": [[0, 24], [128, 134], [255, 250]]},
    },
    "cinematic_teal_orange": {
        "label": "电影青橙 Cinematic teal & orange",
        "description": "Teal shadows, warm skin and highlights, muted greens; night / city / sport.",
        "params": {"contrast": 10, "hsl": {"orange": {"saturation": 10}, "aqua": {"hue": -10, "saturation": 15},
                                           "blue": {"hue": -15, "saturation": -10},
                                           "green": {"hue": 20, "saturation": -25}},
                   "grading": {"shadows": {"hue": 195, "saturation": 25}, "highlights": {"hue": 35, "saturation": 18}},
                   "curve": [[0, 10], [255, 255]]},
    },
    "moody_dark": {
        "label": "暗调情绪 Moody dark",
        "description": "Darker, desaturated, cool shadows, held highlights; dramatic / low-key.",
        "params": {"exposure": -0.3, "contrast": 15, "highlights": -30, "shadows": -15, "blacks": -10,
                   "saturation": -15, "grading": {"shadows": {"hue": 220, "saturation": 12}}},
    },
    "bw_high_contrast": {
        "label": "高反差黑白 High-contrast B&W",
        "description": "Black and white with bright skin and dark skies; graphic scenes, portraits with character.",
        "params": {"contrast": 35, "whites": 15, "blacks": -15,
                   "hsl": {"red": {"luminance": 10}, "orange": {"luminance": 15}, "blue": {"luminance": -25}}},
        # applied as-is whatever the strength: a "70% black-and-white" would just be a sepia wash
        "fixed": {"saturation": -100, "vibrance": 0},
    },
}


# --------------------------------------------------------------------------- #
# Parameters
# --------------------------------------------------------------------------- #
def normalize_params(params: dict | None) -> dict:
    """Validate and fill defaults. Raises ValueError on unknown keys so the model learns fast."""
    params = copy.deepcopy(params or {})
    out: dict = {k: 0.0 for k in SCALARS}
    out.update(hsl={}, grading={}, curve=None)
    for k, v in params.items():
        if k in SCALARS:
            lo, hi = SCALARS[k]
            out[k] = float(np.clip(float(v), lo, hi))
        elif k == "hsl":
            for color, adj in (v or {}).items():
                if color not in COLORS:
                    raise ValueError(f"unknown HSL colour {color!r}; use one of {COLORS}")
                bad = set(adj) - {"hue", "saturation", "luminance"}
                if bad:
                    raise ValueError(f"unknown HSL keys {bad}")
                out["hsl"][color] = {kk: float(np.clip(float(adj.get(kk, 0)), -100, 100))
                                     for kk in ("hue", "saturation", "luminance")}
        elif k == "grading":
            for band, adj in (v or {}).items():
                if band not in BANDS:
                    raise ValueError(f"unknown grading band {band!r}; use {BANDS}")
                out["grading"][band] = {"hue": float(adj.get("hue", 0)) % 360,
                                        "saturation": float(np.clip(float(adj.get("saturation", 0)), 0, 100))}
        elif k == "curve":
            if v:
                pts = sorted((float(a), float(b)) for a, b in v)
                out["curve"] = [[float(np.clip(a, 0, 255)), float(np.clip(b, 0, 255))] for a, b in pts]
        else:
            raise ValueError(f"unknown parameter {k!r}; valid: {list(SCALARS) + ['hsl', 'grading', 'curve']}")
    return out


def _merge(base: dict, recipe: str | None, strength: float) -> tuple[dict, list | None]:
    """Effective params = base + strength * recipe. Returns (params, recipe_curve_scaled)."""
    p = normalize_params(base)
    if not recipe:
        return p, None
    if recipe not in RECIPES:
        raise ValueError(f"unknown recipe {recipe!r}; available: {list(RECIPES)}")
    s = float(np.clip(strength, 0.0, 1.0))
    r = normalize_params(RECIPES[recipe]["params"])
    for k, (lo, hi) in SCALARS.items():
        p[k] = float(np.clip(p[k] + s * r[k], lo, hi))
    for color, adj in r["hsl"].items():
        cur = p["hsl"].setdefault(color, {"hue": 0.0, "saturation": 0.0, "luminance": 0.0})
        for kk in cur:
            cur[kk] = float(np.clip(cur[kk] + s * adj[kk], -100, 100))
    for band, adj in r["grading"].items():   # add as vectors on the colour wheel
        cur = p["grading"].get(band, {"hue": 0.0, "saturation": 0.0})
        vx = cur["saturation"] * math.cos(math.radians(cur["hue"])) + s * adj["saturation"] * math.cos(math.radians(adj["hue"]))
        vy = cur["saturation"] * math.sin(math.radians(cur["hue"])) + s * adj["saturation"] * math.sin(math.radians(adj["hue"]))
        p["grading"][band] = {"hue": math.degrees(math.atan2(vy, vx)) % 360,
                              "saturation": float(min(100.0, math.hypot(vx, vy)))}
    for k, v in RECIPES[recipe].get("fixed", {}).items():
        p[k] = float(v)
    if RECIPES[recipe].get("lut"):
        p["lut"] = {"name": recipe, "strength": s}     # film LUT: strength blends the LUT result
    rc = None
    if r["curve"]:
        rc = [[x, x + s * (y - x)] for x, y in r["curve"]]
    return p, rc


# --------------------------------------------------------------------------- #
# Film LUTs (HaldCLUT). Pat David's film-emulation collection, CC BY-SA 4.0,
# from github.com/NatronGitHub/clut (see luts/SOURCE_README.md). Approximations of
# the film stocks, not Fujifilm products; names are for information only.
# --------------------------------------------------------------------------- #
LUT_DIR = Path(__file__).resolve().parent.parent / "luts"
FILM_LUTS = {
    "fuji_provia_100f": "富士 Provia 100F 正片: neutral, clean slide-film colour",
    "fuji_velvia_50": "富士 Velvia 50 正片: very saturated, deep blues/greens, high contrast; landscapes",
    "fuji_astia_100f": "富士 Astia 100F 正片: soft contrast, gentle skin tones; portraits",
    "fuji_superia_reala_100": "富士 Superia Reala 100 负片: natural, accurate, slightly soft",
    "fuji_superia_200": "富士 Superia 200 负片: everyday consumer film, a touch warm/green",
    "fuji_400h": "富士 Pro 400H 负片: airy pastel, cool-minty greens, soft skin; weddings/portraits",
    "fuji_160c": "富士 Pro 160C 负片: punchy negative, saturated but smooth",
    "fuji_neopan_acros_100": "富士 Neopan Acros 100 黑白: fine-grain black and white",
}


@lru_cache(maxsize=16)
def load_haldclut(name: str) -> np.ndarray:
    """HaldCLUT PNG -> 3D LUT array indexed [b, g, r] -> RGB float32 (cube size N = level^2)."""
    path = LUT_DIR / f"{name}.png"
    img = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"LUT not found: {path}")
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    scale = 65535.0 if img.dtype == np.uint16 else 255.0
    flat = img[..., ::-1].reshape(-1, 3).astype(np.float32) / scale
    n = round(len(flat) ** (1 / 3))
    if n ** 3 != len(flat):
        raise ValueError(f"{path} is not a HaldCLUT")
    return flat.reshape(n, n, n, 3)          # red varies fastest -> index [b, g, r]


def apply_lut(rgb: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """Trilinear 3D-LUT lookup on float sRGB [0,1]."""
    n = lut.shape[0]
    x = np.clip(rgb, 0, 1) * (n - 1)
    i0 = np.floor(x).astype(np.int32)
    i0 = np.minimum(i0, n - 2)
    f = x - i0
    r0, g0, b0 = i0[..., 0], i0[..., 1], i0[..., 2]
    fr, fg, fb = f[..., 0:1], f[..., 1:2], f[..., 2:3]
    out = np.zeros_like(rgb, dtype=np.float32)
    for db in (0, 1):
        wb = fb if db else 1 - fb
        for dg in (0, 1):
            wg = fg if dg else 1 - fg
            for dr in (0, 1):
                wr = fr if dr else 1 - fr
                out += wb * wg * wr * lut[b0 + db, g0 + dg, r0 + dr]
    return out


def profile_lut(name: str) -> dict:
    """'Learn' a film LUT's colour style by measuring what it does to a test chart:
    tone curve on greys, white-balance shift of mid grey, and per-hue saturation /
    hue / brightness changes. Gives the model (and people) a readable description."""
    lut = load_haldclut(name)
    greys = np.linspace(0.05, 0.95, 10, dtype=np.float32)
    g_in = np.repeat(greys[:, None], 3, 1)[None]
    g_out = apply_lut(g_in, lut)
    L_in, L_out = _lab(g_in)[0, :, 0], _lab(g_out)[0, :, 0]
    mid = _lab(apply_lut(np.full((1, 1, 3), 0.5, np.float32), lut))[0, 0]
    slope = float(np.polyfit(L_in[3:7], L_out[3:7], 1)[0])       # midtone contrast
    bands = {}
    for color, hc in zip(COLORS, HUE_CENTERS):
        c_in = cv2.cvtColor(np.array([[[hc, 0.6, 0.7]]], np.float32), cv2.COLOR_HSV2RGB)
        c_out = apply_lut(c_in, lut)
        hi, ho = cv2.cvtColor(c_in, cv2.COLOR_RGB2HSV)[0, 0], cv2.cvtColor(c_out, cv2.COLOR_RGB2HSV)[0, 0]
        bands[color] = {"hue_shift_deg": round(float((ho[0] - hi[0] + 180) % 360 - 180), 1),
                        "saturation_x": round(float(ho[1] / max(hi[1], 1e-6)), 2),
                        "brightness_x": round(float(ho[2] / max(hi[2], 1e-6)), 2)}
    sat_mean = float(np.mean([b["saturation_x"] for b in bands.values()]))
    mono = sat_mean < 0.1
    words = []
    words.append("high contrast" if slope > 1.12 else "soft contrast" if slope < 0.92 else "normal contrast")
    words.append("black and white" if mono else
                 ("very saturated" if sat_mean > 1.25 else "saturated" if sat_mean > 1.08
                  else "muted" if sat_mean < 0.92 else "natural saturation"))
    if not mono:
        if mid[2] > 2.5:
            words.append("warm")
        elif mid[2] < -2.5:
            words.append("cool")
        if mid[1] > 2.0:
            words.append("magenta tint")
        elif mid[1] < -2.0:
            words.append("green tint")
        big = sorted(bands.items(), key=lambda kv: -abs(kv[1]["hue_shift_deg"]))[:2]
        for c, b in big:
            if abs(b["hue_shift_deg"]) >= 4:
                words.append(f"{c} shifted {b['hue_shift_deg']:+.0f} deg")
    return {"name": name, "summary": ", ".join(words),
            "midtone_contrast_x": round(slope, 2),
            "black_L": round(float(L_out[0]), 1), "white_L": round(float(L_out[-1]), 1),
            "mid_grey_a_b": [round(float(mid[1]), 2), round(float(mid[2]), 2)],
            "mean_saturation_x": round(sat_mean, 2), "bands": bands}


def _register_film_luts():
    prof_path = LUT_DIR / "profiles.json"
    profiles = json.loads(prof_path.read_text(encoding="utf-8")) if prof_path.is_file() else {}
    for name, desc in FILM_LUTS.items():
        if (LUT_DIR / f"{name}.png").is_file():
            measured = profiles.get(name, {}).get("summary")
            RECIPES[name] = {"label": desc.split(":")[0],
                             "description": desc.split(":", 1)[-1].strip()
                             + (f"; measured: {measured}" if measured else "")
                             + " (film LUT, CC BY-SA 4.0 Pat David)", "params": {}, "lut": True}


_register_film_luts()


# --------------------------------------------------------------------------- #
# A: the develop pipeline
# --------------------------------------------------------------------------- #
def _s2l(x):
    return np.where(x <= 0.04045, x / 12.92, ((np.maximum(x, 0) + 0.055) / 1.055) ** 2.4).astype(np.float32)


def _l2s(x):
    x = np.maximum(x, 0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055).astype(np.float32)


def _luma(rgb):
    return rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722


def _smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def _wb_gains(p) -> np.ndarray:
    t, ti = p["temperature"] / 100.0, p["tint"] / 100.0
    g = np.array([math.exp(0.5 * t), math.exp(-0.3 * ti), math.exp(-0.5 * t)], np.float32)
    return g / float(g @ np.array([0.2126, 0.7152, 0.0722], np.float32))   # keep grey brightness


def _pchip(xs, ys, x):
    """Monotone cubic interpolation (Fritsch-Carlson) for smooth, non-overshooting curves."""
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    if len(xs) == 1:
        return np.full_like(x, ys[0])
    h = np.diff(xs)
    d = np.diff(ys) / h
    m = np.zeros_like(xs)
    m[0], m[-1] = d[0], d[-1]
    for i in range(1, len(xs) - 1):
        if d[i - 1] * d[i] <= 0:
            m[i] = 0.0
        else:
            w1, w2 = 2 * h[i] + h[i - 1], h[i] + 2 * h[i - 1]
            m[i] = (w1 + w2) / (w1 / d[i - 1] + w2 / d[i])
    idx = np.clip(np.searchsorted(xs, x) - 1, 0, len(xs) - 2)
    t = (x - xs[idx]) / h[idx]
    h00, h10 = 2 * t ** 3 - 3 * t ** 2 + 1, t ** 3 - 2 * t ** 2 + t
    h01, h11 = -2 * t ** 3 + 3 * t ** 2, t ** 3 - t ** 2
    y = h00 * ys[idx] + h10 * h[idx] * m[idx] + h01 * ys[idx + 1] + h11 * h[idx] * m[idx + 1]
    return np.where(x <= xs[0], ys[0] + (x - xs[0]) * m[0], np.where(x >= xs[-1], ys[-1] + (x - xs[-1]) * m[-1], y))


def _curve_lut(p, recipe_curve, n=4096) -> np.ndarray:
    x = np.linspace(0, 1, n)
    bp = -p["blacks"] / 100 * 0.06          # blacks -100 -> crush, +100 -> lift (negative bp)
    wp = 1 - p["whites"] / 100 * 0.12
    y = (x - bp) / (wp - bp)
    c = p["contrast"] / 100
    y = y + c * 0.15 * np.sin(2 * np.pi * (np.clip(y, 0, 1) - 0.5))
    for pts in (p["curve"], recipe_curve):
        if pts:
            xs = [a / 255 for a, _ in pts]
            ys = [b / 255 for _, b in pts]
            if xs[0] > 0:
                xs, ys = [0.0] + xs, [0.0] + ys
            if xs[-1] < 1:
                xs, ys = xs + [1.0], ys + [1.0]
            y = _pchip(xs, ys, np.clip(y, 0, 1))
    return np.clip(y, 0, 1).astype(np.float32)


def _band_weights(h):
    """Partition of unity over the 8 HSL bands (smooth tents in hue)."""
    ws = []
    n = len(HUE_CENTERS)
    for i, ci in enumerate(HUE_CENTERS):
        left = (ci - HUE_CENTERS[i - 1]) % 360
        right = (HUE_CENTERS[(i + 1) % n] - ci) % 360
        d = (h - ci + 180) % 360 - 180
        w = np.where(d < 0, 1 + d / left, 1 - d / right)
        w = np.clip(w, 0, 1)
        ws.append(w * w * (3 - 2 * w))
    return ws


def _log_luma(lin):
    return np.log2(_luma(lin) + 1e-4).astype(np.float32)


def _local_base(rgb_small, p) -> tuple[np.ndarray, np.ndarray]:
    """Edge-aware local brightness for highlights/shadows: a self-guided filter (He et al.)
    on log-luminance at low resolution, returned as the linear coefficients (a, b).

    A plain blur smears a dark subject's brightness into the bright sky next to it, so the sky
    there is treated as 'shadow' and lifted: a halo. The guided filter keeps strong edges
    (a ~ 1 where the variance is large) and only smooths within regions. Returning (a, b)
    lets the caller upsample them and apply base = a * I + b with the FULL-resolution
    luminance I (fast guided filter), so edges stay sharp at any output size."""
    lin = _s2l(rgb_small) * _wb_gains(p) * (2 ** p["exposure"])
    I = _log_luma(lin)
    r = max(2, int(round(max(I.shape) * 0.03)))
    eps = 0.1                              # in log2 units^2: edges of > ~1 stop are preserved
    box = lambda x: cv2.boxFilter(x, cv2.CV_32F, (2 * r + 1, 2 * r + 1))
    mean_i = box(I)
    var_i = box(I * I) - mean_i * mean_i
    a = var_i / (var_i + eps)
    b = mean_i - a * mean_i
    return box(a), box(b)


def _upsample_ab(ab, shape_hw):
    h, w = shape_hw
    return tuple(cv2.resize(x, (w, h), interpolation=cv2.INTER_LINEAR) for x in ab)


def _render(rgb, p, base_ab, lut, recipe_curve=None, seed=0):
    """Pointwise pipeline on float32 sRGB [0,1]; base_ab are the upsampled guided-filter coefficients."""
    lin = _s2l(rgb) * _wb_gains(p) * np.float32(2 ** p["exposure"])
    if p["highlights"] or p["shadows"]:
        a, b = base_ab
        bd = _l2s(np.exp2(a * _log_luma(lin) + b))
        ev = (p["shadows"] / 100 * 1.2 * (1 - _smoothstep(0.0, 0.5, bd))
              + p["highlights"] / 100 * 1.0 * _smoothstep(0.5, 1.0, bd))
        lin *= np.exp2(ev)[..., None].astype(np.float32)
    d = _l2s(lin)
    if p["exposure"] > 0 or p["highlights"] > 0 or p["shadows"] > 0:   # soft shoulder instead of hard clip
        k = 0.9
        d = np.where(d > k, k + (1 - k) * np.tanh((d - k) / (1 - k)), d)
    d = np.clip(d, 0, 1)
    d = lut[np.clip((d * (len(lut) - 1)).astype(np.int32), 0, len(lut) - 1)]

    hsl = p["hsl"]
    if any(any(v) for v in (a.values() for a in hsl.values())):
        hsv = cv2.cvtColor(d, cv2.COLOR_RGB2HSV)
        H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        ws = _band_weights(H)
        dh = np.zeros_like(H)
        ds = np.zeros_like(H)
        dv = np.zeros_like(H)
        for color, w in zip(COLORS, ws):
            a = hsl.get(color)
            if a:
                dh += w * a["hue"] / 100 * 30
                ds += w * a["saturation"] / 100
                dv += w * a["luminance"] / 100 * 0.4
        hsv[..., 0] = (H + dh) % 360
        hsv[..., 2] = np.clip(V * (1 + dv * S), 0, 1)
        hsv[..., 1] = np.clip(S * (1 + ds), 0, 1)
        d = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    if p["saturation"] or p["vibrance"]:
        y = _luma(d)[..., None]
        mx, mn = d.max(-1), d.min(-1)
        s_now = ((mx - mn) / (mx + 1e-6))[..., None]
        f = 1 + p["saturation"] / 100
        if p["vibrance"]:
            hue = cv2.cvtColor(d, cv2.COLOR_RGB2HSV)[..., 0]
            protect = 1 - 0.6 * _band_weights(hue)[1]           # orange band ~ skin
            f = f * (1 + p["vibrance"] / 100 * (1 - s_now) * protect[..., None])
        d = np.clip(y + (d - y) * f, 0, 1)

    if p.get("lut"):
        s = p["lut"]["strength"]
        d = d + s * (apply_lut(d, load_haldclut(p["lut"]["name"])) - d)

    if p["grading"]:
        L = _luma(d)
        wts = {"shadows": 1 - _smoothstep(0.0, 0.6, L),
               "midtones": np.clip(1 - np.abs(L - 0.5) / 0.35, 0, 1),
               "highlights": _smoothstep(0.4, 1.0, L)}
        for band, adj in p["grading"].items():
            if adj["saturation"] <= 0:
                continue
            col = cv2.cvtColor(np.array([[[adj["hue"], 1.0, 1.0]]], np.float32), cv2.COLOR_HSV2RGB)[0, 0]
            off = col - float(col @ np.array([0.2126, 0.7152, 0.0722], np.float32))
            d = d + (wts[band] * adj["saturation"] / 100 * 0.3)[..., None] * off
        d = np.clip(d, 0, 1)

    if p["grain"] > 0:
        L = _luma(d)
        rng = np.random.default_rng(seed)
        n = rng.normal(0, p["grain"] / 100 * 0.05, L.shape).astype(np.float32) * (4 * L * (1 - L))
        d = np.clip(d + n[..., None], 0, 1)
    return d.astype(np.float32)


# --------------------------------------------------------------------------- #
# Loading / rendering helpers
# --------------------------------------------------------------------------- #
def _to_rgb_float(img) -> np.ndarray:
    scale = 65535.0 if img.dtype == np.uint16 else 255.0
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    return (img[..., ::-1].astype(np.float32) / scale)


@lru_cache(maxsize=8)
def _load_preview_cached(path: str, mtime: float, side: int) -> np.ndarray:
    img = read_image(path)
    s = side / max(img.shape[:2])
    if s < 1:
        img = cv2.resize(img, (round(img.shape[1] * s), round(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
    return _to_rgb_float(img)


def _load_preview(path: str, side: int = PREVIEW_SIDE) -> np.ndarray:
    return _load_preview_cached(str(path), Path(path).stat().st_mtime, side)


def _render_small(rgb, params, recipe=None, strength=1.0):
    p, rc = _merge(params, recipe, strength)
    lut = _curve_lut(p, rc)
    small = cv2.resize(rgb, None, fx=512 / max(rgb.shape[:2]), fy=512 / max(rgb.shape[:2]),
                       interpolation=cv2.INTER_AREA) if max(rgb.shape[:2]) > 512 else rgb
    base_ab = _upsample_ab(_local_base(small, p), rgb.shape[:2])
    return _render(rgb, p, base_ab, lut, rc), p


def _save_rgb8(path, rgb, quality=90):
    write_image(path, (np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8)[..., ::-1], jpeg_quality=quality)


def _label(img_bgr, text):
    cv2.rectangle(img_bgr, (0, 0), (img_bgr.shape[1], 32), (255, 255, 255), -1)
    cv2.putText(img_bgr, text, (6, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    return img_bgr


def _sheet(tiles, path, cols=3):
    th = max(t.shape[0] for t in tiles)
    tw = max(t.shape[1] for t in tiles)
    rows = math.ceil(len(tiles) / cols)
    sheet = np.full((rows * (th + 8), min(cols, len(tiles)) * (tw + 8), 3), 255, np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        sheet[r * (th + 8):r * (th + 8) + t.shape[0], c * (tw + 8):c * (tw + 8) + t.shape[1]] = t
    write_image(path, sheet, jpeg_quality=88)


# --------------------------------------------------------------------------- #
# Measurements (shared by analyze_color, render_adjustments and check_color)
# --------------------------------------------------------------------------- #
def _lab(rgb):
    return cv2.cvtColor(np.ascontiguousarray(rgb, dtype=np.float32), cv2.COLOR_RGB2Lab)


def _wb_slope():
    """How many Lab b / a units one temperature / tint slider unit moves a mid-grey (for suggestions)."""
    grey = np.full((1, 1, 3), 0.5, np.float32)
    base = _lab(grey)[0, 0]
    pt = normalize_params({"temperature": 50})
    pn = normalize_params({"tint": 50})
    bt = _lab(_l2s(_s2l(grey) * _wb_gains(pt)))[0, 0]
    bn = _lab(_l2s(_s2l(grey) * _wb_gains(pn)))[0, 0]
    return (bt[2] - base[2]) / 50, (bn[1] - base[1]) / 50


_DB_PER_TEMP, _DA_PER_TINT = _wb_slope()


def measure(rgb: np.ndarray) -> dict:
    from .composition import detect_faces   # local import: composition imports horizon, not colour

    lab = _lab(rgb)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    chroma = np.hypot(a, b)
    hi_clip = float((rgb.max(-1) >= 0.995).mean() * 100)
    lo_clip = float((rgb.max(-1) <= 0.005).mean() * 100)
    mono = float(chroma.mean()) < 3.0
    neutral = (chroma < 12) & (L > 25) & (L < 90)
    cast = None
    if neutral.mean() > 0.02 and not mono:
        ca, cb = float(a[neutral].mean()), float(b[neutral].mean())
        cast = {"a_green_magenta": round(ca, 2), "b_blue_yellow": round(cb, 2),
                "suggested_temperature": round(float(np.clip(-cb / _DB_PER_TEMP, -100, 100)), 1),
                "suggested_tint": round(float(np.clip(-ca / _DA_PER_TINT, -100, 100)), 1)}
    faces = detect_faces((np.clip(rgb, 0, 1)[..., ::-1] * 255).astype(np.uint8))
    skin = None
    if faces and not mono:
        x, y, w, h = faces[0]["box"]
        x0, y0 = int(x + 0.25 * w), int(y + 0.3 * h)
        x1, y1 = int(x + 0.75 * w), int(y + 0.8 * h)
        patch = lab[max(y0, 0):y1, max(x0, 0):x1].reshape(-1, 3)
        if len(patch):
            Ls, As, Bs = (float(np.median(patch[:, i])) for i in range(3))
            skin = {"L": round(Ls, 1), "a": round(As, 1), "b": round(Bs, 1),
                    "hue_angle": round(math.degrees(math.atan2(Bs, As)), 1),
                    "chroma": round(math.hypot(As, Bs), 1)}
    small = cv2.resize(lab, (64, 64), interpolation=cv2.INTER_AREA).reshape(-1, 3)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    _, lbl, cent = cv2.kmeans(small, 5, None, crit, 2, cv2.KMEANS_PP_CENTERS)
    dom = []
    for i, c in enumerate(cent):
        frac = float((lbl == i).mean())
        rgb_c = cv2.cvtColor(np.array([[c]], np.float32), cv2.COLOR_Lab2RGB)[0, 0]
        hue = cv2.cvtColor(np.array([[np.clip(rgb_c, 0, 1)]], np.float32), cv2.COLOR_RGB2HSV)[0, 0, 0]
        name = "neutral" if math.hypot(c[1], c[2]) < 10 else \
            COLORS[int(np.argmin([min(abs(hue - hc), 360 - abs(hue - hc)) for hc in HUE_CENTERS]))]
        dom.append({"name": name, "hex": "#%02x%02x%02x" % tuple(int(v * 255) for v in np.clip(rgb_c, 0, 1)),
                    "fraction": round(frac, 3)})
    dom.sort(key=lambda d_: -d_["fraction"])
    # sky: bright, bluish pixels in the top fifth of the frame
    top = lab[: max(1, lab.shape[0] // 5)]
    skym = (top[..., 0] > 55) & (top[..., 2] < 0)
    sky = None
    if skym.mean() > 0.3 and not mono:
        sky = {"a": round(float(top[..., 1][skym].mean()), 2), "b": round(float(top[..., 2][skym].mean()), 2)}
    p = np.percentile(L, [1, 5, 50, 95, 99])
    return {
        "luminance_L": {"p1": round(float(p[0]), 1), "p5": round(float(p[1]), 1), "median": round(float(p[2]), 1),
                        "p95": round(float(p[3]), 1), "p99": round(float(p[4]), 1)},
        "key": "low-key" if p[2] < 30 else ("high-key" if p[2] > 70 else "normal"),
        "contrast_std_L": round(float(L.std()), 1),
        "clipped_highlights_pct": round(hi_clip, 2),
        "crushed_shadows_pct": round(lo_clip, 2),
        "monochrome": mono,
        "mean_chroma": round(float(chroma.mean()), 1),
        "highly_saturated_pct": round(float((chroma > 60).mean() * 100), 2),
        "neutral_cast": cast,
        "sky": sky,
        "skin": skin,
        "dominant_colors": dom,
    }


def color_issues(m: dict) -> list[dict]:
    """Classical correctness checks on a measured image. Each: rule, status, value, detail."""
    out = []

    def add(rule, ok, value, detail):
        out.append({"rule": rule, "status": "pass" if ok else "fail", "value": value, "detail": detail})

    add("highlights not clipped", m["clipped_highlights_pct"] <= 1.0, m["clipped_highlights_pct"],
        "% pixels at pure white (<= 1%)")
    add("shadows not crushed", m["crushed_shadows_pct"] <= 2.0, m["crushed_shadows_pct"],
        "% pixels at pure black (<= 2%)")
    if m["skin"]:
        s = m["skin"]
        add("subject exposed well", 35 <= s["L"] <= 85, s["L"], "face lightness L* 35..85")
        add("natural skin hue", 25 <= s["hue_angle"] <= 72, s["hue_angle"],
            "Lab hue angle 25..72 deg (below = too red/magenta, above = too yellow/green)")
        add("skin not oversaturated", s["chroma"] <= 38, s["chroma"], "skin chroma <= 38")
    else:
        med = m["luminance_L"]["median"]
        add("overall exposure plausible", 12 <= med <= 88, med, "median L* 12..88 (low/high key allowed)")
    if m.get("sky"):
        sk = m["sky"]
        add("blue sky stays blue (not purple/grey)", sk["a"] <= 0.3, [sk["a"], sk["b"]],
            "sky Lab a <= 0.3 (clean blue skies read a < 0; above that, with negative b, it looks lavender)")
    if m["neutral_cast"]:
        c = m["neutral_cast"]
        add("no strong colour cast on neutrals",
            abs(c["a_green_magenta"]) <= 5 and abs(c["b_blue_yellow"]) <= 10,
            [c["a_green_magenta"], c["b_blue_yellow"]], "near-grey areas |a|<=5, |b|<=10 (some warmth is fine)")
    return out


# The face may follow a style only so far: beyond this the skin itself looks cold, green or orange.
# Expressed in white-balance slider units (temperature / tint) as measured on the face skin.
FACE_SHIFT_LIMIT = {"temperature": 20.0, "tint": 20.0}


def _skin_ab(rgb: np.ndarray, box) -> tuple[float, float] | None:
    x, y, w, h = box
    x0, y0, x1, y1 = int(x + 0.25 * w), int(y + 0.3 * h), int(x + 0.75 * w), int(y + 0.8 * h)
    patch = _lab(rgb[max(y0, 0):y1, max(x0, 0):x1]).reshape(-1, 3)
    patch = patch[(patch[:, 0] > 8) & (patch[:, 0] < 97)]          # skip black / blown pixels
    if len(patch) < 20:
        return None
    return float(np.median(patch[:, 1])), float(np.median(patch[:, 2]))


def face_shift(before: np.ndarray, after: np.ndarray) -> dict | None:
    """How far the colour edit moved the main face's skin colour, in temperature/tint units.

    Same face box on both images (detected on `before`). When over FACE_SHIFT_LIMIT, gives the
    local correction for the people stage (apply_retouch local, region 'skin') that brings the
    face back to the limit while the rest of the photo keeps the style."""
    from .composition import detect_faces
    if before.shape[:2] != after.shape[:2]:
        after = cv2.resize(after, (before.shape[1], before.shape[0]), interpolation=cv2.INTER_AREA)
    faces = [f for f in detect_faces((np.clip(before, 0, 1)[..., ::-1] * 255).astype(np.uint8)) if f["score"] >= 0.7]
    if not faces:
        return None
    ab0, ab1 = _skin_ab(before, faces[0]["box"]), _skin_ab(after, faces[0]["box"])
    if ab0 is None or ab1 is None:
        return None
    t = (ab1[1] - ab0[1]) / _DB_PER_TEMP          # + = face got warmer (yellower)
    ti = (ab1[0] - ab0[0]) / _DA_PER_TINT         # + = face got more magenta
    lt, lti = FACE_SHIFT_LIMIT["temperature"], FACE_SHIFT_LIMIT["tint"]
    over_t = t - np.clip(t, -lt, lt)
    over_ti = ti - np.clip(ti, -lti, lti)
    out = {"temperature": round(float(t), 1), "tint": round(float(ti), 1),
           "limit": FACE_SHIFT_LIMIT, "ok": bool(abs(t) <= lt and abs(ti) <= lti)}
    if not out["ok"]:
        # Target: back to half the limit (the skin mask is soft, so aim inside, not at the edge).
        # Solve on the face's own pixels: skin reacts to the sliders differently from grey.
        tgt_b = ab0[1] + 0.5 * float(np.clip(t, -lt, lt)) * _DB_PER_TEMP
        tgt_a = ab0[0] + 0.5 * float(np.clip(ti, -lti, lti)) * _DA_PER_TINT
        ct, cti = _solve_skin_wb(after, faces[0]["box"], tgt_a, tgt_b, -over_t, -over_ti)
        out["skin_correction"] = {"region": "skin", "temperature": ct, "tint": cti}
    return out


def _solve_skin_wb(rgb, box, tgt_a, tgt_b, t0, ti0, iters=4):
    """Local temperature/tint that moves the face patch's median (a, b) to the target."""
    x, y, w, h = box
    patch = rgb[max(int(y + 0.3 * h), 0):int(y + 0.8 * h), max(int(x + 0.25 * w), 0):int(x + 0.75 * w)]
    if patch.size == 0:
        return round(float(t0), 1), round(float(ti0), 1)
    patch = np.ascontiguousarray(cv2.resize(patch, (48, 48), interpolation=cv2.INTER_AREA))

    def ab(t_, ti_):
        o, _ = _render_small(patch, {"temperature": t_, "tint": ti_})
        lab = _lab(o).reshape(-1, 3)
        return float(np.median(lab[:, 1])), float(np.median(lab[:, 2]))

    t_, ti_ = float(t0), float(ti0)
    for _ in range(iters):
        a_, b_ = ab(t_, ti_)
        da_dt = (ab(t_, ti_ + 10)[0] - a_) / 10 or _DA_PER_TINT
        db_dt = (ab(t_ + 10, ti_)[1] - b_) / 10 or _DB_PER_TEMP
        t_ = float(np.clip(t_ + (tgt_b - b_) / db_dt, -100, 100))
        ti_ = float(np.clip(ti_ + (tgt_a - a_) / da_dt, -100, 100))
    return round(t_, 1), round(ti_, 1)


def face_shift_issue(fs: dict | None) -> list[dict]:
    if not fs:
        return []
    return [{"rule": "face colour close to the original", "status": "pass" if fs["ok"] else "fail",
             "value": [fs["temperature"], fs["tint"]],
             "detail": (f"face skin moved by temperature {fs['temperature']:+.0f}, tint {fs['tint']:+.0f} "
                        f"(limit +-{FACE_SHIFT_LIMIT['temperature']:.0f} / +-{FACE_SHIFT_LIMIT['tint']:.0f}). "
                        + ("" if fs["ok"] else
                           "Keep the style for the scene and correct the face in the people stage: "
                           f"apply_retouch local [{json.dumps(fs['skin_correction'])}]"))}]


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #
def analyze_color(image_path: str) -> dict:
    """Measure exposure, clipping, colour cast (+ WB suggestion), skin tone, dominant colours."""
    rgb = _load_preview(image_path)
    m = measure(rgb)
    return {**m, "issues": [i for i in color_issues(m) if i["status"] == "fail"],
            "recipes": {k: v["label"] + " - " + v["description"] for k, v in RECIPES.items()}}


def _set_path(params: dict, key: str, value):
    """Set e.g. 'exposure', 'hsl.blue.saturation', 'grading.shadows.hue' on a params dict copy."""
    p = copy.deepcopy(params or {})
    parts = key.split(".")
    cur = p
    for part in parts[:-1]:
        cur = cur.setdefault(part, {})
    cur[parts[-1]] = value
    return p


def preview_variants(image_path: str, param: str, values: list[float], params: dict | None = None,
                     recipe: str | None = None, recipe_strength: float = 1.0,
                     sheet_path: str | None = None) -> dict:
    """Render the same photo with one parameter at several values, side by side, to choose visually."""
    if not 2 <= len(values) <= 6:
        raise ValueError("give 2 to 6 values")
    if param == "recipe_strength":
        if not recipe:
            raise ValueError("param 'recipe_strength' needs a recipe")
    rgb = _load_preview(image_path, 640)
    tiles, labels = [], []
    for i, v in enumerate(values):
        if param == "recipe_strength":
            out, _ = _render_small(rgb, params, recipe, float(v))
        else:
            out, _ = _render_small(rgb, _set_path(params, param, v), recipe, recipe_strength)
        tag = chr(ord("A") + i)
        labels.append({"tile": tag, param: v})
        tiles.append(_label((out[..., ::-1] * 255).astype(np.uint8).copy(), f"{tag}: {param}={v:g}"))
    if sheet_path is None:
        p_ = Path(image_path)
        sheet_path = str(p_.with_name(f"{p_.stem}_var_{param.replace('.', '_')}.jpg"))
    _sheet(tiles, sheet_path, cols=3)
    return {"sheet_path": sheet_path, "tiles": labels,
            "note": "Pick the tile that looks best; use its value."}


def preview_looks(image_path: str, params: dict | None = None, recipes: list[str] | None = None,
                  strength: float = 0.7, sheet_path: str | None = None) -> dict:
    """Render the base correction plus each look recipe on top, as a labelled contact sheet."""
    names = recipes or list(RECIPES)
    rgb = _load_preview(image_path, 520)
    base, _ = _render_small(rgb, params)
    tiles = [_label((base[..., ::-1] * 255).astype(np.uint8).copy(), "0: base (no look)")]
    for i, name in enumerate(names, 1):
        out, _ = _render_small(rgb, params, name, strength)
        tiles.append(_label((out[..., ::-1] * 255).astype(np.uint8).copy(), f"{i}: {name}"))
    if sheet_path is None:
        p_ = Path(image_path)
        sheet_path = str(p_.with_name(f"{p_.stem}_looks.jpg"))
    _sheet(tiles, sheet_path, cols=3)
    return {"sheet_path": sheet_path, "strength": strength,
            "tiles": [{"tile": 0, "recipe": None}] + [{"tile": i, "recipe": n, "label": RECIPES[n]["label"]}
                                                      for i, n in enumerate(names, 1)]}


def render_adjustments(image_path: str, params: dict | None = None, recipe: str | None = None,
                       recipe_strength: float = 1.0, preview_path: str | None = None) -> dict:
    """Fast preview of the full adjustment, with measurements and rule checks of the result."""
    rgb = _load_preview(image_path)
    out, eff = _render_small(rgb, params, recipe, recipe_strength)
    if preview_path is None:
        p_ = Path(image_path)
        preview_path = str(p_.with_name(f"{p_.stem}_color_preview.jpg"))
    _save_rgb8(preview_path, out)
    m = measure(out)
    fs = face_shift(rgb, out)
    return {"preview_path": preview_path, "effective_params": _compact(eff), "measurements": m,
            "face_shift": fs,
            "issues": [i for i in color_issues(m) + face_shift_issue(fs) if i["status"] == "fail"]}


def apply_color(image_path: str, params: dict | None = None, recipe: str | None = None,
                recipe_strength: float = 1.0, output_path: str | None = None) -> dict:
    """Render at full resolution (16-bit TIFF) in strips, plus a JPEG preview and a params sidecar."""
    img = read_image(image_path)
    H, W = img.shape[:2]
    p, rc = _merge(params, recipe, recipe_strength)
    lut = _curve_lut(p, rc)
    small = cv2.resize(img, None, fx=512 / max(H, W), fy=512 / max(H, W), interpolation=cv2.INTER_AREA)
    a_full, b_full = _upsample_ab(_local_base(_to_rgb_float(small), p), (H, W))
    out = np.empty((H, W, 3), np.uint16)
    strip = 512   # pointwise ops run in strips to keep memory low on 24+ MP files
    for i, y0 in enumerate(range(0, H, strip)):
        y1 = min(H, y0 + strip)
        d = _render(_to_rgb_float(img[y0:y1]), p, (a_full[y0:y1], b_full[y0:y1]), lut, rc, seed=i)
        out[y0:y1] = (np.clip(d, 0, 1)[..., ::-1] * 65535 + 0.5).astype(np.uint16)
    src = Path(image_path)
    if output_path is None:
        output_path = str(src.with_name(f"{src.stem}_color.tif"))
    write_image(output_path, out)
    prev = cv2.resize(out, None, fx=min(1.0, 1600 / max(H, W)), fy=min(1.0, 1600 / max(H, W)),
                      interpolation=cv2.INTER_AREA)
    preview_path = str(Path(output_path).with_name(Path(output_path).stem + "_preview.jpg"))
    write_image(preview_path, (prev / 257).astype(np.uint8), jpeg_quality=90)
    sidecar = str(Path(output_path).with_suffix(".params.json"))
    Path(sidecar).write_text(json.dumps({"params": params or {}, "recipe": recipe,
                                         "recipe_strength": recipe_strength,
                                         "effective": _compact(p)}, indent=1), encoding="utf-8")
    k = 1024 / max(H, W)
    fs = face_shift(_to_rgb_float(cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)),
                    _to_rgb_float(cv2.resize(out, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)))
    return {"output_path": output_path, "preview_path": preview_path, "params_path": sidecar,
            "output_size": {"w": W, "h": H}, "effective_params": _compact(p), "face_shift": fs,
            "issues": [i for i in face_shift_issue(fs) if i["status"] == "fail"]}


# --------------------------------------------------------------------------- #
# Reference matching (样片追色): fit OUR sliders so the photo takes on a sample's look.
# Like PixCake's AI追色 with 精准控制, the result is ordinary editable sliders: white balance,
# exposure / highlights / shadows / whites / blacks, saturation, 8-colour HSL and a gentle
# tone curve. It matches statistics, not content, so the strength and a tone limit keep a
# night photo from being forced to daylight brightness.
# --------------------------------------------------------------------------- #
_Q = (1, 10, 25, 50, 75, 90, 99)


def _Y(Lstar: float) -> float:
    """CIE L* -> relative luminance (floored so dark night pixels stay finite in stops)."""
    f = (max(Lstar, 0.5) + 16) / 116
    return max(f ** 3 if f > 6 / 29 else 3 * (6 / 29) ** 2 * (f - 4 / 29), 1e-4)


def _Lstar(Y: float) -> float:
    Y = max(Y, 1e-6)
    return 116 * Y ** (1 / 3) - 16 if Y > (6 / 29) ** 3 else Y * (29 / 3) ** 3


def _look_stats(rgb: np.ndarray) -> dict:
    lab = _lab(rgb)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    q = dict(zip(_Q, np.percentile(L, _Q)))
    mid = (L > 15) & (L < 90)
    hsv = cv2.cvtColor(np.clip(rgb, 0, 1).astype(np.float32), cv2.COLOR_RGB2HSV)
    chroma = np.hypot(a, b)
    coloured = mid & (hsv[..., 1] > 0.12)
    bands = {}
    n = L.size
    for color, w in zip(COLORS, _band_weights(hsv[..., 0])):
        w = w * coloured
        tot = float(w.sum())
        if tot / n < 0.01:
            continue
        ang = np.radians(hsv[..., 0])
        hue = math.degrees(math.atan2(float((w * np.sin(ang)).sum()), float((w * np.cos(ang)).sum()))) % 360
        bands[color] = {"frac": tot / n, "hue": hue, "chroma": float((w * chroma).sum() / tot),
                        "L": float((w * L).sum() / tot)}
    return {"q": q, "a": float(a[mid].mean()) if mid.any() else 0.0, "b": float(b[mid].mean()) if mid.any() else 0.0,
            "chroma": float(chroma[mid].mean()) if mid.any() else 0.0, "bands": bands}


# How far reference matching may move each slider. A sample usually shows different content
# (a sunny street vs. a night portrait), so its statistics are only a guide: these limits keep
# the photo's own character, and orange (skin) is protected most.
MATCH_LIMITS = {"temperature": 50, "tint": 30, "exposure": 0.8, "contrast": 40, "highlights": 50,
                "shadows": 50, "whites": 30, "blacks": 30, "saturation": 40,
                "hsl_hue": 20, "hsl_sat": 35, "hsl_lum": 20, "skin_hue": 8, "skin_sat": 20, "skin_lum": 12,
                "curve": 20}


def fit_reference(src: np.ndarray, ref: np.ndarray, iters: int = 7, match_tone: bool = True,
                  match_color: bool = True) -> tuple[dict, dict]:
    """Iteratively adjust slider values until src's look statistics approach ref's.

    Colour is matched directly (midtone balance, chroma, per-colour hue/saturation/lightness).
    Tone is matched by SHAPE, relative to each photo's own median: contrast from the spread
    (p90 - p10), highlights / shadows from their distance to the median, and only a damped,
    limited exposure shift, so a night photo keeps being a night photo."""
    R = _look_stats(ref)
    L = MATCH_LIMITS
    p: dict = {"hsl": {}}
    lim = lambda k, v: float(np.clip(v, -L[k], L[k]))
    for it in range(iters):
        out, _ = _render_small(src, p)
        S = _look_stats(out)
        rq, sq = R["q"], S["q"]
        if match_color:
            p["temperature"] = lim("temperature", p.get("temperature", 0) + 0.7 * (R["b"] - S["b"]) / _DB_PER_TEMP)
            p["tint"] = lim("tint", p.get("tint", 0) + 0.7 * (R["a"] - S["a"]) / _DA_PER_TINT)
        if match_tone:   # brightness in stops (log), damped and capped: a night photo stays a night photo
            p["exposure"] = lim("exposure", p.get("exposure", 0) + 0.5 * math.log2(_Y(rq[50]) / _Y(sq[50])))
        if match_color and S["chroma"] > 1:
            p["saturation"] = lim("saturation", p.get("saturation", 0) + 50 * (R["chroma"] / S["chroma"] - 1))
        if match_color and it >= 1:                   # per-colour refinement once the global balance is close
            for color, rb in R["bands"].items():
                sb = S["bands"].get(color)
                if not sb:
                    continue
                skin = color == "orange"
                hmax = L["skin_hue"] if skin else L["hsl_hue"]
                smax = L["skin_sat"] if skin else L["hsl_sat"]
                lmax = L["skin_lum"] if skin else L["hsl_lum"]
                cur = p["hsl"].setdefault(color, {"hue": 0.0, "saturation": 0.0, "luminance": 0.0})
                dh = (rb["hue"] - sb["hue"] + 180) % 360 - 180
                cur["hue"] = float(np.clip(cur["hue"] + 0.4 * dh / 30 * 100, -hmax, hmax))
                cur["saturation"] = float(np.clip(cur["saturation"] + 30 * (rb["chroma"] / max(sb["chroma"], 1) - 1),
                                                  -smax, smax))
                cur["luminance"] = float(np.clip(cur["luminance"] + 0.8 * ((rb["L"] - rq[50]) - (sb["L"] - sq[50])),
                                                 -lmax, lmax))
    # light/dark curve: match the SHAPE in stops around each photo's own median (scale-free,
    # so a dark photo gets the sample's contrast character without being forced brighter)
    if match_tone:
        out, _ = _render_small(src, p)
        S = _look_stats(out)
        y50 = _Y(S["q"][50])
        pts = []
        for qq in (10, 25, 50, 75, 90):
            st_s = math.log2(_Y(S["q"][qq]) / y50)
            st_r = math.log2(_Y(R["q"][qq]) / _Y(R["q"][50]))
            target_L = _Lstar(y50 * 2 ** (st_s + 0.6 * (st_r - st_s)))
            x = S["q"][qq] / 100 * 255
            y = x + float(np.clip((target_L - S["q"][qq]) / 100 * 255, -L["curve"], L["curve"]))
            pts.append([round(float(np.clip(x, 1, 254)), 1), round(float(np.clip(y, 1, 254)), 1)])
        pts.sort()
        mono = []
        for x, y in pts:                              # keep it monotone and strictly increasing in x
            if mono and (x <= mono[-1][0] + 4 or y <= mono[-1][1]):
                continue
            mono.append([x, y])
        if len(mono) >= 2 and any(abs(y - x) > 3 for x, y in mono):
            p["curve"] = mono
    p["hsl"] = {c: {k: round(v, 1) for k, v in a.items()} for c, a in p["hsl"].items() if any(abs(v) > 0.5 for v in a.values())}
    p = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in p.items() if v not in (0, 0.0, {}, None)}
    out, _ = _render_small(src, p)
    S = _look_stats(out)
    report = {"lightness_quantiles_ref": {k: round(v, 1) for k, v in R["q"].items()},
              "lightness_quantiles_result": {k: round(v, 1) for k, v in S["q"].items()},
              "midtone_ab_ref": [round(R["a"], 2), round(R["b"], 2)],
              "midtone_ab_result": [round(S["a"], 2), round(S["b"], 2)],
              "chroma_ref_result": [round(R["chroma"], 1), round(S["chroma"], 1)]}
    return p, report


def scale_params(p: dict, s: float) -> dict:
    """Scale every slider toward 0 (and the curve toward identity) by strength s."""
    out = {}
    for k, v in p.items():
        if k in SCALARS:
            out[k] = round(v * s, 3)
        elif k == "hsl":
            out[k] = {c: {kk: round(vv * s, 1) for kk, vv in a.items()} for c, a in v.items()}
        elif k == "curve":
            out[k] = [[x, round(x + s * (y - x), 1)] for x, y in v]
        else:
            out[k] = v
    return out


def match_reference(image_path: str, reference_path: str, strength: float = 0.8, match_tone: bool = True,
                    sheet_path: str | None = None) -> dict:
    """Generate slider values that give image_path the colour / exposure / contrast of a sample photo."""
    src = _load_preview(image_path, 512)
    ref = _load_preview(reference_path, 512)
    full_p, report = fit_reference(src, ref, match_tone=match_tone)
    p = scale_params(full_p, float(np.clip(strength, 0, 1)))
    big = _load_preview(image_path, 640)
    out, _ = _render_small(big, p)
    if sheet_path is None:
        p_ = Path(image_path)
        sheet_path = str(p_.with_name(f"{p_.stem}_match.jpg"))
    th = 420
    tiles = []
    for im, t in ((_load_preview(reference_path, 640), "sample"), (big, "before"),
                  (out, f"after (strength {strength:g})")):
        k = th / im.shape[0]
        tiles.append(_label(cv2.resize((np.clip(im, 0, 1)[..., ::-1] * 255).astype(np.uint8),
                                       (max(1, round(im.shape[1] * k)), th)), t))
    _sheet(tiles, sheet_path, cols=3)
    m = measure(out)
    fs = face_shift(big, out)
    return {"params": p, "full_strength_params": full_p, "strength": strength, "match_report": report,
            "sheet_path": sheet_path, "face_shift": fs,
            "issues": [i for i in color_issues(m) + face_shift_issue(fs) if i["status"] == "fail"],
            "note": "Use params as the base for render_adjustments / apply_color (a look recipe can still "
                    "be added). Statistics are matched, not content: check the faces stay natural."}


def check_color(image_path: str, min_score: float = 0.8) -> dict:
    """Evaluator: classical correctness checks on a finished image (not exposed to the model)."""
    m = measure(_load_preview(image_path))
    checks = color_issues(m)
    score = sum(c["status"] == "pass" for c in checks) / len(checks) if checks else 1.0
    return {"status": "pass" if score >= min_score else "fail", "score": round(score, 3),
            "failed_rules": [c["rule"] for c in checks if c["status"] == "fail"],
            "checks": checks, "measurements": m}


def _compact(p: dict) -> dict:
    """Drop zero / empty entries so effective params read like a short recipe."""
    out = {k: round(v, 3) for k, v in p.items() if k in SCALARS and v}
    hsl = {c: {k: round(v, 1) for k, v in a.items() if v} for c, a in p["hsl"].items()}
    hsl = {c: a for c, a in hsl.items() if a}
    if hsl:
        out["hsl"] = hsl
    gr = {b: {"hue": round(a["hue"], 1), "saturation": round(a["saturation"], 1)}
          for b, a in p["grading"].items() if a["saturation"]}
    if gr:
        out["grading"] = gr
    if p["curve"]:
        out["curve"] = p["curve"]
    if p.get("lut"):
        out["film_lut"] = {"name": p["lut"]["name"], "strength": round(p["lut"]["strength"], 2)}
    return out
