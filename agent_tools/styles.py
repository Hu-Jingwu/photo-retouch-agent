"""Style library (样片库) and a judge for which style fits a photo.

Eleven reference photos live in samples/ (credits in samples/SOURCES.md). suggest_styles
reads the photo's situation - day or night, people or landscape, sky, season cues in the
colours (fresh green and pink blossom, deep summer blue, orange foliage, snow) - scores how
well each style fits, and renders the top choices with match_reference so the model can
look before choosing. The scores are reasons to look, not the decision: the model decides.
"""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from . import color as C
from .composition import detect_faces

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "samples"

STYLES = {
    # samples chosen by the owner (spring..blue_hour, night_portrait, street*, wedding_light)
    "spring": {"label": "春 Spring", "file": "spring.jpg",
               "look": "pink cherry blossom against a clear sky-blue, bright and fresh"},
    "summer": {"label": "夏 Summer", "file": "summer.jpg",
               "look": "vivid saturated colour, deep blue sky and sea, strong sunlight"},
    "autumn": {"label": "秋 Autumn", "file": "autumn.jpg",
               "look": "golden autumn: rich orange-gold and red foliage, blue sky, white clouds"},
    "winter": {"label": "冬 Winter", "file": "winter.jpg",
               "look": "clean bright snow whites, cool shadows, turquoise water"},
    "film": {"label": "胶片感 Film", "file": "film.jpg",
             "look": "Portra-like soft contrast, warm natural skin, gentle greens"},
    "epic_landscape": {"label": "风光大片 Epic landscape", "file": "epic_landscape.jpg",
                       "look": "rich greens, golden sunset sky, deep contrast"},
    "blue_hour": {"label": "蓝调 Blue hour", "file": "blue_hour.jpg",
                  "look": "deep blue night, cool navy shadows and snow, teal-green highlights"},
    "night_portrait": {"label": "夜景人像 Night portrait", "file": "night_portrait.jpg",
                       "look": "close night portrait: bright clean skin against a dark background with warm bokeh"},
    "street_night": {"label": "街头夜景 Night street", "file": "street_night.jpg",
                     "look": "person in a neon-lit street at night: warm orange-red lights, glowing bokeh"},
    "street": {"label": "街景 Street", "file": "street.jpg",
               "look": "daylight city street portrait: soft muted tones, clean neutral greys, warm skin"},
    "wedding_light": {"label": "婚礼浅色 Light wedding", "file": "wedding_light.jpg",
                      "look": "bright airy high-key: white and pale grey-blue, low saturation, soft contrast"},
}


def sample_path(style: str) -> str:
    if style not in STYLES:
        raise ValueError(f"unknown style {style!r}; use one of {list(STYLES)}")
    return str(SAMPLES / STYLES[style]["file"])


def available_styles() -> list[str]:
    """Styles whose sample photo is present in samples/ (the public repo ships only some)."""
    return [s for s in STYLES if (SAMPLES / STYLES[s]["file"]).exists()]


def _need_sample(style: str) -> None:
    if not (SAMPLES / STYLES[style]["file"]).exists():
        raise FileNotFoundError(
            f"style {style!r} has no sample photo: add your own as samples/{STYLES[style]['file']} "
            f"(see samples/README.md). Styles available now: {available_styles()}")


def add_params(base: dict, extra: dict, s: float = 1.0) -> dict:
    """base + s * extra (sliders add; HSL adds per colour; grading bands from extra are blended in)."""
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in base.items()}
    for k, v in extra.items():
        if k in C.SCALARS:
            lo, hi = C.SCALARS[k]
            out[k] = round(float(np.clip(out.get(k, 0) + s * v, lo, hi)), 3)
        elif k == "hsl":
            hsl = {c: dict(a) for c, a in out.get("hsl", {}).items()}
            for color, adj in v.items():
                cur = hsl.setdefault(color, {})
                for kk, vv in adj.items():
                    cur[kk] = round(float(np.clip(cur.get(kk, 0) + s * vv, -100, 100)), 1)
            out["hsl"] = hsl
        elif k == "grading":
            g = dict(out.get("grading", {}))
            for band, adj in v.items():
                g[band] = {"hue": adj["hue"], "saturation": round(adj["saturation"] * s, 1)}
            out["grading"] = g
    return out


def style_params(image_path: str, style: str, strength: float = 0.8, sheet_path: str | None = None) -> dict:
    """Slider values for one library style: match its sample, then add the style's own bias."""
    if style not in STYLES:
        raise ValueError(f"unknown style {style!r}; use one of {list(STYLES)}")
    _need_sample(style)
    src = C._load_preview(image_path, 512)
    params, report = C.fit_reference(src, C._load_preview(sample_path(style), 512),
                                     match_color=STYLES[style].get("match_color", True))
    params = C.scale_params(params, strength)
    if STYLES[style].get("bias"):
        params = add_params(params, STYLES[style]["bias"], strength)
    big = C._load_preview(image_path, 640)
    out, _ = C._render_small(big, params)
    if sheet_path is None:
        p = Path(image_path)
        sheet_path = str(p.with_name(f"{p.stem}_style_{style}.jpg"))
    th = 420
    tiles = []
    for im, t in ((C._load_preview(sample_path(style), 640), f"sample: {style}"), (big, "before"),
                  (out, f"after: {style} {strength:g}")):
        k = th / im.shape[0]
        tiles.append(C._label(cv2.resize((np.clip(im, 0, 1)[..., ::-1] * 255).astype(np.uint8),
                                         (max(1, round(im.shape[1] * k)), th)), t))
    C._sheet(tiles, sheet_path, cols=3)
    m = C.measure(out)
    fs = C.face_shift(big, out)
    return {"style": style, "params": params, "sheet_path": sheet_path, "match_report": report,
            "face_shift": fs,
            "issues": [i for i in C.color_issues(m) + C.face_shift_issue(fs) if i["status"] == "fail"]}


def situation(rgb: np.ndarray) -> dict:
    """Measured cues about the scene, all 0..1 unless noted."""
    h, w = rgb.shape[:2]
    lab = C._lab(rgb)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    chroma = np.hypot(a, b)
    hsv = cv2.cvtColor(np.clip(rgb, 0, 1).astype(np.float32), cv2.COLOR_RGB2HSV)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    med = float(np.median(L))
    faces = [f for f in detect_faces((np.clip(rgb, 0, 1)[..., ::-1] * 255).astype(np.uint8)) if f["score"] >= 0.7]
    coloured = (S > 0.2) & (V > 0.15)
    frac = lambda m: float(m.mean())
    top = slice(0, h // 3)
    sky_blue = frac(((H[top] > 190) & (H[top] < 250) & (S[top] > 0.25) & (V[top] > 0.35)))
    cues = {
        "median_L": round(med, 1),
        "night": float(np.clip((30 - med) / 15, 0, 1)),                    # dark overall
        "dusk": float(np.clip(1 - abs(med - 32) / 18, 0, 1)),
        "people": min(1.0, len(faces)),
        "face_size": round(max((f["box"][3] / h for f in faces), default=0.0), 3),
        "bright_lights": frac((L > 92) & (L < 101)),                       # lamps / lit windows
        "sky_blue": sky_blue,
        "green": frac(coloured & (H > 70) & (H < 160)),
        "foliage_warm": frac(coloured & (H > 15) & (H < 50) & (L > 20)),
        "blossom": frac((L > 60) & (chroma > 4) & (chroma < 60) & ((H > 300) | (H < 15))),   # pale to saturated pink
        # snow lies on the ground: only the lower two thirds count (bright clouds are not snow)
        "snow": frac((L[h // 3:] > 78) & (chroma[h // 3:] < 8)),
        "turquoise_blue": frac(coloured & (H > 170) & (H < 230) & (S > 0.35)),
        "saturation": float(chroma.mean()),
        "landscape_wide": 1.0 if w / h > 1.3 else 0.0,
        "built": _built(rgb),                                              # straight-line architecture
    }
    return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in cues.items()}


def _built(rgb: np.ndarray) -> float:
    """How much of the frame is man-made: total length of long straight lines (0..1)."""
    g = cv2.cvtColor((np.clip(rgb, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    h, w = g.shape
    segs = cv2.createLineSegmentDetector().detect(g)[0]
    if segs is None:
        return 0.0
    seg = segs.reshape(-1, 4)
    ln = np.hypot(seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1])
    return float(np.clip(ln[ln > 0.06 * max(h, w)].sum() / (6 * (h + w)), 0, 1))


def score_styles(c: dict, only_available: bool = True) -> list[dict]:
    day = 1 - c["night"]
    people = c["people"]
    big_person = float(np.clip(c["face_size"] / 0.06, 0, 1)) if people else 0.0
    built = c.get("built", 0.0)
    city_lights = min(1.0, c["bright_lights"] * 15)
    rules = {
        "night_portrait": (c["night"] * people * (0.5 + 0.5 * big_person) + 0.1 * city_lights * people,
                           "dark scene with a person close to the camera"),
        "street_night": (c["night"] * people * (0.4 + 0.3 * built + 0.4 * city_lights) * (1 - 0.3 * big_person),
                         "person in a street at night with city lights"),
        "street": (day * people * (0.2 + 0.8 * built) * (1 - min(1, 2 * c["green"])),
                   "person in a daylight city street (buildings, straight lines)"),
        "wedding_light": (people * float(np.clip((c["median_L"] - 50) / 25, 0, 1))
                          * float(np.clip((22 - c["saturation"]) / 12, 0, 1)),
                          "bright, pale, low-saturation scene with people (white dress, indoor light)"),
        "blue_hour": (0.7 * max(c["night"], c["dusk"] * 0.6) * (1 - 0.6 * big_person)
                      + 0.6 * c["sky_blue"] + 0.2 * min(1, c["bright_lights"] * 20),
                      "dusk/night tones, blue sky or city lights, no close-up person"),
        "spring": (day * (1.5 * c["green"] + 4 * c["blossom"]), "fresh green and pink/white blossom in daylight"),
        "summer": (day * (1.2 * c["sky_blue"] + 2.5 * c["turquoise_blue"] + 0.01 * c["saturation"])
                   * max(0.0, 1 - 3 * c["blossom"] - 1.5 * c["foliage_warm"] - 2 * c["snow"]),
                   "bright blue sky / sea, saturated daylight"),
        "autumn": (day * 2.0 * c["foliage_warm"], "warm orange/brown foliage"),
        "winter": (day * 3.0 * c["snow"], "snow and cool whites"),
        "epic_landscape": (day * (1 - 0.7 * big_person) * (1 - c["foliage_warm"]) * (0.5 * c["landscape_wide"] + 0.8 * c["sky_blue"]
                                                           + 0.8 * c["green"]), "open landscape with sky"),
        "film": (0.35 + 0.35 * day * people, "general-purpose soft film look; good for daylight people"),
    }
    out = []
    have = set(available_styles()) if only_available else set(STYLES)
    for name, (s, why) in rules.items():
        if name not in have:
            continue
        out.append({"style": name, "label": STYLES[name]["label"], "score": round(float(np.clip(s, 0, 1)), 3),
                    "why": why, "look": STYLES[name]["look"], "sample_path": sample_path(name)})
    return sorted(out, key=lambda r: -r["score"])


def suggest_styles(image_path: str, top_k: int = 3, strength: float = 0.8,
                   sheet_path: str | None = None) -> dict:
    """Rank the library styles for this photo and preview the best ones (sample -> matched result)."""
    rgb = C._load_preview(image_path, 640)
    cues = situation(rgb)
    ranked = score_styles(cues)
    if not ranked:
        raise FileNotFoundError("no style samples found in samples/ (see samples/README.md)")
    top = ranked[:max(1, min(top_k, len(ranked)))]
    if sheet_path is None:
        p = Path(image_path)
        sheet_path = str(p.with_name(f"{p.stem}_styles.jpg"))
    src_small = C._load_preview(image_path, 512)
    th = 300
    tiles = [C._label(cv2.resize((rgb[..., ::-1] * 255).astype(np.uint8),
                                 (round(rgb.shape[1] * th / rgb.shape[0]), th)), "photo now")]
    for r in top:
        ref = C._load_preview(r["sample_path"], 512)
        params, _ = C.fit_reference(src_small, ref, match_color=STYLES[r["style"]].get("match_color", True))
        params = C.scale_params(params, strength)
        if STYLES[r["style"]].get("bias"):
            params = add_params(params, STYLES[r["style"]]["bias"], strength)
        out, _ = C._render_small(rgb, params)
        r["params"] = params
        smp = C._load_preview(r["sample_path"], 400)
        k = th / smp.shape[0]
        tiles.append(C._label(cv2.resize((smp[..., ::-1] * 255).astype(np.uint8),
                                         (round(smp.shape[1] * k), th)), f"sample: {r['style']}"))
        tiles.append(C._label(cv2.resize((np.clip(out, 0, 1)[..., ::-1] * 255).astype(np.uint8),
                                         (round(out.shape[1] * th / out.shape[0]), th)),
                              f"{r['style']} {r['score']:.2f}"))
    C._sheet(tiles, sheet_path, cols=3)
    return {"situation": cues, "ranking": [{k: v for k, v in r.items() if k != "params"} for r in ranked],
            "previews": [{"style": r["style"], "params": r["params"]} for r in top],
            "sheet_path": sheet_path,
            "note": "Scores are cues, not the decision. Look at the previews; then call apply_style for the "
                    "chosen style (or use its preview params directly in render_adjustments)."}
