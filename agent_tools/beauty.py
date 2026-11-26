"""Beauty retouch (网红脸 / 像素蛋糕写真 look) as ONE deterministic tool.

The orchestrator LLM only chooses the level (natural / xiezhen / wanghong) and what to avoid.
Everything else is Python:

  1. measure each face (face.measure_face)
  2. plan_beauty: turn measurements into slider values (fill only the gap to a target, per face)
  3. render at work size and run the guards
  4. auto-repair: a failed guard lowers only the sliders that caused it (up to 3 times)
  5. render the final sliders at full resolution and return a close-up before/after sheet

Rendering and the identity guards come from face.py; this module adds the plan, the new guards
(plastic skin, symmetry, proportions, bent background lines) and the repair loop.
See docs/beauty_agent_design.md.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from . import color as C
from . import face as F
from . import people as P
from .io_utils import write_image

# ---- levels: the MOST each slider may reach (a face that needs less gets less) --------------
_SHAPE0 = {"face_slim": 0, "v_face": 0, "jaw": 0, "small_face": 0, "chin_length": 0,
           "eye_size": 0, "nose_size": 0, "nose_wings": 0, "mouth_size": 0}
LEVELS = {
    # 自然: clean skin and bright eyes, the face shape is never changed
    "natural": {**_SHAPE0, "acne_spots": 80, "dark_circles": 40, "oil_shine": 40, "nasolabial": 0,
                "smoothing": 35, "texture_keep": 60, "tone_unify": 30, "luminous": 10, "whitening": 0,
                "rosy": 0, "eye_brighten": 25, "iris": 0, "eye_whites": 20, "teeth_whiten": 25,
                "lipstick": 0, "contour": 0},
    # 写真: PixCake's own 写真 one-click values (face.PIXCAKE_XIEZHEN)
    "xiezhen": {**_SHAPE0, **F.PIXCAKE_XIEZHEN, "texture_keep": 45, "rosy": 0, "v_face": 0, "chin_length": 0,
                "lipstick": 30, "contour": 45},
    # 网红: the strongest look the identity caps allow
    "wanghong": {"acne_spots": 100, "dark_circles": 85, "oil_shine": 40, "nasolabial": 60,
                 "smoothing": 75, "texture_keep": 40, "tone_unify": 55, "luminous": 30, "whitening": 20,
                 "rosy": 20, "face_slim": 35, "v_face": 30, "jaw": 45, "small_face": 30, "chin_length": 15,
                 "eye_size": 55, "nose_size": 45, "nose_wings": 40, "mouth_size": 30,
                 "eye_brighten": 50, "iris": 80, "eye_whites": 35, "teeth_whiten": 45,
                 "lipstick": 50, "contour": 65},
}

# ---- target face (the "网红" proportions) and how big a gap counts as a full gap -------------
# Ratios are MediaPipe-landmark widths over the cheekbone width (face.measure_face).
TARGET = {"eye_width_rel": (0.205, 0.03),     # larger eyes
          "nose_width_rel": (0.24, 0.04),     # narrower nose
          "jaw_to_cheek_ratio": (0.70, 0.12),  # narrower jaw (V-face)
          "face_aspect_h_w": (1.20, 0.15),    # slightly longer, oval face
          "mouth_width_rel": (0.33, 0.05),    # slightly smaller mouth
          "lip_chroma": (34.0, 12.0),         # richer lip colour (C*)
          "face_relief": (7.0, 4.0)}          # some light-and-shadow shape (flat light -> contour)
LIP_SHADE = {"natural": "natural", "xiezhen": "rose", "wanghong": "red"}   # default per level
LIP_FLOOR = {"natural": 25, "xiezhen": 35, "wanghong": 50}   # lipstick when the user names a shade

GROUPS = {   # names the LLM can pass in `avoid`
    "skin": ["acne_spots", "dark_circles", "eye_bags", "oil_shine", "nasolabial", "smoothing", "tone_unify"],
    "brighten": ["luminous", "whitening"],
    "makeup": ["lipstick", "rosy"],
    "contour": ["contour"],
    "shape": ["face_slim", "v_face", "jaw", "small_face", "chin_length", "cheekbone"],
    "eyes": ["eye_size"],
    "eye_color": ["eye_brighten", "iris", "eye_whites"],
    "nose": ["nose_size", "nose_wings"],
    "mouth": ["mouth_size"],
    "teeth": ["teeth_whiten"],
}
SHAPE_KEYS = GROUPS["shape"] + GROUPS["eyes"] + GROUPS["nose"] + GROUPS["mouth"]

# proportions that no edit may produce (stops 蛇精脸 when several warps stack)
PLAUSIBLE = {"eye_width_rel": (None, 0.235), "nose_width_rel": (0.20, None),
             "jaw_to_cheek_ratio": (0.62, None), "face_aspect_h_w": (None, 1.35)}


def _gap(value, key, direction):
    """0..1: how far `value` is from the target, in units of the 'full gap' span.
    direction +1: we want the value larger (eyes); -1: smaller (nose, jaw). 0 = already there."""
    tgt, span = TARGET[key]
    return float(np.clip(direction * (tgt - value) / span, 0, 1))


def plan_beauty(meas: dict, level: str = "xiezhen") -> tuple[dict, dict]:
    """Sliders for ONE face from its measurements. Returns (params, reasons)."""
    if level not in LEVELS:
        raise ValueError(f"level must be one of {list(LEVELS)}")
    L = LEVELS[level]
    p, why = dict(L), {}

    def setk(k, factor, reason):
        p[k] = round(L.get(k, 0) * factor, 1)
        why[k] = reason

    # skin: only as much as the skin needs
    dark = meas.get("under_eye_darkness_L")
    if dark is not None:
        setk("dark_circles", float(np.clip((dark - 3) / 5, 0, 1)), f"under-eye {dark} L* darker than cheeks")
    shine = meas.get("shine_pct")
    if shine is not None:
        setk("oil_shine", float(np.clip((shine - 0.5) / 3, 0, 1)), f"{shine}% of face skin shiny")
    if not meas.get("teeth_visible"):
        setk("teeth_whiten", 0, "teeth not visible")

    # shape: fill only the gap to the target face
    eye = (meas["eye_width_rel_right"] + meas["eye_width_rel_left"]) / 2
    f_eye = _gap(eye, "eye_width_rel", +1)
    opened = min(meas["eye_open_right"], meas["eye_open_left"])
    if opened < 0.2:
        f_eye *= 0.5            # enlarging half-closed / smiling eyes looks odd
    setk("eye_size", f_eye, f"eye width {eye:.3f} of face (target {TARGET['eye_width_rel'][0]})"
         + (", eyes half closed" if opened < 0.2 else ""))
    f_nose = _gap(meas["nose_width_rel"], "nose_width_rel", -1)
    for k in ("nose_size", "nose_wings"):
        setk(k, f_nose, f"nose width {meas['nose_width_rel']} of face (target {TARGET['nose_width_rel'][0]})")
    f_jaw = _gap(meas["jaw_to_cheek_ratio"], "jaw_to_cheek_ratio", -1)
    for k in ("face_slim", "jaw", "v_face"):
        setk(k, f_jaw, f"jaw {meas['jaw_to_cheek_ratio']} of cheek width (target {TARGET['jaw_to_cheek_ratio'][0]})")
    setk("small_face", 0.5 + 0.5 * f_jaw, "half by level, half by jaw width")
    f_len = _gap(meas["face_aspect_h_w"], "face_aspect_h_w", +1)
    setk("chin_length", f_len, f"face height/width {meas['face_aspect_h_w']} (target {TARGET['face_aspect_h_w'][0]})")
    setk("mouth_size", _gap(meas["mouth_width_rel"], "mouth_width_rel", -1),
         f"mouth width {meas['mouth_width_rel']} of face (target {TARGET['mouth_width_rel'][0]})")
    if "lip_chroma" in meas:
        setk("lipstick", _gap(meas["lip_chroma"], "lip_chroma", +1),
             f"lip colour C* {meas['lip_chroma']} (target {TARGET['lip_chroma'][0]})")
    if "face_relief" in meas:
        setk("contour", _gap(meas["face_relief"], "face_relief", +1),
             f"face light-and-shadow {meas['face_relief']} L* (target {TARGET['face_relief'][0]})")

    # pose and size: warps on turned or tiny faces are risky and hardly visible
    if meas.get("facing", "camera") != "camera":
        for k in SHAPE_KEYS:
            if p.get(k):
                p[k] = round(p[k] * 0.5, 1)
        why["pose"] = "face turned: shape edits halved"
    fh = meas.get("face_height_frac", 1)
    if fh < 0.06:
        for k in SHAPE_KEYS + ["teeth_whiten", "iris", "eye_whites", "lipstick", "contour"]:
            p[k] = 0
        why["size"] = f"face only {fh:.0%} of the frame high: shape and eye details skipped"
    return p, why


# ---- guards added on top of face.face_checks -----------------------------------------------
def _hp(img):
    L = C._lab(img)[..., 0]
    return np.abs(L - cv2.GaussianBlur(L, (0, 0), 1.5))


def _moved(lm, dx, dy):
    h, w = dx.shape
    s = np.round(lm[:468]).astype(int)
    s[:, 0] = np.clip(s[:, 0], 0, w - 1)
    s[:, 1] = np.clip(s[:, 1], 0, h - 1)
    return lm[:468] + np.stack([dx[s[:, 1], s[:, 0]], dy[s[:, 1], s[:, 0]]], 1)


_PAIRS = [(172, 397), (136, 365), (150, 379), (234, 454), (33, 263), (133, 362), (129, 358), (61, 291)]


def _asymmetry(lm):
    cx = (lm[168, 0] + lm[1, 0] + lm[152, 0]) / 3      # face mid-line: nose bridge, tip, chin
    fw = float(np.linalg.norm(lm[234] - lm[454]))
    return float(np.mean([abs(abs(lm[a, 0] - cx) - abs(lm[b, 0] - cx)) for a, b in _PAIRS]) / max(fw, 1))


def _ratios(lm):
    fw = float(np.linalg.norm(lm[234] - lm[454]))
    eye = (np.linalg.norm(lm[33] - lm[133]) + np.linalg.norm(lm[263] - lm[362])) / 2
    return {"eye_width_rel": float(eye / fw),
            "nose_width_rel": float(np.linalg.norm(lm[129] - lm[358]) / fw),
            "jaw_to_cheek_ratio": float(np.linalg.norm(lm[172] - lm[397]) / fw),
            "face_aspect_h_w": float(np.linalg.norm(lm[10] - lm[152]) / fw)}


def beauty_checks(before, after, ana, info, face_ids) -> list[dict]:
    checks = F.face_checks(before, after, ana, info)

    def add(rule, ok, value, detail):
        checks.append({"rule": rule, "status": "pass" if ok else "fail", "value": value, "detail": detail})

    m = ana["masks"]
    dx, dy = info["_dx"], info["_dy"]
    for i in face_ids:
        lm = ana["people"][i]["landmarks"]
        x, y, bw, bh = [int(v) for v in ana["people"][i]["face"]["box"]]
        box = np.zeros(m["face_skin"].shape, bool)
        box[max(0, y):y + bh, max(0, x):x + bw] = True
        skin = (m["face_skin"] > 0.7) & box
        if skin.sum() > 200:
            kept = float(_hp(after)[skin].mean() / max(_hp(before)[skin].mean(), 1e-6))
            add(f"face {i}: skin not plastic (pores kept)", kept >= 0.45, round(kept, 2),
                "fine detail on face skin after/before (>= 0.45)")
        mv = _moved(lm, dx, dy)
        a0, a1 = _asymmetry(lm), _asymmetry(mv)
        add(f"face {i}: symmetry not worse", a1 <= a0 + 0.02, [round(a0, 3), round(a1, 3)],
            "left/right asymmetry before, after (after <= before + 0.02)")
        r0, r1 = _ratios(lm), _ratios(mv)
        bad = []
        for k, (lo, hi) in PLAUSIBLE.items():
            # only an edit can fail this: a face that already was outside the band may stay there
            if lo is not None and r1[k] < lo and r1[k] < r0[k] - 1e-3:
                bad.append(f"{k} {r1[k]:.3f} < {lo}")
            if hi is not None and r1[k] > hi and r1[k] > r0[k] + 1e-3:
                bad.append(f"{k} {r1[k]:.3f} > {hi}")
        add(f"face {i}: proportions plausible", not bad, bad or {k: round(v, 3) for k, v in r1.items()},
            f"edited ratios must stay human: {PLAUSIBLE}")
    # warps bend real straight lines (door frames, walls) beside the cheek or waist
    bend, n = line_bend(before, m["person"], dx, dy)
    if n:
        add("background lines stay straight", bend <= 1.0, round(bend, 2),
            f"largest bow of {n} straight background lines near the warp, px at "
            f"{max(before.shape[:2])} px (<= 1.0)")
    return checks


def line_bend(rgb, person, dx, dy, min_len_frac=0.03):
    """Bow (px) the warp puts into straight background lines: for each LSD segment in the background,
    the spread of the displacement component perpendicular to the line along its length.
    A uniform shift of a whole line is not a bend; only the variation along it is."""
    if not (np.any(dx) or np.any(dy)):
        return 0.0, 0
    h, w = dx.shape
    moving = np.hypot(dx, dy) > 0.05
    gray = cv2.cvtColor((np.clip(rgb, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    lines = cv2.createLineSegmentDetector().detect(gray)[0]
    worst, n = 0.0, 0
    for x1, y1, x2, y2 in ([] if lines is None else lines.reshape(-1, 4)):
        length = float(np.hypot(x2 - x1, y2 - y1))
        if length < min_len_frac * max(h, w):
            continue
        t = np.linspace(0, 1, 24)
        xs = np.clip(np.round(x1 + t * (x2 - x1)).astype(int), 0, w - 1)
        ys = np.clip(np.round(y1 + t * (y2 - y1)).astype(int), 0, h - 1)
        if person[ys, xs].mean() > 0.2 or not moving[ys, xs].any():
            continue                                    # a line on the person, or out of the warp's reach
        nx, ny = -(y2 - y1) / length, (x2 - x1) / length
        perp = dx[ys, xs] * nx + dy[ys, xs] * ny
        worst = max(worst, float(perp.max() - perp.min()))
        n += 1
    return worst, n


# which sliders to lower when a guard fails (first match wins)
REPAIR = [
    ("eyes not over-enlarged", ["eye_size"], 0.7),
    ("nose change small", ["nose_size", "nose_wings"], 0.7),
    ("mouth change small", ["mouth_size"], 0.7),
    ("teeth natural", ["teeth_whiten"], 0.6),
    ("lip colour natural", ["lipstick"], 0.6),
    ("lipstick stays on the lips", ["lipstick"], 0.6),
    ("contour subtle", ["contour"], 0.7),
    ("skin not plastic", ["smoothing"], 0.7),
    ("skin texture kept", ["smoothing"], 0.7),
    ("face not lit separately", ["luminous", "whitening"], 0.6),
    ("no halo", ["luminous", "whitening"], 0.7),
    ("proportions plausible", SHAPE_KEYS, 0.75),
    ("identity kept", SHAPE_KEYS, 0.75),
    ("symmetry not worse", SHAPE_KEYS, 0.8),
    ("background", SHAPE_KEYS, 0.7),
]


def _render_all(rgb, ana, plans, lip_shade="natural"):
    """Render each face with its own sliders. Brightening of all skin (luminous / whitening) acts on
    every person at once, so it runs only with the first face (with the strongest planned value)."""
    out, merged = rgb, {"identity": [], "faces_edited": [], "blemishes_removed": 0,
                        "_dx": np.zeros(rgb.shape[:2], np.float32), "_dy": np.zeros(rgb.shape[:2], np.float32),
                        "background_max_shift_px_frac": 0.0}
    lum = max(p["luminous"] for p in plans.values())
    wht = max(p["whitening"] for p in plans.values())
    for n, (i, p) in enumerate(plans.items()):
        q = F.normalize({**p, "luminous": lum if n == 0 else 0, "whitening": wht if n == 0 else 0,
                         "lip_shade": lip_shade})
        out, info = F.render_faces(out, ana, q, face_ids=[i])
        merged["identity"] += info["identity"]
        if "contour_max_ev" in info:
            merged["contour_max_ev"] = max(merged.get("contour_max_ev", 0.0), info["contour_max_ev"])
        merged["faces_edited"] += info["faces_edited"]
        merged["blemishes_removed"] += info.get("blemishes_removed", 0)
        merged["_dx"] += info["_dx"]
        merged["_dy"] += info["_dy"]
        merged["background_max_shift_px_frac"] = max(merged["background_max_shift_px_frac"],
                                                     info["background_max_shift_px_frac"])
    return out, merged


def beauty_retouch(image_path: str, level: str = "xiezhen", avoid: list[str] | None = None,
                   params: dict | None = None, lip_shade: str | None = None, face_ids: list[int] | None = None,
                   output_path: str | None = None, sheet_path: str | None = None) -> dict:
    """Plan, render, guard and repair, then render at full resolution."""
    if level not in LEVELS:
        raise ValueError(f"level must be one of {list(LEVELS)}")
    bad = set(avoid or []) - set(GROUPS)
    if bad:
        raise ValueError(f"unknown avoid groups {bad}; valid: {list(GROUPS)}")
    shade_asked = lip_shade is not None          # the user named a colour: it must show
    lip_shade = lip_shade or LIP_SHADE[level]
    if lip_shade not in F.LIP_SHADES:
        raise ValueError(f"lip_shade must be one of {list(F.LIP_SHADES)}")
    for k in (params or {}):
        if k not in F.ALL_KEYS:
            raise ValueError(f"unknown face parameter {k!r}; valid: {list(F.ALL_KEYS)}")

    rgb, ana = P._load(image_path)
    faces = [i for i, p in enumerate(ana["people"]) if p["landmarks"] is not None
             and (face_ids is None or i in face_ids)]
    if not faces:
        return {"status": "skipped", "reason": "no face with landmarks found", "output_path": None}

    plans, reasons, meas_all = {}, {}, {}
    for i in faces:
        p_ = ana["people"][i]
        meas = F.measure_face(rgb, p_["landmarks"], ana["masks"]["face_skin"])
        meas["facing"] = {1: "image right", -1: "image left", 0: "camera"}[p_["face"]["facing"]]
        meas["face_height_frac"] = round(p_["face"]["box"][3] / rgb.shape[0], 3)
        plan, why = plan_beauty(meas, level)
        for g in avoid or []:
            for k in GROUPS[g]:
                if k in plan:
                    plan[k] = 0
        if shade_asked and "makeup" not in (avoid or []):
            floor = LIP_FLOOR[level]
            if plan.get("lipstick", 0) < floor:
                plan["lipstick"] = floor
                why["lipstick"] = f"shade {lip_shade} was asked for: at least {floor}"
        plan.update(params or {})           # explicit values from the LLM win
        plans[i], reasons[i], meas_all[i] = plan, why, meas

    # guard + repair loop at work size
    repairs = []
    for attempt in range(4):
        out, info = _render_all(rgb, ana, plans, lip_shade)
        checks = beauty_checks(rgb, out, ana, info, faces)
        failed = [c for c in checks if c["status"] == "fail"]
        if not failed or attempt == 3:
            break
        did = set()
        for c in failed:
            for key, keys, factor in REPAIR:
                if key in c["rule"] and key not in did:
                    did.add(key)
                    for pl in plans.values():
                        for k in keys:
                            v = pl.get(k, 0)
                            pl[k] = [round(x * factor, 1) for x in v] if isinstance(v, list) else round(v * factor, 1)
                    repairs.append({"round": attempt + 1, "failed": c["rule"], "lowered": keys, "by": factor})
                    break
        if not did:
            break                            # nothing we know how to lower: report it

    src = Path(image_path)
    if sheet_path is None:
        sheet_path = str(src.with_name(src.stem + "_beauty_sheet.jpg"))
    F._closeups([rgb, out], ana, ["before", f"after ({level})"], sheet_path, faces)

    # full resolution with the repaired sliders
    full, ana_full = P.load_full(image_path)
    out_full, info_full = _render_all(full, ana_full, plans, lip_shade)
    if output_path is None:
        output_path = str(src.with_name(f"{src.stem}_beauty.tif"))
    write_image(output_path, (np.clip(out_full, 0, 1)[..., ::-1] * 65535 + 0.5).astype(np.uint16))
    H, W = full.shape[:2]
    kp = min(1.0, 1600 / max(H, W))
    prev = cv2.resize(out_full, None, fx=kp, fy=kp, interpolation=cv2.INTER_AREA)
    preview_path = str(Path(output_path).with_name(Path(output_path).stem + "_preview.jpg"))
    C._save_rgb8(preview_path, prev)

    final_checks = checks
    return {"status": "done" if not failed else "checks_failed",
            "level": level, "lip_shade": lip_shade, "output_path": output_path, "preview_path": preview_path, "sheet_path": sheet_path,
            "faces": {str(i): {"measured": meas_all[i], "sliders": {k: v for k, v in plans[i].items()
                                                                   if (any(v) if isinstance(v, list) else v)},
                               "why": reasons[i]} for i in faces},
            "repairs": repairs, "identity": info["identity"],
            "checks_failed": [c for c in final_checks if c["status"] == "fail"],
            "checks_passed": len([c for c in final_checks if c["status"] == "pass"]),
            "note": "Checks were run at work size (1600 px); the full-resolution file uses the same sliders."}
