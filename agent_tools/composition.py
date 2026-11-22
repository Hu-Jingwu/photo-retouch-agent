"""Composition + auto-crop tools, based on classical framing rules.

Same division of labour as leveling: CV proposes candidate crops and scores
them on interpretable rules; the multimodal model looks at the contact sheet
and decides; apply_crop executes.

Rules (see docs/composition_rules.md for the full list and sources):
  all photos  rule of thirds, clean edges, no amputated people, keep what matters
  people      eyes on the upper third, headroom, never cut at joints, lead room,
              shot size, vertical orientation for a single person
  landscape   horizon on a third line, sky:land ratio chosen by which part is
              more interesting, reflections may be centred

Runs AFTER leveling, on the leveled frame, and never crops into the blank
corners outside `valid_rect` returned by apply_rotation.
"""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from .horizon import _detect_segments, _dominant_horizon_line
from .io_utils import is_raw, read_image, to_bgr_u8, to_gray_u8, write_image

ROOT = Path(__file__).resolve().parent.parent
YUNET_PATH = ROOT / "models" / "face_detection_yunet_2023mar.onnx"
_face_detector = None

ASPECTS = {"1:1": 1.0, "4:5": 4 / 5, "5:4": 5 / 4, "2:3": 2 / 3, "3:2": 3 / 2, "3:4": 3 / 4,
           "4:3": 4 / 3, "9:16": 9 / 16, "16:9": 16 / 9, "2:1": 2.0}
WEIGHTS = {
    "people": {"placement": 0.22, "lead_room": 0.12, "joints": 0.14, "saliency_kept": 0.17,
               "edge_clean": 0.13, "headroom": 0.10, "area_kept": 0.12},
    "landscape": {"horizon": 0.25, "placement": 0.20, "saliency_kept": 0.20,
                  "edge_clean": 0.15, "area_kept": 0.20},
    "general": {"placement": 0.35, "saliency_kept": 0.30, "edge_clean": 0.20, "area_kept": 0.15},
}
SCALES = (1.0, 0.9, 0.8, 0.7, 0.6)
# Classical figure proportions: a standing adult is ~8 head-heights tall,
# measured from the crown. Joints, where a frame edge should never fall:
JOINTS = {"neck": 1.0, "elbows/waist": 3.0, "wrists/hips": 4.0, "knees": 6.0, "ankles": 7.7}
FULL_BODY = 8.3


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #
def detect_faces(bgr: np.ndarray, min_score: float = 0.7) -> list[dict]:
    """YuNet faces: box, eye line, head box, facing direction. Largest first."""
    global _face_detector
    if not YUNET_PATH.exists():
        return []
    if _face_detector is None:
        _face_detector = cv2.FaceDetectorYN.create(str(YUNET_PATH), "", (320, 320), min_score)
    h, w = bgr.shape[:2]
    _face_detector.setInputSize((w, h))
    _, found = _face_detector.detect(bgr)
    faces = []
    for r in (found if found is not None else []):
        x, y, bw, bh = (float(v) for v in r[:4])
        eye_mid = (float(r[4]) + float(r[6])) / 2.0
        eye_dist = abs(float(r[6]) - float(r[4])) + 1e-6
        yaw = (float(r[8]) - eye_mid) / eye_dist     # nose offset: + = turned to image right
        faces.append({
            "box": [x, y, bw, bh],
            "eye_y": (float(r[5]) + float(r[7])) / 2.0,
            "center_x": eye_mid,
            "score": float(r[14]),
            "facing": 1 if yaw > 0.12 else (-1 if yaw < -0.12 else 0),
            # head incl. hair and chin; the main head is never cut
            "head": [x - 0.25 * bw, y - 0.45 * bh, x + 1.25 * bw, y + 1.1 * bh],
            # head-height unit for figure proportions (crown to chin)
            "hu": 1.3 * bh,
            "crown": y + bh - 1.3 * bh,
        })
    faces.sort(key=lambda f: -f["box"][2] * f["box"][3])
    return faces


def saliency_map(gray: np.ndarray, faces: list[dict]) -> np.ndarray:
    """Spectral-residual saliency (Hou & Zhang 2007) + a prior for people."""
    h, w = gray.shape
    s = 128.0 / max(h, w)
    g = cv2.resize(gray, (max(8, round(w * s)), max(8, round(h * s))),
                   interpolation=cv2.INTER_AREA).astype(np.float32)
    F = np.fft.fft2(g)
    log_amp = np.log(np.abs(F) + 1e-8)
    residual = log_amp - cv2.blur(log_amp, (3, 3))
    sal = np.abs(np.fft.ifft2(np.exp(residual + 1j * np.angle(F)))) ** 2
    sal = cv2.GaussianBlur(sal.astype(np.float32), (0, 0), 3)
    sal = cv2.resize(sal, (w, h), interpolation=cv2.INTER_LINEAR)
    sal = (sal - sal.min()) / (np.ptp(sal) + 1e-8)

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    for f in faces:  # people are what viewers look at first
        x, y, bw, bh = f["box"]
        cx, cy = x + bw / 2, y + bh / 2
        sal += 0.6 * np.exp(-(((xx - cx) / bw) ** 2 + ((yy - cy) / bh) ** 2))
        sal += 0.3 * np.exp(-(((xx - cx) / (1.6 * bw)) ** 2 + ((yy - cy - 3 * bh) / (3 * bh)) ** 2))
    return sal / (sal.max() + 1e-8)


def detect_objects(bgr: np.ndarray, faces: list[dict], min_area: float = 0.03,
                   min_fill: float = 0.45) -> list[dict]:
    """Prominent non-person objects (a glowing lamp, a sign, a statue) that a crop must not cut in half.

    Frequency-tuned saliency (Achanta et al. 2009): how far each smoothed pixel's Lab colour is from
    the frame's mean colour. Big, solid, strongly distinct regions survive; people are removed with
    the person segmenter so their legs or sleeves are not counted (the joint rules handle people).
    Boxes are in `bgr` pixel coordinates.
    """
    h, w = bgr.shape[:2]
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    ft = np.linalg.norm(cv2.GaussianBlur(lab, (0, 0), 3) - lab.reshape(-1, 3).mean(0), axis=2)
    ft /= ft.max() + 1e-6
    m = (ft >= max(0.5, float(np.percentile(ft, 90)))).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    if faces:
        person = _person_mask(bgr)
        if person is not None:
            m[cv2.dilate(person, np.ones((9, 9), np.uint8)) > 0] = 0
    n, _, st, _ = cv2.connectedComponentsWithStats(m)
    out = []
    for i in range(1, n):
        x, y, bw, bh, a = (int(v) for v in st[i])
        if a >= min_area * h * w and a / (bw * bh) >= min_fill:
            out.append({"box": [float(x), float(y), float(bw), float(bh)], "area": a / (h * w)})
    return sorted(out, key=lambda o: -o["area"])


def _person_mask(bgr: np.ndarray) -> np.ndarray | None:
    try:
        from . import people as P          # lazy: people imports this module
        rgb8 = np.ascontiguousarray(bgr[..., ::-1])
        res = P._model("seg").segment(P._mp_image(rgb8))
        bg = res.confidence_masks[0].numpy_view().squeeze()
        if bg.shape[:2] != bgr.shape[:2]:
            bg = cv2.resize(bg, (bgr.shape[1], bgr.shape[0]))
        return (bg < 0.5).astype(np.uint8)
    except Exception:
        return None


def _inside_frac(box, x, y, cw, ch) -> float:
    bx, by, bw, bh = box
    ix = max(0.0, min(bx + bw, x + cw) - max(bx, x))
    iy = max(0.0, min(by + bh, y + ch) - max(by, y))
    return ix * iy / (bw * bh + 1e-9)


def analyse_landscape(gray: np.ndarray, vr_work: tuple[float, float, float, float]) -> dict | None:
    """Find a level horizon and decide whether sky or land deserves 2/3 of the frame."""
    h, w = gray.shape
    seg, _ = _detect_segments(cv2.GaussianBlur(gray, (3, 3), 0), 0.04 * max(h, w))
    idx = np.where(np.abs(seg.angle) <= 10)[0]
    line = _dominant_horizon_line(seg, idx, w, h)
    if not line or line["width_coverage"] < 0.6 or abs(line["tilt_deg"]) > 2.0:
        return None
    hy = line["y_at_center_frac"] * h
    vx, vy, vw, vh = vr_work
    top, bot = int(max(vy, 0)), int(min(vy + vh, h))
    if not (top + 0.1 * h < hy < bot - 0.1 * h):
        return None
    g = gray.astype(np.float32)
    tex = cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    x0, x1 = int(vx), int(vx + vw)
    sky_tex = float(tex[top:int(hy) - 2, x0:x1].mean())
    land_tex = float(tex[int(hy) + 2:bot, x0:x1].mean())
    band = int(min(hy - top, bot - hy, 0.25 * h))
    above = g[int(hy) - band:int(hy), x0:x1]
    below = g[int(hy):int(hy) + band, x0:x1][::-1]
    reflection = False
    if above.size and above.shape == below.shape:
        a, b = above - above.mean(), below - below.mean()
        corr = float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-6))
        reflection = corr > 0.6
    sky_interesting = sky_tex > 0.8 * land_tex
    return {
        "horizon_y": hy,
        "sky_interesting": sky_interesting,
        # dramatic sky -> horizon on the lower third; plain sky -> upper third
        "target_v": 2 / 3 if sky_interesting else 1 / 3,
        "reflection": reflection,
        "sky_texture": round(sky_tex, 2),
        "land_texture": round(land_tex, 2),
    }


def _iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    return inter / (aw * ah + bw * bh - inter + 1e-9)


def _aspect_value(a: str, w: float, h: float) -> float:
    if a == "original":
        return w / h
    if a in ASPECTS:
        return ASPECTS[a]
    num, den = a.split(":")
    return float(num) / float(den)


def shot_size(k_bottom: float) -> str:
    """Name the shot by where the bottom edge falls, in head-heights from the crown."""
    if k_bottom < 1.6:
        return "close-up"
    if k_bottom < 2.6:
        return "medium close-up (chest)"
    if k_bottom < 3.8:
        return "medium (waist)"
    if k_bottom < 5.5:
        return "cowboy (thigh)"
    if k_bottom < FULL_BODY:
        return "three-quarter (legs cut)"
    return "full body"


# --------------------------------------------------------------------------- #
# Tool 1: propose_crops
# --------------------------------------------------------------------------- #
def propose_crops(
    image_path: str,
    valid_rect: dict | None = None,
    aspects: list[str] | None = None,
    num_candidates: int = 8,
    min_scale: float = 0.5,
    work_size: int = 768,
    contact_sheet_path: str | None = None,
) -> dict:
    ctx = _analyse(image_path, valid_rect, work_size)
    W, H, vr, s, small, gray = (ctx[k_] for k_ in ("W", "H", "vr", "s", "small", "gray"))
    faces, scene, land, single = ctx["faces"], ctx["scene"], ctx["land"], ctx["single"]
    vx, vy, vw, vh = ctx["vr_work"]
    focal_src = ctx["focal_src"]
    weights = WEIGHTS[scene]

    if aspects is None:
        if scene == "people" and single:
            aspects = (["original", "4:5", "3:4", "1:1", "16:9"] if W > H
                       else ["original", "4:5", "1:1", "3:4"])
        elif scene == "landscape":
            aspects = ["original", "16:9", "2:1", "4:5"]
        else:
            aspects = ["original", "4:5", "1:1", "9:16"] if H > W else ["original", "16:9", "1:1", "4:5"]

    return _propose(ctx, aspects, weights, num_candidates, min_scale, contact_sheet_path, image_path,
                    W, H, vr, s, small, gray, faces, scene, land, single, vx, vy, vw, vh, focal_src)


def _analyse(image_path: str, valid_rect: dict | None, work_size: int = 768) -> dict:
    """Shared scene analysis for propose_crops and check_composition (work-resolution coords)."""
    img = read_image(image_path)
    H, W = img.shape[:2]
    vr = valid_rect or {"x": 0, "y": 0, "w": W, "h": H}

    s = work_size / max(H, W)
    bgr8 = to_bgr_u8(img)
    small = cv2.resize(bgr8, (round(W * s), round(H * s)), interpolation=cv2.INTER_AREA)
    gray = to_gray_u8(small)
    h, w = gray.shape
    fs = 1024 / max(H, W)
    faces = detect_faces(cv2.resize(bgr8, None, fx=fs, fy=fs, interpolation=cv2.INTER_AREA))
    k = s / fs                           # face coords -> work coords
    for f in faces:
        f["box"] = [v * k for v in f["box"]]
        f["head"] = [v * k for v in f["head"]]
        for key in ("eye_y", "center_x", "hu", "crown"):
            f[key] *= k
    vx, vy, vw, vh = vr["x"] * s, vr["y"] * s, vr["w"] * s, vr["h"] * s

    # ---- scene type decides which rules apply ----
    land = None
    if faces:
        scene = "people"
        main_area = faces[0]["box"][2] * faces[0]["box"][3]
        must_keep = [f for f in faces if f["box"][2] * f["box"][3] >= 0.5 * main_area]
        minor = [f for f in faces if f not in must_keep]
        single = len(must_keep) == 1
    else:
        land = analyse_landscape(gray, (vx, vy, vw, vh))
        scene = "landscape" if land else "general"
        must_keep, minor, single = [], [], False

    sal = saliency_map(gray, faces)
    if scene == "landscape":
        # Spectral saliency flags whatever is RARE, which in a landscape is often the plain sky.
        # Detail (texture) is a better measure of where the interest is, so blend it in.
        g32 = gray.astype(np.float32)
        tex = cv2.magnitude(cv2.Sobel(g32, cv2.CV_32F, 1, 0), cv2.Sobel(g32, cv2.CV_32F, 0, 1))
        tex = cv2.GaussianBlur(tex, (0, 0), 8)
        sal = 0.4 * sal + 0.6 * tex / (tex.max() + 1e-8)
    ii = cv2.integral(sal.astype(np.float64))

    def mass(x0, y0, x1, y1):
        x0, y0 = int(max(0, x0)), int(max(0, y0))
        x1, y1 = int(min(w, x1)), int(min(h, y1))
        if x1 <= x0 or y1 <= y0:
            return 0.0
        return ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0]

    valid_mass = mass(vx, vy, vx + vw, vy + vh) + 1e-9
    ctx = {"faces": faces, "must_keep": must_keep, "minor": minor, "land": land, "mass": mass,
           "objects": detect_objects(small, faces),
           "valid_mass": valid_mass, "valid_density": valid_mass / (vw * vh), "valid_area": vw * vh}

    if faces:
        ctx["focal"] = (faces[0]["center_x"], faces[0]["eye_y"])
        focal_src = "main face eye line"
    else:
        m = sal >= 0.6
        ys, xs = np.nonzero(m)
        ctx["focal"] = (float(np.average(xs, weights=sal[m])), float(np.average(ys, weights=sal[m])))
        focal_src = "centre of visual interest"
    ctx.update(W=W, H=H, vr=vr, s=s, small=small, gray=gray, scene=scene, single=single,
               vr_work=(vx, vy, vw, vh), focal_src=focal_src, work_wh=(w, h))
    return ctx


def _propose(ctx, aspects, weights, num_candidates, min_scale, contact_sheet_path, image_path,
             W, H, vr, s, small, gray, faces, scene, land, single, vx, vy, vw, vh, focal_src):
    cands = []
    seen = []
    for a_name in aspects:
        ar = _aspect_value(a_name, W, H)
        if any(abs(ar - x) < 0.02 for x in seen):
            continue
        seen.append(ar)
        cw, ch = (vh * ar, vh) if vw / vh > ar else (vw, vw / ar)
        for sc in SCALES + (0.5,):
            if sc < min_scale - 1e-9:
                continue
            cw_s, ch_s = cw * sc, ch * sc
            nx = 1 if vw - cw_s < 2 else 15
            ny = 1 if vh - ch_s < 2 else 15
            for x0 in np.linspace(vx, vx + vw - cw_s, nx):
                for y0 in np.linspace(vy, vy + vh - ch_s, ny):
                    rect = (x0, y0, cw_s, ch_s)
                    feats, notes = _score(rect, scene, ctx)
                    if feats is None:
                        continue
                    total = sum(weights[k_] * feats[k_] for k_ in weights) * feats["objects_whole"]
                    cands.append((total, a_name, rect, feats, notes))

    if not cands:
        raise RuntimeError("No crop keeps the main subject whole inside the valid area; skip cropping.")
    cands.sort(key=lambda c: -c[0])
    chosen = _pick_diverse(cands, num_candidates)

    out = []
    for i, (total, a_name, (x, y, cw_, ch_), feats, notes) in enumerate(chosen, 1):
        rx, ry = max(int(round(x / s)), vr["x"]), max(int(round(y / s)), vr["y"])
        rw = min(int(round(cw_ / s)), vr["x"] + vr["w"] - rx)
        rh = min(int(round(ch_ / s)), vr["y"] + vr["h"] - ry)
        crop_g = gray[int(y):int(y + ch_), int(x):int(x + cw_)]
        cand = {
            "id": i,
            "aspect": a_name,
            "orientation": "vertical" if ch_ > cw_ * 1.02 else ("square" if abs(ch_ - cw_) <= cw_ * 0.02
                                                                 else "horizontal"),
            "rect": {"x": rx, "y": ry, "w": rw, "h": rh},
            "score": round(total, 3),
            "features": {k_: round(float(v), 3) for k_, v in feats.items()},
            "symmetry": round(_symmetry(crop_g), 3),
            "notes": notes,
        }
        if faces:
            cand["shot_size"] = shot_size((y + ch_ - faces[0]["crown"]) / faces[0]["hu"])
        out.append(cand)

    if contact_sheet_path is None:
        p = Path(image_path)
        contact_sheet_path = str(p.with_name(p.stem + "_crops.jpg"))
    _contact_sheet(small, s, vr, out, contact_sheet_path, ctx["objects"])

    scene_info: dict = {"type": scene, "num_faces": len(faces)}
    if faces:
        scene_info["main_subject_facing"] = {1: "image right", -1: "image left", 0: "camera"}[faces[0]["facing"]]
        scene_info["single_main_subject"] = single
    if land:
        scene_info.update(horizon_y=round(land["horizon_y"] / s), sky_interesting=land["sky_interesting"],
                          reflection=land["reflection"],
                          horizon_target=("lower third (more sky)" if land["sky_interesting"]
                                          else "upper third (more land)"))
    return {
        "image_size": {"w": W, "h": H},
        "valid_rect": vr,
        "scene": scene_info,
        "faces": [{"box": [round(v / s) for v in f["box"]], "score": round(f["score"], 2)} for f in faces],
        "prominent_objects": [{"box": [round(v / s) for v in o["box"]], "area_pct": round(100 * o["area"], 1)}
                              for o in ctx["objects"]],
        "focal_point": {"x": round(ctx["focal"][0] / s), "y": round(ctx["focal"][1] / s), "source": focal_src},
        "candidates": out,
        "weights": weights,
        "contact_sheet_path": contact_sheet_path,
        "rules": "Classical framing rules, see docs/composition_rules.md. Scores are guidance; "
                 "notes list rule problems per candidate.",
    }


def _pick_diverse(cands, n):
    """Best overall, widest, best per aspect, best vertical; then fill avoiding near-duplicates."""
    chosen = [cands[0]]

    def add(c):
        if c is not None and all(_iou(c[2], d[2]) < 0.85 for d in chosen):
            chosen.append(c)

    widest = max(c[3]["area_kept"] for c in cands)
    add(max((c for c in cands if c[3]["area_kept"] >= widest - 1e-6), key=lambda c: c[0]))
    for a_name in dict.fromkeys(c[1] for c in cands):
        add(next(c for c in cands if c[1] == a_name))
    add(next((c for c in cands if c[2][3] > c[2][2] * 1.02), None))      # best vertical
    for c in cands:
        if len(chosen) >= n:
            break
        if all(_iou(c[2], d[2]) < 0.6 for d in chosen):
            chosen.append(c)
    return sorted(chosen, key=lambda c: -c[0])[:n]


def _score(rect, scene, ctx):
    x, y, cw, ch = rect
    notes: list[str] = []
    faces, mass = ctx["faces"], ctx["mass"]

    # hard rule: the main subject(s) stay whole
    for f in ctx["must_keep"]:
        hx0, hy0, hx1, hy1 = f["head"]
        if hx0 < x or hy0 < y or hx1 > x + cw or hy1 > y + ch:
            return None, notes
    fx, fy = ctx["focal"]
    u, v = (fx - x) / cw, (fy - y) / ch
    if not (0 <= u <= 1 and 0 <= v <= 1):
        return None, notes

    # rule of thirds
    d_power = min(math.hypot(u - a, v - b) for a in (1 / 3, 2 / 3) for b in (1 / 3, 2 / 3))
    placement = math.exp(-(d_power / 0.1) ** 2)
    if faces:  # eyes belong in the upper part: on the upper third line, centred or on a vertical third
        eye_line = math.exp(-((v - 1 / 3) / 0.06) ** 2) if v >= 1 / 3 else math.exp(-((v - 1 / 3) / 0.1) ** 2)
        horiz = max(math.exp(-((u - 0.5) / 0.08) ** 2), math.exp(-((min(abs(u - 1 / 3), abs(u - 2 / 3))) / 0.08) ** 2))
        placement = eye_line * horiz

    # clean edges
    m = mass(x, y, x + cw, y + ch)
    band = 0.03 * min(cw, ch)
    inner = mass(x + band, y + band, x + cw - band, y + ch - band)
    band_area = cw * ch - (cw - 2 * band) * (ch - 2 * band)
    r = ((m - inner) / band_area) / (ctx["valid_density"] + 1e-9)
    edge = float(np.clip(1.5 - r, 0.0, 1.0))
    for f in ctx["must_keep"]:
        hx0, hy0, hx1, hy1 = f["head"]
        if hx0 < x + band or hx1 > x + cw - band or hy0 < y + band:
            edge *= 0.3
            notes.append("head touches the edge")
    for f in ctx["minor"]:  # no amputated people: fully in or fully out
        hx0, hy0, hx1, hy1 = f["head"]
        ix = max(0.0, min(hx1, x + cw) - max(hx0, x))
        iy = max(0.0, min(hy1, y + ch) - max(hy0, y))
        frac = ix * iy / ((hx1 - hx0) * (hy1 - hy0) + 1e-9)
        if 0.05 < frac < 0.95:
            edge *= 0.3
            notes.append("cuts through a secondary person")
    # prominent objects (lamp, sign, statue): fully in or fully out, never sliced in half
    whole = 1.0
    for o in ctx.get("objects", ()):
        frac = _inside_frac(o["box"], x, y, cw, ch)
        if 0.02 < frac < 0.9:
            whole *= 0.6
            notes.append(f"cuts through a prominent object ({frac:.0%} of it kept)")

    feats = {
        "placement": placement,
        "saliency_kept": m / ctx["valid_mass"],
        "edge_clean": edge,
        "area_kept": cw * ch / ctx["valid_area"],
        "objects_whole": whole,      # multiplies the total score (not a weighted term)
    }

    if scene == "people":
        f0 = faces[0]
        hr = (f0["head"][1] - y) / ch
        feats["headroom"] = 1.0 if hr >= 0.04 else math.exp(-((0.04 - hr) / 0.02) ** 2)
        if hr < 0.04:
            notes.append("cramped headroom")
        # never cut at joints: bottom edge position in head-heights from the crown
        kb = (y + ch - f0["crown"]) / f0["hu"]
        if kb < 3.8 and hr > 0.25:   # tight shot with lots of empty space above the head
            feats["headroom"] *= math.exp(-((hr - 0.25) / 0.08) ** 2)
            notes.append("too much headroom for a tight shot")
        if kb >= FULL_BODY:
            feats["joints"] = 1.0
        else:
            name, kj = min(JOINTS.items(), key=lambda j: abs(kb - j[1]))
            dist = abs(kb - kj)
            feats["joints"] = min(1.0, dist / 0.4)
            if dist < 0.25:
                notes.append(f"bottom edge cuts at the {name}")
        # lead room: more space in front of the face than behind it
        if f0["facing"]:
            fu = (f0["center_x"] - x) / cw
            ahead, behind = (1 - fu, fu) if f0["facing"] > 0 else (fu, 1 - fu)
            ratio = ahead / (behind + 1e-6)
            feats["lead_room"] = float(np.clip((ratio - 0.6) / 0.6, 0.0, 1.0))
            if ratio < 0.8:
                notes.append("subject looks out of the frame (little lead room)")
        else:
            feats["lead_room"] = 1.0

    elif scene == "landscape":
        land = ctx["land"]
        hv = (land["horizon_y"] - y) / ch
        target, other = land["target_v"], 1 - land["target_v"]
        score = max(math.exp(-((hv - target) / 0.06) ** 2), 0.7 * math.exp(-((hv - other) / 0.06) ** 2))
        if abs(hv - 0.5) < 0.06:
            score = max(score, 1.0 if land["reflection"] else 0.3)
            if not land["reflection"]:
                notes.append("horizon through the middle")
        feats["horizon"] = score

    return feats, notes


def _symmetry(gray_crop: np.ndarray) -> float:
    if gray_crop.size == 0:
        return 0.0
    g = cv2.resize(gray_crop, (128, 128)).astype(np.float32)
    e = cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    diff = np.abs(e - e[:, ::-1]).mean()
    return float(max(0.0, 1.0 - diff / (e.mean() * 2 + 1e-6)))


def _contact_sheet(small, s, vr, cands, path, objects=(), tile_h=420):
    colors = [(0, 0, 255), (0, 200, 0), (255, 120, 0), (0, 200, 255), (255, 0, 255), (255, 255, 0),
              (128, 0, 255), (0, 128, 255)]
    overview = small.copy()
    for o in objects:   # prominent objects in white: keep them whole
        ox, oy, ow, oh = (int(v) for v in o["box"])
        cv2.rectangle(overview, (ox, oy), (ox + ow, oy + oh), (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(overview, "object", (ox + 3, oy + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    vx, vy, vw, vh = (int(round(vr[k] * s)) for k in ("x", "y", "w", "h"))
    cv2.rectangle(overview, (vx, vy), (vx + vw, vy + vh), (160, 160, 160), 1)
    for c in cands:
        r = c["rect"]
        x0, y0 = int(r["x"] * s), int(r["y"] * s)
        x1, y1 = int((r["x"] + r["w"]) * s), int((r["y"] + r["h"]) * s)
        col = colors[(c["id"] - 1) % len(colors)]
        cv2.rectangle(overview, (x0, y0), (x1, y1), col, 2)
        cv2.putText(overview, f"#{c['id']}", (x0 + 6, y0 + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
    tiles = [_fit_h(overview, tile_h, "overview")]
    for c in cands:
        r = c["rect"]
        crop = small[int(r["y"] * s):int((r["y"] + r["h"]) * s), int(r["x"] * s):int((r["x"] + r["w"]) * s)]
        crop = crop.copy()
        ch, cw = crop.shape[:2]
        for f in (1 / 3, 2 / 3):    # thirds grid
            cv2.line(crop, (int(cw * f), 0), (int(cw * f), ch), (255, 255, 255), 1, cv2.LINE_AA)
            cv2.line(crop, (0, int(ch * f)), (cw, int(ch * f)), (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(_fit_h(crop, tile_h, f"#{c['id']} {c['aspect']} {c['score']:.2f}",
                            colors[(c["id"] - 1) % len(colors)]))
    rows, row, row_w = [], [], 0
    for t in tiles:
        if row and row_w + t.shape[1] > 1800:
            rows.append(row)
            row, row_w = [], 0
        row.append(t)
        row_w += t.shape[1] + 10
    rows.append(row)
    width = max(sum(t.shape[1] + 10 for t in r) for r in rows)
    sheet = np.full((len(rows) * (tile_h + 10), width, 3), 255, np.uint8)
    for i, r in enumerate(rows):
        xo = 0
        for t in r:
            sheet[i * (tile_h + 10):i * (tile_h + 10) + tile_h, xo:xo + t.shape[1]] = t
            xo += t.shape[1] + 10
    write_image(path, sheet, jpeg_quality=88)


def _fit_h(img, th, label, color=(0, 0, 0)):
    sc = th / img.shape[0]
    t = cv2.resize(img, (max(1, round(img.shape[1] * sc)), th), interpolation=cv2.INTER_AREA)
    cv2.rectangle(t, (0, 0), (t.shape[1], 34), (255, 255, 255), -1)
    cv2.putText(t, label, (6, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return t


# --------------------------------------------------------------------------- #
# Evaluator: check_composition  (deterministic rule grading of a chosen crop)
# --------------------------------------------------------------------------- #
def check_composition(image_path: str, crop_rect: dict | None, valid_rect: dict | None = None,
                      min_score: float = 0.8) -> dict:
    """Grade a crop against the classical rules in docs/composition_rules.md.

    image_path: the frame the crop was taken from (leveled image, or the original).
    crop_rect:  the chosen crop in that frame's pixels (None = the whole valid area).
    PASS = no hard violation and at least `min_score` of the applicable rules pass.
    """
    ctx = _analyse(image_path, valid_rect)
    s, vr, scene, faces = ctx["s"], ctx["vr"], ctx["scene"], ctx["faces"]
    cr = crop_rect or vr
    x, y, cw, ch = cr["x"] * s, cr["y"] * s, cr["w"] * s, cr["h"] * s
    checks: list[dict] = []

    def add(rule, ok, value=None, hard=False, detail=""):
        checks.append({"rule": rule, "status": "pass" if ok else "fail",
                       "value": None if value is None else round(float(value), 3),
                       "hard": hard, "detail": detail})

    tol = 2 * s   # 2 px at full resolution
    add("inside valid area", cr["x"] >= vr["x"] - 2 and cr["y"] >= vr["y"] - 2
        and cr["x"] + cr["w"] <= vr["x"] + vr["w"] + 2 and cr["y"] + cr["h"] <= vr["y"] + vr["h"] + 2,
        hard=True, detail="no blank rotation corners")

    # clean edges: interest in a 3% border band vs the valid-area average
    mass = ctx["mass"]
    m = mass(x, y, x + cw, y + ch)
    band = 0.03 * min(cw, ch)
    inner = mass(x + band, y + band, x + cw - band, y + ch - band)
    band_area = cw * ch - (cw - 2 * band) * (ch - 2 * band)
    r = ((m - inner) / band_area) / (ctx["valid_density"] + 1e-9)
    add("clean edges", r <= 1.2, r, detail="interest at the borders / frame average (<= 1.2)")

    for o in ctx["objects"]:
        frac = _inside_frac(o["box"], x, y, cw, ch)
        add("prominent object not cut in half", not (0.02 < frac < 0.9), frac, hard=o["area"] >= 0.05,
            detail=f"object of {100 * o['area']:.0f}% of the frame: fully in or fully out")

    fx, fy = ctx["focal"]
    u, v = (fx - x) / cw, (fy - y) / ch
    shot = None

    if scene == "people":
        f0 = faces[0]
        for f in ctx["must_keep"]:
            hx0, hy0, hx1, hy1 = f["head"]
            whole = hx0 >= x - tol and hy0 >= y - tol and hx1 <= x + cw + tol and hy1 <= y + ch + tol
            add("main subject's head whole", whole, hard=True)
        for f in ctx["minor"]:
            hx0, hy0, hx1, hy1 = f["head"]
            ix = max(0.0, min(hx1, x + cw) - max(hx0, x))
            iy = max(0.0, min(hy1, y + ch) - max(hy0, y))
            frac = ix * iy / ((hx1 - hx0) * (hy1 - hy0) + 1e-9)
            add("no amputated people", not (0.05 < frac < 0.95), frac,
                detail="secondary person fully in or fully out")
        hr = (f0["head"][1] - y) / ch
        kb = (y + ch - f0["crown"]) / f0["hu"]
        shot = shot_size(kb)
        add("headroom not cramped", hr >= 0.03, hr, detail="space above head / frame height (>= 0.03)")
        if kb < 3.8:   # close-up .. medium: lots of empty space above the head looks unintended
            add("headroom not excessive (tight shot)", hr <= 0.25, hr, detail="<= 0.25 for close shots")
        if kb < FULL_BODY:
            name, kj = min(JOINTS.items(), key=lambda j: abs(kb - j[1]))
            add("not cut at a joint", abs(kb - kj) >= 0.25, abs(kb - kj),
                detail=f"bottom edge {abs(kb - kj):.2f} head-heights from the {name} (>= 0.25)")
        if kb < 5.5:   # close-up .. cowboy: eye line on the upper third
            add("eyes near the upper third", 0.2 <= v <= 0.45, v, detail="eye line height in the frame")
        else:          # three-quarter / full length: the head naturally sits high in the frame
            add("eyes in the upper part (long shot)", 0.08 <= v <= 0.45, v, detail="eye line height in the frame")
        if f0["facing"]:
            fu = (f0["center_x"] - x) / cw
            ahead, behind = (1 - fu, fu) if f0["facing"] > 0 else (fu, 1 - fu)
            add("lead room", ahead / (behind + 1e-6) >= 0.8, ahead / (behind + 1e-6),
                detail="space in front of the face / behind it (>= 0.8)")
    else:
        d_line = min(abs(u - 1 / 3), abs(u - 2 / 3), abs(v - 1 / 3), abs(v - 2 / 3))
        centred = abs(u - 0.5) < 0.08 and _symmetry(ctx["gray"][int(y):int(y + ch), int(x):int(x + cw)]) > 0.6
        add("subject on a third (or centred if symmetric)", d_line <= 0.1 or centred, d_line)

    if scene == "landscape":
        land = ctx["land"]
        hv = (land["horizon_y"] - y) / ch
        if land["reflection"]:
            add("horizon placement", True, hv, detail="reflection: centred horizon allowed")
        else:
            add("horizon not through the middle", abs(hv - 0.5) >= 0.08, hv)
            add("sky:land ratio favours the interesting part", abs(hv - land["target_v"]) <= 0.1, hv,
                detail=("dramatic sky: horizon near the lower third" if land["sky_interesting"]
                        else "plain sky: horizon near the upper third"))

    hard_fail = [c["rule"] for c in checks if c["hard"] and c["status"] == "fail"]
    soft = [c for c in checks if not c["hard"]]
    score = sum(c["status"] == "pass" for c in soft) / len(soft) if soft else 1.0
    return {
        "status": "pass" if not hard_fail and score >= min_score else "fail",
        "score": round(score, 3),
        "hard_violations": hard_fail,
        "failed_rules": [c["rule"] for c in checks if c["status"] == "fail"],
        "scene": scene,
        "shot_size": shot,
        "checks": checks,
    }


# --------------------------------------------------------------------------- #
# Tool 2: apply_crop
# --------------------------------------------------------------------------- #
def apply_crop(image_path: str, rect: dict, output_path: str | None = None,
               valid_rect: dict | None = None) -> dict:
    img = read_image_keep_alpha(image_path)
    H, W = img.shape[:2]
    x, y, w, h = (int(rect[k]) for k in ("x", "y", "w", "h"))
    if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > W or y + h > H:
        raise ValueError(f"rect {rect} is outside the {W}x{H} image")
    if valid_rect:
        vx, vy, vw, vh = (int(valid_rect[k]) for k in ("x", "y", "w", "h"))
        tol = 2
        if x < vx - tol or y < vy - tol or x + w > vx + vw + tol or y + h > vy + vh + tol:
            raise ValueError(f"rect {rect} reaches outside valid_rect {valid_rect} "
                             "(would include blank corners from the rotation)")
    out = img[y:y + h, x:x + w]
    if out.ndim == 3 and out.shape[2] == 4:
        out = out[:, :, :3]          # inside valid_rect everything is opaque
    src = Path(image_path)
    if output_path is None:
        ext = ".tif" if is_raw(src) or out.dtype != np.uint8 else src.suffix
        output_path = str(src.with_name(f"{src.stem}_cropped{ext}"))
    write_image(output_path, out)
    prev = to_bgr_u8(out)
    sc = 1600 / max(prev.shape[:2])
    if sc < 1:
        prev = cv2.resize(prev, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
    preview_path = str(Path(output_path).with_name(Path(output_path).stem + "_preview.jpg"))
    write_image(preview_path, prev, jpeg_quality=90)
    g = math.gcd(w, h)
    return {
        "output_path": output_path,
        "preview_path": preview_path,
        "rect": {"x": x, "y": y, "w": w, "h": h},
        "output_size": {"w": w, "h": h},
        "aspect": f"{w // g}:{h // g}" if g > 20 else round(w / h, 3),
        "area_kept_fraction": round(w * h / (W * H), 4),
    }


def read_image_keep_alpha(path: str | Path) -> np.ndarray:
    """Like read_image, but keeps a 4th (alpha) channel if the file has one."""
    path = Path(path)
    if path.suffix.lower() not in (".png", ".tif", ".tiff"):
        return read_image(path)      # JPEG/RAW: no alpha; keep EXIF-orientation handling
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    buf = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"OpenCV could not decode: {path}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img
