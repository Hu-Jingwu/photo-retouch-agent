"""People: detection, edge-aware masks, local adjustments and portrait retouch.

Models (Google MediaPipe, in models/):
  face_landmarker.task              478 face points (run on an enlarged crop around each
                                    YuNet face, because it only finds large faces)
  selfie_multiclass_256x256.tflite  background / hair / body-skin / face-skin / clothes / other
  pose_landmarker_lite.task         33 body joints

Halo safety: every mask is snapped to real edges with a guided filter (guide = the
photo itself), so an adjustment never spills a soft glow past the subject's outline,
and check_retouch measures the ring just outside each person to prove it.

Retouch style ("像素蛋糕写真" -like, but written from general retouching practice):
texture-keeping skin smoothing (frequency separation), even skin tone, light skin
brightening, blemish removal, eye brightening, and *slight* face / body slimming by
smooth warps that are limited in size and checked for background bending.
"""
from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from . import color as C
from .composition import detect_faces
from .io_utils import read_image, write_image

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
WORK_SIDE = 1600

# MediaPipe face-mesh landmark indices (ordered contours)
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377,
             152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
LEFT_BROW = [336, 296, 334, 293, 300, 276, 283, 282, 295, 285]
RIGHT_BROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
JAW = [93, 132, 58, 172, 136, 150, 149, 176, 148, 377, 400, 378, 379, 365, 397, 288, 361, 323]
SEG_CLASSES = ("background", "hair", "body_skin", "face_skin", "clothes", "other")

PRESETS = {
    "pixcake_natural": {    # 轻修写真: clean skin with pores kept, very light shaping
        "skin_smoothing": 35, "skin_even_tone": 30, "skin_brighten": 12, "blemish_removal": True,
        "eye_brighten": 15, "face_slim": 15, "body_slim": 10},
    "pixcake_beauty": {     # 精修写真: smoother, brighter, a little more shaping (still subtle)
        "skin_smoothing": 55, "skin_even_tone": 45, "skin_brighten": 22, "blemish_removal": True,
        "eye_brighten": 25, "face_slim": 25, "body_slim": 20},
}
RETOUCH_KEYS = ("skin_smoothing", "skin_even_tone", "skin_brighten", "blemish_removal",
                "eye_brighten", "face_slim", "body_slim", "clothes_smooth", "local")
REGIONS = ("subject_light", "person", "face", "skin", "hair", "clothes", "background")
LOCAL_KEYS = ("exposure", "contrast", "highlights", "shadows", "saturation", "vibrance",
              "temperature", "tint")


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #
_models: dict = {}


def _close_models():
    """Close MediaPipe tasks before interpreter shutdown (otherwise their destructors
    print harmless but noisy TypeErrors)."""
    for m in _models.values():
        try:
            m.close()
        except Exception:
            pass
    _models.clear()


import atexit  # noqa: E402

atexit.register(_close_models)


def _mp():
    import mediapipe as mp
    from mediapipe.tasks import python as mpt
    from mediapipe.tasks.python import vision
    return mp, mpt, vision


def _model(kind: str):
    if kind in _models:
        return _models[kind]
    mp, mpt, vision = _mp()
    if kind == "face":
        m = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
            base_options=mpt.BaseOptions(model_asset_path=str(MODELS / "face_landmarker.task")), num_faces=1))
    elif kind == "seg":
        m = vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(
            base_options=mpt.BaseOptions(model_asset_path=str(MODELS / "selfie_multiclass_256x256.tflite")),
            output_confidence_masks=True, output_category_mask=False))
    else:
        m = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
            base_options=mpt.BaseOptions(model_asset_path=str(MODELS / "pose_landmarker_lite.task")), num_poses=1))
    _models[kind] = m
    return m


def _mp_image(rgb8):
    mp, _, _ = _mp()
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb8))


# --------------------------------------------------------------------------- #
# Guided filter (He et al.) - edge-aware smoothing used for masks and skin
# --------------------------------------------------------------------------- #
def guided_filter(I: np.ndarray, p: np.ndarray, r: int, eps: float) -> np.ndarray:
    box = lambda x: cv2.boxFilter(x, cv2.CV_32F, (2 * r + 1, 2 * r + 1))
    mi, mp_ = box(I), box(p)
    cov = box(I * p) - mi * mp_
    var = box(I * I) - mi * mi
    a = cov / (var + eps)
    b = mp_ - a * mi
    return box(a) * I + box(b)


def _refine(mask: np.ndarray, gray: np.ndarray, r: int) -> np.ndarray:
    """Snap a soft mask to the photo's real edges (no glow past the outline)."""
    return np.clip(guided_filter(gray, mask.astype(np.float32), r, 1e-3), 0, 1)


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #
def _crop_box(cx, cy, half_w, top, bottom, w, h):
    x0, x1 = int(max(0, cx - half_w)), int(min(w, cx + half_w))
    y0, y1 = int(max(0, top)), int(min(h, bottom))
    return x0, y0, x1, y1


def analyze(rgb: np.ndarray) -> dict:
    """Faces (with 478 landmarks), pose, and refined masks at the resolution of `rgb`."""
    h, w = rgb.shape[:2]
    rgb8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    bgr8 = np.ascontiguousarray(rgb8[..., ::-1])
    gray = cv2.cvtColor(rgb8, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    faces = [f for f in detect_faces(bgr8) if f["score"] >= 0.7][:4]

    conf = np.zeros((6, h, w), np.float32)
    conf[0] = 1.0
    seg_runs = [(0, 0, w, h)]
    people = []
    for f in faces:
        x, y, bw, bh = f["box"]
        cx, cy = x + bw / 2, y + bh / 2
        # landmarks on an enlarged square crop around the face
        side = 2.4 * max(bw, bh)
        fx0, fy0, fx1, fy1 = _crop_box(cx, cy, side / 2, cy - side / 2, cy + side / 2, w, h)
        crop = rgb8[fy0:fy1, fx0:fx1]
        lm = None
        if crop.size:
            k = 384 / max(crop.shape[:2])
            res = _model("face").detect(_mp_image(cv2.resize(crop, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)))
            if res.face_landmarks:
                lm = np.array([[p.x * crop.shape[1] + fx0, p.y * crop.shape[0] + fy0] for p in res.face_landmarks[0]],
                              np.float32)
        # body crop for pose / segmentation of small people
        bx0, by0, bx1, by1 = _crop_box(cx, cy, 4.5 * bw, y - 1.2 * bh, y + 9.5 * bh, w, h)
        pose = None
        bcrop = rgb8[by0:by1, bx0:bx1]
        if bcrop.size:
            res = _model("pose").detect(_mp_image(bcrop))
            if res.pose_landmarks:
                pose = np.array([[p.x * bcrop.shape[1] + bx0, p.y * bcrop.shape[0] + by0, p.visibility]
                                 for p in res.pose_landmarks[0]], np.float32)
        if bh < 0.15 * h:            # small in the frame: segment on the body crop for detail
            seg_runs.append((bx0, by0, bx1, by1))
        people.append({"face": f, "landmarks": lm, "pose": pose,
                       "face_width": float(np.linalg.norm(lm[234] - lm[454])) if lm is not None else bw})

    for i, (x0, y0, x1, y1) in enumerate(seg_runs):
        if not faces and i == 0:
            break
        res = _model("seg").segment(_mp_image(rgb8[y0:y1, x0:x1]))
        cm = np.stack([m.numpy_view().reshape(y1 - y0, x1 - x0) if m.numpy_view().shape[:2] == (y1 - y0, x1 - x0)
                       else cv2.resize(m.numpy_view().squeeze(), (x1 - x0, y1 - y0)) for m in res.confidence_masks])
        if i == 0:
            conf[:, y0:y1, x0:x1] = cm
        else:   # body crops refine the full-frame result for small people
            conf[:, y0:y1, x0:x1] = np.where(cm[0:1] < conf[0:1, y0:y1, x0:x1], cm, conf[:, y0:y1, x0:x1])

    r = max(2, round(0.004 * max(h, w)))
    masks = {
        "person": _refine(1 - conf[0], gray, r),
        "hair": _refine(conf[1], gray, r),
        "body_skin": _refine(conf[2], gray, r),
        "face_skin": _refine(conf[3], gray, r),
        "clothes": _refine(conf[4], gray, r),
    }
    features = np.zeros((h, w), np.float32)
    eyes = np.zeros((h, w), np.float32)
    face_region = np.zeros((h, w), np.float32)
    for p in people:
        lm = p["landmarks"]
        if lm is None:
            x, y, bw, bh = p["face"]["box"]
            cv2.ellipse(face_region, (int(x + bw / 2), int(y + bh / 2)), (int(bw * 0.55), int(bh * 0.65)),
                        0, 0, 360, 1.0, -1)
            continue
        cv2.fillConvexPoly(face_region, cv2.convexHull(lm[FACE_OVAL].astype(np.int32)), 1.0)
        for idx in (LEFT_EYE, RIGHT_EYE, LEFT_BROW, RIGHT_BROW, LIPS):
            cv2.fillConvexPoly(features, cv2.convexHull(lm[idx].astype(np.int32)), 1.0)
        for idx in (LEFT_EYE, RIGHT_EYE):
            cv2.fillConvexPoly(eyes, cv2.convexHull(lm[idx].astype(np.int32)), 1.0)
    fw = np.median([p["face_width"] for p in people]) if people else 0
    if fw:
        k = max(3, int(fw * 0.03)) | 1
        features = cv2.GaussianBlur(cv2.dilate(features, np.ones((k, k), np.uint8)), (k, k), 0)
        eyes = cv2.GaussianBlur(eyes, (k, k), 0)
    # "face" region for local edits: skin plus eyes/brows/lips inside the face outline
    masks["face"] = _refine(face_region * np.maximum(masks["face_skin"], features), gray, r) \
        if face_region.any() else masks["face_skin"]
    masks["skin"] = np.clip(masks["face_skin"] + masks["body_skin"], 0, 1) * (1 - features)
    masks["background"] = 1 - masks["person"]
    masks["subject_light"] = subject_light_mask(masks["person"])
    masks["features"] = features
    masks["eyes"] = eyes
    return {"size": (h, w), "people": people, "masks": masks, "face_width": float(fw), "gray": gray}


def subject_light_mask(person: np.ndarray) -> np.ndarray:
    """Relight the whole subject like a real light would: the full person (edge-snapped
    mask, weight 1) plus a soft, wide fall-off into the nearby scene (0.65 at the outline,
    fading over ~a third of the person's height). Brightening only the face looks pasted on;
    a narrow spill looks like a halo; a wide gentle fall-off looks like light."""
    h, w = person.shape
    rows = np.nonzero((person > 0.5).any(1))[0]
    if len(rows) == 0:
        return person.copy()
    hp = rows.max() - rows.min() + 1
    k = 4                                         # build the wide fall-off at 1/4 resolution
    small = cv2.resize(person, (max(1, w // k), max(1, h // k)), interpolation=cv2.INTER_AREA)
    sigma = max(0.3 * hp, 0.15 * max(h, w)) / k
    env = cv2.GaussianBlur(small, (0, 0), sigma)
    env = np.clip(env / max(float(env[small > 0.5].mean()) if (small > 0.5).any() else 1.0, 1e-6), 0, 1)
    env = cv2.resize(env, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.clip(np.maximum(person, 0.65 * np.clip(env * 2, 0, 1)), 0, 1).astype(np.float32)


@lru_cache(maxsize=4)
def _analyze_cached(path: str, mtime: float, side: int):
    rgb = C._load_preview(path, side)
    return rgb, analyze(rgb)


def _load(path: str, side: int = WORK_SIDE):
    return _analyze_cached(str(path), Path(path).stat().st_mtime, side)


# --------------------------------------------------------------------------- #
# Retouch operations (all on float32 sRGB [0,1])
# --------------------------------------------------------------------------- #
def normalize_retouch(params: dict | None, preset: str | None = None) -> dict:
    out = {k: 0.0 for k in RETOUCH_KEYS if k not in ("blemish_removal", "local")}
    out.update(blemish_removal=False, local=[])
    if preset:
        if preset not in PRESETS:
            raise ValueError(f"unknown preset {preset!r}; use one of {list(PRESETS)}")
        out.update(PRESETS[preset])
    for k, v in (params or {}).items():
        if k not in RETOUCH_KEYS:
            raise ValueError(f"unknown retouch parameter {k!r}; valid: {RETOUCH_KEYS}")
        if k == "blemish_removal":
            out[k] = bool(v)
        elif k == "local":
            out[k] = []
            for adj in v or []:
                reg = adj.get("region")
                if reg not in REGIONS:
                    raise ValueError(f"local region must be one of {REGIONS}")
                bad = set(adj) - set(LOCAL_KEYS) - {"region"}
                if bad:
                    raise ValueError(f"unknown local keys {bad}; valid: {LOCAL_KEYS}")
                out[k].append(dict(adj))
        else:
            out[k] = float(np.clip(float(v), 0, 100))
    return out


def _local_adjust(rgb, masks, local):
    out = rgb
    for adj in local:
        params = {k: v for k, v in adj.items() if k != "region"}
        if not any(params.values()):
            continue
        adjusted, _ = C._render_small(out, params)
        m = masks[adj["region"]][..., None]
        out = out + m * (adjusted - out)
    return np.clip(out, 0, 1)


def _skin_smooth(rgb, skin, fw, amount):
    """Frequency-separation smoothing: edge-aware low band from a guided filter, then
    put the finest texture (pores) back so skin never looks plastic."""
    if amount <= 0 or fw <= 0:
        return rgb
    ys, xs = np.nonzero(skin > 0.02)
    if len(ys) == 0:
        return rgb
    pad = int(fw * 0.3)
    y0, y1 = max(0, ys.min() - pad), min(rgb.shape[0], ys.max() + pad)
    x0, x1 = max(0, xs.min() - pad), min(rgb.shape[1], xs.max() + pad)
    sub = rgb[y0:y1, x0:x1]
    r = max(2, int(fw * 0.045))
    low = np.stack([guided_filter(sub[..., c], sub[..., c], r, 0.0025) for c in range(3)], -1)
    sig = max(0.6, fw * 0.006)
    fine = sub - cv2.GaussianBlur(sub, (0, 0), sig)
    smoothed = low + 0.55 * fine
    a = amount / 100 * 0.85
    out = rgb.copy()
    m = skin[y0:y1, x0:x1, None]
    out[y0:y1, x0:x1] = sub + a * m * (smoothed - sub)
    return np.clip(out, 0, 1)


def _even_tone(rgb, skin, fw, amount):
    if amount <= 0 or fw <= 0:
        return rgb
    lab = C._lab(rgb)
    sig = max(2.0, fw * 0.15)
    m = skin.astype(np.float32)
    den = cv2.GaussianBlur(m, (0, 0), sig) + 1e-4
    for ch in (1, 2):
        avg = cv2.GaussianBlur(lab[..., ch] * m, (0, 0), sig) / den
        lab[..., ch] = lab[..., ch] + amount / 100 * 0.7 * m * (avg - lab[..., ch])
    out = cv2.cvtColor(lab, cv2.COLOR_Lab2RGB)
    return np.clip(out, 0, 1)


def _nconv(x, m, sig):
    """Gaussian blur of x using only pixels where m > 0 (normalised convolution)."""
    return cv2.GaussianBlur(x * m, (0, 0), sig) / np.maximum(cv2.GaussianBlur(m, (0, 0), sig), 1e-4)


def _clothes_smooth(rgb, clothes, fw, amount):
    """衣服褶皱: a crease is soft shading at a middle scale. Split the clothes' lightness into fine
    (fabric texture, kept), middle (creases, reduced) and large (the garment's light, kept) bands.
    Prints, seams and logos have sharp or coloured edges and are protected; deep folds (arm against
    body) are only softened a little."""
    if amount <= 0 or fw <= 0 or clothes.max() < 0.5:
        return rgb, 0.0
    m = np.clip(clothes, 0, 1).astype(np.float32) ** 2
    lab = C._lab(rgb)
    L = lab[..., 0]
    s_fine, s_large = max(1.0, fw * 0.012), max(4.0, fw * 0.35)
    fine_less = _nconv(L, m, s_fine)
    mid = fine_less - _nconv(L, m, s_large)
    edge = cv2.GaussianBlur(np.abs(L - fine_less), (0, 0), s_fine * 3)            # print / seam detail
    ga = np.hypot(cv2.Sobel(cv2.GaussianBlur(lab[..., 1], (0, 0), 1), cv2.CV_32F, 1, 0),
                  cv2.Sobel(cv2.GaussianBlur(lab[..., 1], (0, 0), 1), cv2.CV_32F, 0, 1))
    gb = np.hypot(cv2.Sobel(cv2.GaussianBlur(lab[..., 2], (0, 0), 1), cv2.CV_32F, 1, 0),
                  cv2.Sobel(cv2.GaussianBlur(lab[..., 2], (0, 0), 1), cv2.CV_32F, 0, 1))
    colour_edge = cv2.GaussianBlur(np.hypot(ga, gb), (0, 0), s_fine * 3)
    protect = np.maximum(np.clip((edge - 1.5) / 2.5, 0, 1), np.clip((colour_edge - 3.0) / 6.0, 0, 1))
    deep = 1.0 / (1.0 + (np.abs(mid) / 15.0) ** 2)
    delta = -amount / 100 * 0.95 * mid * (1 - protect) * deep * m
    lab[..., 0] = L + delta
    out = np.clip(cv2.cvtColor(lab, cv2.COLOR_Lab2RGB), 0, 1)
    sel = m > 0.5
    return out, float(np.abs(delta)[sel].mean()) if sel.any() else 0.0


def _brighten(rgb, mask, ev):
    if ev == 0:
        return rgb
    lin = C._s2l(rgb) * (2 ** (ev * mask))[..., None]
    return np.clip(C._l2s(lin), 0, 1)


def _blemishes(rgb, face_skin, features, fw, people=None):
    """Small, round, dark spots on plain face skin -> inpaint. Returns (image, count).

    Eyes, brows, lips and the nostrils are excluded with a generous margin (dark by nature,
    not blemishes). Inpainting runs on 16-bit channels: OpenCV's inpaint does not support
    float images properly and produced white speckles."""
    if fw <= 0:
        return rgb, 0
    h, w = rgb.shape[:2]
    L = C._lab(rgb)[..., 0]
    s1 = max(1.0, fw * 0.02)
    dog = cv2.GaussianBlur(L, (0, 0), s1 * 2) - cv2.GaussianBlur(L, (0, 0), s1 * 0.5)
    k = max(3, int(fw * 0.08)) | 1
    near_features = cv2.dilate((features > 0.05).astype(np.uint8), np.ones((k, k), np.uint8))
    allowed = (face_skin > 0.7) & (near_features == 0)
    for p in people or []:                           # nostrils / nose tip are dark by nature
        lm = p.get("landmarks")
        if lm is not None:
            cv2.circle(allowed.view(np.uint8), tuple(int(v) for v in lm[1]), int(p["face_width"] * 0.14), 0, -1)
    spots = (dog > 5.0) & allowed
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(spots.astype(np.uint8), 8)
    area_max = (fw * 0.05) ** 2
    keep = np.zeros((h, w), np.uint8)
    count = 0
    for i in range(1, n):
        a = stats[i, cv2.CC_STAT_AREA]
        bw_, bh_ = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        if 2 <= a <= area_max and max(bw_, bh_) <= 2.5 * min(bw_, bh_) and count < 60:   # round-ish
            keep[lbl == i] = 1
            count += 1
    if not count:
        return rgb, 0
    keep = cv2.dilate(keep, np.ones((3, 3), np.uint8))
    out = rgb.copy()
    for c in range(3):
        ch16 = (np.clip(rgb[..., c], 0, 1) * 65535).astype(np.uint16)
        out[..., c] = cv2.inpaint(ch16, keep, 3, cv2.INPAINT_TELEA).astype(np.float32) / 65535
    return out, count


def _warp_field(ana, face_slim, body_slim, shape):
    """Smooth displacement field (dx only) for slight face and body slimming."""
    h, w = shape
    dx = np.zeros((h, w), np.float32)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    stats = {"face_max_shift_frac": 0.0, "body_max_shift_frac": 0.0}
    for p in ana["people"]:
        lm, fw = p["landmarks"], p["face_width"]
        if face_slim > 0 and lm is not None:
            cx = float(lm[1, 0])                      # nose tip: the face's centre line
            sig = fw * 0.12
            num = np.zeros((h, w), np.float32)
            den = np.zeros((h, w), np.float32)
            x0, x1 = int(max(0, cx - fw)), int(min(w, cx + fw))
            y0, y1 = int(max(0, lm[10, 1])), int(min(h, lm[152, 1] + fw * 0.4))
            sx, sy = xx[y0:y1, x0:x1], yy[y0:y1, x0:x1]
            for i in JAW:
                px, py = lm[i]
                g = np.exp(-((sx - px) ** 2 + (sy - py) ** 2) / (2 * sig * sig))
                # 0.038: strength calibrated against PixCake's real 瘦脸 effect (see face.GAIN)
                num[y0:y1, x0:x1] += g * (cx - px) * 0.038 * face_slim / 100
                den[y0:y1, x0:x1] += g
            field = num / np.maximum(den, 1.0)
            dx += field
            stats["face_max_shift_frac"] = max(stats["face_max_shift_frac"], float(np.abs(field).max() / fw))
        pose = p["pose"]
        if body_slim > 0 and pose is not None and min(pose[[11, 12, 23, 24], 2]) > 0.5:
            sh, hp = pose[[11, 12], :2], pose[[23, 24], :2]
            knee_ok = min(pose[[25, 26], 2]) > 0.5
            pts_y = [sh[:, 1].mean(), hp[:, 1].mean()] + ([pose[[25, 26], 1].mean()] if knee_ok else [])
            pts_x = [sh[:, 0].mean(), hp[:, 0].mean()] + ([pose[[25, 26], 0].mean()] if knee_ok else [])
            hw_sh, hw_hp = abs(sh[0, 0] - sh[1, 0]) / 2, abs(hp[0, 0] - hp[1, 0]) / 2 * 1.4
            top = pts_y[0] + 0.15 * (pts_y[1] - pts_y[0])
            bot = pts_y[-1] if knee_ok else pts_y[1] + 0.8 * (pts_y[1] - pts_y[0])
            ys = np.arange(h, dtype=np.float32)
            xc = np.interp(ys, pts_y, pts_x).astype(np.float32)
            hw = np.interp(ys, [pts_y[0], pts_y[1]], [hw_sh, hw_hp]).astype(np.float32)
            span = max(bot - top, 1.0)
            wy = np.clip(np.minimum((ys - top) / (0.25 * span), (bot - ys) / (0.25 * span)), 0, 1)
            wy = wy * wy * (3 - 2 * wy)
            rel = (xx - xc[:, None]) / np.maximum(hw[:, None], 1.0)
            wx = np.exp(-(rel / 1.6) ** 4)
            field = -0.05 * body_slim / 100 * (xx - xc[:, None]) * wx * wy[:, None]
            dx += field
            stats["body_max_shift_frac"] = max(stats["body_max_shift_frac"],
                                               float(np.abs(field).max() / max(2 * max(hw_sh, hw_hp), 1)))
    return dx, stats


def _apply_warp(rgb, dx, dy=None):
    """Backward-map warp: output pixel p samples the input at p - d(p) (small, smooth d)."""
    if not np.any(dx) and (dy is None or not np.any(dy)):
        return rgb
    h, w = dx.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    return cv2.remap(rgb, xx - dx, yy - (dy if dy is not None else 0), cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REFLECT)


def render_retouch(rgb, ana, params: dict) -> tuple[np.ndarray, dict]:
    m, fw = ana["masks"], ana["face_width"]
    info = {}
    out = _local_adjust(rgb, m, params["local"])
    if params["blemish_removal"]:
        out, info["blemishes_removed"] = _blemishes(out, m["face_skin"], m["features"], fw, ana["people"])
    out = _skin_smooth(out, m["skin"], fw, params["skin_smoothing"])
    out = _even_tone(out, m["skin"], fw, params["skin_even_tone"])
    out = _brighten(out, m["skin"], 0.35 * params["skin_brighten"] / 100)
    out = _brighten(out, m["eyes"], 0.4 * params["eye_brighten"] / 100)
    if params["clothes_smooth"] > 0:
        out, info["clothes_smoothed_L"] = _clothes_smooth(out, m["clothes"], fw, params["clothes_smooth"])
    dx, st = _warp_field(ana, params["face_slim"], params["body_slim"], out.shape[:2])
    out = _apply_warp(out, dx)
    info.update(st)
    bg = m["person"] < 0.2
    info["background_max_shift_px_frac"] = float(np.abs(dx[bg]).max() / dx.shape[1]) if bg.any() and np.any(dx) else 0.0
    info["_dx"] = dx
    return np.clip(out, 0, 1).astype(np.float32), info


# --------------------------------------------------------------------------- #
# Checks (used by the tools and the evaluation harness)
# --------------------------------------------------------------------------- #
def retouch_checks(before, after, ana, info) -> list[dict]:
    out = []

    def add(rule, ok, value, detail):
        out.append({"rule": rule, "status": "pass" if ok else "fail", "value": value, "detail": detail})

    m = ana["masks"]
    person = (m["person"] > 0.5).astype(np.uint8)
    if person.any():
        h, w = person.shape
        r1 = max(3, int(0.02 * max(h, w)))
        d1 = cv2.dilate(person, np.ones((r1, r1), np.uint8))
        d2 = cv2.dilate(person, np.ones((3 * r1, 3 * r1), np.uint8))
        d3 = cv2.dilate(person, np.ones((6 * r1, 6 * r1), np.uint8))
        near = (d1 > 0) & (person == 0)
        far = (d3 > 0) & (d2 == 0)
        if near.any() and far.any():
            dL = C._lab(after)[..., 0] - C._lab(before)[..., 0]
            dn, df = float(dL[near].mean()), float(dL[far].mean())
            diff = abs(dn - df)
            # A halo is a NARROW glow: almost all of the change sits right at the outline and is
            # gone a little further out. A deliberate relight (subject_light) falls off gently,
            # so the near and far rings change by similar amounts.
            ratio = diff / max(abs(dn), 1.0)
            add("no halo around people", diff <= 2.5 or ratio <= 0.4, [round(diff, 2), round(ratio, 2)],
                "[L* change near outline minus further out, share of the near change]; "
                "fails only if > 2.5 AND > 40% (a narrow glow)")
    # the face must not be lit separately from the rest of the person
    yb, ya = C._luma(C._s2l(before)), C._luma(C._s2l(after))
    # Only pixels that can show their gain: not near black before, not near white after
    # (a white shirt relit by +2 EV clips and would under-count the body's lift).
    usable = (yb > 0.003) & (ya < 0.8)
    face = (m["face_skin"] > 0.6) & usable
    body = (m["body_skin"] > 0.6) & usable
    if body.sum() < 200:      # no visible body skin: fall back to the rest of the person
        body = (m["person"] > 0.6) & (m["face_skin"] < 0.1) & (m["hair"] < 0.3) & usable
    if face.sum() > 50 and body.sum() > 200:
        ratio = np.log2(np.maximum(ya, 1e-6) / np.maximum(yb, 1e-6))
        ev = lambda sel: float(np.median(ratio[sel]))
        gap = ev(face) - ev(body)
        add("face not lit separately", gap <= 0.8, round(gap, 2),
            "extra brightening on the face beyond the body, EV (<= 0.8); relight the whole person instead")
    skin = m["face_skin"] > 0.7
    if skin.sum() > 200:
        hp = lambda im: np.abs(C._lab(im)[..., 0] - cv2.GaussianBlur(C._lab(im)[..., 0], (0, 0), 1.5))
        kept = float(hp(after)[skin].mean() / max(hp(before)[skin].mean(), 1e-6))
        add("skin texture kept", kept >= 0.35, round(kept, 2), "fine detail on face skin after/before (>= 0.35)")
    if "clothes_smoothed_L" in info:
        cl = m["clothes"] > 0.7
        if cl.sum() > 200:
            hp = lambda im: np.abs(C._lab(im)[..., 0] - cv2.GaussianBlur(C._lab(im)[..., 0], (0, 0), 1.5))
            kept = float(hp(after)[cl].mean() / max(hp(before)[cl].mean(), 1e-6))
            add("fabric texture kept", kept >= 0.85, round(kept, 2), "fine detail on clothes after/before (>= 0.85)")
            la, lb = C._lab(after), C._lab(before)
            dab = float(np.hypot(la[..., 1] - lb[..., 1], la[..., 2] - lb[..., 2])[cl].mean())
            add("clothes colours and prints kept", dab <= 1.0, round(dab, 2), "mean colour change on clothes, Δab (<= 1)")
    add("face slimming subtle", info.get("face_max_shift_frac", 0) <= 0.04,
        round(info.get("face_max_shift_frac", 0), 4), "max jaw shift / face width (<= 4%)")
    add("body slimming subtle", info.get("body_max_shift_frac", 0) <= 0.03,
        round(info.get("body_max_shift_frac", 0), 4), "max shift / body width (<= 3%)")
    add("background not warped", info.get("background_max_shift_px_frac", 0) <= 0.002,
        round(info.get("background_max_shift_px_frac", 0), 5), "max shift of background pixels / image width (<= 0.2%)")
    return out


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #
def detect_people(image_path: str, overlay_path: str | None = None) -> dict:
    rgb, ana = _load(image_path)
    h, w = ana["size"]
    m = ana["masks"]
    lab = C._lab(rgb)
    people = []
    for i, p in enumerate(ana["people"]):
        x, y, bw, bh = p["face"]["box"]
        fm = (m["face_skin"] > 0.6)
        reg = np.zeros((h, w), bool)
        reg[int(y):int(y + bh), int(x):int(x + bw)] = True
        sel = fm & reg
        face_L = float(np.median(lab[..., 0][sel])) if sel.sum() > 20 else None
        pose = p["pose"]
        people.append({
            "id": i, "face_box": [round(v / w, 3) if j % 2 == 0 else round(v / h, 3) for j, v in enumerate([x, y, bw, bh])],
            "face_height_frac": round(bh / h, 3),
            "facing": {1: "image right", -1: "image left", 0: "camera"}[p["face"]["facing"]],
            "landmarks": p["landmarks"] is not None,
            "body_visible": bool(pose is not None and min(pose[[11, 12, 23, 24], 2]) > 0.5),
            "face_lightness_L": round(face_L, 1) if face_L is not None else None,
        })
    bg = m["background"] > 0.7
    bg_L = float(np.median(lab[..., 0][bg])) if bg.any() else None
    if overlay_path is None:
        p_ = Path(image_path)
        overlay_path = str(p_.with_name(p_.stem + "_people.jpg"))
    vis = (rgb[..., ::-1] * 255).astype(np.uint8).copy()
    tint = np.zeros_like(vis)
    tint[..., 2] = (m["face_skin"] * 255).astype(np.uint8)
    tint[..., 1] = (m["body_skin"] * 200).astype(np.uint8)
    tint[..., 0] = (m["hair"] * 255).astype(np.uint8)
    vis = cv2.addWeighted(vis, 0.65, tint, 0.35, 0)
    edge = cv2.Canny((m["person"] > 0.5).astype(np.uint8) * 255, 50, 150)
    vis[edge > 0] = (255, 255, 255)
    for p in ana["people"]:
        if p["landmarks"] is not None:
            for x_, y_ in p["landmarks"][::6]:
                cv2.circle(vis, (int(x_), int(y_)), 1, (0, 255, 255), -1)
        if p["pose"] is not None:
            for x_, y_, v in p["pose"]:
                if v > 0.5:
                    cv2.circle(vis, (int(x_), int(y_)), 4, (0, 255, 0), -1)
    write_image(overlay_path, vis, jpeg_quality=88)
    return {"num_people": len(people), "people": people, "background_lightness_L": bg_L and round(bg_L, 1),
            "overlay_path": overlay_path, "presets": PRESETS,
            "note": "Overlay: red = face skin, green = body skin, blue = hair, white outline = person mask, "
                    "yellow dots = face landmarks, green dots = body joints."}


def _side_by_side(before, after, ana, path):
    h, w = before.shape[:2]
    tiles = [(before, "before"), (after, "after")]
    if ana["people"]:
        x, y, bw, bh = ana["people"][0]["face"]["box"]
        s = int(max(bw, bh) * 1.6)
        cx, cy = int(x + bw / 2), int(y + bh / 2)
        x0, y0 = max(0, cx - s), max(0, cy - s)
        x1, y1 = min(w, cx + s), min(h, cy + s)
        tiles += [(before[y0:y1, x0:x1], "before (face)"), (after[y0:y1, x0:x1], "after (face)")]
    th = 520
    ims = []
    for im, lab_ in tiles:
        k = th / im.shape[0]
        t = cv2.resize((im[..., ::-1] * 255).astype(np.uint8), (max(1, round(im.shape[1] * k)), th),
                       interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_CUBIC)
        ims.append(C._label(t, lab_))
    C._sheet(ims, path, cols=4)


def preview_retouch(image_path: str, params: dict | None = None, preset: str | None = None,
                    sheet_path: str | None = None) -> dict:
    rgb, ana = _load(image_path)
    p = normalize_retouch(params, preset)
    out, info = render_retouch(rgb, ana, p)
    if sheet_path is None:
        p_ = Path(image_path)
        sheet_path = str(p_.with_name(p_.stem + "_retouch_preview.jpg"))
    _side_by_side(rgb, out, ana, sheet_path)
    checks = retouch_checks(rgb, out, ana, info)
    m = C.measure(out)
    return {"sheet_path": sheet_path, "effective": {k: v for k, v in p.items() if v},
            "info": {k: v for k, v in info.items() if not k.startswith("_")},
            "checks_failed": [c for c in checks if c["status"] == "fail"],
            "colour_issues": [i for i in C.color_issues(m) if i["status"] == "fail"]}


def apply_retouch(image_path: str, params: dict | None = None, preset: str | None = None,
                  output_path: str | None = None) -> dict:
    """Full-resolution retouch. Analysis runs at 2048 px; masks and warp fields are upsampled
    and re-snapped to full-resolution edges."""
    full, ana_full = load_full(image_path)
    H, W = full.shape[:2]
    p = normalize_retouch(params, preset)
    out, info = render_retouch(full, ana_full, p)
    src = Path(image_path)
    if output_path is None:
        output_path = str(src.with_name(f"{src.stem}_retouch.tif"))
    write_image(output_path, (np.clip(out, 0, 1)[..., ::-1] * 65535 + 0.5).astype(np.uint16))
    kp = min(1.0, 1600 / max(H, W))
    prev = cv2.resize(out, None, fx=kp, fy=kp, interpolation=cv2.INTER_AREA)
    preview_path = str(Path(output_path).with_name(Path(output_path).stem + "_preview.jpg"))
    C._save_rgb8(preview_path, prev)
    sm_before = cv2.resize(full, (prev.shape[1], prev.shape[0]), interpolation=cv2.INTER_AREA)
    ana_prev = _load(image_path, max(prev.shape[:2]))[1]
    checks = retouch_checks(sm_before, prev, ana_prev, info)
    return {"output_path": output_path, "preview_path": preview_path,
            "effective": {k_: v for k_, v in p.items() if v},
            "info": {k_: v for k_, v in info.items() if not k_.startswith("_")},
            "checks": checks}


def load_full(image_path: str):
    """Full-resolution float image plus an analysis scaled to it. Detection runs at 2048 px;
    masks are upsampled and re-snapped to full-resolution edges, points are scaled."""
    img = read_image(image_path)
    H, W = img.shape[:2]
    full = C._to_rgb_float(img)
    small, ana = _load(image_path, 2048)
    k = W / small.shape[1]
    gray = cv2.cvtColor((full * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    r = max(2, round(0.002 * max(H, W)))
    masks = {}
    for name, mk in ana["masks"].items():
        up = cv2.resize(mk, (W, H), interpolation=cv2.INTER_LINEAR)
        masks[name] = _refine(up, gray, r) if name in ("person", "face_skin", "body_skin", "skin", "face",
                                                        "hair", "clothes", "background", "subject_light") else up
    ana_full = {"size": (H, W), "masks": masks, "face_width": ana["face_width"] * k, "gray": gray,
                "people": [{**pp, "landmarks": None if pp["landmarks"] is None else pp["landmarks"] * k,
                            "pose": None if pp["pose"] is None else pp["pose"] * np.array([k, k, 1], np.float32),
                            "face_width": pp["face_width"] * k,
                            "face": {**pp["face"], "box": [v * k for v in pp["face"]["box"]],
                                     "head": [v * k for v in pp["face"]["head"]]}}
                           for pp in ana["people"]]}
    return full, ana_full
