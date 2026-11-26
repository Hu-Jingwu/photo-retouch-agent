"""Export an edit as a Lightroom / Camera Raw XMP sidecar, so the RAW can be finished in Lightroom.

The agent's decisions are slider values, so they can be written as Camera Raw settings. Lightroom
then applies them to the RAW itself, and every value stays editable. Our renderer and Adobe's are
different programs, so the result in Lightroom is a close starting point, not the same picture:

- Exact: which sliders were used and in which direction; the tone curve points; HSL; colour grading.
- Approximate: slider strengths (each program has its own formula behind a slider of the same name);
  white balance, which Lightroom needs in Kelvin while the agent stores a shift from the camera's
  setting (converted below with the camera's colour matrix); rotation and crop (Camera Raw's
  conventions are assumed as documented in `_geometry` and should be checked once on a test photo).
- Not exported: people retouch, local relighting, skin colour fixes, clothing wrinkles, the beauty
  retouch (face shape, lipstick, contour), background blur and film LUTs. Camera Raw settings cannot
  express them; they stay in the 16-bit TIFF of that stage.

    python -m agent_tools.xmp runs/<photo>_<time>            # writes <photo>.xmp in the run
    python -m agent_tools.xmp runs/<photo>_<time> --next-to-raw

Lightroom reads a sidecar for proprietary RAW files (.NEF, .CR2, .ARW...): put `<photo>.xmp` beside
the RAW, then Metadata > Read Metadata from File. It ignores sidecars for DNG, JPEG and TIFF, whose
settings live inside the file; for those the .xmp is a record of the settings only.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
from functools import lru_cache
from pathlib import Path
from xml.sax.saxutils import quoteattr

import cv2
import numpy as np

from . import color
from .io_utils import is_raw, read_image

CRS = "http://ns.adobe.com/camera-raw-settings/1.0/"
PROCESS_VERSION = "11.0"          # Lightroom's current process version family (PV2012 sliders)
LR_COLORS = ("Red", "Orange", "Yellow", "Green", "Aqua", "Blue", "Purple", "Magenta")
# Camera Raw tint unit: the light's uv offset from the Planckian locus toward green, x 3000. A greenish
# light (fluorescent; Lightroom's Fluorescent preset is +21) has a positive tint, and daylight about +10.
TINT_SCALE = 3000.0
BRADFORD = np.array([[0.8951, 0.2664, -0.1614], [-0.7502, 1.7135, 0.0367], [0.0389, -0.0685, 1.0296]])
D65 = np.array([0.95047, 1.0, 1.08883])
GREY_Y = float(color._s2l(np.float32(0.5)))      # the mid-grey the slider slopes were measured on


# --------------------------------------------------------------------------- #
# White balance: our relative shift -> Camera Raw Kelvin and tint
# --------------------------------------------------------------------------- #
def _planck_uv(T: float) -> np.ndarray:
    """CIE 1960 uv of a black body (Krystek's approximation, 1000-15000 K)."""
    u = (0.860117757 + 1.54118254e-4 * T + 1.28641212e-7 * T * T) / (1 + 8.42420235e-4 * T + 7.08145163e-7 * T * T)
    v = (0.317398726 + 4.22806245e-5 * T + 4.20481691e-8 * T * T) / (1 - 2.89741816e-5 * T + 1.61456053e-7 * T * T)
    return np.array([u, v])


def _normal(T: float) -> np.ndarray:
    """Unit normal to the locus at T, pointing toward green (+v), the direction of a positive tint."""
    t = _planck_uv(T * 1.01) - _planck_uv(T * 0.99)
    n = np.array([-t[1], t[0]]) / np.hypot(*t)
    return n if n[1] > 0 else -n


def white_xyz(kelvin: float, tint: float) -> np.ndarray:
    """XYZ (Y = 1) of the white Camera Raw means by Temperature = kelvin, Tint = tint."""
    u, v = _planck_uv(kelvin) + _normal(kelvin) * (tint / TINT_SCALE)
    x, y = 3 * u / (2 * u - 8 * v + 4), 2 * v / (2 * u - 8 * v + 4)
    return np.array([x / y, 1.0, (1 - x - y) / y])


def kelvin_tint(xyz: np.ndarray) -> tuple[float, float]:
    """Inverse of white_xyz: the nearest locus temperature and the signed offset as a tint."""
    X, Y, Z = xyz / xyz[1]
    uv = np.array([4 * X / (X + 15 * Y + 3 * Z), 6 * Y / (X + 15 * Y + 3 * Z)])
    mireds = np.linspace(1e6 / 15000, 1e6 / 1667, 2000)
    d = [np.hypot(*(uv - _planck_uv(1e6 / m))) for m in mireds]
    m = mireds[int(np.argmin(d))]
    for step in (0.5, 0.05):                       # refine the search around the best mired
        cand = np.arange(m - 20 * step, m + 20 * step, step)
        m = cand[int(np.argmin([np.hypot(*(uv - _planck_uv(1e6 / c))) for c in cand]))]
    T = 1e6 / m
    return T, float((uv - _planck_uv(T)) @ _normal(T)) * TINT_SCALE


def _grey_ab(scene_white: np.ndarray, assumed_white: np.ndarray) -> np.ndarray:
    """Lab a*, b* of a neutral grey lit by scene_white when the converter assumes assumed_white."""
    lms = BRADFORD @ (GREY_Y * scene_white)
    xyz = np.linalg.solve(BRADFORD, lms * (BRADFORD @ D65) / (BRADFORD @ assumed_white))
    f = lambda t: np.where(t > 216 / 24389, np.cbrt(t), (24389 / 27 * t + 16) / 116)
    fx, fy, fz = f(xyz / D65)
    return np.array([500 * (fx - fy), 200 * (fy - fz)])


def solve_white_balance(as_shot: tuple[float, float], temperature: float, tint: float) -> tuple[float, float]:
    """Camera Raw (Kelvin, tint) that shifts a grey as far as our temperature/tint sliders do.

    Our sliders move a mid-grey by `temperature * _DB_PER_TEMP` in b* and `tint * _DA_PER_TINT` in a*.
    Telling Camera Raw the light was (K, tint) instead of the as-shot white tints the same grey; solve
    for the setting that gives the same a*, b* (Newton steps on mired and tint)."""
    scene = white_xyz(*as_shot)
    target = np.array([tint * color._DA_PER_TINT, temperature * color._DB_PER_TEMP])
    x = np.array([1e6 / as_shot[0], as_shot[1]])                    # mired, tint
    ab = lambda v: _grey_ab(scene, white_xyz(1e6 / v[0], v[1]))
    for _ in range(30):
        r = ab(x) - target
        if np.abs(r).max() < 1e-3:
            break
        h = 0.5
        J = np.column_stack([(ab(x + e) - ab(x - e)) / (2 * h) for e in (np.array([h, 0]), np.array([0, h]))])
        x = x - np.linalg.lstsq(J, r, rcond=None)[0]
        x[0] = float(np.clip(x[0], 1e6 / 50000, 1e6 / 2000))
    return float(np.clip(1e6 / x[0], 2000, 50000)), float(np.clip(x[1], -150, 150))


def as_shot_white(raw_path: str | Path) -> dict | None:
    """The camera's white balance as Kelvin and tint, from its multipliers and colour matrix."""
    p = Path(raw_path)
    return dict(_as_shot_white(str(p), p.stat().st_mtime))


@lru_cache(maxsize=16)
def _as_shot_white(raw_path: str, mtime: float) -> dict | None:
    import rawpy
    with open(raw_path, "rb") as fh, rawpy.imread(fh) as raw:
        mult = np.array(raw.camera_whitebalance[:3], float)
        cam_xyz = np.array(raw.rgb_xyz_matrix[:3], float)
        flip = int(raw.sizes.flip)
    if mult.min() <= 0 or not np.any(cam_xyz):
        return {"flip": flip, "kelvin": None, "tint": None}
    neutral = mult[1] / mult                     # the camera's neutral, green = 1
    xyz = np.linalg.solve(cam_xyz, neutral)
    T, tint = kelvin_tint(xyz / xyz[1])
    return {"flip": flip, "kelvin": T, "tint": tint}


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=16)
def _picture_size(path: str, mtime: float) -> tuple[int, int]:
    """(height, width) of the upright picture; cached, because decoding a RAW takes seconds and one
    run writes a sidecar for every stage."""
    return tuple(read_image(path).shape[:2])

def _to_sensor(x: float, y: float, flip: int) -> tuple[float, float]:
    """A normalised point of the upright picture -> the same point in the RAW's stored orientation.

    LibRaw `flip`: 0 none, 3 = 180 degrees, 5 = stored 90 degrees counter-clockwise, 6 = 90 clockwise.
    Camera Raw keeps the crop in the stored orientation, so the orientation flag can change freely."""
    return {0: (x, y), 3: (1 - x, 1 - y), 5: (1 - y, x), 6: (y, 1 - x)}.get(flip, (x, y))


def _geometry(size: tuple[int, int], angle: float, rect: dict | None, flip: int) -> dict:
    """Camera Raw crop settings for our rotation (about the centre, counter-clockwise positive, same
    canvas) followed by a crop in the rotated frame.

    Assumed Camera Raw conventions, to check once on a test photo: CropAngle is positive clockwise, and
    CropLeft/Top/Right/Bottom are the crop box before its rotation about its own centre, normalised to
    the picture in the RAW's stored orientation."""
    W, H = size
    if not rect and not angle:
        return {"HasCrop": False}
    rect = rect or {"x": 0, "y": 0, "w": W, "h": H}
    cx, cy = rect["x"] + rect["w"] / 2, rect["y"] + rect["h"] / 2
    back = cv2.invertAffineTransform(cv2.getRotationMatrix2D((W / 2, H / 2), angle, 1.0))
    ox, oy = back @ np.array([cx, cy, 1.0])            # crop centre in the unrotated picture
    corners = [_to_sensor((ox + sx * rect["w"] / 2) / W, (oy + sy * rect["h"] / 2) / H, flip)
               for sx in (-1, 1) for sy in (-1, 1)]
    xs, ys = zip(*corners)
    return {"HasCrop": True, "CropLeft": min(xs), "CropTop": min(ys), "CropRight": max(xs),
            "CropBottom": max(ys), "CropAngle": -angle}


# --------------------------------------------------------------------------- #
# Colour
# --------------------------------------------------------------------------- #
def _compose_curves(first: list | None, second: list | None) -> list[list[int]] | None:
    """One point curve equal to applying `first` then `second` (both 0-255 point lists)."""
    curves = [c for c in (first, second) if c]
    if not curves:
        return None
    if len(curves) == 1:
        pts = curves[0]
    else:
        def full(c):
            xs, ys = [a / 255 for a, _ in c], [b / 255 for _, b in c]
            if xs[0] > 0:
                xs, ys = [0.0] + xs, [0.0] + ys
            if xs[-1] < 1:
                xs, ys = xs + [1.0], ys + [1.0]
            return xs, ys
        x = np.linspace(0, 1, 17)
        y = color._pchip(*full(curves[0]), x)
        y = color._pchip(*full(curves[1]), np.clip(y, 0, 1))
        pts = [[a * 255, b * 255] for a, b in zip(x, np.clip(y, 0, 1))]
    return [[int(round(a)), int(round(b))] for a, b in pts]


def _signed(v: float, digits: int = 0) -> str:
    return f"{v:+.{digits}f}" if round(v, digits) else ("0" if digits == 0 else f"{0:.{digits}f}")


def settings(src: str | Path, state: dict) -> tuple[dict, dict, dict]:
    """(crs attributes, crs sequences, report) for an agent state (agent.new_state() layout)."""
    src = Path(src)
    attrs: dict = {"Version": "15.0", "ProcessVersion": PROCESS_VERSION, "HasSettings": True,
                   "AlreadyApplied": False}
    seqs: dict = {}
    report: dict = {"exported": [], "approximate": [], "not_exported": [], "checks": []}

    raw_info = as_shot_white(src) if is_raw(src) else None
    flip = raw_info["flip"] if raw_info else 0

    cp = state.get("color_params")
    if cp:
        p, recipe_curve = color._merge(cp.get("params"), cp.get("recipe"), cp.get("recipe_strength", 1.0))
        attrs["Exposure2012"] = _signed(p["exposure"], 2)
        for ours, theirs in (("contrast", "Contrast2012"), ("highlights", "Highlights2012"),
                             ("shadows", "Shadows2012"), ("whites", "Whites2012"), ("blacks", "Blacks2012"),
                             ("vibrance", "Vibrance"), ("saturation", "Saturation")):
            attrs[theirs] = _signed(p[ours])
        if p["grain"]:
            attrs["GrainAmount"] = int(round(p["grain"]))
        report["approximate"].append("exposure, contrast, highlights, shadows, whites, blacks, vibrance, "
                                     "saturation: same sliders, different formulas")
        for ours, theirs in zip(color.COLORS, LR_COLORS):
            adj = p["hsl"].get(ours, {})
            attrs[f"HueAdjustment{theirs}"] = _signed(adj.get("hue", 0))
            attrs[f"SaturationAdjustment{theirs}"] = _signed(adj.get("saturation", 0))
            attrs[f"LuminanceAdjustment{theirs}"] = _signed(adj.get("luminance", 0))
        for band, name in (("shadows", "Shadow"), ("midtones", "Midtone"), ("highlights", "Highlight")):
            g = p["grading"].get(band)
            if g and g["saturation"] > 0:
                attrs[f"ColorGrade{name}Hue"] = int(round(g["hue"])) % 360
                attrs[f"ColorGrade{name}Sat"] = int(round(g["saturation"]))
        curve = _compose_curves(p["curve"], recipe_curve)
        if curve:
            attrs["ToneCurveName2012"] = "Custom"
            seqs["ToneCurvePV2012"] = [f"{a}, {b}" for a, b in curve]
            report["exported"].append(f"tone curve ({len(curve)} points)")
        report["exported"].append("HSL and colour grading values")
        if p.get("lut"):
            report["not_exported"].append(f"film LUT {p['lut']['name']} (Camera Raw cannot load a HaldCLUT)")

        t, ti = p["temperature"], p["tint"]
        if not (t or ti):
            attrs["WhiteBalance"] = "As Shot"
            report["exported"].append("white balance: As Shot")
        elif raw_info and raw_info["kelvin"]:
            k, tn = solve_white_balance((raw_info["kelvin"], raw_info["tint"]), t, ti)
            attrs.update(WhiteBalance="Custom", Temperature=int(round(k / 50) * 50), Tint=_signed(tn))
            report["approximate"].append(
                f"white balance: as shot about {raw_info['kelvin']:.0f} K / tint {raw_info['tint']:+.0f} "
                f"(from the camera's colour matrix), shifted by temperature {t:+.0f}, tint {ti:+.0f} "
                f"-> {attrs['Temperature']} K / {attrs['Tint']}")
        elif raw_info:
            attrs["WhiteBalance"] = "As Shot"
            report["not_exported"].append("white balance shift: this RAW has no colour matrix to convert it")
        else:                                   # JPEG / TIFF: Camera Raw uses relative units here
            attrs.update(WhiteBalance="Custom", IncrementalTemperature=_signed(t), IncrementalTint=_signed(ti))
            report["approximate"].append("white balance as Camera Raw's relative -100..100 units")
    else:
        report["checks"].append("no colour edit was applied")

    angle, rect = float(state.get("applied_angle") or 0), state.get("crop_rect") or (
        state.get("valid_rect") if state.get("applied_angle") else None)
    if angle or rect:
        h, w = _picture_size(str(src), src.stat().st_mtime)
        geo = _geometry((w, h), angle, rect, flip)
        attrs.update({k: (round(v, 6) if isinstance(v, float) else v) for k, v in geo.items()})
        report["approximate"].append(
            f"rotation {angle:+.2f} degrees and crop: Camera Raw conventions assumed (CropAngle clockwise "
            "positive, crop in the RAW's stored orientation); check once on a test photo")
        if flip:
            report["checks"].append(f"the RAW is stored rotated (flip {flip}); the crop was mapped to that orientation")

    for stage, name in (("retouch", "people retouch, local light and clothing wrinkles"),
                        ("beauty", "beauty retouch (skin, face shape, lipstick, contour)"),
                        ("blur", "background blur")):
        if state.get(stage):
            report["not_exported"].append(f"{name}: kept in the TIFF only")
    report["lightroom_reads_sidecar"] = is_raw(src) and src.suffix.lower() != ".dng"
    return attrs, seqs, report


def _value(v) -> str:
    return ("True" if v else "False") if isinstance(v, bool) else str(v)


def xmp_text(attrs: dict, seqs: dict) -> str:
    lines = ['<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="photo-retouch-agent">',
             ' <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">',
             f'  <rdf:Description rdf:about="" xmlns:crs="{CRS}"']
    lines += [f"    crs:{k}={quoteattr(_value(v))}" for k, v in attrs.items()]
    lines[-1] += ">"
    for name, items in seqs.items():
        lines += [f"   <crs:{name}>", "    <rdf:Seq>"]
        lines += [f"     <rdf:li>{item}</rdf:li>" for item in items]
        lines += ["    </rdf:Seq>", f"   </crs:{name}>"]
    lines += ["  </rdf:Description>", " </rdf:RDF>", "</x:xmpmeta>", ""]
    return "\n".join(lines)


def export_xmp(src: str | Path, state: dict, out_path: str | Path) -> dict:
    """Write the sidecar for one edit and return what was exported, approximated and left out."""
    attrs, seqs, report = settings(src, state)
    out_path = Path(out_path)
    out_path.write_text(xmp_text(attrs, seqs), encoding="utf-8")
    return {"xmp_path": str(out_path), **report}


def place_next_to_raw(xmp_path: str | Path, raw_path: str | Path) -> Path:
    """Copy the sidecar beside the RAW, refusing to replace one that is already there (it may hold
    the photographer's own Lightroom edits)."""
    dest = Path(raw_path).with_suffix(".xmp")
    if dest.exists():
        raise FileExistsError(f"{dest} already exists; not replacing it. Copy {xmp_path} by hand if you want to.")
    shutil.copyfile(xmp_path, dest)
    return dest


def main() -> None:
    ap = argparse.ArgumentParser(description="Write a Lightroom XMP sidecar for a finished agent run")
    ap.add_argument("run_dir", type=Path, help="run folder that holds trace.json")
    ap.add_argument("--next-to-raw", action="store_true", help="also copy it beside the RAW (never replaces one)")
    a = ap.parse_args()
    result = json.loads((a.run_dir / "trace.json").read_text(encoding="utf-8"))["result"]
    src = Path(result["image"])
    state = {"applied_angle": result.get("angle_deg"), "crop_rect": result.get("crop_rect"),
             "valid_rect": result.get("valid_rect"), "color_params": result.get("color_params"),
             "retouch": result.get("retouch"), "beauty": result.get("beauty")}
    report = export_xmp(src, state, a.run_dir / f"{src.stem}.xmp")
    if a.next_to_raw:
        report["copied_to"] = str(place_next_to_raw(report["xmp_path"], src))
    print(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
