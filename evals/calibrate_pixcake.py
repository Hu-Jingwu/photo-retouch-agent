"""Calibrate face-warp strengths against a real PixCake export.

Measures, from face landmarks, how much PixCake's 写真 preset actually changed the face
(eye area, nose / mouth / jaw width, all relative to the distance between the pupils),
then adjusts agent_tools.face.GAIN until our pixcake_xiezhen_100 preset reproduces it.

    .venv\\Scripts\\python.exe -m evals.calibrate_pixcake TEST-c\\DSC_6390.NEF "TEST-c\\DSC_6390_导出\\DSC_6390.jpg"
"""
from __future__ import annotations

import json
import sys

import cv2
import numpy as np

from agent_tools import face as F
from agent_tools import people as P

# feature measured -> gains that drive it
FEATURES = {"eye_area": ["eye_size"], "nose_w": ["nose_size", "nose_wings"],
            "mouth_w": ["mouth_size"], "jaw_w": ["jaw", "face_slim"]}


def geom(lm) -> dict:
    ipd = float(np.linalg.norm(lm[468] - lm[473]))
    area = lambda idx: abs(cv2.contourArea(lm[idx].astype(np.float32)))
    d = lambda a, b: float(np.linalg.norm(lm[a] - lm[b])) / ipd
    return {"eye_area": (area(F.R_EYE) + area(F.L_EYE)) / 2 / ipd ** 2,
            "nose_w": d(F.NOSE_WING_R, F.NOSE_WING_L), "mouth_w": d(F.MOUTH_R, F.MOUTH_L),
            "jaw_w": d(172, 397)}


def ours(rgb, ana, preset="pixcake_xiezhen_100"):
    """Our shape change, measured EXACTLY: original landmarks moved by our own warp field
    (re-detecting landmarks on our output is noisier than the changes being calibrated)."""
    _, info = F.render_faces(rgb, ana, F.normalize({}, preset))
    lm = ana["people"][0]["landmarks"]
    h, w = info["_dx"].shape
    xi = np.clip(np.round(lm[:, 0]).astype(int), 0, w - 1)
    yi = np.clip(np.round(lm[:, 1]).astype(int), 0, h - 1)
    moved = lm + np.stack([info["_dx"][yi, xi], info["_dy"][yi, xi]], 1)
    return geom(moved)


def main(raw_path, pixcake_path, iters=5):
    rgb, ana = P._load(raw_path, 2048)
    _, pana = P._load(pixcake_path, 2048)
    g0 = geom(ana["people"][0]["landmarks"])
    target = {k: v / g0[k] for k, v in geom(pana["people"][0]["landmarks"]).items()}
    print("PixCake target ratios:", {k: round(v, 4) for k, v in target.items()})
    for it in range(iters):
        g = ours(rgb, ana)
        got = {k: g[k] / g0[k] for k in g}
        print(f"iter {it}: ours", {k: round(v, 4) for k, v in got.items()},
              "gains", {k: round(v, 4) for k, v in F.GAIN.items()})
        done = True
        for feat, gains in FEATURES.items():
            want, have = target[feat] - 1, got[feat] - 1
            if abs(want - have) < 0.003:
                continue
            done = False
            factor = float(np.clip(want / have, 0.25, 4.0)) if abs(have) > 1e-4 else 2.0
            for gname in gains:
                F.GAIN[gname] *= factor
        if done:
            break
    g = ours(rgb, ana)
    final = {k: round(g[k] / g0[k], 4) for k in g}
    g70 = ours(rgb, ana, "pixcake_xiezhen_70")
    at70 = {k: round(g70[k] / g0[k], 4) for k in g70}
    result = {"target": {k: round(v, 4) for k, v in target.items()}, "ours_100": final, "ours_70": at70,
              "gains": {k: round(v, 4) for k, v in F.GAIN.items()}}
    print(json.dumps(result, indent=1))
    return result


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
