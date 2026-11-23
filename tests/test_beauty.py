import os
from pathlib import Path

import numpy as np
import pytest

from agent_tools import beauty as B
from agent_tools.registry import TOOLS

ROOT = Path(__file__).resolve().parent.parent
# a portrait with one clear face; set BEAUTY_TEST_PHOTO to your own photo to run the real-photo test
PHOTO = Path(os.getenv("BEAUTY_TEST_PHOTO", ROOT / "TEST-c" / "portrait.NEF"))
HAVE = PHOTO.exists() and (ROOT / "models" / "face_landmarker.task").exists()

FACE = {"eye_width_rel_right": 0.18, "eye_width_rel_left": 0.18, "eye_open_right": 0.3, "eye_open_left": 0.3,
        "nose_width_rel": 0.28, "jaw_to_cheek_ratio": 0.78, "face_aspect_h_w": 1.1, "mouth_width_rel": 0.36,
        "under_eye_darkness_L": 9.0, "shine_pct": 0.0, "teeth_visible": False,
        "facing": "camera", "face_height_frac": 0.15}


def test_natural_never_changes_shape():
    p, _ = B.plan_beauty(FACE, "natural")
    assert all(p.get(k, 0) == 0 for k in B.SHAPE_KEYS)
    assert p["smoothing"] > 0 and p["acne_spots"] > 0


def test_plan_fills_only_the_gap():
    p, why = B.plan_beauty(FACE, "xiezhen")
    assert 0 < p["eye_size"] <= B.LEVELS["xiezhen"]["eye_size"]
    big_eyes, _ = B.plan_beauty({**FACE, "eye_width_rel_right": 0.22, "eye_width_rel_left": 0.22}, "xiezhen")
    assert big_eyes["eye_size"] == 0                      # already above the target: no enlarging
    narrow, _ = B.plan_beauty({**FACE, "nose_width_rel": 0.22}, "xiezhen")
    assert narrow["nose_size"] == 0 and p["nose_size"] > 0
    assert p["teeth_whiten"] == 0 and "teeth_whiten" in why   # teeth not visible
    assert p["oil_shine"] == 0                                # no shine measured


def test_levels_are_ordered():
    n, _ = B.plan_beauty(FACE, "natural")
    x, _ = B.plan_beauty(FACE, "xiezhen")
    w, _ = B.plan_beauty(FACE, "wanghong")
    for k in ("eye_size", "nose_size", "face_slim", "smoothing"):
        assert n.get(k, 0) <= x[k] <= w[k], k


def test_turned_and_tiny_faces_are_protected():
    front, _ = B.plan_beauty(FACE, "wanghong")
    turned, why = B.plan_beauty({**FACE, "facing": "image left"}, "wanghong")
    assert turned["eye_size"] == pytest.approx(front["eye_size"] * 0.5, abs=0.1) and "pose" in why
    tiny, why = B.plan_beauty({**FACE, "face_height_frac": 0.04}, "wanghong")
    assert all(tiny.get(k, 0) == 0 for k in B.SHAPE_KEYS) and "size" in why


def test_line_bend_measures_bow_not_shift():
    h, w = 200, 200
    rgb = np.full((h, w, 3), 0.8, np.float32)
    rgb[:, 100:103] = 0.1                                   # one vertical straight line
    person = np.zeros((h, w), np.float32)
    dy = np.zeros((h, w), np.float32)
    shift = np.full((h, w), 2.0, np.float32)                # whole line moved sideways: not a bend
    assert B.line_bend(rgb, person, shift, dy)[0] < 0.01
    yy = np.arange(h, dtype=np.float32)[:, None]
    bump = np.repeat(3.0 * np.exp(-((yy - 100) / 20) ** 2), w, 1)   # middle of the line pushed by 3 px
    bend, n = B.line_bend(rgb, person, bump, dy)
    assert n >= 1 and bend > 2.5


def test_tool_registered_and_validates():
    t = next(t for t in TOOLS if t["name"] == "beauty_retouch")
    assert t["input_schema"]["properties"]["level"]["enum"] == list(B.LEVELS)
    with pytest.raises(ValueError):
        B.beauty_retouch("x.jpg", level="extreme")
    with pytest.raises(ValueError):
        B.beauty_retouch("x.jpg", avoid=["ears"])
    with pytest.raises(ValueError):
        B.beauty_retouch("x.jpg", params={"eye_sizee": 10})


@pytest.mark.skipif(not HAVE, reason="test photo / models not available")
def test_beauty_on_real_photo(tmp_path):
    r = B.beauty_retouch(str(PHOTO), level="xiezhen", avoid=["mouth"],
                         output_path=str(tmp_path / "b.tif"), sheet_path=str(tmp_path / "s.jpg"))
    assert r["status"] == "done", r["checks_failed"]
    s = r["faces"]["0"]["sliders"]
    assert "mouth_size" not in s and s["eye_size"] > 0
    assert Path(r["output_path"]).exists() and Path(r["sheet_path"]).exists()
    idn = r["identity"][0]
    assert idn["max_shift_frac"] <= 0.06 and 0.8 <= idn["eye_area_ratio"] <= 1.2
