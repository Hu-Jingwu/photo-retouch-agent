from pathlib import Path

import numpy as np
import pytest

from agent_tools import face as F
from agent_tools import people as P
from agent_tools.registry import TOOLS

ROOT = Path(__file__).resolve().parent.parent
PHOTO = ROOT / "TEST-c" / "DSC_6390.NEF"
HAVE = PHOTO.exists() and (ROOT / "models" / "face_landmarker.task").exists()


def test_params_validation_and_pairs():
    p = F.normalize({"face_slim": [20, 30], "eye_size": -10, "nose_size": 15})
    assert p["face_slim"] == [20.0, 30.0] and p["eye_size"] == [-10.0, -10.0]
    assert F.normalize({}, "natural")["face_slim"] == [0.0, 0.0]          # natural never reshapes
    with pytest.raises(ValueError):
        F.normalize({"eye_sizee": 10})
    with pytest.raises(ValueError):
        F.normalize({}, "nope")


def test_radial_warp_is_local():
    dx = np.zeros((200, 200), np.float32)
    dy = np.zeros_like(dx)
    F._radial(dx, dy, np.array([100, 100], np.float32), 30, 0.2)
    assert np.abs(dx[:, :60]).max() == 0 and np.abs(dx[:, 140:]).max() == 0   # nothing outside the radius
    assert dx[100, 115] > 0 and dx[100, 85] < 0                              # bulge pushes outward


def test_face_subagent_removed_from_agent():
    names = {t["name"] for t in TOOLS}
    assert not names & {"retouch_faces", "inspect_faces", "preview_face_edit", "apply_face_edit"}


@pytest.mark.skipif(not HAVE, reason="test photo / models not available")
def test_face_edit_on_real_photo(tmp_path):
    insp = F.inspect_faces(str(PHOTO), sheet_path=str(tmp_path / "i.jpg"))
    assert insp["num_faces"] == 1 and "under_eye_darkness_L" in insp["faces"][0]
    pv = F.preview_face_edit(str(PHOTO), preset="pixcake_female_default", params={"dark_circles": 50},
                             sheet_path=str(tmp_path / "p.jpg"))
    assert pv["checks_failed"] == [], pv["checks_failed"]
    idn = pv["identity"][0]
    assert idn["max_shift_frac"] <= 0.06 and 1.0 < idn["eye_area_ratio"] <= 1.2


@pytest.mark.skipif(not HAVE, reason="test photo / models not available")
def test_extreme_settings_are_caught(tmp_path):
    rgb, ana = P._load(str(PHOTO))
    p = F.normalize({"eye_size": 100, "nose_size": 100, "face_slim": 100, "small_face": 100, "v_face": 100})
    # push beyond the slider range on purpose to prove the identity guard catches it
    p["eye_size"] = [300.0, 300.0]
    out, info = F.render_faces(rgb, ana, p)
    failed = {c["rule"] for c in F.face_checks(rgb, out, ana, info) if c["status"] == "fail"}
    assert any("eyes not over-enlarged" in r for r in failed)
