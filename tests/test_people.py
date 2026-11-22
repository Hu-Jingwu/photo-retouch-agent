from pathlib import Path

import numpy as np
import pytest

from agent_tools import color as C
from agent_tools import people as P

ROOT = Path(__file__).resolve().parent.parent


# ---------- film LUTs ----------
def _identity_hald(level=4):
    n = level * level
    r, g, b = np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing="xy")
    # HaldCLUT order: red fastest, then green, then blue
    idx = np.arange(n ** 3)
    rr, gg, bb = idx % n, (idx // n) % n, idx // (n * n)
    return (np.stack([rr, gg, bb], -1) / (n - 1)).astype(np.float32).reshape(n, n, n, 3)


def test_apply_lut_identity_is_lossless():
    lut = _identity_hald(4)
    rng = np.random.default_rng(0)
    img = rng.random((20, 30, 3)).astype(np.float32)
    assert np.abs(C.apply_lut(img, lut) - img).max() < 1e-5


@pytest.mark.skipif(not (ROOT / "luts" / "fuji_velvia_50.png").exists(), reason="LUTs not downloaded")
def test_film_luts_registered_and_velvia_saturates():
    assert "fuji_velvia_50" in C.RECIPES and C.RECIPES["fuji_velvia_50"].get("lut")
    prof = C.profile_lut("fuji_velvia_50")
    assert prof["mean_saturation_x"] > 1.1
    acros = C.profile_lut("fuji_neopan_acros_100")
    assert "black and white" in acros["summary"]


# ---------- masks / halo ----------
def test_guided_filter_keeps_edges():
    I = np.zeros((100, 100), np.float32)
    I[:, 50:] = 1.0
    rough = I.copy()
    rough[:, 45:55] = 0.5                      # blurry, misaligned mask edge
    q = P.guided_filter(I, rough, 4, 1e-3)
    assert q[:, 40].mean() < 0.1 and q[:, 60].mean() > 0.9


def _fake_analysis(h=300, w=400):
    person = np.zeros((h, w), np.float32)
    person[60:280, 150:250] = 1
    face = np.zeros((h, w), np.float32)
    face[70:110, 180:220] = 1
    masks = {"person": person, "background": 1 - person, "face": face, "face_skin": face,
             "subject_light": P.subject_light_mask(person),
             "body_skin": np.zeros_like(face), "skin": face, "hair": np.zeros_like(face),
             "clothes": np.zeros_like(face), "features": np.zeros_like(face), "eyes": np.zeros_like(face)}
    return {"size": (h, w), "people": [], "masks": masks, "face_width": 40.0, "gray": person}


def test_local_person_exposure_has_no_halo():
    ana = _fake_analysis()
    before = np.full((300, 400, 3), 0.7, np.float32)
    before[60:280, 150:250] = 0.1             # dark person on bright background
    p = P.normalize_retouch({"local": [{"region": "person", "exposure": 1.5}]})
    after, info = P.render_retouch(before, ana, p)
    checks = {c["rule"]: c for c in P.retouch_checks(before, after, ana, info)}
    assert checks["no halo around people"]["status"] == "pass"
    assert after[150, 200].mean() > before[150, 200].mean() + 0.05   # the person did get brighter


def test_halo_check_catches_a_blurred_mask():
    ana = _fake_analysis()
    before = np.full((300, 400, 3), 0.7, np.float32)
    before[60:280, 150:250] = 0.1
    import cv2
    bad = dict(ana["masks"])
    bad["person"] = cv2.GaussianBlur(ana["masks"]["person"], (0, 0), 15)   # the old bug: soft spill
    after, info = P.render_retouch(before, {**ana, "masks": bad},
                                   P.normalize_retouch({"local": [{"region": "person", "exposure": 1.5}]}))
    checks = {c["rule"]: c for c in P.retouch_checks(before, after, ana, info)}
    assert checks["no halo around people"]["status"] == "fail"


def _dark_person_scene():
    before = np.full((300, 400, 3), 0.25, np.float32)
    before[60:280, 150:250] = 0.08            # dark person on a dim background (night scene)
    before[70:110, 180:220] = 0.08            # face
    return before


def test_subject_relight_is_not_a_halo_and_not_a_pasted_face():
    ana = _fake_analysis()
    before = _dark_person_scene()
    after, info = P.render_retouch(before, ana, P.normalize_retouch(
        {"local": [{"region": "subject_light", "exposure": 1.6}]}))
    checks = {c["rule"]: c for c in P.retouch_checks(before, after, ana, info)}
    assert checks["no halo around people"]["status"] == "pass", checks["no halo around people"]
    assert checks["face not lit separately"]["status"] == "pass"
    assert after[200, 200].mean() > before[200, 200].mean() + 0.05      # body brightened too
    assert after[150, 290].mean() > before[150, 290].mean() + 0.01      # and some nearby scene


def test_face_only_lift_is_flagged():
    ana = _fake_analysis()
    before = _dark_person_scene()
    after, info = P.render_retouch(before, ana, P.normalize_retouch(
        {"local": [{"region": "face", "exposure": 2.5}]}))
    checks = {c["rule"]: c for c in P.retouch_checks(before, after, ana, info)}
    assert checks["face not lit separately"]["status"] == "fail"


def test_unknown_retouch_keys_rejected():
    with pytest.raises(ValueError):
        P.normalize_retouch({"skin_smooth": 10})
    with pytest.raises(ValueError):
        P.normalize_retouch({"local": [{"region": "sky", "exposure": 1}]})
    assert P.normalize_retouch({}, "pixcake_natural")["skin_smoothing"] == 35


# ---------- real photo (skipped if the test photo or models are missing) ----------
PHOTO = ROOT / "TEST-c" / "DSC_5409.NEF"


@pytest.mark.skipif(not PHOTO.exists() or not (ROOT / "models" / "face_landmarker.task").exists(),
                    reason="test photo / models not available")
def test_real_portrait_retouch_passes_checks(tmp_path):
    det = P.detect_people(str(PHOTO), overlay_path=str(tmp_path / "people.jpg"))
    assert det["num_people"] == 1 and det["people"][0]["landmarks"] and det["people"][0]["body_visible"]
    r = P.preview_retouch(str(PHOTO), preset="pixcake_beauty", sheet_path=str(tmp_path / "r.jpg"))
    assert r["checks_failed"] == [], r["checks_failed"]
    assert r["info"]["face_max_shift_frac"] <= 0.04
