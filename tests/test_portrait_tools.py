import cv2
import numpy as np
import pytest

import agent as A
from agent_tools import beauty as B
from agent_tools import depth as D
from agent_tools import face as F
from agent_tools import people as P
from agent_tools.registry import TOOLS


# ---- lipstick / contour ------------------------------------------------------------------
def test_lip_shade_validated_and_defaulted():
    assert F.normalize({"lipstick": 40})["lip_shade"] == "natural"
    assert F.normalize({"lip_shade": "red"})["lip_shade"] == "red"
    with pytest.raises(ValueError):
        F.normalize({"lip_shade": "purple"})


def test_lipstick_and_contour_fill_only_the_gap():
    face = {"eye_width_rel_right": 0.2, "eye_width_rel_left": 0.2, "eye_open_right": 0.3, "eye_open_left": 0.3,
            "nose_width_rel": 0.24, "jaw_to_cheek_ratio": 0.7, "face_aspect_h_w": 1.2, "mouth_width_rel": 0.33,
            "facing": "camera", "face_height_frac": 0.15}
    pale, _ = B.plan_beauty({**face, "lip_chroma": 20.0, "face_relief": 3.0}, "xiezhen")
    rich, _ = B.plan_beauty({**face, "lip_chroma": 40.0, "face_relief": 12.0}, "xiezhen")
    assert pale["lipstick"] > 0 and pale["contour"] > 0
    assert rich["lipstick"] == 0 and rich["contour"] == 0          # lips already rich, light already shapes the face
    nat, _ = B.plan_beauty({**face, "lip_chroma": 20.0, "face_relief": 3.0}, "natural")
    assert nat["lipstick"] == 0 and nat["contour"] == 0


def test_beauty_tool_offers_lip_shade_and_makeup_groups():
    t = next(t for t in TOOLS if t["name"] == "beauty_retouch")
    assert t["input_schema"]["properties"]["lip_shade"]["enum"] == list(F.LIP_SHADES)
    assert {"makeup", "contour"} <= set(B.GROUPS)
    with pytest.raises(ValueError):
        B.beauty_retouch("x.jpg", lip_shade="purple")


# ---- depth and blur ----------------------------------------------------------------------
def _scene(bg_blur: float):
    rng = np.random.default_rng(1)
    h, w = 400, 600
    tex = rng.random((h, w)).astype(np.float32)
    tex = cv2.resize(cv2.resize(tex, (60, 40)), (w, h), interpolation=cv2.INTER_NEAREST)   # blocky detail
    bg = np.repeat(tex[..., None], 3, 2) * 0.8 + 0.1
    if bg_blur:
        bg = cv2.GaussianBlur(bg, (0, 0), bg_blur)
    person = np.zeros((h, w), np.float32)
    cv2.ellipse(person, (300, 220), (80, 170), 0, 0, 360, 1.0, -1)
    subj = np.repeat(cv2.resize(rng.random((34, 16)).astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)[..., None], 3, 2)
    rgb = person[..., None] * (subj * 0.6 + 0.2) + (1 - person[..., None]) * bg
    return rgb.astype(np.float32), person


def test_defocus_tells_soft_background_from_sharp():
    sharp_rgb, person = _scene(0)
    soft_rgb, _ = _scene(4.0)
    ana = {"people": [{"face": {"box": [0, 0, 50, 60]}}]}
    j_sharp = D._judge(D.defocus_map(sharp_rgb, person), person, ana)
    j_soft = D._judge(D.defocus_map(soft_rgb, person), person, ana)
    assert j_soft["background_extra_defocus_px"] > j_sharp["background_extra_defocus_px"] + 0.5
    assert j_sharp["verdict"] == "sharp_background" and j_soft["verdict"] == "already_shallow"


def test_blur_keeps_subject_and_has_no_halo():
    rgb, person = _scene(0)
    z = np.ones(person.shape, np.float32)
    out, info = D.render_blur(rgb, person, z, 80, base=0.35)
    checks = {c["rule"]: c for c in D.blur_checks(rgb, out, person)}
    assert info["max_radius_px"] > 5
    assert checks["subject untouched"]["status"] == "pass"
    assert checks["no halo around the subject"]["status"] == "pass"
    assert checks["background softer"]["status"] == "pass"
    same, _ = D.render_blur(rgb, person, z, 0, base=0.35)
    assert np.array_equal(same, rgb)


def test_blur_is_the_last_stage():
    assert A.STAGES[-1] == ("blur", "apply_background_blur")
    s = A.new_state()
    s.update(beauty_path="b.tif", beauty={"level": "xiezhen"}, blur_path="bl.tif", blur={"strength": 55})
    assert A.final_path(s) == "bl.tif"
    A.invalidate_after(s, "beauty")
    assert s["stale"]["blur"] == {"strength": 55} and s["blur_path"] is None


# ---- clothing wrinkles --------------------------------------------------------------------
def test_clothes_smooth_flattens_creases_but_keeps_prints():
    h, w = 300, 300
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    crease = 0.08 * np.sin(xx / 9.0)                       # soft shading bands: wrinkles
    plain = np.repeat((0.5 + crease)[..., None], 3, 2)
    clothes = np.ones((h, w), np.float32)
    out, amount = P._clothes_smooth(plain, clothes, fw=150, amount=80)
    assert amount > 0
    assert out[..., 0].std() < 0.6 * plain[..., 0].std()   # creases mostly gone
    printed = plain.copy()
    printed[::30, :] = [0.9, 0.1, 0.1]                     # sharp red stripes: a print
    printed[1::30, :] = [0.9, 0.1, 0.1]
    out2, _ = P._clothes_smooth(printed, clothes, fw=150, amount=80)
    stripe = np.zeros((h, w), bool)
    stripe[::30] = stripe[1::30] = True
    assert np.abs(out2[stripe] - printed[stripe]).mean() < 0.03   # the print is protected


def test_named_shade_always_shows(monkeypatch):
    """A colour the user names must be visible even when the lips are already rich."""
    calls = {}

    def fake_render(rgb, ana, plans, lip_shade="natural"):
        calls["plans"], calls["shade"] = plans, lip_shade
        raise RuntimeError("stop")

    monkeypatch.setattr(B, "_render_all", fake_render)
    monkeypatch.setattr(B.P, "_load", lambda path: (np.zeros((10, 10, 3), np.float32), {
        "people": [{"landmarks": np.zeros((478, 2), np.float32), "face": {"facing": 0, "box": [0, 0, 5, 5]}}],
        "masks": {"face_skin": np.zeros((10, 10), np.float32)}}))
    monkeypatch.setattr(B.F, "measure_face", lambda *a: {
        "eye_width_rel_right": 0.2, "eye_width_rel_left": 0.2, "eye_open_right": 0.3, "eye_open_left": 0.3,
        "nose_width_rel": 0.24, "jaw_to_cheek_ratio": 0.7, "face_aspect_h_w": 1.2, "mouth_width_rel": 0.33,
        "lip_chroma": 40.0, "face_relief": 12.0})
    with pytest.raises(RuntimeError):
        B.beauty_retouch("x.jpg", level="xiezhen", lip_shade="rose")
    assert calls["plans"][0]["lipstick"] == B.LIP_FLOOR["xiezhen"] and calls["shade"] == "rose"
    with pytest.raises(RuntimeError):
        B.beauty_retouch("x.jpg", level="xiezhen")              # no shade named: the plan decides (0 here)
    assert calls["plans"][0]["lipstick"] == 0
