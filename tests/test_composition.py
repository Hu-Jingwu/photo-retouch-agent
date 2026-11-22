import cv2
import numpy as np
import pytest

from agent_tools import apply_crop, apply_rotation, dispatch, propose_crops
from agent_tools.composition import _iou
from agent_tools.io_utils import read_image, write_image


def _scene(tmp_path, name="scene.png", obj_xy=(1100, 380)):
    """Plain textured background with one strong object off-centre."""
    rng = np.random.default_rng(0)
    img = np.clip(rng.normal(120, 4, (900, 1500, 3)), 0, 255).astype(np.uint8)
    cv2.circle(img, obj_xy, 70, (30, 60, 230), -1)
    cv2.rectangle(img, (obj_xy[0] - 20, obj_xy[1] + 70), (obj_xy[0] + 20, obj_xy[1] + 250), (20, 20, 20), -1)
    p = tmp_path / name
    write_image(p, img)
    return str(p)


def test_candidates_stay_inside_valid_rect_and_keep_object(tmp_path):
    src = _scene(tmp_path)
    valid = {"x": 60, "y": 40, "w": 1380, "h": 820}
    res = propose_crops(src, valid_rect=valid, contact_sheet_path=str(tmp_path / "sheet.jpg"))
    assert (tmp_path / "sheet.jpg").is_file()
    assert 1 <= len(res["candidates"]) <= 8
    assert res["focal_point"]["source"] == "centre of visual interest"
    for c in res["candidates"]:
        r = c["rect"]
        assert r["x"] >= valid["x"] and r["y"] >= valid["y"]
        assert r["x"] + r["w"] <= valid["x"] + valid["w"]
        assert r["y"] + r["h"] <= valid["y"] + valid["h"]
        assert r["x"] < 1100 < r["x"] + r["w"] and r["y"] < 380 < r["y"] + r["h"]
    widest = max(c["features"]["area_kept"] for c in res["candidates"])
    assert widest > 0.9                       # a "keep the scene" option is always offered


def test_candidates_are_diverse(tmp_path):
    res = propose_crops(_scene(tmp_path), contact_sheet_path=str(tmp_path / "s.jpg"))
    rects = [tuple(c["rect"].values()) for c in res["candidates"]]
    for i in range(len(rects)):
        for j in range(i + 1, len(rects)):
            assert _iou(rects[i], rects[j]) < 0.9


def test_apply_crop_on_rotated_output(tmp_path):
    src = _scene(tmp_path)
    rot = apply_rotation(src, 3.0, output_path=str(tmp_path / "lev.png"))
    vr = rot["valid_rect"]
    res = propose_crops(rot["output_path"], valid_rect=vr, contact_sheet_path=str(tmp_path / "s.jpg"))
    out = apply_crop(rot["output_path"], res["candidates"][0]["rect"],
                     output_path=str(tmp_path / "final.png"), valid_rect=vr)
    img = read_image(out["output_path"])
    assert img.shape[1] == out["output_size"]["w"] and img.shape[0] == out["output_size"]["h"]
    raw = cv2.imdecode(np.fromfile(out["output_path"], np.uint8), cv2.IMREAD_UNCHANGED)
    assert raw.shape[2] == 3                  # alpha dropped: nothing transparent left
    assert (tmp_path / "final_preview.jpg").is_file()


def test_apply_crop_refuses_blank_corners(tmp_path):
    src = _scene(tmp_path)
    rot = apply_rotation(src, 5.0, output_path=str(tmp_path / "lev.png"))
    with pytest.raises(ValueError, match="valid_rect"):
        apply_crop(rot["output_path"], {"x": 0, "y": 0, "w": 400, "h": 300},
                   output_path=str(tmp_path / "bad.png"), valid_rect=rot["valid_rect"])


def _landscape(tmp_path, cloudy_sky: bool):
    """Horizon exactly in the middle; either the sky or the land carries the texture."""
    rng = np.random.default_rng(3)
    h, w = 900, 1500
    img = np.zeros((h, w, 3), np.uint8)
    img[: h // 2] = (225, 190, 140)
    img[h // 2:] = (60, 110, 70)
    tex = rng.normal(0, 40, (h // 2, w, 3))
    tex = cv2.GaussianBlur(tex, (0, 0), 3) * 3
    if cloudy_sky:
        img[: h // 2] = np.clip(img[: h // 2] + tex, 0, 255)
    else:
        img[h // 2:] = np.clip(img[h // 2:] + tex, 0, 255)
    p = tmp_path / f"land_{cloudy_sky}.png"
    write_image(p, img)
    return str(p)


@pytest.mark.parametrize("cloudy, target", [(False, 1 / 3), (True, 2 / 3)])
def test_landscape_puts_horizon_on_the_right_third(tmp_path, cloudy, target):
    res = propose_crops(_landscape(tmp_path, cloudy), contact_sheet_path=str(tmp_path / "s.jpg"))
    assert res["scene"]["type"] == "landscape"
    assert res["scene"]["sky_interesting"] is cloudy
    top = res["candidates"][0]
    v = (res["scene"]["horizon_y"] - top["rect"]["y"]) / top["rect"]["h"]
    assert v == pytest.approx(target, abs=0.08), top


def _person_ctx(facing=0):
    # face box 100 px tall at the top of an 1100 px frame; crown ~ y=70, head unit 130
    f = {"box": [450, 100, 90, 100], "eye_y": 140, "center_x": 495, "score": 0.9, "facing": facing,
         "head": [427.5, 55, 562.5, 210], "hu": 130.0, "crown": 70.0}
    flat = lambda x0, y0, x1, y1: max(0.0, (x1 - x0) * (y1 - y0))
    return {"faces": [f], "must_keep": [f], "minor": [], "land": None, "mass": flat,
            "valid_mass": 1500 * 1100, "valid_density": 1.0, "valid_area": 1500 * 1100,
            "focal": (495, 140)}


def test_joint_rule_penalises_cutting_at_the_knees():
    from agent_tools.composition import _score
    ctx = _person_ctx()
    knees_y = 70 + 6.0 * 130          # bottom edge exactly on the knees
    mid_thigh = 70 + 5.0 * 130
    f_knee, notes = _score((300, 20, 400, knees_y - 20), "people", ctx)
    f_thigh, _ = _score((300, 20, 400, mid_thigh - 20), "people", ctx)
    assert f_knee["joints"] < 0.2 and any("knees" in n for n in notes)
    assert f_thigh["joints"] == 1.0


def test_lead_room_prefers_space_in_front_of_the_face():
    from agent_tools.composition import _score
    ctx = _person_ctx(facing=1)       # looking to image right
    space_ahead, _ = _score((350, 20, 600, 700), "people", ctx)    # face on the left of the crop
    space_behind, notes = _score((0, 20, 600, 700), "people", ctx)  # face on the right of the crop
    assert space_ahead["lead_room"] > 0.9
    assert space_behind["lead_room"] < 0.3 and any("lead room" in n for n in notes)


def test_check_composition_landscape(tmp_path):
    from agent_tools.composition import check_composition
    p = _landscape(tmp_path, cloudy_sky=False)          # 1500x900, horizon at y=450, plain sky
    good = {"x": 0, "y": 300, "w": 1000, "h": 450}       # horizon at 1/3 -> land gets 2/3
    middle = {"x": 0, "y": 225, "w": 1000, "h": 450}     # horizon dead centre
    g = check_composition(p, good)
    b = check_composition(p, middle)
    assert g["scene"] == "landscape"
    assert "horizon not through the middle" not in g["failed_rules"]
    assert "sky:land ratio favours the interesting part" not in g["failed_rules"]
    assert "horizon not through the middle" in b["failed_rules"]
    assert g["score"] > b["score"]


def test_check_composition_hard_rule_valid_area(tmp_path):
    from agent_tools.composition import check_composition
    p = _scene(tmp_path)
    res = check_composition(p, {"x": 0, "y": 0, "w": 800, "h": 600},
                            valid_rect={"x": 50, "y": 50, "w": 1400, "h": 800})
    assert res["status"] == "fail" and "inside valid area" in res["hard_violations"]


def test_dispatch_lists_crop_tools(tmp_path):
    text, is_err = dispatch("propose_crops", {"image_path": _scene(tmp_path),
                                              "contact_sheet_path": str(tmp_path / "s.jpg")})
    assert not is_err and "candidates" in text


def test_prominent_object_is_kept_whole_or_left_out(tmp_path):
    from agent_tools.composition import check_composition, detect_objects
    img = np.full((600, 900, 3), 40, np.uint8)
    cv2.rectangle(img, (100, 300), (260, 580), (0, 220, 255), -1)    # bright yellow "lamp"
    p = tmp_path / "lamp.png"
    cv2.imwrite(str(p), img)
    objs = detect_objects(img, [])
    assert len(objs) == 1 and objs[0]["area"] > 0.05
    sliced = check_composition(str(p), {"x": 0, "y": 0, "w": 900, "h": 440})
    assert "prominent object not cut in half" in sliced["hard_violations"]
    whole = check_composition(str(p), {"x": 0, "y": 100, "w": 900, "h": 500})
    assert "prominent object not cut in half" not in whole["failed_rules"]
