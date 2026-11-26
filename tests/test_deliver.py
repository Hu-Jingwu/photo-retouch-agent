import json
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import pytest

import agent
from agent_tools import deliver as D
from tests.scripted import FINAL, photo, scripted_agent

RDF = "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}"
CRS = "{http://ns.adobe.com/camera-raw-settings/1.0/}"


def _xmp_attrs(path):
    desc = ET.parse(path).getroot().find(f".//{RDF}Description")
    return {k.replace(CRS, ""): v for k, v in desc.attrib.items()}


def test_stage_words_in_both_languages():
    assert D.stage_name("调色") == D.stage_name("colour") == "color"
    assert D.stage_name("美颜") == "beauty" and D.stage_name("虚化") == "blur" and D.stage_name("拉平") == "rotation"
    assert D.stage_name("最终") == "final"
    with pytest.raises(ValueError):
        D.stage_name("滤镜大师")


def test_social_export_lands_in_the_size_band(tmp_path):
    rng = np.random.default_rng(0)
    big = cv2.GaussianBlur((rng.random((3000, 4000, 3)) * 255).astype(np.uint8), (0, 0), 1.2)
    src = tmp_path / "big.png"
    cv2.imwrite(str(src), big)
    res = D.export_image(src, "social", tmp_path / "out" / "big_social")
    assert max(res["size"]) == 2048 and 1.0 <= res["mb"] <= 2.5
    raw = open(res["path"], "rb").read()
    assert b"Exif" not in raw[:200] and b"GPS" not in raw              # no metadata leaves with it
    pr = D.export_image(src, "print", tmp_path / "out" / "big_print")
    assert pr["size"] == [4000, 3000] and pr["jpeg_quality"] == 95
    full = D.export_image(src, "full", tmp_path / "out" / "big_full")
    assert full["path"].endswith(".png") and open(full["path"], "rb").read() == open(src, "rb").read()


def test_small_flat_picture_reports_why_it_is_small(tmp_path):
    src = photo(tmp_path)
    res = D.export_image(src, "social", tmp_path / "s")
    assert res["mb"] < 1 and "note" in res


def _fake_run(tmp_path):
    """A finished run folder with every stage's file (copies of a synthetic photo)."""
    src = photo(tmp_path, "shot.png")
    run = tmp_path / "runs" / "shot_1"
    run.mkdir(parents=True)
    state = agent.new_state()
    files = {}
    for key in ("output_path", "final_path", "color_path", "retouch_path", "beauty_path", "blur_path"):
        files[key] = str(run / f"{key}.png")
        cv2.imwrite(files[key], cv2.imread(str(src)))
    state.update(files, applied_angle=1.5, valid_rect={"x": 5, "y": 5, "w": 350, "h": 230},
                 crop_rect={"x": 20, "y": 10, "w": 300, "h": 200},
                 color_params={"params": {"exposure": 0.4, "temperature": 10}, "recipe": None, "recipe_strength": 1.0},
                 retouch={"preset": None, "params": {"clothes_smooth": 50}},
                 beauty={"level": "xiezhen", "lip_shade": "rose"}, blur={"strength": 45})
    (run / "trace.json").write_text(json.dumps({"result": {"image": str(src)}, "state": state}), "utf-8")
    return src, run, state


def test_footprints_give_one_sidecar_per_stage(tmp_path):
    src, run, state = _fake_run(tmp_path)
    fp = D.write_footprints(src, state, run)
    assert "error" not in fp and list(fp["stages"]) == D.ORDER
    rot = _xmp_attrs(fp["stages"]["rotation"])
    col = _xmp_attrs(fp["stages"]["color"])
    assert rot["CropAngle"] == "-1.5" and "Exposure2012" not in rot               # rotation stage: no colour yet
    assert col["Exposure2012"] == "+0.40" and col["HasCrop"] == "True"
    assert any("background blur" in s for s in fp["final"]["not_exported"])
    assert any("clothing wrinkles" in s for s in fp["final"]["not_exported"])


def test_get_stage_and_list_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(D, "INDEX", tmp_path / "runs" / "index.jsonl")
    src, run, state = _fake_run(tmp_path)
    D.record_run({"run_dir": str(run), "image": str(src), "instruction": "写真精修"}, state)
    found = D.list_runs("shot")
    assert found["total"] == 1 and found["runs"][0]["stages"] == D.ORDER
    assert D.list_runs("nothing-like-this")["total"] == 0

    res = D.get_stage(str(run), "调色", "social")
    assert res["stage"] == "color" and res["path"].endswith("_3_color_social.jpg")
    lr = D.get_stage(str(run), "裁切", "lightroom")
    assert lr["xmp_path"].endswith("2_crop\\shot.xmp") or lr["xmp_path"].endswith("2_crop/shot.xmp")
    assert D.get_stage(str(run), "final", "social")["stage"] == "blur"
    # copying next to the original never replaces an existing sidecar
    (tmp_path / "shot.xmp").write_text("mine", encoding="utf-8")
    with pytest.raises(FileExistsError):
        D.get_stage(str(run), "调色", "lightroom", next_to_raw=True)
    assert (tmp_path / "shot.xmp").read_text(encoding="utf-8") == "mine"


def test_stage_not_in_run_is_refused(tmp_path):
    src, run, state = _fake_run(tmp_path)
    state["blur_path"], state["blur"] = None, None
    (run / "trace.json").write_text(json.dumps({"result": {"image": str(src)}, "state": state}), "utf-8")
    with pytest.raises(ValueError, match="not applied"):
        D.get_stage(str(run), "虚化")


def test_agent_run_delivers(tmp_path, monkeypatch):
    """A whole (scripted) agent run ends with the sidecar, per-stage footprints, an export and an index line."""
    monkeypatch.setattr(D, "INDEX", tmp_path / "index.jsonl")
    a = scripted_agent(monkeypatch, [[("apply_rotation", {"angle_deg": 1.0})], FINAL], export=("social",))
    res = a.run(str(photo(tmp_path)), run_dir=tmp_path / "run")
    assert (tmp_path / "run" / "photo.xmp").exists()
    assert list(res["lightroom"]["stages"]) == ["rotation"]
    assert res["exports"][0]["quality"] == "social" and (tmp_path / "run" / "exports").exists()
    line = json.loads((tmp_path / "index.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert line["stages"] == ["rotation"] and line["run_dir"].endswith("run")
    saved = json.loads((tmp_path / "run" / "trace.json").read_text(encoding="utf-8"))
    assert saved["result"]["lightroom"]["final"]["xmp_path"].endswith("photo.xmp")
