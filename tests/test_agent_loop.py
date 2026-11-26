import json

import agent as A


def _img():
    return {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AAAA"}}


def test_prune_keeps_only_newest_images():
    msgs = [{"role": "system", "content": "x"},
            {"role": "user", "content": [{"type": "text", "text": "photo"}, _img()]}]   # the photo itself
    marks = []
    for i in range(3):
        msgs.append({"role": "tool", "content": "{}"})
        msgs.append({"role": "user", "content": [{"type": "text", "text": "Images"}, _img()]})
        marks.append((len(msgs) - 1, f"tool{i} (f{i}.jpg)"))
    A.prune_images(msgs, marks, keep=1)
    assert isinstance(msgs[1]["content"], list)                 # original photo untouched
    assert "tool0 (f0.jpg)" in msgs[marks[0][0]]["content"] and "removed" in msgs[marks[1][0]]["content"]
    assert isinstance(msgs[marks[2][0]]["content"], list)       # newest kept
    A.prune_images(msgs, marks, keep=1)                         # idempotent
    msgs2 = [{"role": "user", "content": [_img()]}]
    A.prune_images(msgs2, [(0, "x")], keep=0)                   # 0 = keep everything
    assert isinstance(msgs2[0]["content"], list)


def _edited_state():
    s = A.new_state()
    s.update(output_path="lev.tif", applied_angle=1.3, final_path="crop.tif", crop_rect={"x": 0, "y": 0, "w": 9, "h": 9},
             color_path="color.tif", color_params={"params": {"tint": 10}, "recipe": None, "recipe_strength": 1.0},
             retouch_path="ret.tif", retouch={"preset": None, "params": {"local": []}},
             beauty_path="b.tif", beauty={"level": "xiezhen", "avoid": ["eyes"], "params": None})
    return s


def test_redoing_colour_makes_later_stages_stale():
    s = _edited_state()
    A.invalidate_after(s, "color")
    assert s["stale"]["retouch"] == {"params": {"local": []}}
    assert s["stale"]["beauty"] == {"level": "xiezhen", "avoid": ["eyes"]}
    assert s["retouch_path"] is None and s["beauty_path"] is None
    assert s["color_path"] == "color.tif" and "color" not in s["stale"]
    assert A.final_path(s) == "color.tif"


def test_redoing_a_stale_stage_unmarks_it():
    s = _edited_state()
    A.invalidate_after(s, "color")
    s["retouch_path"] = "ret2.tif"
    A.invalidate_after(s, "retouch")
    assert "retouch" not in s["stale"] and "beauty" in s["stale"]


def test_replay_reapplies_stale_stages_in_order(monkeypatch):
    s = _edited_state()
    A.invalidate_after(s, "crop")
    calls = []

    def fake(name, args, src, run_dir, state):
        calls.append((name, args))
        if name == "beauty_retouch":
            return json.dumps({"error": "boom"}), True
        return "{}", False

    monkeypatch.setattr(A.LevelingAgent, "_call_tool", staticmethod(fake))
    done = A.replay_stale(None, None, s)
    assert [c[0] for c in calls] == ["apply_color", "apply_retouch", "beauty_retouch"]
    assert calls[0][1] == {"params": {"tint": 10}, "recipe_strength": 1.0}
    assert done[-1]["ok"] is False and done[-1]["error"] == "boom"
    assert s["stale"] == {}


def test_restore_state_new_and_old_traces(tmp_path):
    st = _edited_state()
    (tmp_path / "trace.json").write_text(json.dumps({"result": {"image": "C:/p/x.jpg"}, "state": st}), "utf-8")
    src, s, r = A.restore_state(tmp_path)
    assert src.name == "x.jpg" and s["beauty_path"] == "b.tif" and s["stale"] == {}

    old = tmp_path / "old"
    old.mkdir()
    (old / "x_color.tif").write_bytes(b"")
    (old / "x_beauty.tif").write_bytes(b"")
    (old / "trace.json").write_text(json.dumps({"result": {
        "image": "C:/p/x.jpg", "angle_deg": 1.0, "rotations": 1, "output_path": "lev.tif",
        "cropped_path": "crop.tif", "crop_rect": {"x": 0}, "color_params": {"params": {}},
        "retouch": None, "beauty": {"level": "natural"}}}), "utf-8")
    src, s, r = A.restore_state(old)
    assert s["color_path"].endswith("x_color.tif") and s["beauty_path"].endswith("x_beauty.tif")
    assert s["retouch_path"] is None and s["final_path"] == "crop.tif"


def test_revision_folders_stay_flat(tmp_path):
    first = tmp_path / "photo_20261126"
    assert A.next_rev_dir(first).name == "photo_20261126_rev1"
    (tmp_path / "photo_20261126_rev1").mkdir()
    assert A.next_rev_dir(tmp_path / "photo_20261126_rev1").name == "photo_20261126_rev2"
