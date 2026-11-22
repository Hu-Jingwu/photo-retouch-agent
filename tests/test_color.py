import numpy as np
import pytest

from agent_tools import color as C
from agent_tools.io_utils import read_image, write_image


def _patches(tmp_path, name="patches.png", tint=(0, 0, 0)):
    """Six colour patches + a grey ramp, 16-bit."""
    h, w = 300, 600
    img = np.zeros((h, w, 3), np.float32)
    cols = [(0.8, 0.2, 0.2), (0.9, 0.6, 0.3), (0.2, 0.7, 0.3), (0.2, 0.4, 0.85), (0.5, 0.5, 0.5), (0.9, 0.9, 0.9)]
    for i, c in enumerate(cols):
        img[:150, i * 100:(i + 1) * 100] = c
    img[150:] = np.linspace(0.02, 0.98, w)[None, :, None]
    img = np.clip(img + np.array(tint, np.float32), 0, 1)
    p = tmp_path / name
    write_image(p, (img[..., ::-1] * 65535).astype(np.uint16))
    return str(p), img


def _render(img, params, recipe=None, strength=1.0):
    out, _ = C._render_small(img, params, recipe, strength)
    return out


def test_identity_is_lossless(tmp_path):
    _, img = _patches(tmp_path)
    out = _render(img, {})
    assert np.abs(out - img).max() < 2 / 255


def test_exposure_and_temperature_directions(tmp_path):
    _, img = _patches(tmp_path)
    assert _render(img, {"exposure": 1}).mean() > img.mean() + 0.05
    warm = _render(img, {"temperature": 40})
    grey = img[:150, 400:500].mean((0, 1))
    g2 = warm[:150, 400:500].mean((0, 1))
    assert g2[0] / g2[2] > grey[0] / grey[2] * 1.1      # R/B up = warmer


def test_saturation_minus_100_is_greyscale(tmp_path):
    _, img = _patches(tmp_path)
    out = _render(img, {"saturation": -100})
    assert np.abs(out[..., 0] - out[..., 1]).max() < 1e-3
    assert np.abs(out[..., 1] - out[..., 2]).max() < 1e-3


def test_hsl_targets_only_its_colour(tmp_path):
    _, img = _patches(tmp_path)
    out = _render(img, {"hsl": {"blue": {"saturation": -100}}})
    blue_in, blue_out = img[:150, 300:400].mean((0, 1)), out[:150, 300:400].mean((0, 1))
    red_in, red_out = img[:150, 0:100].mean((0, 1)), out[:150, 0:100].mean((0, 1))
    assert np.ptp(blue_out) < 0.25 * np.ptp(blue_in)      # blue patch desaturated
    assert np.abs(red_out - red_in).max() < 0.02           # red untouched


def test_recipe_strength_zero_is_identity_and_bw_is_always_mono(tmp_path):
    _, img = _patches(tmp_path)
    assert np.abs(_render(img, {}, "vivid_positive", 0.0) - img).max() < 2 / 255
    bw = _render(img, {}, "bw_high_contrast", 0.4)
    assert np.abs(bw[..., 0] - bw[..., 2]).max() < 1e-3


def test_no_halo_at_high_contrast_edge():
    """Dark subject against bright sky: strong highlights/shadows must not brighten the sky next to the edge."""
    h, w = 400, 800
    img = np.full((h, w, 3), 0.85, np.float32)            # bright sky
    img[:, :300] = 0.06                                     # dark subject on the left
    out = _render(img, {"highlights": -95, "shadows": 70})
    near = out[:, 302:318].mean()                           # sky right next to the subject
    far = out[:, 650:800].mean()                            # sky far away
    assert abs(near - far) < 0.02, (near, far)
    # and the corrections themselves still happen
    assert out[:, 50:250].mean() > img[:, 50:250].mean() + 0.03   # shadows lifted
    assert far < 0.85                                              # highlights pulled down


def test_match_reference_recovers_a_known_look(tmp_path):
    """A sample made from the same photo with known sliders: matching must reproduce its look."""
    _, img = _patches(tmp_path)
    ref = _render(img, {"temperature": -25, "exposure": 0.4, "saturation": -25})
    params, rep = C.fit_reference(img, ref)
    qr, qs = rep["lightness_quantiles_ref"], rep["lightness_quantiles_result"]
    assert abs(qr[50] - qs[50]) < 3
    assert abs(rep["midtone_ab_ref"][1] - rep["midtone_ab_result"][1]) < 2      # blue-yellow balance
    assert params["temperature"] < 0 and params["saturation"] < 0


def test_match_reference_respects_limits(tmp_path):
    _, img = _patches(tmp_path)
    night = np.clip(img * 0.15, 0, 1)                     # very dark photo vs a bright sample
    params, _ = C.fit_reference(night, img)
    assert abs(params.get("exposure", 0)) <= C.MATCH_LIMITS["exposure"] + 1e-6
    assert C.scale_params(params, 0.5)["exposure"] == pytest.approx(params["exposure"] * 0.5, abs=1e-3)


def test_unknown_param_is_rejected():
    with pytest.raises(ValueError, match="unknown parameter"):
        C.normalize_params({"exposre": 1})
    with pytest.raises(ValueError, match="unknown recipe"):
        C._merge({}, "nope", 1)


def test_analyze_suggests_cooling_a_warm_cast(tmp_path):
    p, _ = _patches(tmp_path, "warm.png", tint=(0.06, 0.0, -0.06))
    a = C.analyze_color(p)
    assert a["neutral_cast"]["b_blue_yellow"] > 3
    assert a["neutral_cast"]["suggested_temperature"] < 0


def test_check_color_flags_clipping(tmp_path):
    p, img = _patches(tmp_path)
    bright = np.clip(img * 2.5, 0, 1)
    q = tmp_path / "bright.png"
    write_image(q, (bright[..., ::-1] * 65535).astype(np.uint16))
    res = C.check_color(str(q))
    assert "highlights not clipped" in res["failed_rules"]
    assert C.check_color(p)["status"] == "pass"


def test_tools_write_files(tmp_path):
    p, _ = _patches(tmp_path)
    v = C.preview_variants(p, "exposure", [-0.5, 0, 0.5], sheet_path=str(tmp_path / "v.jpg"))
    assert (tmp_path / "v.jpg").is_file() and len(v["tiles"]) == 3
    lk = C.preview_looks(p, sheet_path=str(tmp_path / "l.jpg"))
    assert len(lk["tiles"]) == len(C.RECIPES) + 1
    r = C.render_adjustments(p, {"exposure": 0.3}, preview_path=str(tmp_path / "r.jpg"))
    assert r["effective_params"] == {"exposure": 0.3}
    out = C.apply_color(p, {"contrast": 20}, "warm_portrait", 0.5, output_path=str(tmp_path / "o.tif"))
    img = read_image(out["output_path"])
    assert img.dtype == np.uint16 and img.shape[:2] == (300, 600)
    assert (tmp_path / "o.params.json").is_file()


def test_face_shift_flags_a_cold_face_and_solves_the_correction():
    from pathlib import Path
    from agent_tools import color as C
    import pytest
    p = Path(__file__).resolve().parent.parent / "TEST-c" / "DSC_5409.NEF"
    if not p.exists():
        pytest.skip("photo missing")
    before = C._load_preview(str(p), 1024)
    cold, _ = C._render_small(before, {"temperature": -60})
    fs = C.face_shift(before, cold)
    assert fs and not fs["ok"] and fs["temperature"] < -20
    corr = fs["skin_correction"]
    assert corr["region"] == "skin" and corr["temperature"] > 20       # warm the skin back
    mild, _ = C._render_small(before, {"temperature": -10})
    assert C.face_shift(before, mild)["ok"]
