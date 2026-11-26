"""Lightroom XMP export: white balance conversion, geometry and the XMP itself (the run hook is in test_deliver)."""
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from agent_tools import color, xmp
from tests.scripted import photo as _photo

CRS = "{%s}" % xmp.CRS


def test_known_illuminants_give_their_temperature():
    k, tint = xmp.kelvin_tint(xmp.D65)
    assert k == pytest.approx(6504, abs=15) and tint == pytest.approx(9.6, abs=1)   # daylight sits on the green side
    assert xmp.kelvin_tint(np.array([1.09850, 1.0, 0.35585]))[0] == pytest.approx(2856, abs=15)   # illuminant A
    for kelvin, tint in ((3200, 10), (5500, 0), (7500, -20)):
        back = xmp.kelvin_tint(xmp.white_xyz(kelvin, tint))
        assert back[0] == pytest.approx(kelvin, rel=0.002) and back[1] == pytest.approx(tint, abs=0.2)


@pytest.mark.parametrize("temperature, tint", [(0, 0), (20, 0), (-35, 0), (0, 25), (40, -15)])
def test_white_balance_shift_moves_grey_like_our_sliders(temperature, tint):
    as_shot = (5200.0, 5.0)
    k, t = xmp.solve_white_balance(as_shot, temperature, tint)
    ab = xmp._grey_ab(xmp.white_xyz(*as_shot), xmp.white_xyz(k, t))
    assert ab == pytest.approx([tint * color._DA_PER_TINT, temperature * color._DB_PER_TEMP], abs=0.01)
    # The two axes are not quite aligned, so a pure shift on one needs a little of the other.
    if temperature:
        assert np.sign(k - as_shot[0]) == np.sign(temperature)   # warmer slider -> higher Kelvin, as in Lightroom
    if tint and not temperature:
        assert np.sign(t - as_shot[1]) == np.sign(tint)          # more magenta -> higher tint, as in Lightroom
    if not (temperature or tint):
        assert (k, t) == pytest.approx(as_shot)


def test_geometry():
    assert xmp._geometry((400, 300), 0.0, None, 0) == {"HasCrop": False}
    g = xmp._geometry((400, 300), 0.0, {"x": 40, "y": 30, "w": 200, "h": 150}, 0)
    assert (g["CropLeft"], g["CropTop"], g["CropRight"], g["CropBottom"]) == pytest.approx((0.1, 0.1, 0.6, 0.6))
    g = xmp._geometry((400, 300), 2.0, {"x": 50, "y": 40, "w": 300, "h": 220}, 0)
    assert g["CropAngle"] == -2.0                                       # ours counter-clockwise, theirs clockwise
    assert (g["CropLeft"] + g["CropRight"]) / 2 == pytest.approx(0.5) and (g["CropTop"] + g["CropBottom"]) / 2 == pytest.approx(0.5)
    # A crop at the top left of an upright photo stored rotated clockwise (flip 6) is at the
    # bottom left of the stored picture.
    g = xmp._geometry((300, 400), 0.0, {"x": 0, "y": 0, "w": 150, "h": 100}, 6)
    assert (g["CropLeft"], g["CropTop"], g["CropRight"], g["CropBottom"]) == pytest.approx((0.0, 0.5, 0.25, 1.0))


def _parse(path):
    desc = ET.parse(path).getroot().find(".//{http://www.w3.org/1999/02/22-rdf-syntax-ns#}Description")
    return {k.replace(CRS, ""): v for k, v in desc.attrib.items()}, desc


def test_xmp_for_a_colour_edit_is_valid_and_complete(tmp_path):
    state = {"applied_angle": 0.0, "crop_rect": {"x": 10, "y": 20, "w": 200, "h": 150},
             "color_params": {"params": {"exposure": 0.35, "shadows": -20, "temperature": 12, "tint": -4,
                                         "hsl": {"blue": {"saturation": 18}},
                                         "grading": {"shadows": {"hue": 210, "saturation": 12}},
                                         "curve": [[0, 8], [128, 134], [255, 250]]},
                              "recipe": None, "recipe_strength": 1.0},
             "retouch": {"preset": "pixcake_natural"}}
    report = xmp.export_xmp(_photo(tmp_path), state, tmp_path / "photo.xmp")
    attrs, desc = _parse(tmp_path / "photo.xmp")
    assert attrs["Exposure2012"] == "+0.35" and attrs["Shadows2012"] == "-20" and attrs["Contrast2012"] == "0"
    assert attrs["SaturationAdjustmentBlue"] == "+18" and attrs["ColorGradeShadowHue"] == "210"
    assert attrs["IncrementalTemperature"] == "+12" and attrs["HasCrop"] == "True"   # a PNG: relative units
    points = [li.text for li in desc.iter("{http://www.w3.org/1999/02/22-rdf-syntax-ns#}li")]
    assert points == ["0, 8", "128, 134", "255, 250"]
    assert any("people retouch" in s for s in report["not_exported"])
    assert report["lightroom_reads_sidecar"] is False


def test_recipe_curve_and_lut_are_reported(tmp_path):
    lut = next(name for name, r in color.RECIPES.items() if r.get("lut"))
    state = {"color_params": {"params": {"curve": [[0, 0], [64, 80], [255, 255]]}, "recipe": "airy_japanese",
                              "recipe_strength": 0.5}}
    xmp.export_xmp(_photo(tmp_path), state, tmp_path / "a.xmp")
    _, desc = _parse(tmp_path / "a.xmp")
    ys = [int(li.text.split(",")[1]) for li in desc.iter("{http://www.w3.org/1999/02/22-rdf-syntax-ns#}li")]
    assert len(ys) == 17 and ys == sorted(ys)                         # two curves composed into one
    report = xmp.export_xmp(_photo(tmp_path), {"color_params": {"params": {}, "recipe": lut}}, tmp_path / "b.xmp")
    assert any("LUT" in s for s in report["not_exported"])
    assert _parse(tmp_path / "b.xmp")[0]["WhiteBalance"] == "As Shot"


def test_never_replaces_an_existing_sidecar(tmp_path):
    raw = tmp_path / "DSC_1.NEF"
    raw.write_bytes(b"raw")
    (tmp_path / "DSC_1.xmp").write_text("photographer's own edits", encoding="utf-8")
    ours = tmp_path / "ours.xmp"
    ours.write_text("<x/>", encoding="utf-8")
    with pytest.raises(FileExistsError):
        xmp.place_next_to_raw(ours, raw)
    assert (tmp_path / "DSC_1.xmp").read_text(encoding="utf-8") == "photographer's own edits"
