import math

import cv2
import numpy as np
import pytest

from agent_tools import apply_rotation, check_level, dispatch, estimate_tilt
from agent_tools.horizon import largest_rotated_rect
from agent_tools.io_utils import read_image, write_image


def _scene(w=1600, h=1000, seed=0):
    """Sea horizon + a few 'buildings' + texture noise."""
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w, 3), np.uint8)
    img[: h // 2] = (220, 180, 120)          # sky
    img[h // 2:] = (120, 80, 30)             # sea
    for x0 in (200, 650, 1150):
        cv2.rectangle(img, (x0, 250), (x0 + 180, 499), (90, 90, 90), -1)
        for wy in range(280, 480, 50):
            cv2.rectangle(img, (x0 + 30, wy), (x0 + 150, wy + 20), (200, 220, 240), -1)
    noise = rng.normal(0, 6, img.shape)
    return np.clip(img + noise, 0, 255).astype(np.uint8)


def _tilted(tmp_path, tilt_deg, name="tilted.jpg"):
    """Rotate the scene by tilt_deg (CCW) and crop to the clean centre, like a real photo."""
    big = _scene()
    h, w = big.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), tilt_deg, 1.0)
    rot = cv2.warpAffine(big, M, (w, h), flags=cv2.INTER_CUBIC)
    rw, rh = (int(v) - 4 for v in largest_rotated_rect(w, h, tilt_deg))
    x, y = (w - rw) // 2, (h - rh) // 2
    path = tmp_path / name
    write_image(path, rot[y:y + rh, x:x + rw])
    return str(path)


@pytest.mark.parametrize("tilt", [-6.0, -2.5, -0.8, 1.3, 4.0, 9.0])
def test_estimate_recovers_known_tilt(tmp_path, tilt):
    res = estimate_tilt(_tilted(tmp_path, tilt))
    s = res["suggestion"]
    assert s["tilt_deg"] == pytest.approx(tilt, abs=0.15), res
    assert s["correction_deg"] == pytest.approx(-tilt, abs=0.15)
    assert s["confidence"] > 0.5
    assert res["candidates"]["horizon_line"]["width_coverage"] > 0.5
    assert (tmp_path / "tilted_tilt_overlay.png").is_file()


def test_level_roundtrip(tmp_path):
    src = _tilted(tmp_path, 3.2)
    est = estimate_tilt(src)
    out = apply_rotation(src, est["suggestion"]["correction_deg"])
    assert out["output_path"].endswith("_leveled.png")       # JPEG + alpha -> PNG
    assert read_image(out["output_path"]).shape[:2] == read_image(src).shape[:2]
    chk = check_level(out["output_path"], region=out["valid_rect"])
    assert chk["status"] == "pass", chk
    assert abs(chk["residual_tilt_deg"]) < 0.3


def test_max_rect_crop(tmp_path):
    src = _tilted(tmp_path, 5.0)
    out = apply_rotation(src, -5.0, crop="max_rect", output_path=str(tmp_path / "c.jpg"))
    img = read_image(out["output_path"])
    assert img.shape[1] == out["output_size"]["w"] and out["cropped"]
    assert 0.7 < out["kept_area_fraction"] < 1.0


def test_no_lines_is_flagged(tmp_path):
    rng = np.random.default_rng(1)
    noise = cv2.GaussianBlur(rng.integers(0, 255, (800, 1200, 3), dtype=np.uint8), (31, 31), 0)
    p = tmp_path / "noise.png"
    write_image(p, noise)
    res = estimate_tilt(str(p))
    assert res["suggestion"]["confidence"] < 0.3
    assert check_level(str(p))["status"] == "inconclusive"


def test_unicode_path_and_16bit(tmp_path):
    d = tmp_path / "照片"
    d.mkdir()
    src = _tilted(d, 2.0, name="海边.png")
    img16 = read_image(src).astype(np.uint16) * 257
    p16 = d / "海边16.tif"
    write_image(p16, img16)
    out = apply_rotation(str(p16), -2.0)
    assert read_image(out["output_path"]).dtype == np.uint16
    assert estimate_tilt(str(p16))["suggestion"]["tilt_deg"] == pytest.approx(2.0, abs=0.15)


def test_dispatch_reports_errors():
    text, is_err = dispatch("apply_rotation", {"image_path": "missing.jpg", "angle_deg": 1})
    assert is_err and "FileNotFoundError" in text
    _, is_err = dispatch("nope", {})
    assert is_err


def test_largest_rect_identity():
    assert largest_rotated_rect(100, 50, 0) == pytest.approx((100, 50))
    wr, hr = largest_rotated_rect(100, 50, 10)
    assert wr < 100 and hr < 50 and not math.isnan(wr)
