from pathlib import Path

import numpy as np
import pytest

from agent_tools import styles as S

ROOT = Path(__file__).resolve().parent.parent
HAVE = all((ROOT / "samples" / v["file"]).exists() for v in S.STYLES.values())


def test_library_styles_with_samples():
    assert {"spring", "summer", "autumn", "winter", "film", "epic_landscape", "blue_hour",
            "night_portrait", "street_night", "street", "wedding_light"} == set(S.STYLES)


def _cues(**kw):
    base = {"night": 0.0, "dusk": 0.0, "people": 0.0, "face_size": 0.0, "bright_lights": 0.0,
            "sky_blue": 0.0, "green": 0.0, "foliage_warm": 0.0, "blossom": 0.0, "snow": 0.0,
            "turquoise_blue": 0.0, "saturation": 15.0, "landscape_wide": 0.0, "median_L": 40.0, "built": 0.0}
    base.update(kw)
    return base


@pytest.mark.parametrize("cues, expected", [
    (dict(night=1.0, people=1.0, face_size=0.1, bright_lights=0.1), "night_portrait"),
    (dict(night=1.0, people=1.0, face_size=0.02, bright_lights=0.1, built=0.8), "street_night"),
    (dict(people=1.0, face_size=0.05, built=0.9, saturation=10), "street"),
    (dict(people=1.0, face_size=0.03, median_L=75, saturation=5), "wedding_light"),
    (dict(night=0.8, sky_blue=0.4, bright_lights=0.05, landscape_wide=1.0), "blue_hour"),
    (dict(foliage_warm=0.6), "autumn"),
    (dict(snow=0.4), "winter"),
    (dict(green=0.3, blossom=0.15), "spring"),
    (dict(sky_blue=0.5, turquoise_blue=0.3, saturation=30), "summer"),
])
def test_judge_picks_the_obvious_style(cues, expected):
    assert S.score_styles(_cues(**cues), only_available=False)[0]["style"] == expected


def test_add_params_merges():
    out = S.add_params({"temperature": 10, "hsl": {"orange": {"hue": 2}}},
                       {"temperature": 20, "hsl": {"orange": {"hue": 10, "saturation": 5}}}, 0.5)
    assert out["temperature"] == 20 and out["hsl"]["orange"] == {"hue": 7.0, "saturation": 2.5}


@pytest.mark.skipif(not (HAVE and (ROOT / "TEST-c" / "DSC_6905.NEF").exists()), reason="samples/photo missing")
def test_autumn_is_golden(tmp_path):
    r = S.style_params(str(ROOT / "TEST-c" / "DSC_6905.NEF"), "autumn", sheet_path=str(tmp_path / "a.jpg"))
    p = r["params"]
    # the golden sample: richer orange/yellow leaves and more colour overall, not muted
    assert p["saturation"] > 0 and p["hsl"]["yellow"]["saturation"] > 0 and p["hsl"]["orange"]["saturation"] > 0
