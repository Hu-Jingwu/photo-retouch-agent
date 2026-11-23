"""Tool definitions the orchestrator LLM sees, plus a dispatcher.

Definitions follow the Claude Messages API `tools` format
({name, description, input_schema}). The orchestrator loop (not written yet)
should, for any result that has `overlay_path`, also attach that PNG as an
image block in the tool_result so the model can *see* what the CV measured.
"""
from __future__ import annotations

import json
import traceback
from typing import Any, Callable

from . import beauty, color, composition, horizon, people, styles

_REGION_SCHEMA = {
    "type": "object",
    "description": "Sub-rectangle in full-resolution pixels.",
    "properties": {
        "x": {"type": "integer"}, "y": {"type": "integer"},
        "w": {"type": "integer"}, "h": {"type": "integer"},
    },
    "required": ["x", "y", "w", "h"],
    "additionalProperties": False,
}

TOOLS: list[dict[str, Any]] = [
    {
        "name": "estimate_tilt",
        "description": (
            "Measure how far the photo is rotated (camera roll) using classical computer vision "
            "(line segment detection). It only measures; you decide whether to level. "
            "Returns three independent candidates: `horizon_line` (one long collinear edge such as "
            "a sea or skyline horizon; most trustworthy when width_coverage >= 0.5), "
            "`horizontal_lines` (all near-horizontal structure), and `vertical_lines` (buildings, "
            "poles; can be biased by keystone convergence when the camera points up or down). "
            "Also returns a `suggestion`, `flags` (e.g. no_reliable_lines, "
            "horizontal_vertical_disagree, already_level) and an overlay image showing the detected "
            "lines: green = horizontal, blue = vertical, red = dominant horizon. "
            "Look at the overlay before acting. Do NOT level when the tilt is clearly intentional "
            "(Dutch angle), when the dominant lines are a slope, shoreline curve or perspective "
            "lines rather than a true horizontal, or when confidence is low and the image looks level. "
            "Angles: tilt_deg is counter-clockwise-positive apparent rotation; the fix is "
            "correction_deg = -tilt_deg."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "image_path": {"type": "string", "description": "Path to the image file."},
                "max_tilt_deg": {
                    "type": "number",
                    "description": "Only lines within this many degrees of horizontal/vertical are "
                                   "considered. Default 20.",
                },
                "region": {**_REGION_SCHEMA,
                           "description": "Optional: analyse only this part of the frame, e.g. "
                                          "below a cluttered sky or the valid_rect after rotation."},
            },
            "required": ["image_path"],
            "additionalProperties": False,
        },
    },
    {
        "name": "apply_rotation",
        "description": (
            "Rotate the full-resolution image about its centre by angle_deg "
            "(counter-clockwise positive; pass correction_deg from estimate_tilt, or your own "
            "adjusted value). Limited to +/-45 degrees. With crop='none' (default) the canvas is "
            "kept, the exposed corners are transparent, and `valid_rect` is returned so the "
            "composition/auto-crop step can crop inside it. Use crop='max_rect' only when no "
            "composition step will follow. Returns output_path and kept_area_fraction "
            "(how much of the frame survives a clean crop)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "image_path": {"type": "string"},
                "angle_deg": {"type": "number", "description": "CCW-positive rotation in degrees."},
                "output_path": {"type": "string", "description": "Optional output file path."},
                "crop": {"type": "string", "enum": ["none", "max_rect"]},
                "border": {"type": "string", "enum": ["transparent", "reflect", "black"],
                           "description": "How to fill exposed corners when crop='none'."},
            },
            "required": ["image_path", "angle_deg"],
            "additionalProperties": False,
        },
    },
    {
        "name": "check_level",
        "description": (
            "Deterministic evaluator check: measure residual tilt on an edited image. "
            "Returns status pass / fail / inconclusive (inconclusive = not enough straight lines "
            "to judge; do not treat it as a failure). Pass valid_rect from apply_rotation as "
            "`region` so the transparent corners are ignored."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "image_path": {"type": "string"},
                "tolerance_deg": {"type": "number", "description": "Default 0.3."},
                "region": _REGION_SCHEMA,
            },
            "required": ["image_path"],
            "additionalProperties": False,
        },
    },
]

TOOLS += [
    {
        "name": "propose_crops",
        "description": (
            "Propose crops scored on classical framing rules. Run it AFTER leveling; it stays inside "
            "valid_rect. It classifies the scene (people / landscape / general) and applies:\n"
            "- all: rule of thirds (placement), clean edges, no amputated people (partial faces at the "
            "edge are penalised), keep what matters (saliency_kept), area_kept.\n"
            "- people: eyes on the upper third, headroom, never cut at joints (neck, elbows/waist, "
            "wrists/hips, knees, ankles; estimated from figure proportions), lead room (space in the "
            "direction the face is turned), a shot_size label, and vertical crops (4:5, 3:4) are "
            "offered for a single subject even in a horizontal photo.\n"
            "- landscape: horizon on a third line; the more interesting part (textured sky vs land) "
            "gets 2/3 of the frame; a reflection may be centred.\n"
            "Each candidate lists `notes` with rule problems. Candidates always include the widest "
            "option and the best vertical one. Returns a contact sheet: an overview with numbered "
            "rectangles plus each crop with a thirds grid. Judge the contact sheet yourself."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "image_path": {"type": "string"},
                "valid_rect": {**_REGION_SCHEMA,
                               "description": "valid_rect from apply_rotation (filled in automatically)."},
                "aspects": {"type": "array", "items": {"type": "string"},
                            "description": "Aspect ratios w:h to try, e.g. ['original','4:5','1:1']. "
                                           "Default depends on orientation."},
                "num_candidates": {"type": "integer", "description": "Default 6."},
                "min_scale": {"type": "number",
                              "description": "Smallest crop as a fraction of the largest that fits. "
                                             "Default 0.6; raise it for gentler crops."},
            },
            "required": ["image_path"],
            "additionalProperties": False,
        },
    },
    {
        "name": "apply_crop",
        "description": (
            "Crop the image to rect (full-resolution pixels, same coordinates as propose_crops). "
            "Pass a candidate's rect as is, or adjust it yourself. It is refused if it reaches outside "
            "valid_rect. Writes the final full-quality file plus a preview."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "image_path": {"type": "string"},
                "rect": _REGION_SCHEMA,
            },
            "required": ["image_path", "rect"],
            "additionalProperties": False,
        },
    },
]

_PARAMS_SCHEMA = {
    "type": "object",
    "description": (
        "Develop sliders (all optional, 0 = unchanged). exposure: EV -3..3. "
        "contrast, highlights, shadows, whites, blacks, temperature (+ warmer), tint (+ magenta), "
        "vibrance, saturation: -100..100. grain: 0..100. "
        "hsl: {colour: {hue, saturation, luminance}} with colour in red, orange, yellow, green, aqua, "
        "blue, purple, magenta, values -100..100 (orange covers skin). "
        "grading: {shadows|midtones|highlights: {hue: 0..360, saturation: 0..100}}. "
        "curve: [[x, y], ...] points 0..255 on a tone curve."),
}
_RECIPE_PROPS = {
    "recipe": {"type": "string", "enum": list(color.RECIPES),
               "description": "Optional named look applied on top of params."},
    "recipe_strength": {"type": "number", "description": "0..1, default 1."},
}

TOOLS += [
    {
        "name": "analyze_color",
        "description": (
            "Measure the photo's tone and colour: luminance percentiles and key, clipped highlights / "
            "crushed shadows (%), colour cast on near-grey areas with a suggested temperature/tint "
            "correction, skin tone of the main face (Lab lightness, hue angle, chroma), dominant colours, "
            "and failed correctness checks. Also lists the available look recipes."),
        "input_schema": {"type": "object", "properties": {"image_path": {"type": "string"}},
                         "required": ["image_path"], "additionalProperties": False},
    },
    {
        "name": "preview_variants",
        "description": (
            "Render the photo with ONE parameter at 2-6 different values side by side (tiles A, B, C...) "
            "on top of your current params, so you can choose by eye instead of guessing numbers. "
            "param is a slider name ('exposure', 'temperature', 'shadows'...), a dotted path "
            "('hsl.blue.saturation', 'grading.shadows.saturation'), or 'recipe_strength'."),
        "input_schema": {"type": "object", "properties": {
            "image_path": {"type": "string"},
            "param": {"type": "string"},
            "values": {"type": "array", "items": {"type": "number"}},
            "params": _PARAMS_SCHEMA, **_RECIPE_PROPS},
            "required": ["image_path", "param", "values"], "additionalProperties": False},
    },
    {
        "name": "preview_looks",
        "description": (
            "Contact sheet: your base correction (tile 0) and every look recipe on top of it (tiles 1..n) "
            "at the given strength, to choose a look (or none)."),
        "input_schema": {"type": "object", "properties": {
            "image_path": {"type": "string"},
            "params": _PARAMS_SCHEMA,
            "recipes": {"type": "array", "items": {"type": "string", "enum": list(color.RECIPES)}},
            "strength": {"type": "number", "description": "0..1, default 0.7"}},
            "required": ["image_path"], "additionalProperties": False},
    },
    {
        "name": "suggest_styles",
        "description": (
            "Style memory (风格记忆): use ONLY when the user asks for a style without naming one. "
            "Judges which of the 11 library styles (样片库) fits this photo: spring 春, summer 夏, autumn 秋 "
            "(golden), winter 冬, film 胶片感, epic_landscape 风光大片, blue_hour 蓝调, night_portrait 夜景人像, "
            "street_night 街头夜景, street 街景, wedding_light 婚礼浅色. "
            "Measures the situation (day/night, people and face size, blue sky, greens, warm foliage, "
            "blossom, snow, city lights, buildings, brightness), returns a ranking with reasons, and a sheet: the photo now, "
            "then for each top style its sample and the photo rendered in that style. Scores are cues: "
            "look at the sheet and decide."),
        "input_schema": {"type": "object", "properties": {
            "image_path": {"type": "string"},
            "top_k": {"type": "integer", "description": "how many styles to preview (default 3)"}},
            "required": ["image_path"], "additionalProperties": False},
    },
    {
        "name": "apply_style",
        "description": (
            "Slider values for one library style on this photo (matched to its sample, plus the style's "
            "own adjustments, if any), with a sheet sample | before | after. Use the "
            "params as your base in render_adjustments / apply_color. strength 0..1 (default 0.8)."),
        "input_schema": {"type": "object", "properties": {
            "image_path": {"type": "string"},
            "style": {"type": "string", "enum": list(styles.STYLES)},
            "strength": {"type": "number"}},
            "required": ["image_path", "style"], "additionalProperties": False},
    },
    {
        "name": "match_reference",
        "description": (
            "样片追色: generate slider values that give the photo the colour, exposure and light/dark "
            "curve of a sample photo (white balance, tint, exposure, saturation, 8-colour HSL, tone curve), "
            "like PixCake's AI追色 with 精准控制. Statistics are matched with limits (a night photo stays "
            "a night photo; skin colours move least). Returns params plus a sheet: sample | before | after. "
            "Use the params as your base in render_adjustments / apply_color. strength 0..1 (default 0.8)."),
        "input_schema": {"type": "object", "properties": {
            "image_path": {"type": "string"},
            "reference_path": {"type": "string", "description": "the sample photo to match"},
            "strength": {"type": "number"},
            "match_tone": {"type": "boolean", "description": "also match exposure and curve (default true)"}},
            "required": ["image_path", "reference_path"], "additionalProperties": False},
    },
    {
        "name": "render_adjustments",
        "description": (
            "Fast preview of params (+ optional recipe) with measurements and failed correctness checks "
            "(clipping, skin tone, colour cast, exposure) of the RESULT. Use it to verify before applying."),
        "input_schema": {"type": "object", "properties": {
            "image_path": {"type": "string"}, "params": _PARAMS_SCHEMA, **_RECIPE_PROPS},
            "required": ["image_path"], "additionalProperties": False},
    },
    {
        "name": "apply_color",
        "description": (
            "Render the final full-resolution 16-bit image with params (+ optional recipe), plus a preview "
            "and a JSON sidecar of the settings. Call once you are satisfied with render_adjustments."),
        "input_schema": {"type": "object", "properties": {
            "image_path": {"type": "string"}, "params": _PARAMS_SCHEMA, **_RECIPE_PROPS},
            "required": ["image_path"], "additionalProperties": False},
    },
]

_RETOUCH_SCHEMA = {
    "type": "object",
    "description": (
        "Retouch settings 0..100 (all optional): skin_smoothing (keeps pores), skin_even_tone, "
        "skin_brighten, eye_brighten, face_slim, body_slim; blemish_removal: true/false. "
        "local: list of edge-aware local adjustments, each {region, exposure (EV), contrast, highlights, "
        "shadows, saturation, vibrance, temperature, tint} with region in subject_light (whole person "
        "plus a soft natural fall-off into the nearby scene: use this to relight a dark subject), person, "
        "face (only small extra on top of a relight), skin, hair, clothes, background."),
}

TOOLS += [
    {
        "name": "detect_people",
        "description": (
            "Find people: per person face box, face height in the frame, facing direction, whether 478 face "
            "landmarks and body joints were found, and the face lightness L*; plus background lightness. "
            "Returns an overlay of the masks (face skin, body skin, hair, person outline) so you can check "
            "they fit. Call it after colour; if num_people is 0, skip portrait retouch."),
        "input_schema": {"type": "object", "properties": {"image_path": {"type": "string"}},
                         "required": ["image_path"], "additionalProperties": False},
    },
    {
        "name": "preview_retouch",
        "description": (
            "Preview portrait retouch and local adjustments: returns a before/after sheet (full frame + "
            "face close-up), what was done, failed safety checks (halo around people, skin texture lost, "
            "slimming too strong, background warped) and colour issues of the result. "
            "Presets: pixcake_natural (light 写真 retouch) and pixcake_beauty (stronger but still subtle); "
            "params override a preset. Use local adjustments to fix a dark face or a too-bright background "
            "without touching the rest (masks follow real edges, so no halo)."),
        "input_schema": {"type": "object", "properties": {
            "image_path": {"type": "string"},
            "preset": {"type": "string", "enum": list(people.PRESETS)},
            "params": _RETOUCH_SCHEMA},
            "required": ["image_path"], "additionalProperties": False},
    },
    {
        "name": "apply_retouch",
        "description": "Apply the chosen retouch at full resolution (16-bit) and return a preview plus checks.",
        "input_schema": {"type": "object", "properties": {
            "image_path": {"type": "string"},
            "preset": {"type": "string", "enum": list(people.PRESETS)},
            "params": _RETOUCH_SCHEMA},
            "required": ["image_path"], "additionalProperties": False},
    },
]

TOOLS += [
    {
        "name": "beauty_retouch",
        "description": (
            "Face beauty retouch (美颜: 像素蛋糕写真 / 网红脸), ONLY when the user asks for it. One call does "
            "everything in Python: measures each face, plans slider values that fill only the gap to a "
            "target face (a face that already has big eyes gets no eye enlarging), renders, runs the guards "
            "(same person, eyes/nose/mouth change small, pores kept, symmetry, plausible proportions, "
            "background lines straight, no halo), lowers the sliders that broke a guard, and writes the "
            "full-resolution result. Levels: natural (自然: clean skin, bright eyes, no shape change), "
            "xiezhen (写真: PixCake's 写真 values), wanghong (网红: strongest the guards allow). "
            "Look at the returned close-up sheet. If something looks off, call again (it always starts "
            "from the same input, nothing stacks) with `avoid` groups or explicit `params`."),
        "input_schema": {"type": "object", "properties": {
            "image_path": {"type": "string"},
            "level": {"type": "string", "enum": list(beauty.LEVELS)},
            "avoid": {"type": "array", "items": {"type": "string", "enum": list(beauty.GROUPS)},
                      "description": "slider groups to leave untouched, e.g. the user said 眼睛不要放大 -> ['eyes']"},
            "params": {"type": "object", "description": (
                "explicit slider values 0..100 that override the plan for every face, e.g. "
                "{\"eye_size\": 15, \"smoothing\": 50}. Keys: " + ", ".join(beauty.F.ALL_KEYS))}},
            "required": ["image_path", "level"], "additionalProperties": False},
    },
]

_HANDLERS: dict[str, Callable[..., dict]] = {
    "detect_people": people.detect_people,
    "preview_retouch": people.preview_retouch,
    "apply_retouch": people.apply_retouch,
    "beauty_retouch": beauty.beauty_retouch,
    "analyze_color": color.analyze_color,
    "preview_variants": color.preview_variants,
    "preview_looks": color.preview_looks,
    "render_adjustments": color.render_adjustments,
    "match_reference": color.match_reference,
    "suggest_styles": styles.suggest_styles,
    "apply_style": styles.style_params,
    "apply_color": color.apply_color,
    "estimate_tilt": horizon.estimate_tilt,
    "apply_rotation": horizon.apply_rotation,
    "check_level": horizon.check_level,
    "propose_crops": composition.propose_crops,
    "apply_crop": composition.apply_crop,
}


def dispatch(name: str, tool_input: dict[str, Any]) -> tuple[str, bool]:
    """Run a tool call. Returns (json_text, is_error) ready for a tool_result block."""
    handler = _HANDLERS.get(name)
    if handler is None:
        return json.dumps({"error": f"unknown tool {name!r}"}), True
    try:
        return json.dumps(handler(**tool_input), ensure_ascii=False), False
    except Exception as exc:  # report to the model instead of crashing the loop
        return json.dumps({"error": f"{type(exc).__name__}: {exc}",
                           "trace": traceback.format_exc(limit=3)}), True
