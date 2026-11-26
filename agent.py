"""Orchestrator agent: leveling, crop, colour, people and beauty retouch.

The model (HY4, via an OpenAI-compatible API) decides; the tools in
agent_tools/ measure and execute. Usage:

    .venv\\Scripts\\python.exe agent.py photo.NEF --instruction "Level, crop, correct colour"
    .venv\\Scripts\\python.exe agent.py --revise runs\\photo_20261126_101500 --instruction "脸再亮一点"

Context management: old tool images are replaced by a one-line note (--keep-images, default 1).
Settings come from .env: HY_API_KEY, HY_BASE_URL, HY_MODEL.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import cv2
from dotenv import load_dotenv
from openai import OpenAI

from agent_tools import TOOLS, dispatch
from agent_tools.io_utils import is_raw, read_image, to_bgr_u8

ROOT = Path(__file__).resolve().parent

SYSTEM_PROMPT = """You are the orchestrator of a photo-retouching agent. Your current jobs are, in order:
(A) horizon leveling, then (B) composition and crop on the leveled frame.
The tools measure and execute; you make the decisions.

=== A. Horizon leveling ===

Angle convention: positive = counter-clockwise. estimate_tilt reports tilt_deg (how the scene is rotated)
and correction_deg (= -tilt_deg, the rotation that fixes it).

Workflow:
1. Call estimate_tilt on the photo. Study the photo and the overlay it returns
   (green = near-horizontal lines, blue = near-vertical lines, red = longest straight "horizon" edge).
2. Decide which lines are truly level or plumb in the real world, judging by the evidence for each candidate:
   - A sea/lake horizon, or a long horizontal seen straight on (a table edge, net, shelf or facade
     facing the camera), is excellent: in the numbers it shows as horizontal_lines with small
     |perspective_slope_deg| (< 2) and small spread, often agreeing with horizon_line.
   - Rigid man-made verticals (columns, door/window frames, poles) are excellent too. The
     vertical_lines perspective fit removes keystone lean, but the larger its |perspective_slope_deg|
     (camera pointed up/down), the less precise it is.
   - Horizontals are only untrustworthy when they RECEDE away from the camera (a building side shot at
     an angle, a road): then horizontal_lines has a large |perspective_slope_deg| (> 3) or big spread.
   - Never trust: slopes, shorelines, hills, tree trunks, branches, people, or hanging fabric
     (curtains, banners, drapes are rarely plumb).
   When good horizontal and good vertical evidence disagree, prefer the one with less perspective
   and tighter spread, and look carefully at the overlay before choosing.
3. Decide: rotate by the correction of the reference you trust (adjust it if what you see clearly
   disagrees), or skip when the photo is already within about 0.2 degrees, the tilt is clearly an
   intentional creative angle, or no trustworthy reference exists.
4. If you rotate: call apply_rotation with the TOTAL angle (it always rotates the original photo).
   Then call check_level. If it fails with decent confidence AND its overlay shows it measured a
   trustworthy reference, call apply_rotation once more with angle = previous angle - residual_tilt_deg.
   If check_level measured an untrustworthy line, judge from the overlay yourself. At most 2 rotations.

=== B. Composition and crop (after A is finished) ===
6. Call propose_crops (it automatically works on the leveled image inside the valid area).
   Study the contact sheet and choose by classical framing rules, like a trained photographer:
   - People: never cut at a joint (neck, elbows, waist, wrists, knees, ankles); give lead room in the
     direction the person looks or moves; eyes near the upper third; no cramped headroom; crop out or fully
     include people at the edges, never half of them. Decide the shot size on purpose: a tight vertical
     (4:5 / 3:4) isolates one subject and suits action or portraits; a wide frame tells where they are.
   - Landscapes: horizon on a third line, not the middle (unless a reflection); give 2/3 of the frame
     to the more interesting part (dramatic sky vs. land/foreground).
   - All: main subject near a thirds line or power point (or centred if the scene is symmetric),
     clean edges, nothing distracting at the borders.
   - Prominent objects (`prominent_objects`, white boxes on the overview, e.g. a lit lamp or a statue
     beside the person) are part of the story: keep them whole or leave them out completely, never slice
     one in half. Check where the box actually ends before you say a crop "keeps" it.
   Read each candidate's `notes` (rule problems) and features; the highest score is a hint, not the answer.
   You may call propose_crops again with other aspects or min_scale if nothing fits.
7. Call apply_crop with the rect of the candidate you chose (you may nudge it, staying inside the valid area).
   Look at the preview it returns; if it is clearly wrong, choose again (at most 2 crops).

=== C. Global colour and tone (after B; tools work on the cropped image automatically) ===
8. Call analyze_color. Style is the USER's choice, never yours. Without any style request in the
   instruction, only make the photo technically right (steps 9 and 11; skip the style library and the
   looks in step 10).
   - The instruction gives a sample photo (样片 / reference): call match_reference with it right away and
     use its params as your base for steps 9-11 (adjust exposure on top if the subject needs it; skip
     extra look recipes, the sample IS the look).
   - The instruction names a library style (style memory / 样片库: spring 春, summer 夏, autumn 秋, winter 冬,
     film 胶片感, epic_landscape 风光大片, blue_hour 蓝调, night_portrait 夜景人像, street_night 街头夜景,
     street 街景, wedding_light 婚礼浅色): call apply_style with it and use its params as your base.
   - The instruction asks for a style but not which one ("调个风格", "make it stylish"): call
     suggest_styles, look at its previews, then apply_style for the one that suits the photo.
   - The instruction names some other look ("胶片感", "warm", "Velvia"): use the recipes in step 10.
   Faces must stay natural under any style: every colour tool reports `face_shift`, how far the main
   face's skin moved in white-balance units (limit +-20 temperature, +-20 tint vs. the original photo).
   If it is over, keep the style for the scene and correct the face in step D: pass the given
   `skin_correction` as a local adjustment to preview_retouch / apply_retouch. If detect_people finds
   no person to mask, lower the style strength instead until the face is within the limit.
9. Base correction, in this order: exposure (subject/face well exposed, L* about 50-75 for faces),
   white balance (neutral greys, natural skin; the cast suggestion is a hint, keep pleasant warmth of
   warm light), highlights/shadows (recover clipping, open dark subjects), then contrast.
   With a blue sky in the frame, do NOT warm the white balance to "neutralise" a cool reading: warming
   plus strong highlight recovery turns the sky lavender/grey. Keep the sky blue (analyze_color reports
   it; the 'blue sky stays blue' check fails when it turns purple) and lift dark faces locally in D.
   Do not guess numbers you cannot judge: call preview_variants for that slider and pick a tile.
10. Look (only when the instruction asks for a look): call preview_looks with your base params. Pick the recipe that suits the content and mood
    (or none) and its strength (often 0.4-0.8; use preview_variants with param 'recipe_strength').
11. Call render_adjustments to verify; fix any reported issues (clipping, skin hue, cast) with at most
    2 more renders. Then call apply_color with the same params and recipe.
    Film looks: recipes named fuji_* are film-emulation LUTs (Provia, Velvia, Astia, Reala, Superia,
    Pro 400H, Pro 160C, Acros); their descriptions say what each does to colour.
    If a face stays dark only because the background is bright (backlight, night lamps), do not
    force it with global exposure: fix it locally in step D.

=== D. People: retouch and local edits (after C; works on the colour result automatically) ===
13. Call detect_people. If there are no people, skip to 16.
14. Call preview_retouch, usually with preset pixcake_natural (pixcake_beauty for posed portraits /
    写真), plus local adjustments when needed. Order matters, like a photographer relighting:
    - A dark subject: FIRST relight the whole person with the scene around them,
      {"region": "subject_light", "exposure": 1.0-2.0, "shadows": 20} (aim for face L* about 50-70).
      Never brighten only the face: a face lit on its own looks pasted on (the 'face not lit
      separately' check fails above +0.8 EV extra). A small extra {"region": "face", "exposure": <=0.5}
      on top is fine.
    - A blown background: {"region": "background", "highlights": -40}.
    - Face colour pulled too far by the style (step C's face_shift over the limit): add its
      skin_correction, e.g. {"region": "skin", "temperature": 30, "tint": 5}, so the scene keeps the
      style and the skin looks like skin.
    - Clothing wrinkles (衣服褶皱), only when the user asks for it or for 写真精修: "clothes_smooth": 40-70.
      It keeps fabric texture, prints and seams; check the 'fabric texture kept' result.
    Skin smoothing comes after the relight automatically.
    Judge the face close-up: natural, skin texture still visible, same person, shapes only slightly
    changed. Never exaggerate slimming. If a check fails (halo, texture, warp), lower that setting.
15. Call apply_retouch with your final preset/params.
    If the user asked for 美颜 / 写真 / 网红脸 (step 15b follows), call apply_retouch WITHOUT a preset:
    only your local adjustments and body_slim, because beauty_retouch does the face skin itself
    (smoothing twice makes skin plastic).
15b. Only when the user asks for face beauty (美颜, 精修, 写真, 网红脸): call beauty_retouch with the
    level the user's words mean: 自然 / natural / light -> natural; 写真 / 精修 -> xiezhen;
    网红 / 网红脸 / strong -> wanghong. Put what the user does not want in `avoid` (眼睛不要放大 -> eyes,
    不要瘦脸 -> shape, 不要口红 -> makeup, 不要立体 -> contour). Lipstick (口红) and contour (立体) are part
    of the levels; pass lip_shade when the user names a colour (豆沙 rose, 橘色 coral, 正红 red, 莓果 berry),
    and params {"lipstick": 30-60} / {"contour": 40-80} when they ask for more than the plan gives.
    It plans, checks and repairs by itself; look at its close-up sheet: same person, real skin, natural
    eyes and lips. Call it again with `avoid` / `params` only if something clearly looks off
    (at most 2 calls). Without a beauty request, never call it.
15c. Only when the user asks for background blur (背景虚化, 虚化背景, bokeh): FIRST call analyze_depth and
    read its verdict and depth map. already_shallow: add little (0-35) or skip. sharp_background: decide
    from the photo whether the scene is simply all in focus (blur fits) or the background is right behind
    the person (a wall a step away: keep <= 30 or skip, strong blur would look fake). mixed_depth: use the
    suggested range. Then call apply_background_blur with your strength and look at the sheet, especially
    the edge close-up (hair, shoulders): no cut-out look, no halo. If it looks fake, call again weaker
    (at most 2 calls).

16. Finish with ONLY a JSON object, no other text:
   {"decision": "rotated" or "skipped", "angle_deg": <total applied angle, 0 if skipped>,
    "reference": "<which lines you trusted>", "reason": "<one sentence>",
    "crop": {"candidate_id": <id or null>, "aspect": "<w:h>", "reason": "<one sentence>"},
    "color": {"recipe": "<name or null>", "strength": <0..1>, "reason": "<one sentence>"},
    "retouch": {"preset": "<name or null>", "local": "<what you changed locally or null>", "reason": "<one sentence>"},
    "beauty": {"level": "<natural|xiezhen|wanghong or null>", "avoid": [<groups>], "reason": "<one sentence>"},
    "blur": {"strength": <0..100 or null>, "verdict": "<analyze_depth verdict or null>", "reason": "<one sentence>"}}"""

COLOR_TOOLS = ("analyze_color", "preview_variants", "preview_looks", "render_adjustments", "apply_color",
               "match_reference", "suggest_styles", "apply_style")
PEOPLE_TOOLS = ("detect_people", "preview_retouch", "apply_retouch")


def new_state() -> dict[str, Any]:
    """Where each stage's result lives, so every tool reads the right input (shared with mcp_server.py)."""
    return {"applied_angle": 0.0, "rotations": 0, "valid_rect": None,
            "output_path": None, "check_level": None,
            "crop_rect": None, "final_path": None, "preview_path": None,
            "crops": 0, "color_path": None, "color_params": None,
            "retouch_path": None, "retouch": None, "beauty_path": None, "beauty": None,
            "blur_path": None, "blur": None, "stale": {}}


def final_path(state: dict) -> str | None:
    return (state.get("blur_path") or state["beauty_path"] or state["retouch_path"] or state["color_path"]
            or state["final_path"] or state["output_path"])


# --------------------------------------------------------------------------- #
# Stages: each later stage is built on the file of the one before it. Redoing a stage makes the
# later ones stale; stale stages are re-applied with their previous settings at the end of a run
# (unless the model redid them itself), so a revision never silently drops the beauty retouch.
# --------------------------------------------------------------------------- #
STAGES = [("rotation", "apply_rotation"), ("crop", "apply_crop"), ("color", "apply_color"),
          ("retouch", "apply_retouch"), ("beauty", "beauty_retouch"), ("blur", "apply_background_blur")]
STAGE_OF = {tool: stage for stage, tool in STAGES}
_CLEAR = {"crop": {"crop_rect": None, "final_path": None},
          "color": {"color_path": None, "color_params": None},
          "retouch": {"retouch_path": None, "retouch": None},
          "beauty": {"beauty_path": None, "beauty": None},
          "blur": {"blur_path": None, "blur": None}}


def stage_args(state: dict, stage: str) -> dict | None:
    """The arguments that re-apply `stage` exactly as it was applied, or None if it wasn't."""
    if stage == "rotation":
        return {"angle_deg": state["applied_angle"]} if state["output_path"] else None
    if stage == "crop":
        return {"rect": state["crop_rect"]} if state["final_path"] else None
    if stage == "color":
        return {k: v for k, v in (state["color_params"] or {}).items() if v is not None} \
            if state["color_path"] else None
    if stage == "retouch":
        r = state["retouch"] or {}
        return {k: r[k] for k in ("preset", "params") if r.get(k)} if state["retouch_path"] else None
    if stage == "beauty":
        b = state["beauty"] or {}
        return {"level": b.get("level", "natural"),
                **{k: b[k] for k in ("avoid", "params", "lip_shade") if b.get(k)}} if state["beauty_path"] else None
    if stage == "blur":
        return {"strength": (state.get("blur") or {}).get("strength", 50)} if state.get("blur_path") else None
    raise ValueError(stage)


def invalidate_after(state: dict, stage: str) -> None:
    """`stage` was just (re)applied: later stages built on its old file are stale now."""
    names = [s for s, _ in STAGES]
    for later in names[names.index(stage) + 1:]:
        args = stage_args(state, later)
        if args is not None:
            state.setdefault("stale", {})[later] = args
            state.update(_CLEAR[later])
    state.setdefault("stale", {}).pop(stage, None)


def replay_stale(src: Path, run_dir: Path, state: dict) -> list[dict]:
    """Re-apply stale stages, in order, with their previous settings."""
    done = []
    for stage, tool in STAGES:
        args = state.get("stale", {}).pop(stage, None)
        if args is None:
            continue
        text, err = LevelingAgent._call_tool(tool, dict(args), src, run_dir, state)
        entry = {"stage": stage, "tool": tool, "args": args, "ok": not err}
        if err:
            entry["error"] = json.loads(text).get("error")
        done.append(entry)
    return done


def restore_state(prev_dir: str | Path) -> tuple[Path, dict, dict]:
    """(original photo, state, previous result) of an earlier run folder, for revising it."""
    prev_dir = Path(prev_dir).resolve()
    data = json.loads((prev_dir / "trace.json").read_text(encoding="utf-8"))
    r = data["result"]
    src = Path(r["image"])
    state = new_state()
    if "state" in data:
        state.update(data["state"])
    else:                                    # runs made before the state was saved
        stem = src.stem
        def made(name, applied):
            p = prev_dir / name
            return str(p) if applied and p.exists() else None
        state.update(applied_angle=r.get("angle_deg") or 0.0, rotations=r.get("rotations", 0),
                     output_path=r.get("output_path"), valid_rect=r.get("valid_rect"),
                     check_level=r.get("check_level"), crop_rect=r.get("crop_rect"),
                     final_path=r.get("cropped_path"), color_params=r.get("color_params"),
                     color_path=made(f"{stem}_color.tif", r.get("color_params")),
                     retouch=r.get("retouch"), retouch_path=made(f"{stem}_retouch.tif", r.get("retouch")),
                     beauty=r.get("beauty"), beauty_path=made(f"{stem}_beauty.tif", r.get("beauty")),
                     blur=r.get("blur"), blur_path=made(f"{stem}_blur.tif", r.get("blur")),
                     preview_path=r.get("preview_path"))
    state["stale"] = {}
    return src, state, r


def next_rev_dir(prev_dir: Path) -> Path:
    """runs/x -> runs/x_rev1; runs/x_rev1 -> runs/x_rev2 (revisions of revisions stay flat)."""
    base = re.sub(r"(_rev\d+)+$", "", prev_dir.name)
    n = 1
    while (prev_dir.parent / f"{base}_rev{n}").exists():
        n += 1
    return prev_dir.parent / f"{base}_rev{n}"


def finish_run(src: Path, run_dir: Path, state: dict, result: dict, export=("social",)) -> dict:
    """Delivery after a run (shared with mcp_server.py): Lightroom sidecars for the edit and for every
    stage, the requested export qualities of the final picture, and a line in runs/index.jsonl.
    Nothing here can undo the edit, so failures are reported instead of raised."""
    from agent_tools import deliver
    out: dict = {"lightroom": deliver.write_footprints(src, state, run_dir), "exports": []}
    final = final_path(state)
    for q in export:
        if not final or q == "lightroom":
            continue
        try:
            out["exports"].append(deliver.export_image(final, q, Path(run_dir) / "exports" / f"{Path(src).stem}_final_{q}"))
        except Exception as exc:                                   # noqa: BLE001
            out["exports"].append({"quality": q, "error": f"{type(exc).__name__}: {exc}"})
    try:
        deliver.record_run({**result, **out, "run_dir": str(run_dir), "image": str(src)}, state)
    except OSError as exc:
        out["index_error"] = str(exc)
    return out


def prune_images(messages: list[dict], image_msgs: list[tuple[int, str]], keep: int) -> None:
    """Context management: replace all but the newest `keep` tool-image messages by a one-line note.
    The model has already looked at them and written down what it decided; re-sending every old
    overlay and contact sheet on every call is what made a run cost ~240k input tokens."""
    if keep <= 0:
        return
    for idx, label in image_msgs[:-keep]:
        if isinstance(messages[idx].get("content"), list):
            messages[idx]["content"] = f"[Images removed to save context; you already saw them: {label}]"


REVISE_NOTE = """

=== Revision mode ===
This photo was edited before; you see the previous decisions and the current result. The user now
gives feedback. Redo ONLY the stages the feedback is about, with the same tools as above (preview
first where it helps). Every tool automatically works on the right input file. Stages after the one
you redo are re-applied automatically with their previous settings when you finish, unless the
feedback is about them too; then redo them yourself. Do not redo stages the feedback does not
concern. Finish with the same JSON object as before (keep the previous values for untouched stages)."""


def _openai_tools() -> list[dict]:
    return [{"type": "function",
             "function": {"name": t["name"], "description": t["description"],
                          "parameters": t["input_schema"]}} for t in TOOLS]


def _image_part(path: str | Path, max_side: int = 1024, rect: dict | None = None) -> dict:
    img = to_bgr_u8(read_image(path))
    if rect:
        img = img[rect["y"]:rect["y"] + rect["h"], rect["x"]:rect["x"] + rect["w"]]
    s = max_side / max(img.shape[:2])
    if s < 1:
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    b64 = base64.b64encode(buf.tobytes()).decode()
    return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}


def _parse_final(text: str) -> dict:
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return {"decision": "unknown", "raw_text": text}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"decision": "unknown", "raw_text": text}


class LevelingAgent:
    def __init__(self, model: str | None = None, base_url: str | None = None,
                 api_key: str | None = None, max_steps: int = 36, keep_images: int = 1,
                 export: tuple[str, ...] = ("social",)):
        load_dotenv(ROOT / ".env")
        self.model = model or os.getenv("HY_MODEL")
        api_key = api_key or os.getenv("HY_API_KEY")
        base_url = base_url or os.getenv("HY_BASE_URL")
        if not api_key or not self.model:
            raise RuntimeError("Fill in HY_API_KEY and HY_MODEL in .env first.")
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.max_steps = max_steps
        self.keep_images = keep_images      # tool-image messages kept in context (0 = keep all)
        self.export = tuple(export)         # qualities written for the final picture (deliver.QUALITIES)
        # Some endpoints silently drop the system message for vision models.
        # HY_SYSTEM_IN_USER=1 sends the instructions inside the first user message instead.
        self.system_in_user = os.getenv("HY_SYSTEM_IN_USER", "0") == "1"

    # ------------------------------------------------------------------ #
    def run(self, image_path: str, instruction: str = "Level this photo if it needs it.",
            run_dir: str | Path | None = None, reference: str | None = None) -> dict[str, Any]:
        if reference:
            instruction = (f"{instruction}\nSample photo (样片) to match colour, exposure and curve: "
                           f"{Path(reference).resolve()}")
        src = Path(image_path).resolve()
        run_dir = Path(run_dir or ROOT / "runs" / f"{src.stem}_{time.strftime('%Y%m%d_%H%M%S')}")
        run_dir.mkdir(parents=True, exist_ok=True)
        task_text = f"{instruction}\nPhoto path: {src}"
        messages = self._start_messages(SYSTEM_PROMPT, task_text, [_image_part(src)])
        return self._loop(messages, src, run_dir, new_state(), nudge=(
            "Do not explain how to edit the photo. You ARE the editor and must act through the tools. Start "
            "now by calling estimate_tilt on the photo, then follow the workflow, and finish with only the "
            "JSON object."), extra={"instruction": instruction})

    def revise(self, prev_run: str | Path, feedback: str, run_dir: str | Path | None = None) -> dict[str, Any]:
        """Change an earlier edit after user feedback ("脸再亮一点", "不要瘦脸"): only the stages the
        feedback is about are redone; later stages are re-applied with their previous settings."""
        prev_run = Path(prev_run).resolve()
        src, state, prev = restore_state(prev_run)
        run_dir = Path(run_dir) if run_dir else next_rev_dir(prev_run)
        run_dir.mkdir(parents=True, exist_ok=True)
        before = {k: prev.get(k) for k in ("instruction", "angle_deg", "reference", "reason", "crop_rect",
                                           "crop_choice", "color_params", "color_choice", "retouch_choice",
                                           "beauty", "blur")}
        if before.get("beauty"):
            before["beauty"] = {k: before["beauty"].get(k) for k in ("level", "avoid", "params", "sliders")}
        task_text = (f"User feedback on the previous edit: {feedback}\nPhoto path: {src}\n"
                     f"Previous decisions: {json.dumps(before, ensure_ascii=False, default=str)}\n"
                     "Images: the original photo, then the current result.")
        imgs = [_image_part(src)]
        if final_path(state):
            imgs.append(_image_part(state["preview_path"] or final_path(state)))
        messages = self._start_messages(SYSTEM_PROMPT + REVISE_NOTE, task_text, imgs)
        return self._loop(messages, src, run_dir, state, nudge=(
            "You ARE the editor: act through the tools. Redo only the stages the feedback is about, then "
            "finish with only the JSON object."),
            extra={"revised_from": str(prev_run), "feedback": feedback, "instruction": prev.get("instruction")},
            prev=prev)

    # ------------------------------------------------------------------ #
    def _start_messages(self, system: str, task_text: str, images: list[dict]) -> list[dict]:
        if self.system_in_user:
            return [{"role": "user", "content": [
                {"type": "text", "text": f"{system}\n\n=== Task ===\n{task_text}"}, *images]}]
        return [{"role": "system", "content": system},
                {"role": "user", "content": [{"type": "text", "text": task_text}, *images]}]

    def _loop(self, messages: list[dict], src: Path, run_dir: Path, state: dict, nudge: str,
              extra: dict, prev: dict | None = None) -> dict[str, Any]:
        self._usage = usage = {"prompt_tokens": 0, "completion_tokens": 0}
        trace: list[dict] = []
        image_msgs: list[tuple[int, str]] = []          # (message index, what the images were)
        final: dict = {"decision": "unknown", "reason": "max steps reached"}
        nudges = 0
        for step in range(self.max_steps):
            prune_images(messages, image_msgs, self.keep_images)
            resp = self.client.chat.completions.create(
                model=self.model, messages=messages, tools=_openai_tools(), tool_choice="auto")
            if resp.usage:
                usage["prompt_tokens"] += resp.usage.prompt_tokens or 0
                usage["completion_tokens"] += resp.usage.completion_tokens or 0
            msg = resp.choices[0].message
            messages.append(msg.model_dump(exclude_none=True))
            trace.append({"step": step, "assistant": msg.content,
                          "prompt_tokens": resp.usage.prompt_tokens if resp.usage else None,
                          "tool_calls": [{"name": tc.function.name, "arguments": tc.function.arguments}
                                         for tc in (msg.tool_calls or [])]})

            if not msg.tool_calls:
                final = _parse_final(msg.content or "")
                used_tools = any(t["tool_calls"] for t in trace)
                if (final.get("decision") == "unknown" or not used_tools) and nudges < 2:
                    # The model answered in prose (e.g. a tutorial) instead of acting. Nudge it once or twice.
                    nudges += 1
                    trace[-1]["nudged"] = True
                    messages.append({"role": "user", "content": nudge if not used_tools else
                                     "Finish with ONLY the JSON object described in the instructions, no other text."})
                    continue
                break

            images: list[dict] = []
            labels: list[str] = []
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                    text, is_err = self._call_tool(tc.function.name, args, src, run_dir, state)
                except json.JSONDecodeError as exc:
                    text, is_err = json.dumps({"error": f"bad JSON arguments: {exc}"}), True
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": text})
                trace[-1].setdefault("results", []).append(json.loads(text))
                if not is_err:
                    res = json.loads(text)
                    got = self._result_images(tc.function.name, res, state)
                    if got:
                        images += got
                        shown = next((res[k] for k in ("contact_sheet_path", "sheet_path", "overlay_path",
                                                       "preview_path", "output_path") if res.get(k)), "")
                        labels.append(f"{tc.function.name} ({Path(shown).name})")
            if images:
                messages.append({"role": "user", "content": [
                    {"type": "text", "text": "Images for the tool results above, in order."}, *images]})
                image_msgs.append((len(messages) - 1, ", ".join(labels)))

        replayed = replay_stale(src, run_dir, state)
        if prev:   # untouched stages keep the previous run's reported choices
            for k in ("decision", "reference", "reason"):
                if final.get(k) in (None, "unknown"):
                    final[k] = prev.get(k)
            for k, pk in (("crop", "crop_choice"), ("color", "color_choice"), ("retouch", "retouch_choice")):
                final.setdefault(k, prev.get(pk))
        result = {
            "image": str(src),
            **extra,
            "decision": final.get("decision"),
            # what was actually applied to the pixels (source of truth for evaluation)
            "angle_deg": round(state["applied_angle"], 3),     # 0.0 if never rotated
            "model_reported_angle": final.get("angle_deg"),
            "reference": final.get("reference"),
            "reason": final.get("reason"),
            "rotations": state["rotations"],
            "output_path": state["output_path"],
            "valid_rect": state["valid_rect"],
            "check_level": state["check_level"],
            # crop in the leveled frame's full-res pixels; None = not cropped
            "crop_rect": state["crop_rect"],
            "crop_choice": final.get("crop"),
            "cropped_path": state["final_path"],
            "color_params": state["color_params"],
            "color_choice": final.get("color"),
            "retouch": state["retouch"],
            "retouch_choice": final.get("retouch"),
            "beauty": state["beauty"],
            "blur": state.get("blur"),
            "blur_choice": final.get("blur"),
            "replayed": replayed,
            "final_path": final_path(state),
            "preview_path": state["preview_path"],
            "steps": len(trace),
            "tool_calls": sum(len(t["tool_calls"]) for t in trace),
            "nudges": nudges,
            "usage": usage,
            "keep_images": self.keep_images,
            "model": self.model,
            "run_dir": str(run_dir),
        }
        if final.get("decision") == "skipped" and state["rotations"]:
            result["warning"] = "model said skipped but a rotation was applied"
        result.update(finish_run(src, run_dir, state, result, self.export))
        (run_dir / "trace.json").write_text(
            json.dumps({"result": result, "state": state, "trace": trace}, indent=1, ensure_ascii=False,
                       default=str), encoding="utf-8")
        return result

    # ------------------------------------------------------------------ #
    @staticmethod
    def _call_tool(name: str, args: dict, src: Path, run_dir: Path, state: dict):
        """Run a tool, pinning file paths so the model can't scatter files or compound rotations."""
        if name == "estimate_tilt":
            args.setdefault("image_path", str(src))
            n = len(list(run_dir.glob("overlay_*.png")))
            args["overlay_path"] = str(run_dir / f"overlay_{n}.png")
        elif name == "apply_rotation":
            args["image_path"] = str(src)                       # always from the original
            ext = ".tif" if is_raw(src) or src.suffix.lower() in (".tif", ".tiff") else ".png"
            args["output_path"] = str(run_dir / f"{src.stem}_leveled{ext}")
        elif name == "check_level":
            if state["output_path"]:
                args["image_path"] = state["output_path"]
                if state["valid_rect"] and not args.get("region"):
                    args["region"] = state["valid_rect"]
            else:
                args.setdefault("image_path", str(src))
        elif name in ("propose_crops", "apply_crop"):
            # always on the leveled frame if there is one, never on a previous crop
            args["image_path"] = state["output_path"] or str(src)
            if state["valid_rect"]:
                args["valid_rect"] = state["valid_rect"]
            if name == "propose_crops":
                n = len(list(run_dir.glob("crops_*.jpg")))
                args["contact_sheet_path"] = str(run_dir / f"crops_{n}.jpg")
            else:
                ext = ".tif" if is_raw(src) or src.suffix.lower() in (".tif", ".tiff") else ".png"
                args["output_path"] = str(run_dir / f"{src.stem}_final{ext}")
        elif name in COLOR_TOOLS:
            # colour always works on the newest geometry result (crop > leveled > original), never on
            # a previous colour render, so settings never stack
            args["image_path"] = state["final_path"] or state["output_path"] or str(src)
            n = len(list(run_dir.glob(f"{name}_*.jpg")))
            if name in ("preview_variants", "preview_looks", "match_reference", "suggest_styles", "apply_style"):
                args["sheet_path"] = str(run_dir / f"{name}_{n}.jpg")
            elif name == "render_adjustments":
                args["preview_path"] = str(run_dir / f"{name}_{n}.jpg")
            elif name == "apply_color":
                args["output_path"] = str(run_dir / f"{src.stem}_color.tif")
        elif name in PEOPLE_TOOLS:
            # retouch always starts from the colour result, never from a previous retouch
            args["image_path"] = state["color_path"] or state["final_path"] or state["output_path"] or str(src)
            n = len(list(run_dir.glob(f"{name}_*.jpg")))
            if name == "detect_people":
                args["overlay_path"] = str(run_dir / f"{name}_{n}.jpg")
            elif name == "preview_retouch":
                args["sheet_path"] = str(run_dir / f"{name}_{n}.jpg")
            else:
                args["output_path"] = str(run_dir / f"{src.stem}_retouch.tif")
        elif name == "beauty_retouch":
            # beauty works on the people retouch (relight) if there is one, never on a previous beauty
            # result, so repeated calls never stack
            args["image_path"] = (state["retouch_path"] or state["color_path"] or state["final_path"]
                                  or state["output_path"] or str(src))
            n = len(list(run_dir.glob("beauty_sheet_*.jpg")))
            args["sheet_path"] = str(run_dir / f"beauty_sheet_{n}.jpg")
            args["output_path"] = str(run_dir / f"{src.stem}_beauty.tif")
        elif name in ("analyze_depth", "apply_background_blur"):
            # the blur is the last stage: it works on the beauty / retouch result, never on a previous blur
            args["image_path"] = (state["beauty_path"] or state["retouch_path"] or state["color_path"]
                                  or state["final_path"] or state["output_path"] or str(src))
            if name == "analyze_depth":
                n = len(list(run_dir.glob("depth_*.jpg")))
                args["overlay_path"] = str(run_dir / f"depth_{n}.jpg")
            else:
                n = len(list(run_dir.glob("blur_sheet_*.jpg")))
                args["sheet_path"] = str(run_dir / f"blur_sheet_{n}.jpg")
                args["output_path"] = str(run_dir / f"{src.stem}_blur.tif")

        text, is_err = dispatch(name, args)
        if not is_err:
            res = json.loads(text)
            if name == "apply_rotation":
                state.update(applied_angle=float(res["angle_deg"]), valid_rect=res["valid_rect"],
                             output_path=res["output_path"], rotations=state["rotations"] + 1)
            elif name == "check_level":
                state["check_level"] = {k: res[k] for k in ("status", "residual_tilt_deg", "confidence", "basis")}
            elif name == "apply_crop":
                state.update(crop_rect=res["rect"], final_path=res["output_path"],
                             preview_path=res["preview_path"], crops=state["crops"] + 1)
            elif name == "apply_color":
                state.update(color_path=res["output_path"], preview_path=res["preview_path"],
                             color_params={"params": args.get("params"), "recipe": args.get("recipe"),
                                           "recipe_strength": args.get("recipe_strength", 1.0)})
            elif name == "apply_retouch":
                state.update(retouch_path=res["output_path"], preview_path=res["preview_path"],
                             retouch={"preset": args.get("preset"), "params": args.get("params"),
                                      "checks": res["checks"], "info": res["info"],
                                      "source": args["image_path"]})
            elif name == "beauty_retouch" and res.get("output_path"):
                state.update(beauty_path=res["output_path"], preview_path=res["preview_path"],
                             beauty={"level": res["level"], "avoid": args.get("avoid"), "params": args.get("params"),
                                     "sliders": {f: v["sliders"] for f, v in res["faces"].items()},
                                     "lip_shade": res.get("lip_shade"),
                                     "repairs": res["repairs"], "checks_failed": res["checks_failed"],
                                     "source": args["image_path"]})
            elif name == "apply_background_blur":
                state.update(blur_path=res["output_path"], preview_path=res["preview_path"],
                             blur={"strength": res["strength"], "verdict": res["verdict"], "info": res["info"],
                                   "checks_failed": res["checks_failed"], "source": args["image_path"]})
            if name in STAGE_OF and (name != "beauty_retouch" or res.get("output_path")):
                invalidate_after(state, STAGE_OF[name])
        return text, is_err

    @staticmethod
    def _result_images(name: str, res: dict, state: dict) -> list[dict]:
        if name in ("estimate_tilt", "check_level") and res.get("overlay_path"):
            return [_image_part(res["overlay_path"])]
        if name == "apply_rotation":           # let the model see the leveled frame
            return [_image_part(res["output_path"], rect=res["valid_rect"])]
        if name == "propose_crops":
            return [_image_part(res["contact_sheet_path"], max_side=1600)]
        if name == "apply_crop":
            return [_image_part(res["preview_path"])]
        if name in ("preview_variants", "preview_looks", "match_reference", "suggest_styles", "apply_style"):
            return [_image_part(res["sheet_path"], max_side=1600)]
        if name in ("render_adjustments", "apply_color", "apply_retouch"):
            return [_image_part(res["preview_path"])]
        if name == "detect_people":
            return [_image_part(res["overlay_path"])]
        if name == "preview_retouch":
            return [_image_part(res["sheet_path"], max_side=1600)]
        if name == "beauty_retouch" and res.get("sheet_path"):
            return [_image_part(res["sheet_path"], max_side=1600)]
        if name == "analyze_depth" and res.get("overlay_path"):
            return [_image_part(res["overlay_path"])]
        if name == "apply_background_blur":
            return [_image_part(res["sheet_path"], max_side=1600)]
        return []   # analyze_color: numbers only; the model has just seen the cropped preview


def main() -> None:
    ap = argparse.ArgumentParser(description="Photo-retouching agent")
    ap.add_argument("image", nargs="?", help="photo to edit (not needed with --revise)")
    ap.add_argument("--instruction", default="Level this photo if it needs it.",
                    help="what to do; with --revise: the feedback, e.g. '脸再亮一点'")
    ap.add_argument("--revise", metavar="RUN_DIR", help="change an earlier run (runs/<photo>_<time>) after feedback")
    ap.add_argument("--max-steps", type=int, default=36)
    ap.add_argument("--export", default="social",
                    help="qualities for the final picture, comma separated: social, print, full (default social)")
    ap.add_argument("--keep-images", type=int, default=1,
                    help="tool-image messages kept in the model's context (0 = keep all, the old behaviour)")
    ap.add_argument("--reference", help="sample photo (样片) whose colour / exposure / curve to match")
    a = ap.parse_args()
    agent = LevelingAgent(max_steps=a.max_steps, keep_images=a.keep_images,
                          export=[q for q in a.export.split(",") if q])
    if a.revise:
        res = agent.revise(a.revise, a.instruction)
    elif a.image:
        res = agent.run(a.image, a.instruction, reference=a.reference)
    else:
        ap.error("give a photo, or --revise RUN_DIR")
    sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(res, indent=1, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
