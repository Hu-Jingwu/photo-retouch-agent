# Code report: AI photo-retouching agent

*A first agent project, explained for learning. State as of 2026-11-22.*

## 1. The idea in one sentence

**The model decides, the tools execute, the rules grade.**

A large multimodal model (hy4-flash, called through Tencent's OpenAI-compatible Tokenhub API) looks at the
photo and the tool results and makes the decisions. Python tools built on OpenCV, rawpy and MediaPipe do
the measuring and the pixel work. Deterministic checks and an evaluation harness say whether the result
is good. The model never touches pixels directly, and it never grades its own work in the harness.

This split is the core lesson of the project. A model is good at judgement ("which of these lines is
really level?", "which crop tells the story?"). It is bad at precise numbers and repeatable pixel
operations. Code is the opposite. An agent puts each where it is strong.

## 2. Architecture

```
 Photo + instruction (+ optional sample photo / style)
                │
                ▼
 ┌───────────────────────────────┐      ┌──────────────────────────────┐
 │ Orchestrator  (agent.py)      │◄─────│ Style memory (styles.py)     │
 │ hy4-flash, tool calling loop  │      │ 11-style sample library;     │
 │ stages A → B → C → D          │      │ only when the user asks       │
 └──────────────┬────────────────┘      └──────────────────────────────┘
                │ tool calls (JSON)          ▲ images + JSON results
                ▼                            │
 ┌─────────────────────────────────────────────────────────────────────┐
 │ Tools (agent_tools/, dispatched by registry.py)                     │
 │ A Leveling     estimate_tilt · apply_rotation · check_level         │
 │ B Crop         propose_crops · apply_crop                           │
 │ C Colour       analyze_color · preview_variants · preview_looks ·   │
 │                render_adjustments · apply_color · match_reference · │
 │                suggest_styles · apply_style                         │
 │ D People       detect_people · preview_retouch · apply_retouch      │
 └──────────────┬──────────────────────────────────────────────────────┘
                ▼
 ┌───────────────────────────────┐   pass   ┌──────────────────────────┐
 │ Render & evaluate             │─────────►│ Export: 16-bit TIFF,     │
 │ in-tool checks + harness      │          │ JPEG preview, params JSON│
 └──────────────┬────────────────┘          └──────────────────────────┘
                │ fail: issues go back to the model (2-3 retries max)
                └──────────────► Orchestrator
```

## 3. Files

| File | Lines | What it does |
|---|---|---|
| `agent.py` | ~400 | The agent: system prompt, tool-calling loop, path pinning, final JSON |
| `agent_tools/registry.py` | ~345 | Tool list (name, description, JSON schema) and `dispatch()` |
| `agent_tools/horizon.py` | ~395 | A: tilt from line segments, rotation, residual check |
| `agent_tools/composition.py` | ~675 | B: faces, saliency, prominent objects, crop proposals, crop grader |
| `agent_tools/color.py` | ~990 | C: measurements, the colour pipeline, Fuji LUTs, reference matching, face shift |
| `agent_tools/styles.py` | ~205 | Style memory: 11 sample styles and the judge |
| `agent_tools/people.py` | ~590 | D: MediaPipe masks, relight, smoothing, slimming, retouch checks |
| `agent_tools/face.py` | ~485 | Face tools (skin, shape, eyes, teeth), used by `beauty.py` |
| `agent_tools/io_utils.py` | ~77 | Reading RAW (NEF via rawpy), TIFF, JPEG; writing 8/16-bit |
| `evals/harness.py` | ~245 | Runs the agent on a set of photos and grades every stage |
| `evals/ground_truth.py` | ~57 | Your Lightroom rotation/crop, recovered with SIFT matching |
| `tests/` | ~520 | 63 pytest tests for the tools |
| `samples/` | | Style library photos and `SOURCES.md` (credits and licences) |
| `docs/` | | Composition rules, face-panel study, beauty design, these reports |

## 4. How the agent loop works (`agent.py`)

This is the heart of any agent. Here is the loop, simplified from `LevelingAgent.run()`:

```python
messages = [system_prompt, user(instruction + photo)]
for step in range(max_steps):                      # 36 by default
    reply = model(messages, tools=TOOLS)           # the model may answer with tool calls
    messages.append(reply)
    if not reply.tool_calls:                       # no tool call = it thinks it is done
        final = parse_json(reply.content)
        break
    for call in reply.tool_calls:
        result = dispatch(call.name, call.arguments)   # run the Python tool
        messages.append(tool_message(result))           # JSON result back to the model
    messages.append(user(images_for_those_results))     # let it SEE overlays/previews
```

Six details are worth understanding, because each fixed a real failure:

1. **Tools are described in JSON Schema** (`registry.py`). The model only knows a tool through its name,
   description and parameters, so the description is part of the program. Rules like "use ONLY when the
   user asks for a style" live there as well as in the prompt.
2. **Images are sent after the tool results.** A tool message can only carry text, so overlays,
   contact sheets and previews go in a follow-up user message. Without seeing them, the model would be
   choosing crops blind.
3. **Path pinning** (`_call_tool`). The agent code, not the model, decides which file each tool reads
   and writes. Rotation always starts from the original, colour always from the newest crop, and retouch
   always from the colour result. This stopped two early bugs: rotations stacking up, and the model
   writing files all over the disk.
4. **State lives in code**, not in the model's memory: applied angle, valid rectangle, crop, colour
   params. The final result reports what was really applied to the pixels, not what the model claims.
   In one run the model said a crop "keeps the rabbit" when it had cut it in half. Trust the state.
5. **Nudges.** Some models answer with a tutorial ("first, open Photoshop...") instead of calling
   tools. The loop notices that no tool was used and sends up to 2 reminders.
6. **Usage is counted** per run (about 350k input tokens for a full A-D run on one photo, roughly ¥0.09
   with hy4-flash).

## 5. The four stages

### A. Horizon leveling (`horizon.py`)
- `estimate_tilt` finds straight lines (LSD detector). It fits near-horizontal lines and, separately,
  near-vertical lines with a keystone (perspective) model, and returns candidates with their spread and
  perspective slope, plus an overlay image.
- **The model picks which lines to trust**: a sea horizon or a building edge yes; slopes, branches or
  curtains no. This is the judgement code can't do well.
- `apply_rotation` rotates the original and returns the largest clean rectangle (no blank corners).
  `check_level` measures the residual tilt.
- Graded against your own Lightroom edits: `ground_truth.py` matches the RAW to your export with SIFT
  and recovers your rotation. Pass if the error is under 0.5°.

### B. Composition and crop (`composition.py`)
- `propose_crops` scores many candidate rectangles on classical rules (`docs/composition_rules.md`):
  rule of thirds, eye line on the upper third, headroom, lead room, never cutting at joints (using 8
  head-heights figure proportions), horizon on a third, sky:land by texture, clean edges.
- **Prominent objects** (new): a lit lamp, a statue or a sign is found with frequency-tuned saliency
  (people removed with the person segmenter). A crop may keep it whole or leave it out, never slice it.
- The model chooses from a contact sheet; `check_composition` grades the choice.

### C. Global colour (`color.py`)
- The pipeline, in order: white balance → exposure → highlights/shadows (guided-filter local base, which
  avoids halos) → tone curve → 8-band HSL → vibrance/saturation → film LUT → colour grading → grain.
- Looks: hand-written recipes plus 8 Fuji film-emulation LUTs (HaldCLUT).
- **Sample matching (样片追色)**: `match_reference` fits the slider values so the photo takes on a
  sample's statistics: lightness quantiles in stops, midtone a/b, chroma, per-colour HSL. It is
  damped and capped, so a night photo is not forced to daylight.
- **Style memory (风格记忆)**: 11 library styles; 9 samples are yours, 2 come from Wikimedia
  Commons. `suggest_styles` measures cues (night, people, face size, sky, foliage, blossom, snow,
  buildings, brightness) and ranks the styles. `apply_style` applies one. **Only used when the user
  asks for a style.**
- **Face colour limit** (new): every colour tool reports `face_shift`, how far the main face's skin
  moved, in white-balance units. The limit is ±20 temperature and ±20 tint. When it is over, the tool
  gives a `skin_correction` calculated on the face's own pixels, which stage D applies through the skin
  mask. The scene keeps the style and the skin still looks like skin.
- Checks: clipping, crushed shadows, face lightness, skin hue/saturation, "blue sky stays blue", neutral
  cast, face shift.

### D. People (`people.py`)
- `detect_people`: YuNet faces, MediaPipe 478-point face landmarks, pose, and a 6-class segmentation
  (hair, body skin, face skin, clothes...). Masks are snapped to edges with a guided filter.
- `preview_retouch` / `apply_retouch`: relight the whole person (`subject_light`, not just the face),
  frequency-separation skin smoothing that keeps pores, even tone, light face/body slimming with smooth
  warp fields, and local adjustments per region (incl. the skin colour fix from C).
- Checks: no halo around people, face not lit separately (≤ 0.8 EV more than the body), skin texture
  kept, slimming subtle, background not warped.

## 6. Evaluation (`evals/harness.py`)

`python -m evals.harness --agent hy --folder path\to\photos` runs the agent on every photo and grades:
- **Level**: |agent angle − your Lightroom angle| < 0.5°.
- **Crop**: `check_composition` — no hard violation, ≥ 80% of the rule checks pass.
- **Colour**: `check_color` on the final image.
- **Retouch**: the retouch checks, re-measured.

`--rescore` re-grades a saved run against the current rules at no API cost. The harness, not the model,
decides pass/fail. Letting the model grade itself invites *reward hacking*: it learns to say "looks
great". That is also the research question on the board: does a separate critic beat self-evaluation?

## 7. What went wrong, and what each taught

| Problem | Cause | Fix | Lesson |
|---|---|---|---|
| Rotations compounded | model rotated the already-rotated file | always rotate the original; pin paths | keep state in code |
| A vision model never called tools | it wrote tutorials | nudges; harness counts it as a fail | check behaviour, not just output |
| Glow around people | blurred mask edges | guided-filter masks | measure the artefact (halo check) |
| Purple sky | warm WB + highlight recovery | sky check + prompt rule | turn taste into a measurable rule |
| Face lit alone looked pasted | face-only relight | relight the whole person | encode how photographers work |
| Sample matching washed night photos out | matched absolute brightness | match tone in stops, capped | statistics ≠ content |
| Crop sliced the rabbit | no notion of objects | prominent-object rule, hard fail | the model's claims need checking |
| Faces turned cold/green with styles | WB applied to everyone | face-shift limit + masked skin fix | protect what viewers judge first |
| Face sub-agent | too much complexity for the gain | removed | fewer agents is often better |

## 8. How to run

```powershell
.\.venv\Scripts\python.exe agent.py path\to\photo.NEF --instruction "修一下，夜景人像风格"
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m evals.harness --agent hy --folder path\to\photos
```
Settings are in `.env` (`HY_API_KEY`, `HY_BASE_URL`, `HY_MODEL=hy4-flash`). Each run writes
`runs/<photo>_<time>/` with `trace.json` (every model message and tool call), sheets and outputs.
Reading `trace.json` is the best way to learn what the agent actually did.

## 9. Known limits and next steps

- The face detector misses strongly upturned faces (DSC_6905), so the skin fix can't run there; the
  agent should lower the style strength instead.
- Two of your samples (蓝调, 街头夜景) are brightly edited, so the style judge reads them as day. It works
  on real night RAWs because they are dark.
- The agent class is still called `LevelingAgent` from the first day; it now runs all four stages.
- XMP export for Lightroom is on the board but not built yet (output is TIFF + params JSON).
- Ideas: a separate critic model for the subjective check, and a memory that learns from your own
  Lightroom edits.
