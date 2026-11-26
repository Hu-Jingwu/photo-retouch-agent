# AI agent Photoshop — a photo-retouching agent

**The model decides, the tools execute, the rules grade.**
模型做决策，工具执行，规则打分。

A multimodal LLM (any OpenAI-compatible endpoint with function calling and image input; developed
with Tencent Hunyuan `hy4-flash`) edits a photo the way a photographer would: it levels the horizon,
chooses a crop, corrects colour and tone, and retouches people. The model only makes decisions. Every
measurement and pixel operation is done by deterministic Python tools (OpenCV, rawpy, MediaPipe), and
every result is checked by explicit rules.

![Before and after: the agent's edit of an example photo](docs/images/DSC_7302_before_after.jpg)

*Example photo by the project author, used with permission (not covered by the MIT licence).*

## How it works

```
Photo + instruction (+ optional sample photo / style)
        │
        ▼
Orchestrator LLM  ──tool calls──►  A  Leveling      estimate_tilt · apply_rotation · check_level
(agent.py loop)   ◄─JSON+images──  B  Crop          propose_crops · apply_crop
        ▲                          C  Colour        analyze_color · render_adjustments · apply_color
        │                                           match_reference · suggest_styles · apply_style
        │                          D  People        detect_people · preview_retouch · apply_retouch
        │                                           beauty_retouch (美颜, only on request)
        │                          E  Blur          analyze_depth · apply_background_blur (on request)
        │                                │
        └──── issues (2–3 retries) ◄── checks: tilt, composition rules, clipping, skin, sky,
                                         face colour shift, halo, texture, warping
                                                 │ pass
                                                 ▼
                                  16-bit TIFF + JPEG preview + params JSON
```

- **A · Leveling**: line segments with a perspective (keystone) fit; the model chooses which lines
  are truly level or plumb.
- **B · Composition**: crops scored on classical rules: thirds, eye line, headroom, lead room, no cuts
  at joints, horizon placement, and prominent objects kept whole.
- **C · Colour**: a parametric pipeline (white balance, exposure, highlights/shadows, curve, 8-colour
  HSL, film LUTs, grading). Sample matching (样片追色) and a style library (风格记忆) are used only
  when the user asks for a style. The skin colour of faces is limited to ±20 white-balance units.
- **D · People**: MediaPipe masks for relighting the whole person, natural skin smoothing, light
  slimming, and skin-colour correction. Checks catch halos and faces that are lit separately.
  On request, `beauty_retouch` does a 写真 / 网红脸 style face retouch with lipstick (口红) and contour
  (立体) (see below), and `clothes_smooth` softens clothing wrinkles (衣服褶皱) while keeping fabric
  texture and prints.
- **E · Background blur (背景虚化)**: `analyze_depth` first judges the depth structure from defocus
  (how much softer each area already is than the subject: Zhuo & Sim's edge re-blur method) and says
  whether blur fits. `apply_background_blur` then uses a lens-like disc blur that grows with depth,
  keeps the subject's colours out of the blur (no halo) and puts the grain back.

## Beauty retouch (美颜: 写真 / 网红脸)

![Original, the agent's full edit, and a face close-up before and after the 写真 beauty retouch](docs/images/hangzhou_portrait_before_after.jpg)

*"Asian young woman" by Lu Yu, [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/), from
[Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Asian_young_woman.jpg) /
[Flickr](https://www.flickr.com/photos/yulucs/4930631837/). Changes: leveled, cropped, colour-corrected
and face-retouched by this agent at the 写真 (xiezhen) level.*

Instruction: *"Level, crop and correct this portrait, then do a beauty retouch (写真精修)."*
The model rotated the photo 1.34° (it trusted the vertical lines, not the car body), chose a medium
close-up, removed a green cast and called `beauty_retouch` at the 写真 level.

`beauty_retouch` is one deterministic tool. The model only picks the level (natural 自然 / xiezhen 写真 /
wanghong 网红) and what to avoid ("眼睛不要放大"). Python measures each face and fills only the gap to
a target face: this face was already close to the target, so it got only small shape edits (eye_size 13 of 40, nose_size 11 of 35). Guards check
that she is still the same person, pores are kept, the face stays symmetric with human proportions, and
straight background lines stay straight; a failed guard lowers only the slider that caused it. Design:
[docs/beauty_agent_design.md](docs/beauty_agent_design.md).

Full explanations, written for learning: [English report](docs/REPORT_en.md) · [中文报告](docs/REPORT_zh.md).

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts\setup_models.py     # face / pose / segmentation models, ~25 MB
copy .env.example .env                                  # then put your API key in .env
```

## Use

```powershell
# full edit (no style: technical correction only)
.\.venv\Scripts\python.exe agent.py path\to\photo.NEF --instruction "Level, crop, correct colour, retouch people."
# with a style from the library, or a sample photo to match
.\.venv\Scripts\python.exe agent.py photo.NEF --instruction "胶片感 film style"
.\.venv\Scripts\python.exe agent.py photo.NEF --reference my_sample.jpg
# change an earlier edit: only the stages the feedback is about are redone
.\.venv\Scripts\python.exe agent.py --revise runs\photo_20261126_101500 --instruction "脸再亮一点，不要瘦脸"
```

**Revise (多轮修改)**: `--revise` loads the earlier run's state and decisions and shows the model the
current result. The model redoes only what the feedback is about (for example the beauty retouch), and
stages after the one it redid are re-applied with their previous settings, so a colour change never
silently drops the retouch. The result goes to a new folder `runs/<run>_rev1/`.

**Context management**: the model looks at every overlay and contact sheet once. After that, older
tool images are replaced in its context by a one-line note saying what they were (`--keep-images`,
default 1; 0 keeps everything). On the Hangzhou portrait this cut the input tokens of a 12-step run
from 237,828 to 187,839 (-21%), with the same decisions. Most of what is left is fixed: the system
prompt and the 17 tool descriptions (about 8.7k tokens) are sent again on every step. `trace.json`
records `prompt_tokens` per step.

Each run writes `runs/<photo>_<time>/` with every intermediate image and `trace.json`, which records
every model message and tool call. Reading the trace is the best way to see what the agent decided and why.

**Style library**: two openly licensed samples are included; add your own for the other styles, see
[samples/README.md](samples/README.md).

## Use it from Claude Code, ChatGPT (Codex) or WorkBuddy

`mcp_server.py` serves the same tools over MCP (Model Context Protocol). Your AI app's own model
then does the orchestrator's job (the job `hy4-flash` does in `agent.py`), so no Hunyuan API key is
needed. The tools, the checks and the run folder are the same, and the server always feeds each
tool the right intermediate file. The app's model must be able to see images: overlays, contact
sheets and before/after sheets come back as images.

The server has 22 tools: `start_edit`, `resume_edit` (revise an earlier run after feedback), the
agent's 19 tools, and `finish_edit`. `start_edit` returns
the full workflow, so any MCP client can follow it. Clients that support MCP prompts also get it as
the `retouch_workflow` prompt.

First install the requirements (they include the `mcp` package) and run the model setup, as in
[Setup](#setup). In the commands below, replace `C:\path\to\photo-retouch-agent` with your own folder.
On macOS/Linux, use `.venv/bin/python`.

### Claude Code

```powershell
cd C:\path\to\photo-retouch-agent
claude mcp add photo-retouch -- C:\path\to\photo-retouch-agent\.venv\Scripts\python.exe C:\path\to\photo-retouch-agent\mcp_server.py
```

Start `claude` in the project folder and check that `/mcp` lists `photo-retouch`. The repository
includes a skill (`.claude/skills/photo-retouch/SKILL.md`), so you can simply ask:

> 修一下 D:\photos\portrait.jpg：拉平、裁切、调色，然后做写真美颜

### ChatGPT desktop app / Codex CLI

The ChatGPT desktop app (Codex), the Codex CLI and the Codex IDE extension share one MCP
configuration:

```powershell
codex mcp add photo-retouch -- C:\path\to\photo-retouch-agent\.venv\Scripts\python.exe C:\path\to\photo-retouch-agent\mcp_server.py
```

You can also add it to `~/.codex/config.toml` yourself:

```toml
[mcp_servers.photo-retouch]
command = 'C:\path\to\photo-retouch-agent\.venv\Scripts\python.exe'
args = ['C:\path\to\photo-retouch-agent\mcp_server.py']
```

Then ask: *"Use the photo-retouch tools: start_edit on D:\photos\portrait.jpg and follow its workflow.
Level, crop, correct colour, and do a 写真 beauty retouch."*

ChatGPT on the web (chatgpt.com) cannot start local programs. It only connects to remote MCP servers
over HTTPS, in developer mode. That would mean putting this server on the internet, which these
instructions don't cover.

### WorkBuddy

WorkBuddy connects to MCP servers by URL. Start the server in HTTP mode and keep that window open:

```powershell
C:\path\to\photo-retouch-agent\.venv\Scripts\python.exe C:\path\to\photo-retouch-agent\mcp_server.py --http 8765
```

In WorkBuddy, open **Connections → Custom connections → Configure MCP** (this edits
`~/.workbuddy/mcp.json`), add the server, then **Save**:

```json
{
  "mcpServers": {
    "photo-retouch": { "url": "http://127.0.0.1:8765/mcp" }
  }
}
```

Turn it on under **My MCP**, then ask as above. The server listens only on your own computer
(127.0.0.1).

## Evaluate and test

```powershell
.\.venv\Scripts\python.exe -m pytest -q                                   # tool tests (no API calls)
.\.venv\Scripts\python.exe -m evals.harness --agent cv                    # CV-only baseline
.\.venv\Scripts\python.exe -m evals.harness --agent hy --folder photos\   # the LLM agent
```

The harness grades leveling against your own Lightroom edits when you have them (RAW files in
`data/raw`, your exported JPEGs in `data/exports`): it recovers your rotation with SIFT and passes
under 0.5° error. It also grades the crop, colour and retouch with the same rules as above.

## Project layout

| Path | What |
|---|---|
| `agent.py` | system prompt, tool-calling loop, path pinning, final JSON |
| `mcp_server.py` | the same tools as an MCP server for Claude Code, Codex / ChatGPT, WorkBuddy |
| `.claude/skills/photo-retouch/` | Claude Code skill: when and how to use the tools |
| `agent_tools/` | the tools: `horizon`, `composition`, `color`, `styles`, `people`, `face`, `beauty`, `depth`, `registry` |
| `evals/` | evaluation harness and ground truth from Lightroom exports |
| `tests/` | pytest tests |
| `docs/` | composition rules, reports, face-panel study, beauty design |
| `luts/`, `samples/` | film LUTs and style samples, with their licences |

## Licence and credits

Code: MIT (see [LICENSE](LICENSE); third-party content in [NOTICE.md](NOTICE.md)). Film LUTs: Pat David, CC BY-SA 4.0.
Beauty example photo: Lu Yu, CC BY 2.0 (edited). Style samples: see
[samples/SOURCES.md](samples/SOURCES.md). Models: OpenCV Zoo YuNet (MIT) and Google MediaPipe (Apache 2.0),
downloaded by the setup script.
