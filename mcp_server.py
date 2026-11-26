"""MCP server: use the retouching tools from Claude Code, Codex / ChatGPT, WorkBuddy or any MCP client.

The client's own model takes the orchestrator's place (the job hy4-flash does in agent.py); the tools,
path pinning and checks are exactly the same.

    python mcp_server.py                 # stdio (Claude Code, Codex CLI / ChatGPT desktop app)
    python mcp_server.py --http 8765     # streamable HTTP at http://127.0.0.1:8765/mcp (WorkBuddy, others)

Workflow for the client model: start_edit -> the tools in order A-D -> finish_edit.
Revisions: resume_edit(run_dir, feedback) -> redo only what the feedback is about -> finish_edit.
The `retouch_workflow` prompt (and start_edit's result) carry the same instructions as agent.py.
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any

import anyio
from mcp import types
from mcp.server.fastmcp import FastMCP

import agent
from agent_tools.registry import TOOLS

ROOT = Path(__file__).resolve().parent

WORKFLOW = agent.SYSTEM_PROMPT + """

=== Running through MCP ===
You are the orchestrator. First call start_edit with the photo path. The server keeps track of every
intermediate file and always feeds each tool the right input, so you may leave image_path out of every
other tool. Follow the steps above. Instead of answering with the final JSON, call finish_edit with that
JSON object as `summary`, then tell the user where the result is (final_path)."""

# one edit at a time per server process
SESSION: dict[str, Any] = {}


def _schema(t: dict) -> dict:
    """Registry schema with image_path optional: the server pins it from the session."""
    s = copy.deepcopy(t["input_schema"])
    if "image_path" in s.get("properties", {}):
        s["properties"]["image_path"]["description"] = "optional: the server picks the right input file"
        s["required"] = [r for r in s.get("required", []) if r != "image_path"]
    return s


EXTRA_TOOLS = [
    types.Tool(name="start_edit", description=(
        "Start editing one photo (JPEG, PNG, TIFF or camera RAW such as .NEF/.DNG). Creates a run folder and "
        "returns the full workflow instructions and a preview of the photo. Call this first."),
        inputSchema={"type": "object", "properties": {
            "image_path": {"type": "string", "description": "absolute path of the photo"},
            "instruction": {"type": "string", "description": "what the user asked for, e.g. '写真 beauty retouch'"}},
            "required": ["image_path"], "additionalProperties": False}),
    types.Tool(name="resume_edit", description=(
        "Revise an earlier edit after user feedback (e.g. '脸再亮一点', '不要瘦脸'). Loads a previous run "
        "folder (runs/<photo>_<time>), returns the previous decisions, the current result and the revision "
        "workflow. Redo only the stages the feedback is about; later stages are re-applied automatically "
        "with their previous settings at finish_edit."),
        inputSchema={"type": "object", "properties": {
            "run_dir": {"type": "string", "description": "the previous run folder (absolute path)"},
            "feedback": {"type": "string", "description": "what the user wants changed"}},
            "required": ["run_dir", "feedback"], "additionalProperties": False}),
    types.Tool(name="finish_edit", description=(
        "Finish the edit: writes trace.json into the run folder and returns final_path (the 16-bit TIFF of the "
        "last applied stage) and what was applied."),
        inputSchema={"type": "object", "properties": {
            "summary": {"type": "object", "description": "the final JSON object described in the workflow"}},
            "required": ["summary"], "additionalProperties": False}),
]


def _content(text: str, images: list[dict]) -> list[types.TextContent | types.ImageContent]:
    out: list = [types.TextContent(type="text", text=text)]
    for part in images:   # agent._image_part: {"image_url": {"url": "data:image/jpeg;base64,..."}}
        url = part["image_url"]["url"]
        out.append(types.ImageContent(type="image", data=url.split(",", 1)[1], mimeType="image/jpeg"))
    return out


def start_edit(image_path: str, instruction: str = "") -> list:
    src = Path(image_path).expanduser().resolve()
    if not src.exists():
        raise FileNotFoundError(f"no such photo: {src}")
    run_dir = ROOT / "runs" / f"{src.stem}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    SESSION.clear()
    SESSION.update(src=src, run_dir=run_dir, state=agent.new_state(), trace=[], instruction=instruction)
    text = json.dumps({"run_dir": str(run_dir), "photo": str(src), "instruction": instruction,
                       "workflow": WORKFLOW, "next": "call estimate_tilt"}, ensure_ascii=False)
    return _content(text, [agent._image_part(src)])


def resume_edit(run_dir: str, feedback: str) -> list:
    prev_dir = Path(run_dir).expanduser().resolve()
    src, state, prev = agent.restore_state(prev_dir)
    new_dir = agent.next_rev_dir(prev_dir)
    new_dir.mkdir(parents=True)
    SESSION.clear()
    SESSION.update(src=src, run_dir=new_dir, state=state, trace=[], instruction=prev.get("instruction", ""),
                   feedback=feedback, revised_from=str(prev_dir))
    before = {k: prev.get(k) for k in ("instruction", "angle_deg", "crop_rect", "crop_choice", "color_params",
                                       "color_choice", "retouch_choice", "beauty", "blur")}
    text = json.dumps({"run_dir": str(new_dir), "photo": str(src), "feedback": feedback,
                       "previous_decisions": before, "workflow": WORKFLOW + agent.REVISE_NOTE,
                       "images": "the original photo, then the current result"}, ensure_ascii=False, default=str)
    imgs = [agent._image_part(src)]
    if agent.final_path(state):
        imgs.append(agent._image_part(state["preview_path"] or agent.final_path(state)))
    return _content(text, imgs)


def finish_edit(summary: dict) -> list:
    if not SESSION:
        raise RuntimeError("no edit in progress: call start_edit first")
    st = SESSION["state"]
    replayed = agent.replay_stale(SESSION["src"], SESSION["run_dir"], st)
    result = {"image": str(SESSION["src"]), "instruction": SESSION["instruction"], "summary": summary,
              "angle_deg": round(st["applied_angle"], 3), "crop_rect": st["crop_rect"],
              "color_params": st["color_params"], "retouch": st["retouch"], "beauty": st["beauty"],
              "blur": st.get("blur"),
              "final_path": agent.final_path(st), "preview_path": st["preview_path"],
              "replayed": replayed, "feedback": SESSION.get("feedback"), "revised_from": SESSION.get("revised_from"),
              "tool_calls": len(SESSION["trace"]), "orchestrator": "MCP client", "run_dir": str(SESSION["run_dir"])}
    (SESSION["run_dir"] / "trace.json").write_text(
        json.dumps({"result": result, "state": st, "trace": SESSION["trace"]}, indent=1, ensure_ascii=False,
                   default=str),
        encoding="utf-8")
    return _content(json.dumps(result, ensure_ascii=False, default=str), [])


def call(name: str, args: dict) -> list:
    if name == "start_edit":
        return start_edit(**args)
    if name == "resume_edit":
        return resume_edit(**args)
    if name == "finish_edit":
        return finish_edit(**args)
    if not SESSION:
        return _content(json.dumps({"error": "call start_edit with the photo first"}), [])
    text, is_err = agent.LevelingAgent._call_tool(name, dict(args), SESSION["src"], SESSION["run_dir"],
                                                  SESSION["state"])
    SESSION["trace"].append({"tool": name, "arguments": args, "result": json.loads(text)})
    images = [] if is_err else agent.LevelingAgent._result_images(name, json.loads(text), SESSION["state"])
    return _content(text, images)


def build(host: str = "127.0.0.1", port: int = 8765) -> FastMCP:
    server = FastMCP("photo-retouch-agent", instructions=(
        "Photo retouching tools: leveling, crop, colour, people and 写真 beauty retouch. Call start_edit "
        "first; its result contains the workflow."), host=host, port=port)
    low = server._mcp_server          # tools come from the registry's JSON schemas, so register them directly

    @low.list_tools()
    async def list_tools() -> list[types.Tool]:
        return EXTRA_TOOLS + [types.Tool(name=t["name"], description=t["description"], inputSchema=_schema(t))
                              for t in TOOLS]

    @low.call_tool()
    async def call_tool(name: str, arguments: dict) -> list:
        return await anyio.to_thread.run_sync(call, name, arguments or {})

    @server.prompt(name="retouch_workflow", description="The orchestrator instructions for editing a photo")
    def retouch_workflow(photo: str = "", request: str = "") -> str:
        return f"{WORKFLOW}\n\n=== Task ===\nPhoto: {photo}\nRequest: {request}"

    return server


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--http", type=int, metavar="PORT", help="serve streamable HTTP on 127.0.0.1:PORT/mcp")
    a = ap.parse_args()
    if a.http:
        build(port=a.http).run(transport="streamable-http")
    else:
        build().run(transport="stdio")


if __name__ == "__main__":
    main()
