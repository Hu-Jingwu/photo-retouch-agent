---
name: photo-retouch
description: Edit a photo with this project's retouching tools (leveling, crop, colour, people relight, 写真 / 网红脸 beauty retouch) through the photo-retouch MCP server. Use when the user asks to edit, retouch, level, crop, colour-correct or beautify (修图, 调色, 美颜, 写真) a photo file.
---

# Photo retouch through the MCP tools

You are the orchestrator: the tools measure and execute, you decide, the rules grade.

1. Make sure the `photo-retouch` MCP server is connected (its tools are named like
   `mcp__photo-retouch__start_edit`). If it is not, tell the user to run, in the project folder:
   `claude mcp add photo-retouch -- .venv/Scripts/python.exe mcp_server.py` (Windows) or
   `claude mcp add photo-retouch -- .venv/bin/python mcp_server.py` (macOS/Linux), then restart.
2. Call `start_edit` with the absolute photo path and the user's request. Its result holds the full
   workflow (A leveling → B crop → C colour → D people → 15b beauty → 15c background blur). Follow it
   step by step. To change an earlier edit after feedback, use `resume_edit` instead of `start_edit`.
3. Look at every image a tool returns (overlays, contact sheets, before/after sheets) before deciding.
   Do not skip a step's check tool, and never guess numbers you can preview.
4. Style and beauty are the user's choice: no style library or look without a style request, and
   `beauty_retouch` only when the user asks for 美颜 / 写真 / 网红脸 (level natural / xiezhen / wanghong);
   background blur only on request, and always `analyze_depth` before `apply_background_blur`.
5. Finish with `finish_edit` (summary = the final JSON from the workflow), then tell the user the
   `final_path` and, in a few lines, what you changed and why.

Do not run `agent.py` for this: it calls a different model (Hunyuan) through the API key in `.env`.
