import json

import pytest

pytest.importorskip("mcp")
import mcp_server as M  # noqa: E402
from agent_tools.registry import TOOLS  # noqa: E402


def test_every_registry_tool_is_served_with_image_path_optional():
    for t in TOOLS:
        s = M._schema(t)
        assert "image_path" not in s.get("required", [])
        assert t["input_schema"].get("required") is not s.get("required")   # registry schema untouched
    names = [t.name for t in M.EXTRA_TOOLS]
    assert names == ["start_edit", "resume_edit", "finish_edit"]


def test_tools_need_start_edit_first():
    M.SESSION.clear()
    out = M.call("estimate_tilt", {})
    assert "start_edit" in json.loads(out[0].text)["error"]
    with pytest.raises(RuntimeError):
        M.finish_edit({"decision": "x"})


def test_start_edit_rejects_missing_photo(tmp_path):
    with pytest.raises(FileNotFoundError):
        M.start_edit(str(tmp_path / "nope.jpg"))


def test_workflow_tells_the_client_how_to_finish():
    assert "start_edit" in M.WORKFLOW and "finish_edit" in M.WORKFLOW
    assert M.WORKFLOW.startswith(M.agent.SYSTEM_PROMPT[:40])
