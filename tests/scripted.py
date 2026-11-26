"""Test helpers: a small synthetic photo and a scripted stand-in for the model (no API calls).

ScriptedModel comes from the research copy of this project: it replies from a script and records
every request, so the real agent loop and real tools run offline."""
import json
from types import SimpleNamespace

import cv2
import numpy as np


def photo(tmp_path, name="photo.png"):
    img = np.full((240, 360, 3), 120, np.uint8)
    cv2.line(img, (0, 120), (359, 120), (255, 255, 255), 2)
    path = tmp_path / name
    cv2.imwrite(str(path), img)
    return path


class ScriptedModel:
    """Stands in for the OpenAI client. Each script item is a list of (tool, args) calls or a final text."""

    def __init__(self, script):
        self.script, self.requests = list(script), []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(json.loads(json.dumps(kwargs["messages"])))
        reply = self.script.pop(0)
        calls = [SimpleNamespace(id=f"call_{len(self.requests)}_{i}", type="function",
                                 function=SimpleNamespace(name=name, arguments=json.dumps(args)))
                 for i, (name, args) in enumerate(reply if isinstance(reply, list) else [])]
        content = reply if isinstance(reply, str) else None

        def dump(exclude_none=True):
            out = {"role": "assistant", "content": content}
            if calls:
                out["tool_calls"] = [{"id": c.id, "type": "function", "function": {
                    "name": c.function.name, "arguments": c.function.arguments}} for c in calls]
            return {k: v for k, v in out.items() if v is not None}

        msg = SimpleNamespace(content=content, tool_calls=calls or None, model_dump=dump)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)],
                               usage=SimpleNamespace(prompt_tokens=100, completion_tokens=10))


def scripted_agent(monkeypatch, script, **kw):
    import agent
    monkeypatch.setenv("HY_API_KEY", "test")
    monkeypatch.setenv("HY_MODEL", "scripted")
    a = agent.LevelingAgent(**kw)
    a.client = ScriptedModel(script)
    return a


FINAL = json.dumps({"decision": "skipped", "angle_deg": 0, "reference": "none", "reason": "level"})
