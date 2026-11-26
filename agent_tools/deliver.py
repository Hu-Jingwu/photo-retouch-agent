"""Delivery: export qualities, the run index, every stage on request, Lightroom footprints.

Each agent run keeps every stage's result in its run folder (runs/<photo>_<time>/). This module
makes them reachable later:

- export qualities: social (2048 px, 1-2.5 MB JPEG, no metadata, so no GPS), print (full-size JPEG
  95), full (the 16-bit TIFF itself), lightroom (that stage's Camera Raw sidecar);
- the run index runs/index.jsonl (one line per finished run), searched by list_runs;
- get_stage(run, stage, quality): "give me the colour step as a small JPEG" is a lookup, not a model
  decision, so it is a plain function the orchestrator (or an MCP client) calls;
- lightroom footprints: for every applied stage, lightroom/<n>_<stage>/<photo>.xmp holds the edit up
  to that stage. Put one next to the RAW under the RAW's name and Lightroom opens the photo at that
  stage, every slider still editable.

    python -m agent_tools.deliver reindex          # once, for runs made before the index existed
    python -m agent_tools.deliver list [photo]
    python -m agent_tools.deliver get runs/<run> 调色 --quality social
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import cv2

from . import xmp as X
from .io_utils import read_image, to_bgr_u8

ROOT = Path(__file__).resolve().parent.parent
INDEX = ROOT / "runs" / "index.jsonl"

# stage -> (state key of its file, words a user may use)
STAGES = {
    "rotation": ("output_path", ("rotation", "level", "leveled", "拉平", "水平", "旋转")),
    "crop": ("final_path", ("crop", "composition", "裁切", "构图", "裁剪")),
    "color": ("color_path", ("color", "colour", "tone", "调色", "颜色", "色彩")),
    "retouch": ("retouch_path", ("retouch", "people", "relight", "人像", "补光", "褶皱", "衣服")),
    "beauty": ("beauty_path", ("beauty", "face", "美颜", "写真", "精修", "口红", "立体")),
    "blur": ("blur_path", ("blur", "bokeh", "虚化", "背景虚化")),
}
ORDER = list(STAGES)
QUALITIES = ("social", "print", "full", "lightroom")
SOCIAL = {"long_side": 2048, "min_mb": 1.0, "max_mb": 2.5}


def stage_name(word: str) -> str:
    """'调色' / 'colour' / 'color' -> 'color'; 'final' / '最终' -> 'final'."""
    w = word.strip().lower()
    if w in ("final", "last", "最终", "最后", "成片"):
        return "final"
    for name, (_, words) in STAGES.items():
        if w == name or w in words:
            return name
    raise ValueError(f"unknown stage {word!r}; use one of {ORDER + ['final']} (or 拉平 裁切 调色 人像 美颜 虚化 最终)")


def applied_stages(state: dict) -> list[str]:
    return [s for s in ORDER if state.get(STAGES[s][0])]


# --------------------------------------------------------------------------- #
# Export qualities
# --------------------------------------------------------------------------- #
def _encode(img, q):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(q), cv2.IMWRITE_JPEG_OPTIMIZE, 1])
    return buf.tobytes()


def export_image(src_file: str | Path, quality: str, out_path: str | Path) -> dict:
    """Write one stage's picture at an export quality. JPEGs are written without any metadata, so the
    camera's GPS position never leaves with them."""
    src_file, out_path = Path(src_file), Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if quality == "full":
        out_path = out_path.with_suffix(src_file.suffix)
        shutil.copyfile(src_file, out_path)
        return {"path": str(out_path), "quality": quality, "mb": round(out_path.stat().st_size / 1e6, 2),
                "note": "the 16-bit master as rendered by the agent"}
    img = to_bgr_u8(read_image(src_file))
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[..., :3]
    h, w = img.shape[:2]
    if quality == "print":
        data, q = _encode(img, 95), 95
    elif quality == "social":
        long_side = SOCIAL["long_side"]
        if max(h, w) > long_side:
            k = long_side / max(h, w)
            img = cv2.resize(img, (round(w * k), round(h * k)), interpolation=cv2.INTER_AREA)
        lo, hi = SOCIAL["min_mb"] * 1e6, SOCIAL["max_mb"] * 1e6
        q, data = 90, _encode(img, 90)
        while len(data) < lo and q < 98:                  # too small: spend more bits on detail
            q += 2
            data = _encode(img, q)
        while len(data) > hi and q > 78:                  # too big: compress a bit more
            q -= 3
            data = _encode(img, q)
    else:
        raise ValueError(f"quality must be one of {QUALITIES}")
    out_path = out_path.with_suffix(".jpg")
    out_path.write_bytes(data)
    out = {"path": str(out_path), "quality": quality, "jpeg_quality": q, "size": [img.shape[1], img.shape[0]],
           "mb": round(len(data) / 1e6, 2)}
    if quality == "social" and len(data) < SOCIAL["min_mb"] * 1e6:
        out["note"] = "below 1 MB even at JPEG 98: the picture has little fine detail (or is small)"
    return out


# --------------------------------------------------------------------------- #
# Lightroom footprints
# --------------------------------------------------------------------------- #
def state_up_to(state: dict, stage: str) -> dict:
    """The state as it was right after `stage`: later stages removed."""
    keep = ORDER[: ORDER.index(stage) + 1]
    s = dict(state)
    for later in ORDER:
        if later in keep:
            continue
        s[STAGES[later][0]] = None
        s[{"rotation": "applied_angle", "crop": "crop_rect", "color": "color_params", "retouch": "retouch",
           "beauty": "beauty", "blur": "blur"}[later]] = None if later != "rotation" else 0.0
    return s


def write_footprints(src: str | Path, state: dict, run_dir: str | Path) -> dict:
    """<photo>.xmp for the whole edit plus lightroom/<n>_<stage>/<photo>.xmp for every applied stage.
    Failures are reported, never raised: the edit itself is already done."""
    src, run_dir = Path(src), Path(run_dir)
    out: dict = {"stages": {}}
    try:
        out["final"] = X.export_xmp(src, state, run_dir / f"{src.stem}.xmp")
        for i, stage in enumerate(applied_stages(state), 1):
            folder = run_dir / "lightroom" / f"{i}_{stage}"
            folder.mkdir(parents=True, exist_ok=True)
            rep = X.export_xmp(src, state_up_to(state, stage), folder / f"{src.stem}.xmp")
            out["stages"][stage] = rep["xmp_path"]
    except Exception as exc:                                   # noqa: BLE001
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


# --------------------------------------------------------------------------- #
# Run index
# --------------------------------------------------------------------------- #
def record_run(result: dict, state: dict, index: Path | None = None) -> dict:
    index = index or INDEX
    entry = {"run_dir": result.get("run_dir"), "photo": result.get("image"),
             "time": time.strftime("%Y-%m-%d %H:%M"), "instruction": result.get("instruction"),
             "feedback": result.get("feedback"), "revised_from": result.get("revised_from"),
             "stages": applied_stages(state), "final_path": result.get("final_path"),
             "preview_path": result.get("preview_path")}
    index.parent.mkdir(parents=True, exist_ok=True)
    with open(index, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry


def reindex(runs_dir: Path | None = None, index: Path | None = None) -> dict:
    """Rebuild the index from every runs/*/trace.json (for runs made before the index existed)."""
    import agent
    runs_dir = runs_dir or ROOT / "runs"
    index = index or INDEX
    entries = []
    for tr in sorted(runs_dir.glob("*/trace.json"), key=lambda p: p.stat().st_mtime):
        try:
            src, state, r = agent.restore_state(tr.parent)
        except (KeyError, ValueError, json.JSONDecodeError):
            continue
        entries.append({"run_dir": str(tr.parent), "photo": str(src),
                        "time": time.strftime("%Y-%m-%d %H:%M", time.localtime(tr.stat().st_mtime)),
                        "instruction": r.get("instruction"), "feedback": r.get("feedback"),
                        "revised_from": r.get("revised_from"), "stages": applied_stages(state),
                        "final_path": agent.final_path(state), "preview_path": state.get("preview_path")})
    index.parent.mkdir(parents=True, exist_ok=True)
    index.write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in entries), encoding="utf-8")
    return {"indexed": len(entries), "index": str(index)}


def list_runs(photo: str | None = None, limit: int = 10, index: Path | None = None) -> dict:
    index = index or INDEX
    """Finished runs, newest first, optionally only those whose photo name contains `photo`."""
    runs = []
    if index.exists():
        for line in index.read_text(encoding="utf-8").splitlines():
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if photo and photo.lower() not in Path(e.get("photo") or "").name.lower():
                continue
            if e.get("run_dir") and Path(e["run_dir"]).exists():
                runs.append(e)
    runs.reverse()
    return {"runs": runs[:limit], "total": len(runs),
            "note": "stages lists what each run applied; pass run_dir and a stage to get_stage"}


def get_stage(run_dir: str, stage: str = "final", quality: str = "social", next_to_raw: bool = False,
              out_dir: str | None = None) -> dict:
    """One stage of an earlier run at an export quality, or its Lightroom sidecar (quality 'lightroom')."""
    import agent                                              # the state layout lives in agent.py
    if quality not in QUALITIES:
        raise ValueError(f"quality must be one of {QUALITIES}")
    run = Path(run_dir).resolve()
    src, state, _ = agent.restore_state(run)
    done = applied_stages(state)
    if not done:
        raise ValueError(f"run {run.name} has no applied stage")
    name = stage_name(stage)
    if name == "final":
        name = done[-1]
    if name not in done:
        raise ValueError(f"stage {name!r} was not applied in this run; applied: {done}")
    out_dir = Path(out_dir) if out_dir else run / "exports"
    if quality == "lightroom":
        folder = run / "lightroom" / f"{done.index(name) + 1}_{name}"
        folder.mkdir(parents=True, exist_ok=True)
        rep = X.export_xmp(src, state_up_to(state, name), folder / f"{src.stem}.xmp")
        if next_to_raw:
            rep["copied_to"] = str(X.place_next_to_raw(rep["xmp_path"], src))
        rep["how"] = (f"copy it next to {src.name} as {src.stem}.xmp (or use next_to_raw), then in Lightroom: "
                      "Metadata > Read Metadata from File")
        return {"stage": name, "quality": quality, **rep}
    file = state[STAGES[name][0]]
    res = export_image(file, quality, out_dir / f"{src.stem}_{done.index(name) + 1}_{name}_{quality}")
    return {"stage": name, "source": file, **res}


def main() -> None:
    ap = argparse.ArgumentParser(description="Find earlier runs and export any stage")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("reindex")
    l_ = sub.add_parser("list")
    l_.add_argument("photo", nargs="?")
    g = sub.add_parser("get")
    g.add_argument("run_dir")
    g.add_argument("stage", nargs="?", default="final")
    g.add_argument("--quality", default="social", choices=QUALITIES)
    g.add_argument("--next-to-raw", action="store_true")
    a = ap.parse_args()
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    if a.cmd == "reindex":
        res = reindex()
    elif a.cmd == "list":
        res = list_runs(a.photo)
    else:
        res = get_stage(a.run_dir, a.stage, a.quality, a.next_to_raw)
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
