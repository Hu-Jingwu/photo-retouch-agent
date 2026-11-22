"""Evaluation harness for horizon leveling.

Ground truth = the rotation you applied to each photo in Lightroom, recovered
from (RAW, exported JPEG) pairs. A case PASSES when the angle the agent actually
applied is within --tolerance (default 0.5 deg) of that ground truth.

    # CV-only baseline (no API calls)
    .venv\\Scripts\\python.exe -m evals.harness --agent cv
    # The HY4 agent
    .venv\\Scripts\\python.exe -m evals.harness --agent hy
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from agent_tools import estimate_tilt, propose_crops
from agent_tools import color as color_mod
from agent_tools import people as people_mod
from agent_tools.color import check_color
from agent_tools.composition import _iou, check_composition
from agent_tools.horizon import largest_rotated_rect
from agent_tools.io_utils import RAW_EXTS
from evals.ground_truth import lightroom_rotation

ROOT = Path(__file__).resolve().parent.parent
GT_CACHE = ROOT / "evals" / "ground_truth.json"
# Put your own photos here (not part of the repository): RAW files in data/raw, and the
# JPEGs you exported from Lightroom after leveling/cropping them (same file names) in data/exports.
# Or pass --raw-dir / --ref-dir, or --folder for unlabelled photos.
DEFAULT_REF_DIRS = [ROOT / "data" / "exports"]
DEFAULT_RAW_DIRS = [ROOT / "data" / "raw"]


def find_cases(ref_dirs: list[Path], raw_dirs: list[Path]) -> list[dict]:
    cases = []
    refs = sorted((p for d in ref_dirs if d.is_dir() for p in d.glob("*.jp*g")), key=lambda p: p.stem)
    for ref in refs:
        raw = next((p for d in raw_dirs for p in d.glob(ref.stem + ".*")
                    if p.suffix.lower() in RAW_EXTS), None)
        if raw:
            cases.append({"id": ref.stem, "raw": str(raw), "ref": str(ref)})
    return cases


OVERRIDES = ROOT / "evals" / "label_overrides.json"


def ground_truth(case: dict) -> dict:
    """Lightroom-derived angle, unless a human-reviewed override exists for this case."""
    overrides = json.loads(OVERRIDES.read_text(encoding="utf-8")) if OVERRIDES.exists() else {}
    lr = _lightroom_truth(case) if case.get("ref") else {}
    if case["id"] in overrides:
        # override replaces the angle; the crop still comes from your Lightroom export
        return {**lr, **overrides[case["id"]], "reliable": True, "source": "override"}
    return {**lr, "source": "lightroom"}


def _lightroom_truth(case: dict) -> dict:
    cache = json.loads(GT_CACHE.read_text(encoding="utf-8")) if GT_CACHE.exists() else {}
    key = case["id"]
    mtime = Path(case["ref"]).stat().st_mtime
    if key not in cache or cache[key].get("ref_mtime") != mtime or "crop_rect" not in cache[key]:
        cache[key] = {**lightroom_rotation(case["raw"], case["ref"]), "ref_mtime": mtime}
        GT_CACHE.write_text(json.dumps(cache, indent=1), encoding="utf-8")
    return cache[key]


def run_cv_baseline(raw: str, run_dir: Path) -> dict:
    """Pure CV policy: take the tool's suggestion if confident; take the top-scored crop.

    To avoid writing a 135 MB rotated TIFF per case, crops are proposed on the
    unrotated RAW inside the valid area the rotation would leave (approximation).
    """
    stem = Path(raw).stem
    est = estimate_tilt(raw, overlay_path=str(run_dir / f"{stem}_overlay.png"))
    s = est["suggestion"]
    rotate = s["confidence"] >= 0.3 and abs(s["correction_deg"]) >= 0.2
    angle = s["correction_deg"] if rotate else 0.0
    W, H = est["image_size"]["w"], est["image_size"]["h"]
    wr, hr = largest_rotated_rect(W, H, angle)
    rw, rh = int(wr) - 2, int(hr) - 2
    valid = {"x": (W - rw) // 2, "y": (H - rh) // 2, "w": rw, "h": rh}
    crops = propose_crops(raw, valid_rect=valid, contact_sheet_path=str(run_dir / f"{stem}_crops.jpg"))
    top = crops["candidates"][0]
    return {"source": raw, "angle_deg": angle, "decision": "rotated" if rotate else "skipped",
            "reference": s["basis"], "confidence": s["confidence"], "flags": est["flags"],
            "crop_rect": top["rect"], "valid_rect": valid, "crop_choice": {"candidate_id": top["id"],
                                                                           "aspect": top["aspect"]}}


def crop_iou(res: dict, gt: dict, image_wh: tuple[int, int] | None = None) -> float | None:
    if "crop_rect" not in gt:
        return None
    r = res.get("crop_rect") or res.get("valid_rect")
    if r is None and image_wh:
        r = {"x": 0, "y": 0, "w": image_wh[0], "h": image_wh[1]}
    if r is None:
        return None
    g = gt["crop_rect"]
    return round(_iou((r["x"], r["y"], r["w"], r["h"]), (g["x"], g["y"], g["w"], g["h"])), 3)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", choices=["cv", "hy"], default="cv")
    ap.add_argument("--tolerance", type=float, default=0.5, help="max level error, degrees")
    ap.add_argument("--rule-score", type=float, default=0.8,
                    help="min share of applicable composition rules the crop must pass")
    ap.add_argument("--ref-dir", type=Path, action="append", help="folder of Lightroom exports; repeatable")
    ap.add_argument("--raw-dir", type=Path, action="append", help="repeatable")
    ap.add_argument("--only", nargs="*", help="case ids to run, e.g. DSC_7293")
    ap.add_argument("--reference", type=Path, help="sample photo (样片) for the agent to match")
    ap.add_argument("--folder", type=Path,
                    help="test every RAW in this folder without labels (level not graded), e.g. TEST-c")
    ap.add_argument("--rescore", type=Path,
                    help="re-score an existing results.json against current labels (no API calls)")
    a = ap.parse_args()

    if a.rescore:
        return rescore(a.rescore, a.tolerance, a.rule_score)

    if a.folder:
        cases = [{"id": p.stem, "raw": str(p), "ref": None} for p in sorted(a.folder.iterdir())
                 if p.suffix.lower() in RAW_EXTS]
    else:
        cases = find_cases(a.ref_dir or DEFAULT_REF_DIRS, a.raw_dir or DEFAULT_RAW_DIRS)
    if a.only:
        cases = [c for c in cases if c["id"] in a.only]
    if not cases:
        raise SystemExit(f"No (RAW, export) pairs found under {a.ref_dir}")

    out_dir = ROOT / "runs" / f"eval_{a.agent}_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    agent = None
    if a.agent == "hy":
        from agent import LevelingAgent
        agent = LevelingAgent()

    rows = []
    for c in cases:
        gt = ground_truth(c) if c.get("ref") else {"angle_deg": None}
        t0 = time.time()
        try:
            if agent:
                res = agent.run(c["raw"], run_dir=out_dir / c["id"],
                                reference=str(a.reference) if a.reference else None)
            else:
                res = run_cv_baseline(c["raw"], out_dir)
            row = grade(c["id"], res, gt, a.tolerance, a.rule_score)
            row["seconds"] = round(time.time() - t0, 1)
        except Exception as exc:
            row = {"id": c["id"], "gt_deg": gt["angle_deg"], "pass": False,
                   "pass_level": False if gt.get("angle_deg") is not None else None,
                   "pass_crop": False, "error": f"{type(exc).__name__}: {exc}"}
            if getattr(exc, "status_code", None) in (401, 402, 403):
                rows.append(row)
                _print_row(row)
                print("Stopping: API key / billing problem; the remaining cases would fail the same way.")
                break
        rows.append(row)
        _print_row(row)

    summary = _summary(rows, a.agent, a.tolerance, a.rule_score)
    print(json.dumps(summary))
    (out_dir / "results.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=1, ensure_ascii=False), encoding="utf-8")


def grade(case_id: str, res: dict, gt: dict, tolerance: float, rule_score: float) -> dict:
    """Level: angle within `tolerance` of the label (skipped when there is no label).
    Crop: classical composition rules. Colour: classical correctness checks on the final image."""
    has_gt = gt.get("angle_deg") is not None
    if res.get("decision") == "unknown" or res.get("tool_calls") == 0:
        # the agent never acted / never reached a decision: that is a failure, not a free pass
        return {"id": case_id, "gt_deg": gt.get("angle_deg"), "agent_deg": res["angle_deg"],
                "abs_error_deg": round(abs(res["angle_deg"] - gt["angle_deg"]), 3) if has_gt else None,
                "pass_level": False, "pass_crop": False, "pass_color": False, "pass": False,
                "failed_rules": [], "error": f"agent did not complete (decision={res.get('decision')}, "
                                             f"tool_calls={res.get('tool_calls')})", "detail": res}
    err = abs(res["angle_deg"] - gt["angle_deg"]) if has_gt else None
    pass_level = err < tolerance if has_gt else None
    # grade the crop on the frame it was cut from: the leveled image if the agent wrote one,
    # otherwise the original (the CV baseline approximates with the unrotated RAW)
    frame = res.get("output_path") or res.get("image") or res.get("source")
    comp = check_composition(frame, res.get("crop_rect"), valid_rect=res.get("valid_rect"),
                             min_score=rule_score)
    # colour is graded on the finished file (retouch > colour render > crop > leveled > original)
    col = check_color(res.get("final_path") or frame)
    # retouch: re-measured here from the files (not taken from the agent's own report)
    ret, pass_retouch = None, None
    if res.get("retouch") and res.get("final_path"):
        src = res["retouch"]["source"]
        rgb_b, ana = people_mod._load(src, 1200)
        rgb_a = color_mod._load_preview(res["final_path"], 1200)
        if rgb_a.shape == rgb_b.shape:
            ret = people_mod.retouch_checks(rgb_b, rgb_a, ana, res["retouch"].get("info", {}))
            pass_retouch = all(c["status"] == "pass" for c in ret)
    ok = [x for x in (pass_level, comp["status"] == "pass", col["status"] == "pass", pass_retouch)
          if x is not None]
    return {"id": case_id, "gt_deg": gt.get("angle_deg"), "gt_reliable": gt.get("reliable"),
            "agent_deg": res["angle_deg"], "abs_error_deg": round(err, 3) if has_gt else None,
            "pass_level": pass_level,
            "rule_score": comp["score"], "failed_rules": comp["failed_rules"],
            "shot_size": comp["shot_size"], "scene": comp["scene"],
            "pass_crop": comp["status"] == "pass",
            "color_score": col["score"], "failed_color_rules": col["failed_rules"],
            "pass_color": col["status"] == "pass",
            "pass_retouch": pass_retouch,
            "failed_retouch_rules": [c["rule"] for c in ret or [] if c["status"] == "fail"],
            "retouch_checks": ret,
            "pass": all(ok),
            # your own crop, for reference only (not graded)
            "iou_vs_your_crop": crop_iou(res, gt),
            "composition": comp, "color": {k: col[k] for k in ("status", "score", "checks")}, "detail": res}


def _print_row(row: dict) -> None:
    pf = lambda v: "n/a " if v is None else ("PASS" if v else "FAIL")
    lvl = (f"gt={row['gt_deg']:+6.2f} agent={row.get('agent_deg', float('nan')):+6.2f}"
           f" err={row['abs_error_deg']:4.2f}" if row.get("abs_error_deg") is not None
           else f"agent={row.get('agent_deg', float('nan')):+6.2f} (no label)        ")
    print(f"{row['id']:<10} level: {lvl} {pf(row.get('pass_level'))}"
          f" | crop {row.get('rule_score', float('nan')):.2f} {pf(row.get('pass_crop'))}"
          f" | colour {row.get('color_score', float('nan')):.2f} {pf(row.get('pass_color'))}"
          f" | retouch {pf(row.get('pass_retouch'))}"
          f"  {', '.join(row.get('failed_rules', []) + row.get('failed_color_rules', []) + row.get('failed_retouch_rules', []))}"
          f"{row.get('error', '')}")


def _summary(rows, agent, tolerance, rule_score) -> dict:
    errs = [r["abs_error_deg"] for r in rows if r.get("abs_error_deg") is not None]
    rs = [r["rule_score"] for r in rows if "rule_score" in r]
    fails: dict[str, int] = {}
    for r in rows:
        for f in r.get("failed_rules", []) + r.get("failed_color_rules", []) + r.get("failed_retouch_rules", []):
            fails[f] = fails.get(f, 0) + 1
    return {"agent": agent, "tolerance_deg": tolerance, "rule_score_threshold": rule_score,
            "cases": len(rows), "passed_all": sum(bool(r["pass"]) for r in rows),
            "passed_level": sum(bool(r.get("pass_level")) for r in rows),
            "level_labelled": sum(r.get("pass_level") is not None for r in rows),
            "passed_crop": sum(bool(r.get("pass_crop")) for r in rows),
            "passed_color": sum(bool(r.get("pass_color")) for r in rows),
            "passed_retouch": sum(bool(r.get("pass_retouch")) for r in rows),
            "retouched": sum(r.get("pass_retouch") is not None for r in rows),
            "pass_rate": round(sum(bool(r["pass"]) for r in rows) / len(rows), 3),
            "mean_abs_error_deg": round(sum(errs) / len(errs), 3) if errs else None,
            "mean_rule_score": round(sum(rs) / len(rs), 3) if rs else None,
            "most_broken_rules": dict(sorted(fails.items(), key=lambda kv: -kv[1]))}


def rescore(results_path: Path, tolerance: float, rule_score: float) -> None:
    """Re-grade a saved run against current labels AND current rules (no API calls)."""
    data = json.loads(results_path.read_text(encoding="utf-8"))
    rows = []
    for r in data["rows"]:
        if "detail" not in r:
            rows.append(r)
            _print_row(r)
            continue
        gt = (ground_truth({"id": r["id"], "ref": ""}) if r["id"] in _override_ids()
              else {"angle_deg": r["gt_deg"], "reliable": r.get("gt_reliable")})
        gt.setdefault("crop_rect", r.get("gt_crop"))
        if gt.get("crop_rect") is None:
            gt.pop("crop_rect")
        row = grade(r["id"], r["detail"], gt, tolerance, rule_score)
        rows.append(row)
        _print_row(row)
    summary = {**_summary(rows, data["summary"].get("agent"), tolerance, rule_score), "rescored": True}
    print(json.dumps(summary))
    results_path.with_name("results_rescored.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=1, ensure_ascii=False), encoding="utf-8")


def _override_ids() -> set[str]:
    return set(json.loads(OVERRIDES.read_text(encoding="utf-8"))) if OVERRIDES.exists() else set()


if __name__ == "__main__":
    main()
