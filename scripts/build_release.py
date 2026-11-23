"""Copy exactly the publishable files into github_release/ (the git working copy that is pushed).

    python scripts/build_release.py

Uses an allow-list, so private photos, outputs, labels, models and .env can never slip in.
"""
from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "github_release"

FILES = ["README.md", "LICENSE", "NOTICE.md", "requirements.txt", ".env.example", ".gitignore", "agent.py",
         "samples/README.md", "samples/SOURCES.md", "samples/film.jpg", "samples/epic_landscape.jpg",
         "luts/SOURCE_README.md", "luts/profiles.json", "models/.gitkeep"]
GLOBS = ["agent_tools/*.py", "evals/*.py", "tests/*.py", "scripts/*.py", "docs/*.md", "docs/images/*.jpg",
         "luts/*.png"]
NEVER = (".env", ".NEF", ".nef", ".dng", ".tif", ".tiff", "ground_truth.json", "label_overrides.json",
         "PRIVATE_SOURCES.md")


def main() -> None:
    # Clear the old copy but keep .git: github_release/ is the git working copy that gets pushed.
    if OUT.exists():
        for child in OUT.iterdir():
            if child.name == ".git":
                continue
            shutil.rmtree(child) if child.is_dir() else child.unlink()
    paths = [ROOT / f for f in FILES] + [p for g in GLOBS for p in sorted(ROOT.glob(g))]
    total = 0
    for p in paths:
        rel = p.relative_to(ROOT).as_posix()
        if not p.is_file() or rel.endswith(NEVER) or "__pycache__" in rel:
            continue
        dst = OUT / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dst)
        total += p.stat().st_size
        print(f"{p.stat().st_size / 1024:8.0f} KB  {rel}")
    print(f"\n{total / 1e6:.1f} MB in {OUT}")


if __name__ == "__main__":
    main()
