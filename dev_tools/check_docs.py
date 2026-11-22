"""Every reference to a doc of this repository resolves (`docs/README.md`). Run in CI.

    python dev_tools/check_docs.py

A path `docs/<dir>/X.md` names a file that exists, when a doc of that name exists here at
all (one that does not is Ravel's). A relative markdown link resolves. Every doc is in the
index, and every file of `dev_tools/eval/` is in its README. Exits non-zero, naming each failure.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATH_REF = re.compile(r"(?<![\w/.-])(?:\.\./)*docs/([a-z]+)/([A-Z_]+)\.md")
MD_LINK = re.compile(r"\]\((?!https?:|#|mailto:)([^)\s#]+)(?:#[^)]*)?\)")


def main() -> int:
    listed = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.splitlines()
    files = [
        f for f in listed
        if f.endswith((".md", ".py", ".rs", ".yml", ".yaml", ".toml", ".json", ".sql"))
        and (ROOT / f).is_file()
    ]  # fmt: skip
    here = {p.stem: p.parent.name for p in (ROOT / "docs").glob("*/*.md")}
    failures = []
    for rel in files:
        text = (ROOT / rel).read_text(encoding="utf-8", errors="replace")
        for m in PATH_REF.finditer(text):
            if m[2] in here and here[m[2]] != m[1]:
                failures.append(f"{rel}: docs/{m[1]}/{m[2]}.md is docs/{here[m[2]]}/{m[2]}.md")
        if rel.endswith(".md"):
            for m in MD_LINK.finditer(text):
                if not ((ROOT / rel).parent / m[1]).exists():
                    failures.append(f"{rel}: broken link ({m[1]})")
    index = (ROOT / "docs" / "README.md").read_text(encoding="utf-8")
    for p in sorted((ROOT / "docs").glob("*/*.md")):
        if p.relative_to(ROOT / "docs").as_posix() not in index:
            failures.append(f"docs/README.md does not list {p.relative_to(ROOT).as_posix()}")
    evals = (ROOT / "dev_tools" / "eval" / "README.md").read_text(encoding="utf-8")
    for f in listed:
        if f.startswith("dev_tools/eval/") and f != "dev_tools/eval/README.md":
            name = f.removeprefix("dev_tools/eval/").split("/")[0]
            if f"`{name}" not in evals and f"{name}/" not in evals:
                failures.append(f"dev_tools/eval/README.md does not list {name}")
    print("\n".join(failures) if failures else "docs: every reference resolves")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
