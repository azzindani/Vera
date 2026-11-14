"""BEIR corpus -> one Markdown file per document, the shape Ravel's pipeline reads.

    python -I dev_tools/eval/beir/sources.py <beir dir> <out dir>

Writes `<out>/docs/<id>.md` (`title text` as one paragraph, the way MTEB and BEIR join
them for a dense model; a `# title` line would become a heading the embedder never sees) and `<out>/ids.json`, which maps the
sha256 of each file's bytes to every BEIR id with those bytes: Ravel's `doc_id` is that
hash, so two BEIR documents with identical text become one Vera document, and either id
counts as found.
"""

import hashlib
import json
import re
import sys
from pathlib import Path

SAFE = re.compile(r"^[A-Za-z0-9_.-]+$")


def main(src: Path, out: Path) -> None:
    docs = out / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    by_hash: dict[str, list[str]] = {}
    n = empty = 0
    with (src / "corpus.jsonl").open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            doc_id, title, text = row["_id"], row.get("title", "").strip(), row["text"].strip()
            if not SAFE.match(doc_id):
                raise SystemExit(f"unsafe id {doc_id!r}")
            if not text and not title:
                empty += 1
                continue
            body = (f"{title} {text}".strip() + "\n").encode("utf-8")
            (docs / f"{doc_id}.md").write_bytes(body)
            by_hash.setdefault(hashlib.sha256(body).hexdigest(), []).append(doc_id)
            n += 1
    (out / "ids.json").write_text(json.dumps(by_hash), encoding="utf-8")
    dup = sum(len(v) - 1 for v in by_hash.values())
    print(f"{src.name}: {n} docs written, {empty} empty skipped, {dup} identical to another")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
