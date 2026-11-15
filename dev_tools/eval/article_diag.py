"""Right law, wrong article: where is the right article, and what outranks it?

    VERA_HTTP=... DATABASE_URL=... BM25_VOCAB=... python dev_tools/eval/article_diag.py

Run against an engine started with `TOP_K` = `CANDIDATE_POOL`, so the reply is the whole
scored pool. For every answerable case (exact_ref excluded) whose labelled regulation is
in the top 5 but whose labelled article is not:

* where the target sits: rank 6-10, rank 11+ in the pool, or not in the pool
* what the same-regulation chunks ranked above it are: definitions (`Pasal 1`), preamble
  or closing, explanation, annex, or another operative article
* how many chunks the regulation has (a long law gives ranking more to choose from)
"""

from __future__ import annotations

import collections
import json
import os
import re
import sys
from pathlib import Path

import psycopg

sys.path.insert(0, str(Path(__file__).parent))
from e2e import HttpServer, regulation_hits  # noqa: E402
from run import resolve_targets  # noqa: E402


def kind(chapter: str | None, article: str | None) -> str:
    c, a = (chapter or "").upper(), (article or "").strip()
    if c.startswith("PENJELASAN"):
        return "explanation"
    if c.startswith("LAMPIRAN"):
        return "annex"
    if re.fullmatch(r"Pasal 1", a):
        return "definitions (Pasal 1)"
    if not a.startswith("Pasal"):
        return f"non-article ({a[:12] or '-'})"
    return "other article"


def main() -> None:
    cur = psycopg.connect(os.environ["DATABASE_URL"]).cursor()
    srv = HttpServer(os.environ.get("VERA_HTTP", "http://127.0.0.1:8081"))
    spec = json.loads((Path(__file__).parent / "queries.json").read_text("utf-8"))
    where = collections.Counter()
    above = collections.Counter()
    sizes = []
    per_type = collections.Counter()
    rows = []
    n = 0
    for q in spec["queries"]:
        if q["type"] in ("out_of_domain", "exact_ref", "underspecified"):
            continue
        targets = resolve_targets(cur, q)
        if not targets:
            continue
        ids = [r["id"] for r in srv.search(q["query"]).get("results", [])]
        if not ids:
            continue
        n += 1
        regs = regulation_hits(cur, q, ids)
        hit5 = any(i in targets for i in ids[:5])
        if hit5 or not any(regs[:5]):
            continue
        per_type[q["type"]] += 1
        rank = next((k for k, i in enumerate(ids) if i in targets), None)
        where["6-10" if rank is not None and rank < 10 else "11+ in pool" if rank is not None
              else "not in pool"] += 1
        stop = rank if rank is not None else len(ids)
        same = [i for k, i in enumerate(ids[:stop]) if regs[k]]
        cur.execute("SELECT id, chapter, article, doc_id FROM chunks WHERE id = ANY(%s)", (same,))
        meta = {r[0]: r[1:] for r in cur.fetchall()}
        kinds = [kind(*meta[i][:2]) for i in same if i in meta]
        for k in set(kinds):
            above[k] += 1
        doc = next((meta[i][2] for i in same if i in meta), None)
        if doc:
            cur.execute("SELECT count(*) FROM chunks WHERE doc_id = %s AND indexable", (doc,))
            sizes.append(cur.fetchone()[0])
        rows.append({"id": q["id"], "type": q["type"], "rank": rank, "above": kinds[:5]})
    Path(os.environ.get("OUT", "article_diag.json")).write_text(json.dumps(rows), "utf-8")
    print(f"answerable (no exact_ref, no underspecified): {n} · right law, wrong article: {len(rows)}")
    print("  target is at:", dict(where))
    print("  same-law chunks above the target, cases with at least one of:", dict(above.most_common()))
    sizes.sort()
    if sizes:
        print(f"  chunks in that regulation: p50 {sizes[len(sizes) // 2]}, p90 {sizes[int(len(sizes) * .9)]}")
    print("  by type:", dict(per_type.most_common()))


if __name__ == "__main__":
    main()
