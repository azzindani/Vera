"""Offline: would an ordering rule inside a regulation fix right-law-wrong-article misses?

    VERA_HTTP=... DATABASE_URL=... BM25_VOCAB=... python dev_tools/eval/article_sim.py

Against an engine started with `TOP_K` = `CANDIDATE_POOL`, so each reply is the whole
scored pool in the engine's own final order. Every variant reorders that order; nothing
is re-retrieved except where a variant says so. Reports Recall@5 / @10 / MRR per variant,
all cases and holdout.

* `shipped`     the engine's order
* `operative`   within each regulation, its operative chunks take its slots first
                (explanation, annex, preamble and closing after), order otherwise kept
* `explained`   an explanation of `Pasal N` in the pool pulls in that article's
                operative chunks (fetched if absent) directly above it
* `both`        explained, then operative
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import psycopg

sys.path.insert(0, str(Path(__file__).parent))
from e2e import HttpServer  # noqa: E402
from run import resolve_targets  # noqa: E402

CACHE = Path(os.environ.get("POOLS", "article_pools.json"))


def operative(chapter: str | None, article: str | None) -> bool:
    c = (chapter or "").upper()
    return (article or "").startswith("Pasal") and not c.startswith(("PENJELASAN", "LAMPIRAN"))


def explanation(chapter: str | None) -> bool:
    return (chapter or "").upper().startswith("PENJELASAN")


def pools(cur) -> list[dict]:
    if CACHE.exists():
        return json.loads(CACHE.read_text("utf-8"))
    srv = HttpServer(os.environ.get("VERA_HTTP", "http://127.0.0.1:8081"))
    spec = json.loads((Path(__file__).parent / "queries.json").read_text("utf-8"))
    out = []
    for q in spec["queries"]:
        if q["type"] in ("out_of_domain", "exact_ref"):
            continue
        targets = resolve_targets(cur, q)
        if not targets:
            continue
        ids = [r["id"] for r in srv.search(q["query"]).get("results", [])]
        if q["type"] == "underspecified" and ids:
            cur.execute("SELECT r.id FROM chunks r, chunks t WHERE r.id = ANY(%s) AND t.id = ANY(%s)"
                        " AND r.dense <=> t.dense < 0.03", (ids, list(targets)))
            targets = targets | {r[0] for r in cur.fetchall()}
        out.append({"id": q["id"], "type": q["type"], "holdout": bool(q.get("holdout")),
                    "targets": sorted(targets), "ids": ids})
    CACHE.write_text(json.dumps(out), "utf-8")
    return out


def main() -> None:
    cur = psycopg.connect(os.environ["DATABASE_URL"]).cursor()
    cases = pools(cur)
    every = sorted({i for c in cases for i in c["ids"]})
    cur.execute("SELECT id, doc_id, chapter, article, seq FROM chunks WHERE id = ANY(%s)", (every,))
    meta = {r[0]: r[1:] for r in cur.fetchall()}

    # operative chunks of each (doc, article) the pools' explanations point at
    keys = {(meta[i][0], meta[i][2]) for i in every
            if i in meta and explanation(meta[i][1]) and re.match(r"Pasal\s", meta[i][2] or "")}
    docs = sorted({d for d, _ in keys})
    cur.execute("SELECT id, doc_id, chapter, article, seq FROM chunks WHERE doc_id = ANY(%s)"
                " AND indexable AND article LIKE 'Pasal %%' AND chapter NOT LIKE 'PENJELASAN%%'"
                " AND chapter NOT LIKE 'LAMPIRAN%%'", (docs,))
    article_of: dict[tuple, list[str]] = {}
    for r in cur.fetchall():
        meta.setdefault(r[0], r[1:])
        if (r[1], r[3]) in keys:
            article_of.setdefault((r[1], r[3]), []).append((r[4], r[0]))
    article_of = {k: [i for _, i in sorted(v)] for k, v in article_of.items()}

    def v_operative(ids):
        out = list(ids)
        by_doc: dict[str, list[int]] = {}
        for k, i in enumerate(ids):
            if i in meta:
                by_doc.setdefault(meta[i][0], []).append(k)
        for slots in by_doc.values():
            members = [ids[k] for k in slots]
            ordered = ([i for i in members if operative(meta[i][1], meta[i][2])]
                       + [i for i in members if not operative(meta[i][1], meta[i][2])])
            for k, i in zip(slots, ordered):
                out[k] = i
        return out

    def v_explained(ids):
        out, seen = [], set()
        for i in ids:
            if i in meta and explanation(meta[i][1]):
                for a in article_of.get((meta[i][0], meta[i][2]), [])[:2]:
                    if a not in seen:
                        out.append(a)
                        seen.add(a)
            if i not in seen:
                out.append(i)
                seen.add(i)
        return out[: len(ids)]

    variants = {"shipped": lambda x: x, "operative": v_operative, "explained": v_explained,
                "both": lambda x: v_operative(v_explained(x))}
    for name, f in variants.items():
        for split in ("all", "holdout"):
            cs = [c for c in cases if split == "all" or c["holdout"]]
            h5 = h10 = 0
            mrr = 0.0
            for c in cs:
                t = set(c["targets"])
                ids = f(c["ids"])
                h5 += any(i in t for i in ids[:5])
                h10 += any(i in t for i in ids[:10])
                mrr += next((1 / (k + 1) for k, i in enumerate(ids) if i in t), 0.0)
            print(f"  {name:10} {split:8} n={len(cs)}  Recall@5 {h5 / len(cs):.1%}  "
                  f"Recall@10 {h10 / len(cs):.1%}  MRR {mrr / len(cs):.3f}")


if __name__ == "__main__":
    main()
