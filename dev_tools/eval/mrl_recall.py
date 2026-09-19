"""Does Matryoshka truncation cost recall? 1024 vs 512 vs 256 dims.

Qwen3-Embedding is trained so the first N dimensions are independently meaningful,
so truncating and renormalising is a pure SQL operation -- no GPU, no re-embedding.
That makes narrower dense columns the cheapest memory win available on a box where
`chunks` (3.3 GB) does not fit the Postgres budget (1 GB) and every arm is a scan.

    DATABASE_URL=... EMBED_ENDPOINT=... python dev_tools/eval/mrl_recall.py

! This exists to make a DROP decision on evidence. The `dense_512` and `dense_256`
columns cost 525 MB between them; either they buy recall worth that, or they are
dead weight in the page cache and should go. Keeping an unmeasured column is the
expensive option, not the safe one.

! The QUERY must be truncated the same way. Comparing a 1024-d query against a
512-d corpus is not a smaller space, it is a different one.

! Flat scan, no routing. Centroids are 1024-d and would need rebuilding per width;
this isolates the cost of truncation itself rather than mixing in a clustering
change. It is therefore the SLOW path by design -- one full scan per width per
query -- and says nothing about served latency.

! EMBED_ENDPOINT must be the provider that embedded the corpus (`EMBEDDING.md` §5).
"""

from __future__ import annotations

import json
import math
import os
import urllib.request
from pathlib import Path

import psycopg

QUERIES = Path(__file__).parent / "queries.json"
PG = os.environ["DATABASE_URL"]
EP = os.environ.get("EMBED_ENDPOINT", "http://localhost:8080").rstrip("/")
KS = (5, 10, 20, 50)
WIDTHS = (("dense", 1024), ("dense_512", 512), ("dense_256", 256))
# ! out_of_domain has no answer to find; exact_ref is served by the identifier
# path and never touches a vector, so neither measures truncation.
SKIP = {"out_of_domain", "exact_ref"}


def embed(text: str) -> list[float]:
    req = urllib.request.Request(
        f"{EP}/embed",
        data=json.dumps({"inputs": text, "truncate": False}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.load(r)[0]


def resolve_targets(cur, q) -> set[str]:
    """Chunk ids counting as correct. Labels name ARTICLES, so any chunk of a
    labelled article is a hit -- mirrors dev_tools/eval/run.py."""
    ids = set(q.get("answer_chunks", []))
    for ref in q.get("answer_articles", []):
        cur.execute(
            "SELECT id FROM chunks WHERE regulation_type=%s AND regulation_number=%s"
            " AND year=%s AND about=%s AND article=%s",
            (ref["regulation_type"], ref["regulation_number"], ref["year"],
             ref["about"], ref["article"]),
        )
        ids |= {r[0] for r in cur.fetchall()}
    return ids


def main() -> None:
    cases = [c for c in json.loads(QUERIES.read_text(encoding="utf-8"))["queries"]
             if c["type"] not in SKIP]
    res = {col: dict.fromkeys(KS, 0) for col, _ in WIDTHS}
    scored = 0

    with psycopg.connect(PG) as pg, pg.cursor() as cur:
        for q in cases:
            tgt = resolve_targets(cur, q)
            if not tgt:
                continue
            scored += 1
            full = embed(q["query"])
            for col, d in WIDTHS:
                v = full[:d]
                n = math.sqrt(sum(x * x for x in v)) or 1.0
                lit = "[" + ",".join(f"{x / n:.6g}" for x in v) + "]"
                cur.execute(
                    f"SELECT id FROM chunks WHERE indexable AND {col} IS NOT NULL"
                    f" ORDER BY {col} <=> %s::text::halfvec LIMIT %s",
                    (lit, max(KS)),
                )
                got = [r[0] for r in cur.fetchall()]
                for k in KS:
                    res[col][k] += bool(set(got[:k]) & tgt)
            print(f"  {q['id']} done", flush=True)

        cur.execute(
            "SELECT pg_size_pretty(sum(pg_column_size(dense))),"
            "       pg_size_pretty(sum(pg_column_size(dense_512))),"
            "       pg_size_pretty(sum(pg_column_size(dense_256)))"
            " FROM chunks WHERE dense IS NOT NULL")
        sizes = cur.fetchone()

    print(f"\n{scored} queries · flat scan · full corpus\n")
    print(f"{'dims':<10}{'stored':>10}" + "".join(f"{'@' + str(k):>9}" for k in KS))
    for (col, d), size in zip(WIDTHS, sizes):
        print(f"{d:<10}{size:>10}"
              + "".join(f"{res[col][k] / scored:>8.1%}" for k in KS))

    base = res["dense"]
    print("\ndelta against 1024:")
    for col, d in WIDTHS[1:]:
        print(f"  {d:>4}: " + "  ".join(
            f"@{k} {(res[col][k] - base[k]) / scored:+.1%}" for k in KS))
    print("\n! A width that loses nothing at the k an agent is handed (20-50) is a"
          "\n  free halving of the dense column. One that loses points is dead"
          "\n  weight in the page cache and should be dropped, ✗ kept 'just in case'.")


if __name__ == "__main__":
    main()
