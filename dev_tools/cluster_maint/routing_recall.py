"""Routing recall against the LABELLED answers, and what pruning costs.

Sibling of `eval_routing.py`, and deliberately not a replacement. That one asks
whether the flat-scan top-1 lands inside a probed cluster -- a proxy that needs no
labels and can therefore run on any corpus. This one asks whether the **labelled
answer** does, which is stronger evidence and only possible where an eval set exists.

! This could not be measured while the dense arm carried weight 0.0. Layers 2 and 3
exist to prune the dense search, so with dense contributing nothing they pruned
nothing that mattered. It became meaningful on 2026-09-19 (`EMBEDDING.md` §5d).

Three columns, per `EVAL.md` §3:

  routing recall   is a labelled answer in a probed cluster AT ALL? Low here means
                   the clustering or CLUSTERS_PROBED; high here with flat Recall@5
                   means fusion or the candidate caps, ✗ routing.
  dense Recall@5   the same arm restricted to probed clusters. The gap against the
                   flat row IS the price of routing.
  corpus touched   what that price buys, and the whole reason there is no global
                   ANN index.

    DATABASE_URL=... EMBED_ENDPOINT=... python routing_recall.py

! EMBED_ENDPOINT must be the provider that embedded the corpus. Pointing it at a
different implementation compares orthogonal geometry and returns plausible nonsense
rather than failing -- the defect `EMBEDDING.md` §5 is about.
"""

from __future__ import annotations

import json
import math
import os
import urllib.request
from pathlib import Path

import psycopg

ROOT = Path(__file__).resolve().parents[2]
QUERIES = ROOT / "dev_tools" / "eval" / "queries.json"
PG = os.environ["DATABASE_URL"]
EP = os.environ.get("EMBED_ENDPOINT", "http://localhost:8080").rstrip("/")
WIDTHS = (1, 2, 3, 5, 8, 12, 20, 40)
# ! out_of_domain has no answer to route to; exact_ref bypasses routing entirely
# (CLAUDE.md §7.4), so including either would measure something other than routing.
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

    with psycopg.connect(PG) as pg, pg.cursor() as cur:
        cur.execute("SELECT id, centroid::text, row_count FROM clusters ORDER BY id")
        cl_ids, cents, sizes = [], [], []
        for cid, c, n in cur.fetchall():
            v = [float(x) for x in c.strip("[]").split(",")]
            nrm = math.sqrt(sum(a * a for a in v)) or 1.0
            cl_ids.append(cid)
            cents.append([a / nrm for a in v])
            sizes.append(n)
        total = sum(sizes)
        print(f"{len(cl_ids)} clusters · {total:,} rows", flush=True)

        rows, hits_flat = [], 0
        hits_routed = {w: 0 for w in WIDTHS}
        for c in cases:
            tgt = resolve_targets(cur, c)
            if not tgt:
                print(f"! {c['id']} has no resolvable target — skipped")
                continue
            cur.execute("SELECT DISTINCT cluster_id FROM chunks WHERE id = ANY(%s)",
                        (sorted(tgt),))
            want = {r[0] for r in cur.fetchall() if r[0] is not None}

            qv = embed(c["query"])
            lit = "[" + ",".join(f"{x:.6g}" for x in qv) + "]"
            nrm = math.sqrt(sum(a * a for a in qv)) or 1.0
            qn = [a / nrm for a in qv]
            ranked = sorted(
                ((sum(a * b for a, b in zip(qn, ct)), cid, sz)
                 for ct, cid, sz in zip(cents, cl_ids, sizes)),
                key=lambda t: -t[0])
            order = [cid for _, cid, _ in ranked]
            pos = next((i for i, cid in enumerate(order, 1) if cid in want), None)
            rows.append({"qid": c["id"], "rank": pos,
                         "rows": [sum(sz for _, _, sz in ranked[:w]) for w in WIDTHS]})

            cur.execute("SELECT id FROM chunks WHERE indexable AND dense IS NOT NULL"
                        " ORDER BY dense <=> %s::text::halfvec LIMIT 5", (lit,))
            hits_flat += bool({r[0] for r in cur.fetchall()} & tgt)
            for w in WIDTHS:
                cur.execute(
                    "SELECT id FROM chunks WHERE indexable AND dense IS NOT NULL"
                    " AND cluster_id = ANY(%s)"
                    " ORDER BY dense <=> %s::text::halfvec LIMIT 5", (order[:w], lit))
                hits_routed[w] += bool({r[0] for r in cur.fetchall()} & tgt)

    n = len(rows)
    print(f"\n{n} queries\n")
    print(f"{'probed':>8}{'routing recall':>17}{'dense R@5':>12}{'corpus touched':>17}")
    for w in WIDTHS:
        rr = sum(1 for r in rows if r["rank"] and r["rank"] <= w) / n
        touched = sum(r["rows"][WIDTHS.index(w)] for r in rows) / n / total
        print(f"{w:>8}{rr:>16.1%}{hits_routed[w] / n:>11.1%}{touched:>16.1%}")
    print(f"{'flat':>8}{'100.0%':>16}{hits_flat / n:>11.1%}{'100.0%':>16}")

    missed = [r["qid"] for r in rows if not r["rank"] or r["rank"] > 5]
    print(f"\nanswer outside the 5 probed clusters: {len(missed)}/{n}"
          f"{' · ' + ', '.join(missed) if missed else ''}")
    print("! These are what the GLOBAL sparse and text arms exist for "
          "(CLAUDE.md §5.3).")


if __name__ == "__main__":
    main()
