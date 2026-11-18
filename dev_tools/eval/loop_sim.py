"""Offline: N search rounds, each inside the next regulations the ranking points to.

    POOLS=pools.json VECS=rounds_vecs.json DATABASE_URL=... python -I dev_tools/eval/loop_sim.py

Round 1 is the engine's own 60-deep pool, final order (`article_sim.py`). Every later
round takes the next D regulations of the CURRENT ranking that no round has searched
yet, searches their operative chunks with the same query vector (`rounds_sim.ROUND2`:
dense and stemmed text, fused by RRF inside each regulation), and keeps the best M of
each. After each round the ranking is rebuilt the way `rounds_sim.py` chose on train:
round 1's top K stays, everything below is ordered by dense distance.

A round stops the loop for a question when it admits nothing new under the relevance
floor (the worst dense distance in round 1's top 25). Reports, per round: Recall@5/10/25,
the ceiling (target anywhere in what was found), how many questions are still looping,
and the cumulative SQL time. `exact_ref` counted at rank 1, as `depth_recall.py` does.
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time
from pathlib import Path

import psycopg

sys.path.insert(0, str(Path(__file__).parent))
from article_sim import operative  # noqa: E402
from rounds_sim import ROUND2, RRF_K, SCAN  # noqa: E402

ROUNDS = int(os.environ.get("ROUNDS", "6"))
D = int(os.environ.get("D", "3"))
M = int(os.environ.get("M", "2"))
KEEP = int(os.environ.get("KEEP", "10"))
CUTS = (5, 10, 25)


def main() -> None:
    cur = psycopg.connect(os.environ["DATABASE_URL"]).cursor()
    cases = json.loads(Path(os.environ["POOLS"]).read_text("utf-8"))
    spec = {q["id"]: q for q in json.loads((Path(__file__).parent / "queries.json")
                                           .read_text("utf-8"))["queries"]}
    exact = [q for q in spec.values() if q["type"] == "exact_ref"]
    vecs = json.loads(Path(os.environ["VECS"]).read_text("utf-8"))

    every = sorted({i for c in cases for i in c["ids"]})
    cur.execute("SELECT id, doc_id FROM chunks WHERE id = ANY(%s)", (every,))
    doc_of = dict(cur.fetchall())

    # per round r: the ranking, whether the question is still looping, cumulative ms
    ranked = [[[] for _ in cases] for _ in range(ROUNDS)]
    found = [[set() for _ in cases] for _ in range(ROUNDS)]
    alive = [[False] * len(cases) for _ in range(ROUNDS)]
    spent = [[0.0] * len(cases) for _ in range(ROUNDS)]
    for k, c in enumerate(cases):
        ids, v, q = c["ids"], vecs[c["id"]], spec[c["id"]]["query"]
        cur.execute("SELECT id, dense <=> %s::halfvec FROM chunks WHERE id = ANY(%s)", (v, ids))
        dist = {r[0]: float(r[1]) for r in cur.fetchall()}
        floor = max((dist.get(i, 9.0) for i in ids[:25]), default=9.0)
        order, searched, extra, ms, going = list(ids), set(), [], 0.0, True
        for r in range(ROUNDS):
            if r and going:
                docs = [d for d in dict.fromkeys(doc_of.get(i) for i in order)
                        if d and d not in searched][:D]
                searched.update(docs)
                admitted = 0
                if docs:
                    t0 = time.perf_counter()
                    cur.execute(ROUND2, {"v": v, "t": q, "docs": docs, "n": SCAN})
                    rows = cur.fetchall()
                    ms += (time.perf_counter() - t0) * 1000
                    by_doc: dict[str, list[tuple[float, str, float]]] = {}
                    for i, doc, ch, ar, d, rd, rt in rows:
                        doc_of.setdefault(i, doc)
                        if operative(ch, ar):
                            by_doc.setdefault(doc, []).append(
                                (1 / (RRF_K + rd) + 1 / (RRF_K + rt), i, float(d)))
                    for lst in by_doc.values():
                        lst.sort(reverse=True)
                        for _, i, d in lst[:M]:
                            if i not in dist:
                                dist[i] = d
                                extra.append(i)
                                admitted += d <= floor
                going = admitted > 0
                tail = sorted((dist[i], i) for i in list(ids[KEEP:]) + extra)
                order = list(dict.fromkeys(ids[:KEEP] + [i for _, i in tail]))
            ranked[r][k] = order
            found[r][k] = set(ids) | set(extra)
            alive[r][k] = going
            spent[r][k] = ms

    def first(lst, t):
        return next((j + 1 for j, i in enumerate(lst) if i in t), None)

    for split in ("all", "train", "holdout"):
        idx = [k for k, c in enumerate(cases)
               if split == "all" or c["holdout"] == (split == "holdout")]
        held = sum(1 for q in exact if q.get("holdout"))
        ex = {"all": len(exact), "train": len(exact) - held, "holdout": held}[split]
        n = len(idx) + ex
        print(f"[{split}] n={n}  D={D} M={M} keep top {KEEP}")
        for r in range(ROUNDS):
            ranks = [first(ranked[r][k], set(cases[k]["targets"])) for k in idx]
            at = "  ".join(f"@{cut} {100 * (sum(1 for x in ranks if x and x <= cut) + ex) / n:.1f}"
                           for cut in CUTS)
            ceil = 100 * (sum(bool(found[r][k] & set(cases[k]["targets"])) for k in idx) + ex) / n
            ms = [spent[r][k] for k in idx]
            print(f"  round {r + 1}  {at}  ceiling {ceil:.1f}  still looping "
                  f"{sum(alive[r][k] for k in idx)}  SQL ms p50 {statistics.median(ms):.0f} "
                  f"p95 {sorted(ms)[int(0.95 * len(ms))]:.0f}")


if __name__ == "__main__":
    main()
