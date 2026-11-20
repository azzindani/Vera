"""Offline: more probes, deeper arms, and a tail ordered on two signals instead of one.

    POOLS=best_pools.json CACHE=tail_cache.json VECS=rounds_vecs.json \
        VERA_HTTP=... DATABASE_URL=... BM25_VOCAB=... python -I dev_tools/eval/tail_sim.py

POOLS is harvested by `article_sim.pools` against an engine started with `TOP_K` =
`CANDIDATE_POOL` and the configuration being improved (e.g. `DENSE_TAIL_FROM=10 ROUNDS=2`),
so each list is that engine's own final order. Per question, the cache adds:

* the dense distance of every listed chunk,
* the dense arm's top 200 from the 5 nearest clusters and from the 10 nearest,
* `pg_search`'s BM25 top 200 (`indexable` only, as the sparse arm).

Variants keep the engine's top K and rebuild everything below it:

* `dist`          below K by dense distance: what the engine does now (a control)
* `agree k=R`     below K by RRF over the dense-distance rank and the BM25 rank, both
                  taken within the candidates below K; a chunk BM25 did not return
                  ranks after every chunk it did
* `+deep N`       the dense top N (5 probes) and BM25 top N join the candidates below K
* `+probe10`      the dense top 20 from 10 probes joins them (routing)

Recall@10/25 for all, train and holdout, wins and losses at 25 against the engine's list.
`exact_ref` counted at rank 1 (63/63 measured by `e2e.py`), as `depth_recall.py` does.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import psycopg

sys.path.insert(0, str(Path(__file__).parent))
from article_sim import pools  # noqa: E402

CUTS = (10, 25)


def harvest(cur, cases, vecs, spec) -> dict:
    path = Path(os.environ.get("CACHE", "tail_cache.json"))
    if path.exists():
        return json.loads(path.read_text("utf-8"))
    out = {}
    for c in cases:
        v, q = vecs[c["id"]], spec[c["id"]]["query"]
        cur.execute("SELECT id FROM clusters ORDER BY centroid <=> %s::halfvec LIMIT 10", (v,))
        probes = [r[0] for r in cur.fetchall()]
        dense = {}
        for n in (5, 10):
            cur.execute("SELECT id, dense <=> %s::halfvec FROM chunk_vectors"
                        " WHERE cluster_id = ANY(%s) ORDER BY dense <=> %s::halfvec LIMIT 200",
                        (v, probes[:n], v))
            dense[n] = [(i, float(d)) for i, d in cur.fetchall()]
        cur.execute("SELECT s.id FROM (SELECT id, paradedb.score(id) AS sc FROM chunks"
                    " WHERE id @@@ paradedb.match('body', %s) ORDER BY paradedb.score(id) DESC"
                    " LIMIT 400) s JOIN chunks c ON c.id = s.id WHERE c.indexable"
                    " ORDER BY s.sc DESC LIMIT 200", (q,))
        bm25 = [r[0] for r in cur.fetchall()]
        need = list(dict.fromkeys(c["ids"] + bm25))
        cur.execute("SELECT id, dense <=> %s::halfvec FROM chunks"
                    " WHERE id = ANY(%s) AND dense IS NOT NULL", (v, need))
        dist = {i: float(d) for i, d in cur.fetchall()}
        for n in (5, 10):
            dist.update(dict(dense[n]))
        out[c["id"]] = {"dense5": [i for i, _ in dense[5]], "dense10": [i for i, _ in dense[10]],
                        "bm25": bm25, "dist": dist}
    path.write_text(json.dumps(out), "utf-8")
    return out


def rebuild(ids, extra, keep, dist, bm25_rank, mode, k):
    head = ids[:keep]
    seen = set(head)
    tail = [i for i in dict.fromkeys(ids[keep:] + extra) if i not in seen]
    if mode == "dist":
        tail.sort(key=lambda i: dist.get(i, 9.0))
        return head + tail
    by_d = {i: r for r, i in enumerate(sorted(tail, key=lambda i: dist.get(i, 9.0)), 1)}
    by_b = {i: r for r, i in enumerate(sorted(tail, key=lambda i: bm25_rank.get(i, 10**6)), 1)}
    big = len(tail) + 1

    def score(i):
        b = by_b[i] if i in bm25_rank else big
        return 1 / (k + by_d[i]) + 1 / (k + b)
    tail.sort(key=score, reverse=True)
    return head + tail


def main() -> None:
    cur = psycopg.connect(os.environ["DATABASE_URL"]).cursor()
    cases = pools(cur)
    spec = {q["id"]: q for q in json.loads((Path(__file__).parent / "queries.json")
                                           .read_text("utf-8"))["queries"]}
    exact = [q for q in spec.values() if q["type"] == "exact_ref"]
    vecs = json.loads(Path(os.environ["VECS"]).read_text("utf-8"))
    cache = harvest(cur, cases, vecs, spec)

    variants: dict[str, list[list[str]]] = {"engine": [c["ids"] for c in cases]}
    for keep in (5, 10):
        for name, deep, probe10, mode, k in (
            ("dist", 0, False, "dist", 0),
            ("dist +probe10", 0, True, "dist", 0),
            ("agree k=60", 0, False, "agree", 60),
            ("agree k=10", 0, False, "agree", 10),
            ("agree k=60 +deep 50", 50, False, "agree", 60),
            ("agree k=60 +deep 100", 100, False, "agree", 60),
            ("agree k=10 +deep 50", 50, False, "agree", 10),
            ("agree k=60 +deep 50 +probe10", 50, True, "agree", 60),
            ("agree k=10 +deep 50 +probe10", 50, True, "agree", 10),
            ("dist +deep 50 +probe10", 50, True, "dist", 0),
        ):
            label = f"keep {keep} · {name}"
            lists = []
            for c in cases:
                h = cache[c["id"]]
                extra = (h["dense5"][:deep] + h["bm25"][:deep]) if deep else []
                if probe10:
                    extra += h["dense10"][:20]
                rank = {i: r for r, i in enumerate(h["bm25"], 1)}
                lists.append(rebuild(c["ids"], extra, keep, h["dist"], rank, mode, k))
            variants[label] = lists

    def first(lst, t):
        return next((j + 1 for j, i in enumerate(lst) if i in t), None)

    def score(name, idx, ex):
        n = len(idx) + ex
        rs = [first(variants[name][k][:25], set(cases[k]["targets"])) for k in idx]
        bs = [first(cases[k]["ids"][:25], set(cases[k]["targets"])) for k in idx]
        at = [100 * (sum(1 for r in rs if r and r <= cut) + ex) / n for cut in CUTS]
        win = sum(1 for r, b in zip(rs, bs) if r and not b)
        loss = sum(1 for r, b in zip(rs, bs) if b and not r)
        return at, win, loss

    held = sum(1 for q in exact if q.get("holdout"))
    splits = {}
    for split, ex in (("all", len(exact)), ("train", len(exact) - held), ("holdout", held)):
        splits[split] = ([k for k, c in enumerate(cases)
                          if split == "all" or c["holdout"] == (split == "holdout")], ex)
    print(f"{'variant':40} " + "   ".join(f"{s:>22}" for s in splits))
    rows = []
    for name in variants:
        cells = []
        for split, (idx, ex) in splits.items():
            at, win, loss = score(name, idx, ex)
            cells.append(f"@10 {at[0]:4.1f} @25 {at[1]:4.1f} +{win:<2}-{loss:<2}")
        rows.append((score(name, *splits["train"])[0][1], name, cells))
    for _, name, cells in rows:
        print(f"{name:40} " + "   ".join(cells))
    best = max(rows[1:], key=lambda r: r[0])
    print(f"\nchosen on train by Recall@25: {best[1]}")


if __name__ == "__main__":
    main()
