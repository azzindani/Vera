"""Offline: a "swarm" · one small search per cluster, each with its own quota.

    POOLS=best_pools.json VECS=rounds_vecs.json SWARM=swarm_cache.json DATABASE_URL=... \
        python -I dev_tools/eval/swarm_sim.py

The dense arm today takes the top 20 across the 5 probed clusters, one global order. A
swarm gives every probed cluster its own quota: the best M chunks of each of the P
nearest clusters, whatever the others hold. An answer 88th in the corpus but 3rd in its
own cluster then gets a seat.

POOLS is the engine's own final order (see `tail_sim.py`). Variants keep its top K and
rebuild below it:

* `fill`        the swarm's chunks first, round-robin over clusters in centroid order
                (each cluster's best, then each one's second, ...), then the engine's tail
* `interleave`  one from the engine's tail, one from the swarm, alternately

Also, per missed answer (MISSES from `miss_diag.py`): its rank inside its own cluster and
its cluster's rank, the two numbers a swarm depends on.

Recall@10/25, all, train and holdout, wins and losses at 25 against the engine's list.
`exact_ref` counted at rank 1, as `depth_recall.py` does.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import psycopg

PROBES = (5, 10, 20)
QUOTA = (2, 3, 5)
KEEP = 10


def harvest(cur, cases, vecs) -> dict:
    path = Path(os.environ.get("SWARM", "swarm_cache.json"))
    if path.exists():
        return json.loads(path.read_text("utf-8"))
    out = {}
    for c in cases:
        v = vecs[c["id"]]
        cur.execute("SELECT id FROM clusters ORDER BY centroid <=> %s::halfvec LIMIT %s",
                    (v, max(PROBES)))
        clusters = [r[0] for r in cur.fetchall()]
        per = []
        for cl in clusters:
            cur.execute("SELECT id FROM chunk_vectors WHERE cluster_id = %s"
                        " ORDER BY dense <=> %s::halfvec LIMIT %s", (cl, v, max(QUOTA)))
            per.append([r[0] for r in cur.fetchall()])
        out[c["id"]] = per
    path.write_text(json.dumps(out), "utf-8")
    return out


def round_robin(per: list[list[str]], probes: int, quota: int) -> list[str]:
    out = []
    for depth in range(quota):
        for lst in per[:probes]:
            if depth < len(lst):
                out.append(lst[depth])
    return out


def build(ids: list[str], swarm: list[str], mode: str) -> list[str]:
    head = ids[:KEEP]
    seen = set(head)
    tail = [i for i in ids[KEEP:] if i not in seen]
    extra = [i for i in swarm if i not in seen]
    if mode == "fill":
        merged = extra + tail
    else:
        merged = []
        for a, b in zip(tail, extra):
            merged += [a, b]
        longer = tail if len(tail) > len(extra) else extra
        merged += longer[min(len(tail), len(extra)):]
    return head + list(dict.fromkeys(i for i in merged if i not in seen))


def where_misses(cur, cases, vecs) -> None:
    misses = os.environ.get("MISSES")
    if not misses:
        return
    by_id = {c["id"]: c for c in cases}
    rows = json.loads(Path(misses).read_text("utf-8"))
    inside, cluster = [], []
    for r in rows:
        c, v = by_id.get(r["id"]), vecs.get(r["id"])
        if not c or not v:
            continue
        cur.execute("SELECT cv.cluster_id, cv.dense <=> %s::halfvec FROM chunk_vectors cv"
                    " WHERE cv.id = ANY(%s) ORDER BY 2 LIMIT 1", (v, c["targets"]))
        hit = cur.fetchone()
        if not hit:
            continue
        cl, d = hit
        cur.execute("SELECT count(*) FROM chunk_vectors WHERE cluster_id = %s"
                    " AND dense <=> %s::halfvec < %s", (cl, v, d))
        inside.append(cur.fetchone()[0] + 1)
        cluster.append(r["cluster_rank"])
    for p in PROBES:
        for m in QUOTA:
            n = sum(1 for a, b in zip(inside, cluster) if a <= m and b <= p)
            print(f"  misses a swarm of {p} clusters x {m} would hold: {n} of {len(inside)}")


def main() -> None:
    cur = psycopg.connect(os.environ["DATABASE_URL"]).cursor()
    cases = json.loads(Path(os.environ["POOLS"]).read_text("utf-8"))
    spec = {q["id"]: q for q in json.loads((Path(__file__).parent / "queries.json")
                                           .read_text("utf-8"))["queries"]}
    exact = [q for q in spec.values() if q["type"] == "exact_ref"]
    vecs = json.loads(Path(os.environ["VECS"]).read_text("utf-8"))
    cache = harvest(cur, cases, vecs)
    where_misses(cur, cases, vecs)

    variants = {"engine": [c["ids"] for c in cases]}
    for p in PROBES:
        for m in QUOTA:
            for mode in ("fill", "interleave"):
                variants[f"{p} clusters x {m} · {mode}"] = [
                    build(c["ids"], round_robin(cache[c["id"]], p, m), mode) for c in cases]

    def first(lst, t):
        return next((j + 1 for j, i in enumerate(lst) if i in t), None)

    held = sum(1 for q in exact if q.get("holdout"))
    splits = {s: ([k for k, c in enumerate(cases)
                   if s == "all" or c["holdout"] == (s == "holdout")], ex)
              for s, ex in (("all", len(exact)), ("train", len(exact) - held),
                            ("holdout", held))}

    def score(name, idx, ex):
        n = len(idx) + ex
        rs = [first(variants[name][k][:25], set(cases[k]["targets"])) for k in idx]
        bs = [first(cases[k]["ids"][:25], set(cases[k]["targets"])) for k in idx]
        at = [100 * (sum(1 for r in rs if r and r <= cut) + ex) / n for cut in (10, 25)]
        return at, sum(1 for r, b in zip(rs, bs) if r and not b), \
            sum(1 for r, b in zip(rs, bs) if b and not r)

    print(f"\n{'variant':32} " + "   ".join(f"{s:>22}" for s in splits))
    best, best_train = None, -1.0
    for name in variants:
        cells = []
        for idx, ex in splits.values():
            at, win, loss = score(name, idx, ex)
            cells.append(f"@10 {at[0]:4.1f} @25 {at[1]:4.1f} +{win:<2}-{loss:<2}")
        print(f"{name:32} " + "   ".join(cells))
        t = score(name, *splits["train"])[0][1]
        if name != "engine" and t > best_train:
            best, best_train = name, t
    print(f"\nchosen on train by Recall@25: {best}")


if __name__ == "__main__":
    main()
