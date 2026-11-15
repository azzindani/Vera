"""Offline: does evidence for a whole regulation rank its chunks better?

    VERA_HTTP=... DATABASE_URL=... POOLS2=pools.json python dev_tools/eval/doc_support_sim.py

Against an engine started with `TOP_K` = `CANDIDATE_POOL` (the reply is the scored pool,
final order). Cached to `POOLS2` with each result's score. For each candidate `i` of
regulation `d` (`doc_id`):

* `sum`    score_i + w * (sum of the other pool scores of d)
* `count`  score_i * (1 + w * (pool chunks of d - 1))
* `max`    score_i + w * (max score of d), i.e. a regulation ranks by its best chunk

then re-sorted. Inside a regulation its members keep the engine's order (operative first,
definitions rule) and only which slots the regulation holds changes. Reports Recall@5,
MRR and right regulation in the top 5, train / holdout, wins and losses. TRAIN picks `w`.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import psycopg

sys.path.insert(0, str(Path(__file__).parent))
from e2e import HttpServer  # noqa: E402
from run import resolve_targets  # noqa: E402

CACHE = Path(os.environ.get("POOLS2", "pools2.json"))


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
        res = srv.search(q["query"]).get("results", [])
        ids = [r["id"] for r in res]
        if q["type"] == "underspecified" and ids:
            cur.execute("SELECT r.id FROM chunks r, chunks t WHERE r.id = ANY(%s) AND t.id = ANY(%s)"
                        " AND r.dense <=> t.dense < 0.03", (ids, list(targets)))
            targets = targets | {r[0] for r in cur.fetchall()}
        out.append({"id": q["id"], "type": q["type"], "holdout": bool(q.get("holdout")),
                    "targets": sorted(targets), "ids": ids,
                    "scores": [float(r.get("score") or 0.0) for r in res]})
    CACHE.write_text(json.dumps(out), "utf-8")
    return out


def rescore(ids, scores, doc, mode, w):
    by: dict[str, list[int]] = {}
    for k, i in enumerate(ids):
        by.setdefault(doc.get(i) or i, []).append(k)
    new = []
    for k, i in enumerate(ids):
        mates = by[doc.get(i) or i]
        if mode == "sum":
            s = scores[k] + w * sum(scores[j] for j in mates if j != k)
        elif mode == "count":
            s = scores[k] * (1 + w * (len(mates) - 1))
        else:
            s = scores[k] + w * max(scores[j] for j in mates)
        new.append(s)
    order = sorted(range(len(ids)), key=lambda k: (-new[k], k))
    # a regulation's slots are where its chunks now sort; its members fill them in the
    # engine's own order
    out = [None] * len(ids)
    slots: dict[str, list[int]] = {}
    for pos, k in enumerate(order):
        slots.setdefault(doc.get(ids[k]) or ids[k], []).append(pos)
    for d, ks in by.items():
        for pos, k in zip(slots[d], ks):
            out[pos] = ids[k]
    return out


def main() -> None:
    cur = psycopg.connect(os.environ["DATABASE_URL"]).cursor()
    cases = pools(cur)
    every = sorted({i for c in cases for i in c["ids"] + c["targets"]})
    cur.execute("SELECT id, doc_id FROM chunks WHERE id = ANY(%s)", (every,))
    doc = dict(cur.fetchall())
    grid = [("none", 0.0)] + [(m, float(w)) for m in ("sum", "count", "max")
                              for w in os.environ.get("W", "0.1,0.25,0.5,1").split(",")]
    for mode, w in grid:
        line = []
        for split in ("train", "holdout"):
            idx = [c for c in cases if (split == "holdout") == c["holdout"]]
            h5 = law5 = win = loss = 0
            mrr = 0.0
            for c in idx:
                t = set(c["targets"])
                tdocs = {doc.get(i) for i in t}
                new = rescore(c["ids"], c["scores"], doc, mode, w) if mode != "none" else c["ids"]
                a = any(i in t for i in new[:5])
                b = any(i in t for i in c["ids"][:5])
                h5 += a
                law5 += any(doc.get(i) in tdocs for i in new[:5])
                win += a and not b
                loss += b and not a
                mrr += next((1 / (r + 1) for r, i in enumerate(new) if i in t), 0.0)
            n = len(idx)
            line.append(f"{split} R@5 {h5 / n:.1%} MRR {mrr / n:.3f} law@5 {law5 / n:.1%}"
                        f" +{win}/-{loss}")
        print(f"  {mode:5} w={w:<4} " + "  |  ".join(line))


if __name__ == "__main__":
    main()
