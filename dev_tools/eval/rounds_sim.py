"""Offline: a second search round inside the regulations the first round found.

    POOLS=pools.json DATABASE_URL=... EMBED_ENDPOINT=... python -I dev_tools/eval/rounds_sim.py

Round 1 is the engine's own 60-deep pool (`article_sim.py`, `TOP_K` = `CANDIDATE_POOL`),
final order. The filter keeps the first D regulations it ranks (by first appearance).
Round 2 searches ONLY their operative chunks, with the SAME query vector (no second
embedding): dense distance and an OR'd stemmed text match, ranked inside each
regulation and fused by RRF there. The best M new chunks of each regulation are merged:

* `tail@K`   round 1's top K stays; round-2 chunks follow, regulation by regulation in
             round 1's order; round 1's tail after them
* `floor`    as `tail@K`, admitting a round-2 chunk only if its dense distance is no
             worse than the worst of round 1's top 25 (a relevance gate, ✗ a free pass)
* `swap X`   round 1's top 25 - X, then the X best round-2 chunks (by dense distance,
             across the D regulations), then round 1's tail: the reply stays 25
* `add X`    round 1's top 25 untouched, then the X best round-2 chunks: the reply grows
             to 25 + X, so compare it with round 1 at 25 + X, ✗ at 25
* `dense@K`  round 1's top K stays; below it, round 1's tail and round 2's chunks
             ordered together by dense distance

Reports Recall@5/10/15/20/25, the union ceiling (round 1 + round 2 anywhere), wins and
losses at 25 against round 1 alone, and round 2's SQL time. `exact_ref` cases bypass
routing and are counted at rank 1 (63/63 measured by `e2e.py`), as `depth_recall.py`
does, so the "all" rows compare with the ROADMAP's 604-case table.

VECS caches the question vectors (default rounds_vecs.json), so a rerun embeds nothing.
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
from doc_rescore_sim import embedder  # noqa: E402

CUTS = (5, 10, 15, 20, 25, 30, 35)
DOCS = (3, 5, 8)
PER = (2, 3, 5)
KEEP = (5, 10)
SCAN = 30          # per regulation, per signal, rows considered before fusion
RRF_K = 60

ROUND2 = """
WITH q AS (
    SELECT %(v)s::halfvec AS v,
           NULLIF(array_to_string(tsvector_to_array(to_tsvector('indonesian', %(t)s)),
                                  ' | '), '')::tsquery AS tq
), c AS (
    SELECT id, doc_id, chapter, article, dense <=> q.v AS d,
           COALESCE(ts_rank_cd(tsv, q.tq), 0) AS t
      FROM chunks, q
     WHERE doc_id = ANY(%(docs)s) AND indexable
), r AS (
    SELECT *, row_number() OVER (PARTITION BY doc_id ORDER BY d) AS rd,
              row_number() OVER (PARTITION BY doc_id ORDER BY t DESC, d) AS rt
      FROM c
)
SELECT id, doc_id, chapter, article, d, rd, rt FROM r WHERE rd <= %(n)s OR rt <= %(n)s
"""


def main() -> None:
    cur = psycopg.connect(os.environ["DATABASE_URL"]).cursor()
    cases = json.loads(Path(os.environ["POOLS"]).read_text("utf-8"))
    spec = {q["id"]: q for q in json.loads((Path(__file__).parent / "queries.json")
                                           .read_text("utf-8"))["queries"]}
    exact = [q for q in spec.values() if q["type"] == "exact_ref"]

    vec_path = Path(os.environ.get("VECS", "rounds_vecs.json"))
    vecs = json.loads(vec_path.read_text("utf-8")) if vec_path.exists() else {}
    embed = embedder(cur)
    for c in cases:
        if c["id"] not in vecs:
            vecs[c["id"]] = embed(spec[c["id"]]["query"])
    vec_path.write_text(json.dumps(vecs), "utf-8")

    every = sorted({i for c in cases for i in c["ids"]})
    cur.execute("SELECT id, doc_id FROM chunks WHERE id = ANY(%s)", (every,))
    doc_of = dict(cur.fetchall())

    variants: dict[str, list[list[str]]] = {"round 1": []}
    union_hit: dict[int, list[bool]] = {d: [] for d in DOCS}
    timings: list[float] = []
    for c in cases:
        ids, v = c["ids"], vecs[c["id"]]
        docs = list(dict.fromkeys(doc_of[i] for i in ids if doc_of.get(i)))[: max(DOCS)]
        t0 = time.perf_counter()
        cur.execute(ROUND2, {"v": v, "t": spec[c["id"]]["query"], "docs": docs, "n": SCAN})
        rows = cur.fetchall()
        timings.append((time.perf_counter() - t0) * 1000)

        # rank inside each regulation: RRF of the dense rank and the text rank
        by_doc: dict[str, list[tuple[float, str, float]]] = {}
        for i, doc, ch, ar, d, rd, rt in rows:
            if not operative(ch, ar):
                continue
            by_doc.setdefault(doc, []).append((1 / (RRF_K + rd) + 1 / (RRF_K + rt), i, float(d)))
        for lst in by_doc.values():
            lst.sort(reverse=True)

        cur.execute("SELECT id, dense <=> %s::halfvec FROM chunks WHERE id = ANY(%s)",
                    (v, ids))
        dist1 = {r[0]: float(r[1]) for r in cur.fetchall()}
        floor = max((dist1.get(i, 9.0) for i in ids[:25]), default=9.0)

        variants["round 1"].append(ids)
        for keep in KEEP:  # control: round 1's tail re-sorted by dense distance, no round 2
            tail = sorted((dist1.get(i, 9.0), i) for i in ids[keep:])
            variants.setdefault(f"D0 dense@{keep}", []).append(ids[:keep] + [i for _, i in tail])
        targets = set(c["targets"])
        for nd in DOCS:
            pool2 = {i for doc in docs[:nd] for _, i, _ in by_doc.get(doc, [])}
            union_hit[nd].append(bool(targets & (pool2 | set(ids))))
            for m in PER:
                for keep in KEEP:
                    for gate in (False, True):
                        new = [i for doc in docs[:nd]
                               for _, i, d in by_doc.get(doc, [])[:m]
                               if i not in ids[:keep] and (not gate or d <= floor)]
                        merged = list(dict.fromkeys(ids[:keep] + new + ids[keep:]))
                        name = f"D{nd} M{m} tail@{keep}{' floor' if gate else ''}"
                        variants.setdefault(name, []).append(merged)
                cand = sorted({(d, i) for doc in docs[:nd] for _, i, d in by_doc.get(doc, [])[:m]
                               if i not in ids})
                best = [i for _, i in cand]
                for x in (3, 5, 10):
                    variants.setdefault(f"D{nd} M{m} swap {x}", []).append(
                        list(dict.fromkeys(ids[:25 - x] + best[:x] + ids[25 - x:])))
                    variants.setdefault(f"D{nd} M{m} add {x}", []).append(
                        list(dict.fromkeys(ids[:25] + best[:x] + ids[25:])))
                for keep in KEEP:
                    tail = sorted([(dist1.get(i, 9.0), i) for i in ids[keep:]] + cand)
                    variants.setdefault(f"D{nd} M{m} dense@{keep}", []).append(
                        list(dict.fromkeys(ids[:keep] + [i for _, i in tail])))

    def first(lst, t):
        return next((k + 1 for k, i in enumerate(lst) if i in t), None)

    print(f"round 2 SQL ms: p50 {statistics.median(timings):.0f}  "
          f"p95 {sorted(timings)[int(0.95 * len(timings))]:.0f}  max {max(timings):.0f}  "
          f"(first {max(DOCS)} regulations)")
    def score(name, idx, ex):
        n = len(idx) + ex
        lists = variants[name]
        ranks = [first(lists[k], set(cases[k]["targets"])) for k in idx]
        base = [first(cases[k]["ids"], set(cases[k]["targets"])) for k in idx]
        at = [100 * (sum(1 for r in ranks if r and r <= cut) + ex) / n for cut in CUTS]
        win = sum(1 for r, b in zip(ranks, base) if r and r <= 25 and not (b and b <= 25))
        loss = sum(1 for r, b in zip(ranks, base) if b and b <= 25 and not (r and r <= 25))
        return at, win, loss

    splits = {}
    for split in ("all", "train", "holdout"):
        idx = [k for k, c in enumerate(cases)
               if split == "all" or c["holdout"] == (split == "holdout")]
        held = sum(1 for q in exact if q.get("holdout"))
        ex = {"all": len(exact), "train": len(exact) - held, "holdout": held}[split]
        splits[split] = (idx, ex)
        n = len(idx) + ex
        ceil = "  ".join(f"D{d} {100 * (sum(union_hit[d][k] for k in idx) + ex) / n:.1f}%"
                         for d in DOCS)
        base1 = sum(1 for k in idx if first(cases[k]["ids"], set(cases[k]["targets"])))
        print(f"[{split}] n={n}  ceiling, round 1 alone {100 * (base1 + ex) / n:.1f}%  "
              f"round 1 + round 2: {ceil}")

    # ! Chosen on TRAIN by Recall@25, per family; the holdout row is read once, after.
    idx, ex = splits["train"]
    chosen = ["round 1", "D0 dense@10", "D0 dense@5"]
    for fam in ("swap", "add", "dense@", "tail@"):
        names = [nm for nm in variants if fam in nm and not nm.startswith("D0")]
        chosen.append(max(names, key=lambda nm: (score(nm, idx, ex)[0][4], -len(nm))))
    chosen += [nm for nm in os.environ.get("SHOW", "").split(",") if nm in variants]
    for name in chosen:
        print()
        print(f"  {name}")
        for split, (idx, ex) in splits.items():
            at, win, loss = score(name, idx, ex)
            cells = "  ".join(f"@{cut} {a:.1f}" for cut, a in zip(CUTS, at))
            print(f"    {split:8} {cells}   @25 +{win} -{loss}")

if __name__ == "__main__":
    main()
