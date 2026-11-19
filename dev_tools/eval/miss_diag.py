"""Why is a missed answer missed? Where it ranks against the WHOLE corpus, per signal.

    CASES=cases.json POOLS=pools.json VECS=vecs.json DATABASE_URL=... \
        python -I dev_tools/eval/miss_diag.py

CASES is `e2e.py`'s `EVAL_CASES_OUT` (per-case `rank`), POOLS the 60-deep pools of
`article_sim.py` (labelled targets, twins included), VECS the question vectors
`rounds_sim.py` cached. For every case whose answer is not in the reply's top 25, the
labelled chunk nearest the question is placed:

* `dense rank`    chunks in the whole corpus closer to the question, plus one (exact,
                  a scan of `chunk_vectors`, ✗ an index)
* `cluster rank`  where its cluster stands among all centroids; the engine probes 5
* `bm25 rank`     its place in `pg_search`'s top 1,000 for the question, or none
* `terms`         the share of the question's stemmed terms its text contains
* `law in pool`   whether any chunk of its regulation is in the engine's 60-deep pool

and sorted into one cause, first match wins:

  routing     dense rank <= 20 but its cluster is not probed: more probes would find it
  dense-deep  dense rank <= 200: the vector sees it, the arm's 20 do not reach it
  bm25-deep   bm25 rank <= 200: the words match, the arm's 20 do not reach it
  vocabulary  terms < 0.5: the question and the answer do not share their words
  far         neither signal places it anywhere near
"""

from __future__ import annotations

import json
import os
import statistics
from collections import Counter
from pathlib import Path

import psycopg

PROBES = 5
ARM = 20


def main() -> None:
    cur = psycopg.connect(os.environ["DATABASE_URL"]).cursor()
    cases = json.loads(Path(os.environ["CASES"]).read_text("utf-8"))
    pools = {c["id"]: c for c in json.loads(Path(os.environ["POOLS"]).read_text("utf-8"))}
    vecs = json.loads(Path(os.environ["VECS"]).read_text("utf-8"))
    spec = {q["id"]: q for q in json.loads((Path(__file__).parent / "queries.json")
                                           .read_text("utf-8"))["queries"]}
    missed = [c for c in cases if c["id"] in pools and not (c["rank"] and c["rank"] <= 25)]
    print(f"missed at 25: {len(missed)} of {len(cases)} "
          f"(holdout {sum(c['holdout'] for c in missed)})")

    rows = []
    for c in missed:
        p, v, q = pools[c["id"]], vecs[c["id"]], spec[c["id"]]["query"]
        cur.execute("SELECT id, dense <=> %s::halfvec, cluster_id, doc_id FROM chunks"
                    " WHERE id = ANY(%s) AND dense IS NOT NULL ORDER BY 2 LIMIT 1",
                    (v, p["targets"]))
        hit = cur.fetchone()
        if not hit:
            continue
        tid, td, tcl, tdoc = hit
        cur.execute("SELECT count(*) FROM chunk_vectors WHERE dense <=> %s::halfvec < %s",
                    (v, td))
        dense_rank = cur.fetchone()[0] + 1
        cur.execute("SELECT count(*) FROM clusters WHERE centroid <=> %s::halfvec <"
                    " (SELECT centroid <=> %s::halfvec FROM clusters WHERE id = %s)",
                    (v, v, tcl))
        cluster_rank = cur.fetchone()[0] + 1
        cur.execute("SELECT id FROM chunks WHERE id @@@ paradedb.match('body', %s)"
                    " ORDER BY paradedb.score(id) DESC LIMIT 1000", (q,))
        bm = [r[0] for r in cur.fetchall()]
        bm25_rank = bm.index(tid) + 1 if tid in bm else None
        cur.execute("SELECT array_length(tsvector_to_array(to_tsvector('indonesian', %s)), 1),"
                    " (SELECT count(*) FROM unnest(tsvector_to_array("
                    "   to_tsvector('indonesian', %s))) t"
                    "  WHERE t = ANY(tsvector_to_array((SELECT tsv FROM chunks WHERE id = %s))))",
                    (q, q, tid))
        nq, shared = cur.fetchone()
        terms = (shared or 0) / max(nq or 1, 1)
        cur.execute("SELECT count(*) FROM chunks WHERE id = ANY(%s) AND doc_id = %s",
                    (p["ids"], tdoc))
        law_in_pool = cur.fetchone()[0] > 0
        if dense_rank <= ARM and cluster_rank > PROBES:
            cause = "routing"
        elif dense_rank <= 200:
            cause = "dense-deep"
        elif bm25_rank and bm25_rank <= 200:
            cause = "bm25-deep"
        elif terms < 0.5:
            cause = "vocabulary"
        else:
            cause = "far"
        rows.append({"id": c["id"], "type": c["type"], "holdout": c["holdout"], "cause": cause,
                     "dense_rank": dense_rank, "cluster_rank": cluster_rank,
                     "bm25_rank": bm25_rank, "terms": round(terms, 2),
                     "law_in_pool": law_in_pool, "engine_rank": c["rank"]})

    for split in ("all", "holdout"):
        rs = [r for r in rows if split == "all" or r["holdout"]]
        print(f"\n[{split}] n={len(rs)}")
        for cause, k in Counter(r["cause"] for r in rs).most_common():
            sub = [r for r in rs if r["cause"] == cause]
            print(f"  {cause:11} {k:3}   law in pool {sum(r['law_in_pool'] for r in sub):3}   "
                  f"median dense rank {statistics.median(r['dense_rank'] for r in sub):>8.0f}   "
                  f"median terms {statistics.median(r['terms'] for r in sub):.2f}   "
                  f"types {dict(Counter(r['type'] for r in sub).most_common(4))}")
        print(f"  answer's cluster probed (rank <= {PROBES}): "
              f"{sum(r['cluster_rank'] <= PROBES for r in rs)} of {len(rs)}")
        print(f"  dense rank <= 100: {sum(r['dense_rank'] <= 100 for r in rs)}   "
              f"<= 1000: {sum(r['dense_rank'] <= 1000 for r in rs)}   "
              f"bm25 top 1000: {sum(bool(r['bm25_rank']) for r in rs)}")
    out = os.environ.get("OUT")
    if out:
        Path(out).write_text(json.dumps(rows, indent=1), "utf-8")


if __name__ == "__main__":
    main()
