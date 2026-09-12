"""Measure each retrieval arm against known answers · the first EVAL.md numbers.

Query set is built automatically from the fixture: a regulation's `about` field
is its subject line, so it makes a fair stand-in for "what a user would ask,"
and the correct answer is known by construction -- any chunk of that regulation.

! This favours the lexical arms, because `about` shares wording with the body.
That bias is identical across arms, so it is still a valid A/B for the one
question being asked here: last-token vs mean pooling.

Metrics: Recall@5 (is a correct chunk in the top 5) and MRR (how high).

Usage:
    python eval_arms.py                 # evaluate current arms
    python eval_arms.py --populate-mean # embed via :8081 into dense_mean first
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from batching import batched  # noqa: E402
from sparse import Bm25Vectorizer  # noqa: E402

PG = os.environ.get(
    "VERA_PG", "host=localhost port=5432 dbname=vera user=vera password=vera"
)
TEI_LAST = "http://localhost:8080"
TEI_MEAN = "http://localhost:8081"
VECTORIZER = (
    Path(__file__).resolve().parents[2] / ".test" / "runs" / "fixture-01.bm25.json"
)

N_QUERIES = 60
PER_ARM = 20
TOP_K = 5
RRF_K = 60


def embed(tei: str, texts):
    body = json.dumps({"inputs": texts, "truncate": False}).encode()
    req = urllib.request.Request(
        f"{tei}/embed", data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)


def lit(v):
    return "[" + ",".join(f"{x:.6g}" for x in v) + "]"


def populate_mean(pg):
    with pg.cursor() as cur:
        cur.execute("ALTER TABLE chunks ADD COLUMN IF NOT EXISTS dense_mean halfvec(1024)")
        pg.commit()
        cur.execute("SELECT id, body FROM chunks WHERE indexable ORDER BY id")
        rows = cur.fetchall()
        skipped: list[str] = []
        print(f"embedding {len(rows):,} chunks with mean pooling...")
        done = 0
        for batch in batched(rows, text_of=lambda r: r[1]):
            vecs = embed(TEI_MEAN, [b for _, b in batch])
            for (cid, _), v in zip(batch, vecs):
                # ! Mean pooling sums hidden states; in fp16 a long document
                # overflows to inf/NaN, which TEI serialises as null. Store
                # NULL rather than a corrupt vector -- a NaN vector ranks
                # arbitrarily and would poison the comparison silently.
                if v is None or any(x is None for x in v):
                    skipped.append(cid)
                    continue
                cur.execute("UPDATE chunks SET dense_mean = %s::halfvec WHERE id = %s",
                            (lit(v), cid))
            done += len(batch)
            print(f"\r  {done:,}/{len(rows):,}", end="", flush=True)
        pg.commit()
    print("\n  dense_mean populated")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--populate-mean", action="store_true")
    args = ap.parse_args()
    import psycopg

    vz = Bm25Vectorizer.load(VECTORIZER)
    pg = psycopg.connect(PG)

    if args.populate_mean:
        populate_mean(pg)

    cur = pg.cursor()
    # Build the query set: regulations with enough chunks to be findable.
    cur.execute(
        """
        SELECT about, regulation_type, regulation_number, year, count(*) n
        FROM chunks WHERE indexable AND about IS NOT NULL AND length(about) > 25
        GROUP BY 1,2,3,4 HAVING count(*) >= 5
        ORDER BY md5(about) LIMIT %s
        """,
        (N_QUERIES,),
    )
    queries = cur.fetchall()
    print(f"query set: {len(queries)} regulations (subject line as query)\n")

    has_mean = bool(cur.execute(
        "SELECT 1 FROM information_schema.columns WHERE table_name='chunks'"
        " AND column_name='dense_mean'").fetchone())

    arms = ["dense_last", "sparse", "tsv"] + (["dense_mean"] if has_mean else [])
    hits = {a: 0 for a in arms}
    mrr = {a: 0.0 for a in arms}
    hits["RRF"] = 0
    mrr["RRF"] = 0.0

    qtexts = [q[0] for q in queries]
    last_vecs, mean_vecs = [], []
    for b in batched(qtexts):
        last_vecs += embed(TEI_LAST, b)
        mean_vecs += embed(TEI_MEAN, b) if has_mean else [None] * len(b)

    for (about, rtype, rnum, yr, _), lv, mv in zip(queries, last_vecs, mean_vecs):
        target = (rtype, rnum, yr)

        def run(sql, params):
            cur.execute(sql, params)
            return [tuple(r[1:]) for r in cur.fetchall()]

        results = {
            "dense_last": run(
                "SELECT id, regulation_type, regulation_number, year FROM chunks"
                " WHERE indexable AND dense IS NOT NULL"
                " ORDER BY dense <=> %s::halfvec LIMIT %s", (lit(lv), PER_ARM)),
            "sparse": run(
                "SELECT id, regulation_type, regulation_number, year FROM chunks"
                " WHERE indexable AND sparse IS NOT NULL"
                " ORDER BY sparse <#> %s::sparsevec LIMIT %s",
                (vz.to_sparsevec(vz.query(about)), PER_ARM)),
            "tsv": run(
                "SELECT id, regulation_type, regulation_number, year FROM chunks"
                " WHERE indexable AND tsv @@ plainto_tsquery('indonesian', %s)"
                " ORDER BY ts_rank(tsv, plainto_tsquery('indonesian', %s)) DESC"
                " LIMIT %s", (about, about, PER_ARM)),
        }
        if has_mean:
            results["dense_mean"] = run(
                "SELECT id, regulation_type, regulation_number, year FROM chunks"
                " WHERE indexable AND dense_mean IS NOT NULL"
                " ORDER BY dense_mean <=> %s::halfvec LIMIT %s", (lit(mv), PER_ARM))

        for arm in arms:
            ranked = results[arm]
            for i, r in enumerate(ranked[:TOP_K], 1):
                if r == target:
                    hits[arm] += 1
                    break
            for i, r in enumerate(ranked, 1):
                if r == target:
                    mrr[arm] += 1.0 / i
                    break

        # RRF over the lexical arms plus the better dense one, by rank only.
        fused: dict[tuple, float] = {}
        for arm in arms:
            for i, r in enumerate(results[arm], 1):
                fused[r] = fused.get(r, 0.0) + 1.0 / (RRF_K + i)
        order = [r for r, _ in sorted(fused.items(), key=lambda kv: -kv[1])]
        if target in order[:TOP_K]:
            hits["RRF"] += 1
        if target in order:
            mrr["RRF"] += 1.0 / (order.index(target) + 1)

    n = len(queries)
    print(f"{'arm':<12} {'Recall@5':>9} {'MRR':>7}")
    print("-" * 30)
    for arm in arms + ["RRF"]:
        print(f"{arm:<12} {hits[arm] / n:>8.1%} {mrr[arm] / n:>7.3f}")
    pg.close()


if __name__ == "__main__":
    main()
