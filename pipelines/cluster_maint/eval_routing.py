"""Does layer-2 routing keep recall while pruning the corpus?

This is the measurement CLAUDE.md §3 rests on: probe ~5 of N clusters instead of
scanning everything, and lose nothing worth keeping.

Two metrics, per EVAL.md §3:

  routing recall   is the flat-scan top-1 inside a probed cluster at all?
                   Separates a ROUTING miss from a RANKING miss -- if this is
                   high but overlap is low, fix fusion; if this is low, fix the
                   clustering or probe more.
  top-10 overlap   how much of the exhaustive result survives pruning.

Usage:
    python eval_routing.py --probe 5
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request

PG = os.environ.get(
    "DATABASE_URL", "host=localhost port=5432 dbname=vera user=vera password=vera"
)
TEI = os.environ.get("EMBED_ENDPOINT", "http://localhost:8080")
N_QUERIES = 40
TOP_K = 10


def embed(texts):
    body = json.dumps({"inputs": texts, "truncate": False}).encode()
    req = urllib.request.Request(
        f"{TEI}/embed", data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)


def lit(v):
    return "[" + ",".join(f"{x:.6g}" for x in v) + "]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", type=int, default=5)
    args = ap.parse_args()
    import psycopg

    pg = psycopg.connect(PG)
    cur = pg.cursor()
    cur.execute("SELECT count(*) FROM clusters")
    n_clusters = cur.fetchone()[0]

    cur.execute(
        "SELECT about FROM chunks WHERE indexable AND about IS NOT NULL"
        " AND length(about) > 30 GROUP BY about ORDER BY md5(about) LIMIT %s",
        (N_QUERIES,))
    queries = [r[0] for r in cur.fetchall()]
    qvecs = embed(queries)

    hit_routing = 0
    overlap_total = 0
    t_flat = t_routed = 0.0

    for q, v in zip(queries, qvecs):
        qv = lit(v)

        t0 = time.time()
        cur.execute(
            "SELECT id FROM chunks WHERE indexable AND dense IS NOT NULL"
            " ORDER BY dense <=> %s::halfvec LIMIT %s", (qv, TOP_K))
        flat = [r[0] for r in cur.fetchall()]
        t_flat += time.time() - t0

        t0 = time.time()
        cur.execute(
            "SELECT id FROM clusters ORDER BY centroid <=> %s::halfvec LIMIT %s",
            (qv, args.probe))
        probed = [r[0] for r in cur.fetchall()]
        cur.execute(
            "SELECT id FROM chunks WHERE indexable AND dense IS NOT NULL"
            " AND cluster_id = ANY(%s) ORDER BY dense <=> %s::halfvec LIMIT %s",
            (probed, qv, TOP_K))
        routed = [r[0] for r in cur.fetchall()]
        t_routed += time.time() - t0

        # Routing recall: could the exhaustive winner have been found at all?
        cur.execute("SELECT cluster_id FROM chunks WHERE id = %s", (flat[0],))
        if cur.fetchone()[0] in probed:
            hit_routing += 1
        overlap_total += len(set(flat) & set(routed)) / TOP_K

    n = len(queries)
    cur.execute("SELECT count(*) FROM chunks WHERE indexable")
    total_rows = cur.fetchone()[0]

    print(f"corpus          {total_rows:,} indexable rows in {n_clusters} clusters")
    print(f"probe           {args.probe}/{n_clusters} clusters "
          f"= {args.probe / n_clusters:.1%} of the corpus touched")
    print()
    print(f"routing recall  {hit_routing / n:.1%}   "
          f"(flat top-1 was inside a probed cluster)")
    print(f"top-{TOP_K} overlap  {overlap_total / n:.1%}   "
          f"(exhaustive results surviving pruning)")
    print()
    print(f"latency flat    {t_flat / n * 1000:7.1f} ms/query")
    print(f"latency routed  {t_routed / n * 1000:7.1f} ms/query   "
          f"speedup {t_flat / t_routed:.2f}x")
    pg.close()


if __name__ == "__main__":
    main()
