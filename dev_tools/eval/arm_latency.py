"""Time each arm separately, so the scaling law is measured per arm, ✗ inferred.

`HARDWARE.md` §6 claims dense is flat in `n` while sparse and text are linear.
That is an argument from how each arm is written -- routed scan against global
scan -- and it has never been checked against a corpus of a different size.

Run it against two databases built from the same corpus at different scales
(see `scale_probe.py`) and the ratio between them IS the scaling exponent:

    DATABASE_URL=...dbname=vera2  BM25_VOCAB=... python arm_latency.py
    DATABASE_URL=...dbname=vera5m BM25_VOCAB=... python arm_latency.py

An arm that is 14× the rows and 14× the milliseconds is linear, and confirms the
extrapolation. One that is 14× the rows and 1× the milliseconds is doing what
routing is supposed to do. Anything else means the model is wrong, which is the
useful outcome.

! Times the SQL, ✗ the server. No fusion, no scoring, no embed call -- those are
O(candidate pool) and do not scale with the corpus, so including them would
dilute the one thing being measured.

! Reports the MEDIAN of several runs after a warmup. The first touch of a cold
5M-row table measures the disk, which is worth knowing but is not the arm.
"""

from __future__ import annotations

import json
import math
import os
import statistics
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "dev_tools" / "pre_embed"))

import psycopg  # noqa: E402
from sparse import Bm25Vectorizer  # noqa: E402

QUERIES = Path(__file__).parent / "queries.json"
PG = os.environ["DATABASE_URL"]
EP = os.environ.get("EMBED_ENDPOINT", "http://localhost:8080").rstrip("/")
if not os.environ.get("BM25_VOCAB"):
    sys.exit("set BM25_VOCAB to the vocabulary this corpus was built with")
VOCAB = Path(os.environ["BM25_VOCAB"])

PER_ARM = int(os.environ.get("PER_ARM_K", "20"))
PROBED = int(os.environ.get("CLUSTERS_PROBED", "5"))
REPEATS = int(os.environ.get("REPEATS", "3"))
LIMIT_Q = int(os.environ.get("LIMIT_QUERIES", "12"))
SKIP = {"out_of_domain", "exact_ref"}

# ! Copied from `SearchOps::text`, both branches, because the arm this measures
# is the one the ENGINE runs. An earlier version of this script used the
# `ts_rank` spelling only and reported 3,038 ms against the engine's 404 ms at
# the same scale -- it was measuring the fallback, i.e. `FAILURE_MODES.md` §13
# happening inside the measurement tool.
_Q = ("WITH q AS (SELECT array_to_string("
      "tsvector_to_array(to_tsvector('indonesian', %s)), ' | ')::tsquery AS tq) ")
TEXT_RUM = _Q + ("SELECT id FROM chunks, q WHERE indexable AND tsv @@ q.tq"
                 " ORDER BY tsv <=> q.tq LIMIT %s")
TEXT_RANK = _Q + ("SELECT id FROM chunks, q WHERE indexable AND tsv @@ q.tq"
                  " ORDER BY ts_rank(tsv, q.tq) DESC LIMIT %s")


def embed(text: str) -> list[float]:
    req = urllib.request.Request(
        f"{EP}/embed",
        data=json.dumps({"inputs": text, "truncate": False}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.load(r)[0]


def timed(cur, sql: str, params: tuple) -> float:
    t0 = time.perf_counter()
    cur.execute(sql, params)
    cur.fetchall()
    return (time.perf_counter() - t0) * 1000


def main() -> None:
    cases = [c for c in json.loads(QUERIES.read_text(encoding="utf-8"))["queries"]
             if c["type"] not in SKIP][:LIMIT_Q]
    vz = Bm25Vectorizer.load(VOCAB)
    per_arm: dict[str, list[float]] = {"dense": [], "sparse": [], "text": [],
                                       "route": []}

    with psycopg.connect(PG) as pg, pg.cursor() as cur:
        cur.execute("SELECT EXISTS (SELECT 1 FROM pg_class WHERE relname='chunks_tsv_rum')")
        has_rum = cur.fetchone()[0]
        text_sql = TEXT_RUM if has_rum else TEXT_RANK
        print(f"text arm: {'RUM index-ordered' if has_rum else 'ts_rank FALLBACK'}",
              flush=True)

        cur.execute("SELECT count(*) FROM chunks WHERE indexable")
        rows = cur.fetchone()[0]
        cur.execute("SELECT count(*) FROM clusters")
        kcl = cur.fetchone()[0]
        cur.execute("SELECT pg_size_pretty(pg_database_size(current_database()))")
        size = cur.fetchone()[0]
        print(f"{rows:,} indexable rows · {kcl:,} clusters · {size}", flush=True)

        t0 = time.perf_counter()
        cur.execute("SELECT id, centroid::text FROM clusters ORDER BY id")
        cl, cents = [], []
        for cid, c in cur.fetchall():
            v = [float(x) for x in c.strip("[]").split(",")]
            n = math.sqrt(sum(a * a for a in v)) or 1.0
            cl.append(cid)
            cents.append([a / n for a in v])
        print(f"centroids loaded in {(time.perf_counter() - t0):.1f}s "
              f"({kcl * 1024 * 4 / 1e6:.0f} MB as f32)", flush=True)

        for n, q in enumerate(cases, 1):
            qv = embed(q["query"])
            lit = "[" + ",".join(f"{x:.6g}" for x in qv) + "]"
            nrm = math.sqrt(sum(a * a for a in qv)) or 1.0
            qn = [a / nrm for a in qv]

            # ! Layer 2 itself: the dot products against every centroid. Flat
            # while k is small, linear in n once k ∝ n makes it large.
            t0 = time.perf_counter()
            order = [cid for _, cid in sorted(
                ((sum(a * b for a, b in zip(qn, ct)), cid)
                 for ct, cid in zip(cents, cl)), key=lambda t: -t[0])]
            route_ms = (time.perf_counter() - t0) * 1000

            spv = vz.to_sparsevec(vz.query(q["query"]))
            arms = {
                "dense": ("SELECT id FROM chunks WHERE indexable AND dense IS NOT NULL"
                          " AND cluster_id = ANY(%s)"
                          " ORDER BY dense <=> %s::text::halfvec LIMIT %s",
                          (order[:PROBED], lit, PER_ARM)),
                "sparse": ("SELECT id FROM chunks WHERE indexable AND sparse IS NOT NULL"
                           " ORDER BY sparse <#> %s::text::sparsevec LIMIT %s",
                           (spv, PER_ARM)),
                "text": (text_sql, (q["query"], PER_ARM)),
            }
            for arm, (sql, params) in arms.items():
                timed(cur, sql, params)  # warm
                per_arm[arm].append(
                    statistics.median(timed(cur, sql, params) for _ in range(REPEATS)))
            per_arm["route"].append(route_ms)
            print(f"  {n}/{len(cases)}", flush=True)

    print(f"\n{'arm':<10}{'p50 ms':>10}{'share':>9}")
    med = {a: statistics.median(v) for a, v in per_arm.items()}
    total = sum(med.values())
    for arm in ("route", "dense", "sparse", "text"):
        print(f"{arm:<10}{med[arm]:>10.0f}{med[arm] / total:>8.0%}")
    print(f"{'TOTAL':<10}{total:>10.0f}")
    print("\n! Sequential, as pipeline.rs runs them. Concurrent arms would be"
          "\n  max(), ✗ sum() -- see the note in HARDWARE.md §3.")


if __name__ == "__main__":
    main()
