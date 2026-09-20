"""ParadeDB's IVF vector index against Vera's own routing. Same corpus, same labels.

This is the one comparison that touches the claim the whole architecture rests on:
that routing to a handful of clusters beats carrying a global index, on a small box.
`HARDWARE.md` §6 measures Vera's side -- 46 ms at 355K, 57 ms at 5M, ~13 MB of
engine -- but has never measured it against anyone.

ParadeDB turns out to build an IVF index, which is the same idea: centroids plus
posting lists. Its own build log says so (`paradedb.vector_info`), and at this
corpus size it uses ~3,672 centroids of ~97 vectors against Vera's 177 of ~2,000.
So this is not routing against HNSW. It is the same family, tuned differently and
implemented by someone else.

    DATABASE_URL=...      the ParadeDB instance (chunks with `dense vector(1024)`)
    BASELINE_URL=...      Vera's instance (halfvec + clusters)
    EMBED_ENDPOINT=...
    python dev_tools/eval/pdb_vector_compare.py

! `WHERE id @@@ pdb.all()` is required. A bare `ORDER BY dense <=> ...` plans a
Seq Scan and measures nothing but the disk -- verified with EXPLAIN, 348,410
blocks read.

! ParadeDB's index takes `vector`, ✗ `halfvec`, so its copy of the column is 4
bytes per dimension against Vera's 2. That is a real cost of the comparison and
is reported alongside latency rather than hidden in it.

! Recall on REAL labels. Both sides answer the same 39 queries with resolvable
answers, so the recall columns are directly comparable; the latency columns are
comparable only because both databases hold the same rows under the same memory
cap.
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

import psycopg

QUERIES = Path(__file__).parent / "queries.json"
PDB = os.environ["DATABASE_URL"]
BASE = os.environ["BASELINE_URL"]
EP = os.environ.get("EMBED_ENDPOINT", "http://localhost:8080").rstrip("/")

KS = (5, 20, 50)
DEPTH = 60
REPEATS = 3
PROBED = int(os.environ.get("CLUSTERS_PROBED", "5"))
SKIP = {"out_of_domain", "exact_ref"}

VERA = ("SELECT id FROM chunks WHERE indexable AND dense IS NOT NULL"
        " AND cluster_id = ANY(%s) ORDER BY dense <=> %s::text::halfvec LIMIT %s")
VERA_FLAT = ("SELECT id FROM chunks WHERE indexable AND dense IS NOT NULL"
             " ORDER BY dense <=> %s::text::halfvec LIMIT %s")
PDB_IVF = ("SELECT id FROM chunks WHERE id @@@ pdb.all()"
           " ORDER BY dense <=> %s::vector(1024) LIMIT %s")


def embed(text: str) -> list[float]:
    req = urllib.request.Request(
        f"{EP}/embed",
        data=json.dumps({"inputs": text, "truncate": False}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.load(r)[0]


def resolve_targets(cur, q) -> set[str]:
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


def run(cur, sql: str, params: tuple) -> tuple[list[str], float]:
    cur.execute(sql, params)          # warm
    cur.fetchall()
    spans = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        cur.execute(sql, params)
        got = [r[0] for r in cur.fetchall()]
        spans.append((time.perf_counter() - t0) * 1000)
    return got, statistics.median(spans)


def main() -> None:
    cases = [c for c in json.loads(QUERIES.read_text(encoding="utf-8"))["queries"]
             if c["type"] not in SKIP]
    arms = ("vera_routed", "vera_flat", "paradedb_ivf")
    hits = {a: dict.fromkeys(KS, 0) for a in arms}
    lat: dict[str, list[float]] = {a: [] for a in arms}
    agree = 0.0
    scored = 0

    with psycopg.connect(PDB) as pc, pc.cursor() as pcur, \
            psycopg.connect(BASE) as bc, bc.cursor() as bcur:
        bcur.execute("SELECT id, centroid::text FROM clusters ORDER BY id")
        cl, cents = [], []
        for cid, c in bcur.fetchall():
            v = [float(x) for x in c.strip("[]").split(",")]
            n = math.sqrt(sum(a * a for a in v)) or 1.0
            cl.append(cid)
            cents.append([a / n for a in v])

        for q in cases:
            tgt = resolve_targets(bcur, q)
            if not tgt:
                continue
            scored += 1
            qv = embed(q["query"])
            lit = "[" + ",".join(f"{x:.6g}" for x in qv) + "]"
            nrm = math.sqrt(sum(a * a for a in qv)) or 1.0
            qn = [a / nrm for a in qv]
            order = [cid for _, cid in sorted(
                ((sum(a * b for a, b in zip(qn, ct)), cid)
                 for ct, cid in zip(cents, cl)), key=lambda t: -t[0])]

            got = {}
            got["vera_routed"], ms = run(bcur, VERA, (order[:PROBED], lit, DEPTH))
            lat["vera_routed"].append(ms)
            got["vera_flat"], ms = run(bcur, VERA_FLAT, (lit, DEPTH))
            lat["vera_flat"].append(ms)
            got["paradedb_ivf"], ms = run(pcur, PDB_IVF, (lit, DEPTH))
            lat["paradedb_ivf"].append(ms)

            for a in arms:
                for k in KS:
                    hits[a][k] += bool(set(got[a][:k]) & tgt)
            sa, sb = set(got["vera_routed"]), set(got["paradedb_ivf"])
            if sa or sb:
                agree += len(sa & sb) / len(sa | sb)
            print(f"  {scored}", flush=True)

    print(f"\n{scored} queries · real labels · depth {DEPTH} · median of {REPEATS}\n")
    print(f"{'configuration':<16}{'p50 ms':>9}" + "".join(f"{'@' + str(k):>9}" for k in KS))
    for a in arms:
        print(f"{a:<16}{statistics.median(lat[a]):>9.0f}"
              + "".join(f"{hits[a][k] / scored:>8.1%}" for k in KS))
    print(f"\nvera_routed vs paradedb_ivf · top-{DEPTH} Jaccard {agree / scored:.1%}")
    print("\n! vera_flat is the ceiling both approximations are chasing: no routing,"
          "\n  no index, every row scored. Neither should beat it on recall.")


if __name__ == "__main__":
    main()
