"""What does `CLUSTERS_PROBED` actually buy? Recall and latency at every width.

`HARDWARE.md` §6b put a number on the cost of routing for the first time:

    routed (probe 5)     49 ms    71.8% @20
    flat scan           948 ms    82.1% @20

Ten points, for 899 ms. `CLUSTERS_PROBED=5` was fitted against Recall@5 alone
(`EVAL.md` §5), and §4b warns that k=5 is the eval's strictness knob rather than
the operating point -- an agent is handed 20-50. So the dial may be set for a
metric nobody serves.

This sweeps it. Both arms of the trade, at the k a caller actually sees:

    DATABASE_URL=... EMBED_ENDPOINT=... python dev_tools/eval/probe_sweep.py

! Dense arm only, no fusion. Fusion would blur the thing being measured -- the
other two arms are global and do not move with probe width, so any change here
is routing's alone.

! Latency is the SQL, ✗ the server: `cluster_id = ANY($1)` over the probed set,
which is what `dense_arm` issues per window. The engine's own figure adds embed
and the global arms on top.

! The 177-cluster row IS the flat scan -- probing every cluster is scanning
everything, and it is the ceiling the rest are chasing.
"""

from __future__ import annotations

import json
import math
import os
import statistics
import time
import urllib.request
from pathlib import Path

import psycopg

QUERIES = Path(__file__).parent / "queries.json"
PG = os.environ["DATABASE_URL"]
EP = os.environ.get("EMBED_ENDPOINT", "http://localhost:8080").rstrip("/")

KS = (5, 20, 50)
DEPTH = 60
REPEATS = int(os.environ.get("REPEATS", "2"))
# ! Every width costs a full pass over the eval set, so the default stops at 45.
# The wide end is already known and does not need re-measuring per run: probing
# all 177 IS the flat scan, 948 ms and 82.1% @20 (`HARDWARE.md` §6b). Override
# with WIDTHS=... to see it anyway.
WIDTHS = tuple(int(w) for w in
               os.environ.get("WIDTHS", "1,3,5,8,12,20,30,45").split(","))
SKIP = {"out_of_domain", "exact_ref"}

SQL = ("SELECT id FROM chunks WHERE indexable AND dense IS NOT NULL"
       " AND cluster_id = ANY(%s) ORDER BY dense <=> %s::text::halfvec LIMIT %s")


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


def main() -> None:
    cases = [c for c in json.loads(QUERIES.read_text(encoding="utf-8"))["queries"]
             if c["type"] not in SKIP]
    hits = {w: dict.fromkeys(KS, 0) for w in WIDTHS}
    lat: dict[int, list[float]] = {w: [] for w in WIDTHS}
    touched = {w: 0 for w in WIDTHS}
    scored = 0

    with psycopg.connect(PG) as pg, pg.cursor() as cur:
        cur.execute("SELECT id, centroid::text, row_count FROM clusters ORDER BY id")
        cl, cents, sizes = [], [], []
        for cid, c, n in cur.fetchall():
            v = [float(x) for x in c.strip("[]").split(",")]
            nrm = math.sqrt(sum(a * a for a in v)) or 1.0
            cl.append(cid)
            cents.append([a / nrm for a in v])
            sizes.append(n or 0)
        total = sum(sizes) or 1
        print(f"{len(cl)} clusters · {total:,} rows", flush=True)

        for q in cases:
            tgt = resolve_targets(cur, q)
            if not tgt:
                continue
            scored += 1
            qv = embed(q["query"])
            lit = "[" + ",".join(f"{x:.6g}" for x in qv) + "]"
            nrm = math.sqrt(sum(a * a for a in qv)) or 1.0
            qn = [a / nrm for a in qv]
            ranked = sorted(
                ((sum(a * b for a, b in zip(qn, ct)), cid, sz)
                 for ct, cid, sz in zip(cents, cl, sizes)), key=lambda t: -t[0])
            order = [cid for _, cid, _ in ranked]

            for w in WIDTHS:
                probe = order[:w]
                cur.execute(SQL, (probe, lit, DEPTH))
                cur.fetchall()                      # warm
                spans = []
                for _ in range(REPEATS):
                    t0 = time.perf_counter()
                    cur.execute(SQL, (probe, lit, DEPTH))
                    got = [r[0] for r in cur.fetchall()]
                    spans.append((time.perf_counter() - t0) * 1000)
                lat[w].append(statistics.median(spans))
                touched[w] += sum(sz for _, _, sz in ranked[:w]) / total
                for k in KS:
                    hits[w][k] += bool(set(got[:k]) & tgt)
            print(f"  {scored}", flush=True)

    print(f"\n{scored} queries · dense arm only · depth {DEPTH}\n")
    print(f"{'probed':>7}{'p50 ms':>9}{'corpus':>9}"
          + "".join(f"{'@' + str(k):>9}" for k in KS))
    for w in WIDTHS:
        print(f"{w:>7}{statistics.median(lat[w]):>9.0f}{touched[w] / scored:>8.1%}"
              + "".join(f"{hits[w][k] / scored:>8.1%}" for k in KS))

    base = hits[5] if 5 in hits else hits[WIDTHS[0]]
    ceiling = hits[WIDTHS[-1]]
    print(f"\nagainst the shipped CLUSTERS_PROBED=5")
    for w in WIDTHS:
        if w == 5:
            continue
        print(f"  {w:>3}: "
              + "  ".join(f"@{k} {(hits[w][k] - base[k]) / scored:+.1%}" for k in KS)
              + f"   {statistics.median(lat[w]) - statistics.median(lat[5]):+.0f} ms")
    print(f"\nceiling (probe all) is @20 {ceiling[20] / scored:.1%}; "
          f"5 leaves {(ceiling[20] - base[20]) / scored:.1%} on the table")
    print("! Read @20 and @50. k=5 is the strictness knob, ✗ the operating point.")


if __name__ == "__main__":
    main()
