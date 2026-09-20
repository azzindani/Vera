"""Can `dense + bm25` replace `dense + sparse + text`? Fused recall decides.

`HARDWARE.md` §6 measures the text arm at 6,428 ms and the sparse arm at 6,282 ms
on 5M rows -- together 99% of a query. `bm25_compare.py` shows `pg_search` doing
the sparse arm's job 3.4× faster, and `arm_overlap.py` reports that dropping the
text arm costs **+0.0% at @50** when three arms are fused.

If both hold together, the whole lexical half collapses to one indexed arm:

    route 85 + dense 57 + bm25 1,858  ≈  2,000 ms   against 12,853 ms today

That is a 6.4× claim resting on two measurements taken separately, which is not
the same as having measured it. This measures it.

    DATABASE_URL=...        ParadeDB, holding the same corpus
    BASELINE_URL=...        the normal instance: dense, sparse, text
    EMBED_ENDPOINT=...  BM25_VOCAB=...
    python dev_tools/eval/fused_bm25.py

! The REAL corpus with the REAL labels. Pointed at a replicated one every
configuration scores ~100% and the comparison is worthless.

! Unweighted RRF, matching `run.py` and `arm_overlap.py`, so the rows here are
comparable to those and are a floor against the engine's weighted fusion.

! Read @20 and @50, ✗ @5 (`EVAL.md` §4b). Dropping an arm is exactly the kind of
change that looks fine at k=5 and costs real recall at the width a caller sees.
"""

from __future__ import annotations

import itertools
import json
import math
import os
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "dev_tools" / "pre_embed"))

import psycopg  # noqa: E402
from sparse import Bm25Vectorizer  # noqa: E402

QUERIES = Path(__file__).parent / "queries.json"
PDB = os.environ["DATABASE_URL"]
BASE = os.environ["BASELINE_URL"]
EP = os.environ.get("EMBED_ENDPOINT", "http://localhost:8080").rstrip("/")
VOCAB = Path(os.environ["BM25_VOCAB"])

KS = (5, 20, 50)
DEPTH = 100
RRF_K = 60
PROBED = int(os.environ.get("CLUSTERS_PROBED", "5"))
SKIP = {"out_of_domain", "exact_ref"}

_Q = ("WITH q AS (SELECT array_to_string("
      "tsvector_to_array(to_tsvector('indonesian', %s)), ' | ')::tsquery AS tq) ")
TEXT_RUM = _Q + ("SELECT id FROM chunks, q WHERE indexable AND tsv @@ q.tq"
                 " ORDER BY tsv <=> q.tq LIMIT %s")
SPARSE = ("SELECT id FROM chunks WHERE indexable AND sparse IS NOT NULL"
          " ORDER BY sparse <#> %s::text::sparsevec LIMIT %s")
DENSE = ("SELECT id FROM chunks WHERE indexable AND dense IS NOT NULL"
         " AND cluster_id = ANY(%s) ORDER BY dense <=> %s::text::halfvec LIMIT %s")
BM25 = ("SELECT id FROM chunks WHERE indexable AND id @@@ paradedb.match('body', %s)"
        " ORDER BY paradedb.score(id) DESC LIMIT %s")

# The configurations worth deciding between, ✗ every subset.
CONFIGS = {
    "dense": ("dense",),
    "dense + sparse": ("dense", "sparse"),
    "dense + text": ("dense", "text"),
    "dense + bm25": ("dense", "bm25"),
    "dense + sparse + text  (today)": ("dense", "sparse", "text"),
    "dense + bm25 + text": ("dense", "bm25", "text"),
    "dense + bm25 + sparse": ("dense", "bm25", "sparse"),
}


def embed(text: str) -> list[float]:
    req = urllib.request.Request(
        f"{EP}/embed",
        data=json.dumps({"inputs": text, "truncate": False}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.load(r)[0]


def fuse(*lists: list[str]) -> list[str]:
    f: dict[str, float] = defaultdict(float)
    for ids in lists:
        for i, cid in enumerate(ids, 1):
            f[cid] += 1.0 / (RRF_K + i)
    return [c for c, _ in sorted(f.items(), key=lambda kv: (-kv[1], kv[0]))]


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
    vz = Bm25Vectorizer.load(VOCAB)
    hits = {name: dict.fromkeys(KS, 0) for name in CONFIGS}
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
            bcur.execute(DENSE, (order[:PROBED], lit, DEPTH))
            got["dense"] = [r[0] for r in bcur.fetchall()]
            bcur.execute(SPARSE, (vz.to_sparsevec(vz.query(q["query"])), DEPTH))
            got["sparse"] = [r[0] for r in bcur.fetchall()]
            bcur.execute(TEXT_RUM, (q["query"], DEPTH))
            got["text"] = [r[0] for r in bcur.fetchall()]
            pcur.execute(BM25, (q["query"], DEPTH))
            got["bm25"] = [r[0] for r in pcur.fetchall()]

            for name, arms in CONFIGS.items():
                ranked = fuse(*(got[a] for a in arms))
                for k in KS:
                    hits[name][k] += bool(set(ranked[:k]) & tgt)
            print(f"  {scored}", flush=True)

    print(f"\n{scored} queries · real corpus · real labels · depth {DEPTH}\n")
    width = max(len(n) for n in CONFIGS) + 2
    print(f"{'configuration':<{width}}" + "".join(f"{'@' + str(k):>9}" for k in KS))
    for name in CONFIGS:
        print(f"{name:<{width}}"
              + "".join(f"{hits[name][k] / scored:>8.1%}" for k in KS))

    today = hits["dense + sparse + text  (today)"]
    print(f"\nagainst today's three arms")
    for name in CONFIGS:
        if name.endswith("(today)"):
            continue
        print(f"  {name:<{width}}"
              + "  ".join(f"@{k} {(hits[name][k] - today[k]) / scored:+.1%}"
                          for k in KS))
    print("\n! Each query is one unit, so on ~39 cases a 2.6% step is ONE query."
          "\n  Differences of a few points here are directional, ✗ conclusive.")


if __name__ == "__main__":
    main()
