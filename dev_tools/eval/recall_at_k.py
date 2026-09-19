"""Recall at the width a caller is actually handed, not just the k=5 gate.

`EVAL.md` §3 makes Recall@5 the primary gate, and it should stay that way: a strict
metric is what makes a regression visible. But it is a STRICTNESS KNOB, ✗ the
operating point. An agent receives 20-50 results and reasons over them, and the
ordering between configurations changes with k -- fusion loses to dense-alone at
k=5 and wins from k=20 up.

Tuning on k=5 alone therefore over-weights the dense arm for the way the engine is
actually used. That is the mistake this script exists to prevent.

    DATABASE_URL=... EMBED_ENDPOINT=... BM25_VOCAB=... python recall_at_k.py

! EMBED_ENDPOINT must be the provider that embedded the corpus (`EMBEDDING.md` §5).
! Unweighted RRF, matching run.py. The engine's weighted fusion does better, so
treat the fused row as a conservative floor rather than the shipped number.
"""

from __future__ import annotations

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
PG = os.environ["DATABASE_URL"]
EP = os.environ.get("EMBED_ENDPOINT", "http://localhost:8080").rstrip("/")
if not os.environ.get("BM25_VOCAB"):
    sys.exit("set BM25_VOCAB to the vocabulary this corpus was built with")
VOCAB = Path(os.environ["BM25_VOCAB"])

KS = (5, 10, 20, 50, 100)
PER_ARM = 100
RRF_K = 60
PROBED = int(os.environ.get("CLUSTERS_PROBED", "5"))
ORQ = "array_to_string(tsvector_to_array(to_tsvector('indonesian', %s)), ' | ')::tsquery"
SKIP = {"out_of_domain", "exact_ref"}


def embed(text: str) -> list[float]:
    req = urllib.request.Request(
        f"{EP}/embed", data=json.dumps({"inputs": text, "truncate": False}).encode(),
        headers={"Content-Type": "application/json"})
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
             ref["about"], ref["article"]))
        ids |= {r[0] for r in cur.fetchall()}
    return ids


def main() -> None:
    cases = [c for c in json.loads(QUERIES.read_text(encoding="utf-8"))["queries"]
             if c["type"] not in SKIP]
    vz = Bm25Vectorizer.load(VOCAB)
    res = {n: dict.fromkeys(KS, 0)
           for n in ("lex", "dense_routed", "dense_flat", "all3")}
    scored = 0

    with psycopg.connect(PG) as pg, pg.cursor() as cur:
        cur.execute("SELECT id, centroid::text FROM clusters ORDER BY id")
        cl, cents = [], []
        for cid, c in cur.fetchall():
            v = [float(x) for x in c.strip("[]").split(",")]
            n = math.sqrt(sum(a * a for a in v)) or 1.0
            cl.append(cid)
            cents.append([a / n for a in v])

        for q in cases:
            tgt = resolve_targets(cur, q)
            if not tgt:
                continue
            scored += 1
            qv = embed(q["query"])
            lit = "[" + ",".join(f"{x:.6g}" for x in qv) + "]"
            nrm = math.sqrt(sum(a * a for a in qv)) or 1.0
            qn = [a / nrm for a in qv]
            order = [cid for _, cid in sorted(
                ((sum(a * b for a, b in zip(qn, ct)), cid) for ct, cid in zip(cents, cl)),
                key=lambda t: -t[0])]

            cur.execute("SELECT id FROM chunks WHERE indexable AND sparse IS NOT NULL"
                        " ORDER BY sparse <#> %s::text::sparsevec LIMIT %s",
                        (vz.to_sparsevec(vz.query(q["query"])), PER_ARM))
            sp = [r[0] for r in cur.fetchall()]
            cur.execute(f"SELECT id FROM chunks WHERE indexable AND tsv @@ {ORQ}"
                        f" ORDER BY ts_rank(tsv, {ORQ}) DESC LIMIT %s",
                        (q["query"], q["query"], PER_ARM))
            tx = [r[0] for r in cur.fetchall()]
            cur.execute("SELECT id FROM chunks WHERE indexable AND dense IS NOT NULL"
                        " ORDER BY dense <=> %s::text::halfvec LIMIT %s", (lit, PER_ARM))
            d_flat = [r[0] for r in cur.fetchall()]
            cur.execute("SELECT id FROM chunks WHERE indexable AND dense IS NOT NULL"
                        " AND cluster_id = ANY(%s)"
                        " ORDER BY dense <=> %s::text::halfvec LIMIT %s",
                        (order[:PROBED], lit, PER_ARM))
            d_routed = [r[0] for r in cur.fetchall()]

            got = {"lex": fuse(sp, tx), "dense_routed": d_routed,
                   "dense_flat": d_flat, "all3": fuse(sp, tx, d_routed)}
            for name, ranked in got.items():
                for k in KS:
                    res[name][k] += bool(set(ranked[:k]) & tgt)

    print(f"\n{scored} queries · full corpus · probing {PROBED} clusters\n")
    print(f"{'configuration':<28}" + "".join(f"{'@' + str(k):>9}" for k in KS))
    for key, label in (("lex", "sparse+text"),
                       ("dense_routed", f"dense (routed, {PROBED})"),
                       ("dense_flat", "dense (flat scan)"),
                       ("all3", "all three fused")):
        print(f"{label:<28}" + "".join(f"{res[key][k] / scored:>8.1%}" for k in KS))
    print("\n! Fusion loses at k=5 and wins from k=20 up. Do not fit weights on k=5.")


if __name__ == "__main__":
    main()
