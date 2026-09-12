"""Score the retrieval arms against the labeled query set · EVAL.md §3.

Reports Recall@k and MRR per arm and for the RRF fusion, so a dial change that
helps one arm and hurts the whole is visible rather than averaged away.

! Unlike the throwaway experiments that preceded it, these queries do not
reuse corpus wording, so a contextual-header column can be compared fairly:
the query is not hiding inside the document.

Usage:
    python eval/run.py
    python eval/run.py --dense-column dense_ctx    # compare a candidate column
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipelines" / "pre_embed"))
from sparse import Bm25Vectorizer  # noqa: E402

PG = os.environ.get(
    "DATABASE_URL", "host=localhost port=5432 dbname=vera user=vera password=vera"
)
TEI = os.environ.get("EMBED_ENDPOINT", "http://localhost:8080")
VOCAB = ROOT / ".test" / "runs" / "spike-01.bm25.json"

PER_ARM = 20
RRF_K = 60


def embed(text):
    body = json.dumps({"inputs": text, "truncate": False}).encode()
    req = urllib.request.Request(
        f"{TEI}/embed", data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)[0]


def rank_of(ranked, targets):
    for i, cid in enumerate(ranked, 1):
        if cid in targets:
            return i
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dense-column", default="dense")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()
    import psycopg

    spec = json.loads((Path(__file__).parent / "queries.json").read_text(encoding="utf-8"))
    queries = [q for q in spec["queries"] if q.get("answer_chunks")]
    skipped = len(spec["queries"]) - len(queries)
    vz = Bm25Vectorizer.load(VOCAB)

    pg = psycopg.connect(PG)
    cur = pg.cursor()

    arms = ["dense", "sparse", "tsv", "RRF(all)", "RRF(s+t)", "RRF(s+d)"]
    hits = {a: 0 for a in arms}
    mrr = {a: 0.0 for a in arms}
    per_query = []

    for q in queries:
        targets = set(q["answer_chunks"])
        qv = "[" + ",".join(f"{x:.6g}" for x in embed(q["query"])) + "]"

        cur.execute(
            f"SELECT id FROM chunks WHERE indexable AND {args.dense_column} IS NOT NULL"
            f" ORDER BY {args.dense_column} <=> %s::text::halfvec LIMIT %s",
            (qv, PER_ARM),
        )
        dense = [r[0] for r in cur.fetchall()]

        cur.execute(
            "SELECT id FROM chunks WHERE indexable AND sparse IS NOT NULL"
            " ORDER BY sparse <#> %s::text::sparsevec LIMIT %s",
            (vz.to_sparsevec(vz.query(q["query"])), PER_ARM),
        )
        sparse = [r[0] for r in cur.fetchall()]

        # ! OR, not plainto_tsquery. plainto ANDs every term, so a natural
        # question ("siapa yang berwenang menetapkan kelas jalan provinsi")
        # requires one chunk to contain all seven lexemes and matches nothing.
        # OR restores recall; ts_rank still has no IDF, so this arm is a net,
        # not a precision instrument -- see the fusion comparison below.
        orq = ("array_to_string(tsvector_to_array("
               "to_tsvector('indonesian', %s)), ' | ')::tsquery")
        cur.execute(
            f"SELECT id FROM chunks WHERE indexable AND tsv @@ {orq}"
            f" ORDER BY ts_rank(tsv, {orq}) DESC LIMIT %s",
            (q["query"], q["query"], PER_ARM),
        )
        tsv = [r[0] for r in cur.fetchall()]

        def fuse(*lists):
            f: dict[str, float] = {}
            for ids in lists:
                for i, cid in enumerate(ids, 1):
                    f[cid] = f.get(cid, 0.0) + 1.0 / (RRF_K + i)
            return [c for c, _ in sorted(f.items(), key=lambda kv: -kv[1])]

        combos = {
            "RRF(all)": fuse(dense, sparse, tsv),
            "RRF(s+t)": fuse(sparse, tsv),
            "RRF(s+d)": fuse(sparse, dense),
        }

        ranks = {}
        for name, ids in [("dense", dense), ("sparse", sparse), ("tsv", tsv),
                          *combos.items()]:
            r = rank_of(ids, targets)
            ranks[name] = r
            if r and r <= args.k:
                hits[name] += 1
            if r:
                mrr[name] += 1.0 / r
        per_query.append((q["id"], q["type"], ranks))

    n = len(queries)
    print(f"queries scored: {n}   (skipped {skipped} without answer_chunks)")
    print(f"dense column  : {args.dense_column}\n")
    print(f"{'arm':<8} {'Recall@' + str(args.k):>9} {'MRR':>7}")
    print("-" * 26)
    for a in arms:
        print(f"{a:<8} {hits[a] / n:>8.1%} {mrr[a] / n:>7.3f}")

    print(f"\nper-query rank (- = not in top {PER_ARM}):")
    print(f"{'id':<6} {'dense':>6} {'sparse':>7} {'tsv':>5} {'all':>5} {'s+t':>5} {'s+d':>5}")
    for qid, _qtype, r in per_query:
        f = lambda v: str(v) if v else "-"  # noqa: E731
        print(f"{qid:<6} {f(r['dense']):>6} {f(r['sparse']):>7} {f(r['tsv']):>5} "
              f"{f(r['RRF(all)']):>5} {f(r['RRF(s+t)']):>5} {f(r['RRF(s+d)']):>5}")
    pg.close()


if __name__ == "__main__":
    main()
