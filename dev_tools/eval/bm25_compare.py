"""ParadeDB `pg_search` BM25 against the two lexical arms it would replace.

`HARDWARE.md` §6 measures the wall: at 5M rows the sparse arm is 6,282 ms and the
text arm 6,428 ms, together 99% of a query. Both are lexical, and neither uses an
index built for ranked keyword retrieval -- `sparse` has no index at all and is a
sequential scan of a `sparsevec` column, while `tsv` has RUM but OR semantics
matches a median 62% of the corpus.

`pg_search` is Tantivy (Rust Lucene) as a Postgres index type. This asks whether
it beats them on BOTH axes, because latency alone would not justify the change:

    DATABASE_URL=...        the ParadeDB instance holding the same corpus
    BASELINE_URL=...        the normal instance, for the arms being compared
    python dev_tools/eval/bm25_compare.py

! Recall is measured on the REAL corpus with the REAL labels. A replicated
corpus would make every arm look perfect (`scale_probe.py`), so this must not be
pointed at one.

! Latency here is a lower bound for pg_search and a fair number for the others:
the BM25 index covers `body` only, while the baseline arms run against the full
table. Index-scan cost dominates the `LIMIT 60` heap fetch either way, but the
table is narrower on the ParadeDB side and that is not nothing.

! An arm is only replaceable if it loses nothing at the WIDE k a caller is handed
(`EVAL.md` §4b). A lexical arm that is faster and worse is not an upgrade.
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "dev_tools" / "pre_embed"))

import psycopg  # noqa: E402
from sparse import Bm25Vectorizer  # noqa: E402

QUERIES = Path(__file__).parent / "queries.json"
PDB = os.environ["DATABASE_URL"]
BASE = os.environ.get("BASELINE_URL", PDB)
VOCAB = Path(os.environ["BM25_VOCAB"]) if os.environ.get("BM25_VOCAB") else None

KS = (5, 20, 50)
DEPTH = 100
REPEATS = 3
SKIP = {"out_of_domain", "exact_ref"}

_Q = ("WITH q AS (SELECT array_to_string("
      "tsvector_to_array(to_tsvector('indonesian', %s)), ' | ')::tsquery AS tq) ")
TEXT_RUM = _Q + ("SELECT id FROM chunks, q WHERE indexable AND tsv @@ q.tq"
                 " ORDER BY tsv <=> q.tq LIMIT %s")
SPARSE = ("SELECT id FROM chunks WHERE indexable AND sparse IS NOT NULL"
          " ORDER BY sparse <#> %s::text::sparsevec LIMIT %s")
BM25 = ("SELECT id FROM chunks WHERE indexable AND id @@@ paradedb.match('body', %s)"
        " ORDER BY paradedb.score(id) DESC LIMIT %s")


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
    vz = Bm25Vectorizer.load(VOCAB) if VOCAB else None
    arms = ("bm25", "text", "sparse")
    hits = {a: dict.fromkeys(KS, 0) for a in arms}
    lat: dict[str, list[float]] = {a: [] for a in arms}
    overlap = {"bm25_vs_text": 0.0, "bm25_vs_sparse": 0.0}
    scored = 0

    with psycopg.connect(PDB) as pc, pc.cursor() as pcur, \
            psycopg.connect(BASE) as bc, bc.cursor() as bcur:
        for q in cases:
            tgt = resolve_targets(bcur, q)
            if not tgt:
                continue
            scored += 1
            got = {}
            got["bm25"], ms = run(pcur, BM25, (q["query"], DEPTH))
            lat["bm25"].append(ms)
            got["text"], ms = run(bcur, TEXT_RUM, (q["query"], DEPTH))
            lat["text"].append(ms)
            if vz:
                got["sparse"], ms = run(
                    bcur, SPARSE, (vz.to_sparsevec(vz.query(q["query"])), DEPTH))
                lat["sparse"].append(ms)
            else:
                got["sparse"] = []

            for a in arms:
                for k in KS:
                    hits[a][k] += bool(set(got[a][:k]) & tgt)
            for other in ("text", "sparse"):
                sa, sb = set(got["bm25"]), set(got[other])
                if sa or sb:
                    overlap[f"bm25_vs_{other}"] += len(sa & sb) / len(sa | sb)
            print(f"  {scored}", flush=True)

    print(f"\n{scored} queries · depth {DEPTH} · median of {REPEATS}\n")
    print(f"{'arm':<10}{'p50 ms':>9}" + "".join(f"{'@' + str(k):>9}" for k in KS))
    for a in arms:
        if not lat[a]:
            continue
        print(f"{a:<10}{statistics.median(lat[a]):>9.0f}"
              + "".join(f"{hits[a][k] / scored:>8.1%}" for k in KS))

    print("\noverlap with the arm it would replace (Jaccard, top-100)")
    for key, tot in overlap.items():
        print(f"  {key:<18}{tot / scored:>7.1%}")
    print("\n! Faster AND at least as good at wide k is a replacement."
          "\n  Faster and worse is a different trade, and needs saying so.")


if __name__ == "__main__":
    main()
