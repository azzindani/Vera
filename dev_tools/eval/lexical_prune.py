"""What the two global lexical arms cost, and what they lose, when sent fewer query terms.

    python dev_tools/eval/lexical_prune.py time   <dbname>     # latency per pruning level (first 12 queries)
    python dev_tools/eval/lexical_prune.py recall <dbname>     # arm-alone hit rate vs the eval labels

The engine currently sends EVERY query term to both arms. At 5.1M rows the tsvector arm matches ~62% of the
corpus with an OR query and takes 6.2 s; the cost is the match count, ✗ the ranking. Pruning terms by IDF
(the BM25 artifact already carries it) shrinks the match set. `HARDWARE.md` §8 records the result.

Modes: `full`, `idf>=T` (keep terms whose corpus IDF is at least T), `topN` (keep the N rarest terms).
Unknown words are charged the maximum IDF, exactly as `QueryVectorizer::evidence` does, so they are always kept.

! `time` runs each query twice: the second timing is warm. pg_search is CPU-bound and shows cold == warm.
! `recall` is a hit-rate of the arm ALONE (target in the arm's top-k), ✗ the fused pipeline. A pruned arm that
loses nothing alone can still change fusion; confirm with e2e.py before shipping a change to the engine.
! Read-only. Needs BM25_VOCAB (the corpus artifact) and DATABASE_URL-style host settings below.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

import psycopg

ROOT = Path(__file__).resolve().parents[2]
VOCAB = Path(os.environ.get("BM25_VOCAB", ROOT / ".test" / "runs" / "corpus.bm25.json"))
HOST = os.environ.get("PGHOST", "127.0.0.1")  # ✗ localhost, see load_test.py
MODES = ["full", "idf>=2", "idf>=3", "idf>=4", "top4", "top3"]

IDF = json.loads(VOCAB.read_text(encoding="utf-8"))["idf"]
MAX_IDF = max(IDF.values())
SPEC = json.loads((Path(__file__).parent / "queries.json").read_text(encoding="utf-8"))["queries"]
tok = lambda s: re.findall(r"(?u)\b\w\w+\b", s.lower())  # noqa: E731 -- the engine's tokeniser


def prune(query: str, mode: str) -> str:
    terms = list(dict.fromkeys(tok(query)))
    if mode == "full":
        return query
    if mode.startswith("idf>="):
        keep = [t for t in terms if IDF.get(t, MAX_IDF) >= float(mode[5:])]
    else:
        keep = sorted(terms, key=lambda t: -IDF.get(t, MAX_IDF))[: int(mode[3:])]
    return " ".join(keep) if keep else query


def bm25(cur, q: str) -> list[str]:
    cur.execute("SELECT id FROM chunks WHERE indexable AND id @@@ paradedb.match('body', %s) "
                "ORDER BY paradedb.score(id) DESC LIMIT 60", (q,))
    return [r[0] for r in cur.fetchall()]


def text(cur, q: str) -> list[str]:
    cur.execute("WITH q AS (SELECT array_to_string(tsvector_to_array(to_tsvector('indonesian', %s)), ' | ')::tsquery AS tq) "
                "SELECT id FROM chunks, q WHERE indexable AND tsv @@ q.tq ORDER BY tsv <=> q.tq LIMIT 60", (q,))
    return [r[0] for r in cur.fetchall()]


def timed(fn, cur, q):
    t = time.perf_counter()
    ids = fn(cur, q)
    return ids, (time.perf_counter() - t) * 1000


def targets(cur, q: dict) -> set[str]:
    ids: set[str] = set()
    for r in q.get("answer_articles", []):
        cur.execute("SELECT id FROM chunks WHERE regulation_type=%s AND regulation_number=%s AND year=%s "
                    "AND about=%s AND article=%s",
                    (r["regulation_type"], r["regulation_number"], r["year"], r["about"], r["article"]))
        ids |= {x[0] for x in cur.fetchall()}
    g = q.get("answer_regulation")
    if g:
        cur.execute("SELECT id FROM chunks WHERE regulation_type=%s AND regulation_number=%s AND year=%s",
                    (g["regulation_type"], g["regulation_number"], g["year"]))
        ids |= {x[0] for x in cur.fetchall()}
    return ids


def run_time(cur) -> None:
    queries = SPEC[:12]
    base: dict[tuple[str, str], set[str]] = {}
    med = lambda v: sorted(v)[len(v) // 2]  # noqa: E731
    for m in MODES:
        row: dict[str, list[float]] = {"bc": [], "bw": [], "tc": [], "tw": [], "bo": [], "to": []}
        for q in queries:
            pq = prune(q["query"], m)
            for fn, c, w, o, key in ((bm25, "bc", "bw", "bo", "b"), (text, "tc", "tw", "to", "t")):
                ids, cold = timed(fn, cur, pq)
                _, warm = timed(fn, cur, pq)
                row[c].append(cold)
                row[w].append(warm)
                top = set(ids[:20])
                if m == "full":
                    base[(q["id"], key)] = top
                else:
                    row[o].append(len(top & base[(q["id"], key)]) / max(1, len(base[(q["id"], key)])))
        extra = "" if m == "full" else (f" | top20 overlap with full: bm25={sum(row['bo']) / len(row['bo']):.2f} "
                                        f"text={sum(row['to']) / len(row['to']):.2f}")
        print(f"{m:7s} pg_search cold p50={med(row['bc']):7.0f} warm p50={med(row['bw']):7.0f} ms | "
              f"tsvector cold p50={med(row['tc']):7.0f} warm p50={med(row['tw']):7.0f} ms{extra}", flush=True)


def run_recall(cur) -> None:
    labelled = [(q, targets(cur, q)) for q in SPEC if q["type"] != "out_of_domain"]
    labelled = [(q, t) for q, t in labelled if t]
    print(f"arm-alone hit rates over {len(labelled)} labelled queries (target in the arm's top-k)")
    for m in MODES:
        out = []
        for fn in (bm25, text):
            hit = {5: 0, 20: 0}
            for q, t in labelled:
                ids = fn(cur, prune(q["query"], m))
                for k in hit:
                    hit[k] += any(i in t for i in ids[:k])
            out.append(f"@5 {100 * hit[5] / len(labelled):4.1f}% @20 {100 * hit[20] / len(labelled):4.1f}%")
        print(f"{m:7s} pg_search {out[0]} | tsvector {out[1]}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in ("time", "recall"):
        sys.exit(__doc__)
    dsn = f"host={HOST} port=5432 dbname={sys.argv[2]} user=vera password=vera"
    with psycopg.connect(dsn, autocommit=True) as con, con.cursor() as cur:
        (run_time if sys.argv[1] == "time" else run_recall)(cur)
