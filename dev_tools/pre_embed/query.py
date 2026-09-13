"""Hybrid search against the loaded corpus · a harness, not the engine.

Proves the three arms work and fuse before any Rust is written:

    dense     halfvec cosine        · semantic
    sparse    sparsevec BM25 (IP)   · exact-term, IDF-weighted
    tsvector  ts_rank('indonesian') · stemmed lexical

Fused with Reciprocal Rank Fusion, which is what vera-engine will do. RRF needs
only ranks, so the three arms' incomparable score scales never have to be
calibrated against each other.

Usage:
    python query.py "sanksi administrasi berupa denda di bidang cukai"
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sparse import Bm25Vectorizer  # noqa: E402

TEI = os.environ.get("EMBED_ENDPOINT", "http://localhost:8080")
PG = os.environ.get(
    "DATABASE_URL", "host=localhost port=5432 dbname=vera user=vera password=vera"
)
VECTORIZER = (
    Path(__file__).resolve().parents[2] / ".test" / "runs" / "fixture-01.bm25.json"
)

RRF_K = 60
PER_ARM = 20
TOP_N = 5


def embed_query(text: str) -> list[float]:
    body = json.dumps({"inputs": text, "truncate": False}).encode()
    req = urllib.request.Request(
        f"{TEI}/embed", data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)[0]


def rrf(ranked_lists: dict[str, list[str]]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for ids in ranked_lists.values():
        for rank, cid in enumerate(ids, 1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (RRF_K + rank)
    return scores


def main() -> None:
    query = " ".join(sys.argv[1:]) or "sanksi administrasi berupa denda di bidang cukai"
    import psycopg

    vz = Bm25Vectorizer.load(VECTORIZER)
    qdense = "[" + ",".join(f"{v:.6g}" for v in embed_query(query)) + "]"
    qsparse = vz.to_sparsevec(vz.query(query))

    with psycopg.connect(PG) as pg, pg.cursor() as cur:
        # ! Every arm filters on `indexable`. Boilerplate is stored and readable
        # but must never be a search result.
        cur.execute(
            "SELECT id FROM chunks WHERE indexable AND dense IS NOT NULL"
            " ORDER BY dense <=> %s::halfvec LIMIT %s",
            (qdense, PER_ARM),
        )
        dense_ids = [r[0] for r in cur.fetchall()]

        # Inner product, not cosine: BM25 weights already encode length
        # normalisation, and cosine would undo it.
        cur.execute(
            "SELECT id FROM chunks WHERE indexable AND sparse IS NOT NULL"
            " ORDER BY sparse <#> %s::sparsevec LIMIT %s",
            (qsparse, PER_ARM),
        )
        sparse_ids = [r[0] for r in cur.fetchall()]

        cur.execute(
            "SELECT id FROM chunks WHERE indexable"
            " AND tsv @@ plainto_tsquery('indonesian', %s)"
            " ORDER BY ts_rank(tsv, plainto_tsquery('indonesian', %s)) DESC LIMIT %s",
            (query, query, PER_ARM),
        )
        ts_ids = [r[0] for r in cur.fetchall()]

        arms = {"dense": dense_ids, "sparse": sparse_ids, "tsv": ts_ids}
        fused = sorted(rrf(arms).items(), key=lambda kv: -kv[1])[:TOP_N]

        print(f'QUERY: "{query}"\n')
        for name, ids in arms.items():
            print(f"  {name:<7} returned {len(ids):>2}")
        print()

        for rank, (cid, score) in enumerate(fused, 1):
            cur.execute(
                "SELECT source_title, article, chapter, truncated_at_source,"
                " left(regexp_replace(body, '\\s+', ' ', 'g'), 90)"
                " FROM chunks WHERE id = %s",
                (cid,),
            )
            title, article, chapter, trunc, snippet = cur.fetchone()
            where = [f"{n}#{ids.index(cid) + 1}" for n, ids in arms.items()
                     if cid in ids]
            flag = "  [TRUNCATED AT SOURCE]" if trunc else ""
            print(f"{rank}. rrf={score:.5f}  arms={','.join(where)}{flag}")
            print(f"   {title[:96]}")
            print(f"   locator: {chapter} / {article}")
            print(f"   {snippet}...\n")


if __name__ == "__main__":
    main()
