"""Every signal the domain gate could use, per query, so a gate rule is chosen from data.

    DATABASE_URL=... BM25_VOCAB=... EMBED_ENDPOINT=http://127.0.0.1:8080 \
        python dev_tools/eval/gate_signals.py <queries.json> [more.json ...] > signals.jsonl

The gate today is `lexical >= 0.40 AND centroid >= 0.39` (`pipeline.rs::domain_gate`), and
`EVAL.md` §4c records that no setting of those two floors separates real questions from
off-topic ones: fluent Indonesian about plumbing lands near legal centroids, and BM25 finds
chunks containing every word of it. This measures the candidates beside them:

  centroid      max cosine to a cluster centroid               (the engine's signal today)
  lexical       IDF mass of the query found in the top-5 BM25 bodies  (the engine's other one)
  dense_top1    cosine to the nearest CHUNK in the probed clusters
  dense_mean5   mean cosine of the five nearest chunks
  agree_docs    documents shared by the dense top-20 and the BM25 top-20
  bm25_top      raw pg_search score of the best hit

! Read-only, and the same SQL the engine runs (`chunk_vectors`, pg_search top-k with
`indexable` outside it), the query embedded with the corpus's own instruction.
"""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.request

import numpy as np
import psycopg

PG = os.environ["DATABASE_URL"]
EMBED = os.environ.get("EMBED_ENDPOINT", "http://127.0.0.1:8080").rstrip("/")
VOCAB = json.load(open(os.environ["BM25_VOCAB"], encoding="utf-8"))
IDF = {k: float(v) for k, v in VOCAB["idf"].items()}
UNK = max(IDF.values())
PROBE, K = 5, 20
tok = lambda s: re.findall(r"(?u)\b\w\w+\b", s.lower())  # noqa: E731 -- bm25.rs::tokenize


def evidence(query: str, texts: list[str]) -> float:
    """bm25.rs::evidence: share of the query's IDF mass the pooled texts contain."""
    terms = sorted(set(tok(query)))
    total = sum(IDF.get(t, UNK) for t in terms)
    if total <= 0:
        return 0.0
    pooled = {t for x in texts for t in tok(x)}
    return sum(IDF.get(t, UNK) for t in terms if t in pooled) / total


def main() -> None:
    cases = []
    for path in sys.argv[1:]:
        data = json.load(open(path, encoding="utf-8"))
        cases += data["queries"] if isinstance(data, dict) else data
    with psycopg.connect(PG, autocommit=True) as c:
        cur = c.cursor()
        cur.execute("SELECT to_jsonb(corpus_meta) ->> 'dense_query_instruction' FROM corpus_meta"
                    " ORDER BY created_at DESC LIMIT 1")
        instr = cur.fetchone()[0]
        prefix = f"Instruct: {instr}\nQuery: " if instr else ""
        cur.execute("SELECT id, centroid::text FROM clusters ORDER BY id")
        rows = cur.fetchall()
        ids = np.array([r[0] for r in rows])
        cent = np.array([np.fromstring(r[1].strip("[]"), sep=",", dtype=np.float32) for r in rows])
        cent /= np.linalg.norm(cent, axis=1, keepdims=True)
        for case in cases:
            q = case["query"]
            req = urllib.request.Request(
                f"{EMBED}/embed", data=json.dumps({"inputs": [prefix + q]}).encode(),
                headers={"Content-Type": "application/json"})
            qv = np.array(json.load(urllib.request.urlopen(req, timeout=300))[0], dtype=np.float32)
            sims = cent @ (qv / np.linalg.norm(qv))
            order = np.argsort(-sims, kind="stable")[:PROBE]
            lit = "[" + ",".join(f"{v:.9g}" for v in qv) + "]"
            cur.execute(
                "SELECT v.id, 1.0 - (v.dense <=> %s::text::halfvec), c.doc_id FROM chunk_vectors v"
                " JOIN chunks c ON c.id = v.id WHERE v.cluster_id = ANY(%s)"
                " ORDER BY v.dense <=> %s::text::halfvec LIMIT %s",
                (lit, [int(i) for i in ids[order]], lit, K))
            dense = cur.fetchall()
            cur.execute(
                "SELECT s.id, s.score, c.doc_id, c.body FROM (SELECT id, paradedb.score(id) AS score"
                " FROM chunks WHERE id @@@ paradedb.match('body', %s) ORDER BY paradedb.score(id) DESC"
                " LIMIT %s) s JOIN chunks c ON c.id = s.id WHERE c.indexable ORDER BY s.score DESC",
                (q, K * 2))
            sparse = cur.fetchall()[:K]
            out = {
                "id": case.get("id"), "type": case.get("type"), "query": q,
                "centroid": float(sims[order[0]]),
                "lexical": evidence(q, [r[3] for r in sparse[:5]]),
                "dense_top1": float(dense[0][1]) if dense else 0.0,
                "dense_mean5": float(np.mean([r[1] for r in dense[:5]])) if dense else 0.0,
                "agree_docs": len({r[2] for r in dense} & {r[2] for r in sparse}),
                "bm25_top": float(sparse[0][1]) if sparse else 0.0,
            }
            print(json.dumps(out, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
