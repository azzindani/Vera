"""BEIR through Vera, pgvector and Qdrant: the same chunks, the same query vectors.

    PGPASSWORD=... python -I dev_tools/eval/beir/bench.py <dataset> \
        --raw <beir dir> --src <sources.py output> --dsn "host=... dbname=beir_<dataset> user=..." \
        [--vera http://127.0.0.1:8081] [--embed http://127.0.0.1:8080] [--qdrant http://127.0.0.1:6333] \
        [--systems vera:balanced,vera:literal,pg-exact,pg-hnsw,pg-bm25,qdrant,qdrant-hybrid] \
        --out results/<dataset>.json

What is held equal, so that only the engine differs:

* **Units.** Every system indexes the rows of the Vera database's `chunks` table, which
  Ravel built from the BEIR corpus (`sources.py`, then `ravel extract / chunk / embed /
  sparse / bundle build / db prepare`). A document split into several chunks is scored at
  its best chunk, the usual BEIR treatment of passages.
* **Query vectors.** Each query is embedded once, by the same embedding server Vera calls,
  with the instruction the corpus declares (`corpus_meta.dense_query_instruction`), in
  Vera's own template. pgvector and Qdrant search with those vectors; Vera embeds its own
  and must land on the same ones (its startup canary checks the space).
* **Lexical signal.** `pg-bm25` is the exact SQL of Vera's BM25 arm (`pg_search`).
  `qdrant-hybrid` fuses Qdrant's dense search with the same BM25 document weights Ravel
  stored in `chunks.sparse`, queried the way Vera's sparse vectoriser does (1.0 per known
  term), by Qdrant's own RRF.

Scored as BEIR scores: nDCG@10 with graded gains and the log2 discount, over the test
queries that have judgements, with documents absent from the ranking contributing zero.
Latency is the search call only for pgvector and Qdrant (the query vector is precomputed),
and the whole request for Vera, which embeds the query itself on CPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import sys
import time
import urllib.request
from pathlib import Path

import psycopg

TOKEN = re.compile(r"\b\w\w+\b")
DEPTH = 100


# -- data -------------------------------------------------------------------------------


def load_beir(raw: Path, split: str) -> tuple[dict[str, str], dict[str, dict[str, int]]]:
    qrels: dict[str, dict[str, int]] = {}
    with (raw / "qrels" / f"{split}.tsv").open(encoding="utf-8") as f:
        next(f)
        for line in f:
            q, d, s = line.rstrip("\n").split("\t")
            if int(s) > 0:
                qrels.setdefault(q, {})[d] = int(s)
    queries = {}
    with (raw / "queries.jsonl").open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["_id"] in qrels:
                queries[row["_id"]] = row["text"]
    return queries, qrels


def chunk_docs(conn: psycopg.Connection, src: Path) -> dict[str, list[str]]:
    """Chunk id -> every BEIR id whose text it is (identical texts are one Ravel doc)."""
    by_hash: dict[str, list[str]] = json.loads((src / "ids.json").read_text("utf-8"))
    out: dict[str, list[str]] = {}
    for cid, url in conn.execute("SELECT id, source_url FROM chunks WHERE indexable"):
        stem = url.rsplit("/", 1)[1].removesuffix(".md")
        body = (src / "docs" / f"{stem}.md").read_bytes()
        out[cid] = by_hash[hashlib.sha256(body).hexdigest()]
    return out


def http_json(url: str, body: dict, timeout: float = 300) -> object:
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def query_vectors(
    queries: dict[str, str], embed: str, instruction: str, cache: Path
) -> dict[str, list[float]]:
    prefix = f"Instruct: {instruction}\nQuery: " if instruction else ""
    key = hashlib.sha256((embed + prefix).encode()).hexdigest()[:12]
    if cache.exists():
        cached = json.loads(cache.read_text("utf-8"))
        if cached["key"] == key and set(cached["vectors"]) >= set(queries):
            return cached["vectors"]
    vectors, spent = {}, []
    for qid, text in queries.items():
        t = time.perf_counter()
        vectors[qid] = http_json(f"{embed}/embed", {"inputs": prefix + text, "truncate": False})[0]
        spent.append(time.perf_counter() - t)
    print(f"  embedded {len(vectors)} queries · p50 {statistics.median(spent)*1000:.0f} ms each")
    cache.write_text(json.dumps({"key": key, "vectors": vectors}), "utf-8")
    return vectors


# -- systems ----------------------------------------------------------------------------


def halfvec(v: list[float]) -> str:
    return "[" + ",".join(f"{x:.6g}" for x in v) + "]"


def pg_dense(conn, table: str, ef: int | None):
    def run(qid, text, vec):
        if ef:
            conn.execute(f"SET hnsw.ef_search = {ef}")
        rows = conn.execute(
            f"SELECT id FROM {table} ORDER BY dense <=> %s::halfvec LIMIT {DEPTH}",
            (halfvec(vec),),
        ).fetchall()
        return [r[0] for r in rows]

    return run


def pg_bm25(conn):
    sql = (
        "SELECT s.id FROM (SELECT id, paradedb.score(id) AS score FROM chunks "
        "WHERE id @@@ paradedb.match('body', %s) ORDER BY paradedb.score(id) DESC "
        f"LIMIT {DEPTH * 2}) s JOIN chunks c ON c.id = s.id WHERE c.indexable "
        f"ORDER BY s.score DESC LIMIT {DEPTH}"
    )

    def run(qid, text, vec):
        return [r[0] for r in conn.execute(sql, (text,)).fetchall()]

    return run


def build_hnsw(conn) -> None:
    if conn.execute("SELECT to_regclass('bench_hnsw')").fetchone()[0]:
        return
    t = time.perf_counter()
    conn.execute("CREATE TABLE bench_hnsw AS SELECT id, dense FROM chunks WHERE indexable")
    conn.execute("CREATE INDEX ON bench_hnsw USING hnsw (dense halfvec_cosine_ops)")
    conn.execute("ANALYZE bench_hnsw")
    conn.commit()
    print(f"  pgvector HNSW (m=16, ef_construction=64) built in {time.perf_counter()-t:.1f} s")


def query_sparse(vocab: dict[str, int], text: str) -> dict:
    idx = sorted({vocab[t] for t in TOKEN.findall(text.lower()) if t in vocab})
    return {"indices": idx, "values": [1.0] * len(idx)}


def build_qdrant(conn, base: str, name: str) -> None:
    try:
        info = json.load(urllib.request.urlopen(f"{base}/collections/{name}"))
        if info["result"]["points_count"]:
            return
    except urllib.error.HTTPError:
        pass
    req = urllib.request.Request(
        f"{base}/collections/{name}",
        method="PUT",
        data=json.dumps({
            "vectors": {"dense": {"size": 1024, "distance": "Cosine"}},
            "sparse_vectors": {"bm25": {}},
        }).encode(),
        headers={"Content-Type": "application/json"},
    )
    urllib.request.urlopen(req).read()
    t = time.perf_counter()
    cur = conn.execute("SELECT id, dense::text, sparse::text FROM chunks WHERE indexable")
    n = 0
    while batch := cur.fetchmany(256):
        points = []
        for cid, dense, sparse in batch:
            body = sparse.split("/")[0].strip("{}")
            pairs = [p.split(":") for p in body.split(",") if p]
            points.append({
                "id": str(__import__("uuid").UUID(hex=cid)),
                "payload": {"chunk": cid},
                "vector": {
                    "dense": json.loads(dense),
                    # sparsevec is 1-based; Qdrant indices are whatever we say, kept as-is
                    "bm25": {
                        "indices": [int(i) - 1 for i, _ in pairs],
                        "values": [float(v) for _, v in pairs],
                    },
                },
            })
        req = urllib.request.Request(
            f"{base}/collections/{name}/points?wait=true",
            method="PUT",
            data=json.dumps({"points": points}).encode(),
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=600).read()
        n += len(points)
    while True:
        info = json.load(urllib.request.urlopen(f"{base}/collections/{name}"))["result"]
        if info["status"] == "green":
            break
        time.sleep(1)
    print(f"  qdrant: {n} points loaded and indexed in {time.perf_counter()-t:.1f} s")


def qdrant(base: str, name: str, vocab: dict[str, int] | None):
    def run(qid, text, vec):
        if vocab is None:
            body = {"query": vec, "using": "dense", "limit": DEPTH, "with_payload": True}
        else:
            body = {
                "prefetch": [
                    {"query": vec, "using": "dense", "limit": DEPTH},
                    {"query": query_sparse(vocab, text), "using": "bm25", "limit": DEPTH},
                ],
                "query": {"fusion": "rrf"},
                "limit": DEPTH,
                "with_payload": True,
            }
        r = http_json(f"{base}/collections/{name}/points/query", body)
        return [p["payload"]["chunk"] for p in r["result"]["points"]]

    return run


def vera(base: str, profile: str, top_k: int, mode: str = "hybrid"):
    n = [0]

    def call(method, params):
        n[0] += 1
        return http_json(f"{base}/mcp", {"jsonrpc": "2.0", "id": n[0], "method": method, "params": params})

    call("initialize", {"protocolVersion": "2024-11-05", "capabilities": {},
                        "clientInfo": {"name": "beir-bench", "version": "0"}})

    def run(qid, text, vec):
        r = call("tools/call", {"name": "search_knowledge", "arguments": {
            "query": text, "top_k": top_k, "profile": profile, "mode": mode}})
        body = json.loads(r["result"]["content"][0]["text"])
        if not body.get("success"):
            raise RuntimeError(f"vera refused {qid}: {body.get('error')} · {body.get('hint')}")
        return [x["id"] for x in body["results"]]

    return run


# -- scoring ----------------------------------------------------------------------------


def to_docs(chunks: list[str], owner: dict[str, list[str]]) -> list[list[str]]:
    """Rank positions as documents: a chunk's doc counts at its first (best) position."""
    seen, out = set(), []
    for c in chunks:
        ids = owner.get(c)
        if not ids or ids[0] in seen:
            continue
        seen.add(ids[0])
        out.append(ids)
    return out


def ndcg(ranked: list[list[str]], rel: dict[str, int], k: int = 10) -> float:
    dcg = sum(
        max(rel.get(d, 0) for d in ids) / math.log2(i + 2) for i, ids in enumerate(ranked[:k])
    )
    ideal = sorted(rel.values(), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def recall(ranked: list[list[str]], rel: dict[str, int], k: int) -> float:
    found = {d for ids in ranked[:k] for d in ids if d in rel}
    return len(found) / len(rel)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--dsn", required=True)
    ap.add_argument("--vocab", type=Path, help="corpus.bm25.json, for qdrant-hybrid")
    ap.add_argument("--vera", default="http://127.0.0.1:8081")
    ap.add_argument("--embed", default="http://127.0.0.1:8080")
    ap.add_argument("--qdrant", default="http://127.0.0.1:6333")
    ap.add_argument("--systems", default="vera:balanced,vera:literal,pg-exact,pg-hnsw,pg-bm25,qdrant,qdrant-hybrid")
    ap.add_argument("--vera-top-k", type=int, default=20)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    queries, qrels = load_beir(a.raw, a.split)
    conn = psycopg.connect(a.dsn)
    owner = chunk_docs(conn, a.src)
    instruction = conn.execute(
        "SELECT to_jsonb(corpus_meta) ->> 'dense_query_instruction' FROM corpus_meta"
    ).fetchone()[0]
    print(f"{a.dataset}: {len(queries)} queries, {len(owner)} chunks · instruction {instruction!r}")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    vecs = query_vectors(queries, a.embed, instruction, a.out.with_suffix(".qvec.json"))
    vocab = json.loads(a.vocab.read_text("utf-8"))["vocab"] if a.vocab else None

    results = json.loads(a.out.read_text("utf-8")) if a.out.exists() else {}
    for name in a.systems.split(","):
        if name.startswith("vera:"):
            # vera:<profile>[:<mode>], mode hybrid | semantic | keyword
            parts = name.split(":")
            run = vera(a.vera, parts[1], a.vera_top_k, parts[2] if len(parts) > 2 else "hybrid")
        elif name == "pg-exact":
            run = pg_dense(conn, "chunks", None)
        elif name.startswith("pg-hnsw"):
            build_hnsw(conn)
            ef = int(name.split("@")[1]) if "@" in name else 40
            run = pg_dense(conn, "bench_hnsw", ef)
        elif name == "pg-bm25":
            run = pg_bm25(conn)
        elif name in ("qdrant", "qdrant-hybrid"):
            build_qdrant(conn, a.qdrant, f"beir_{a.dataset}")
            if name == "qdrant-hybrid" and vocab is None:
                sys.exit("qdrant-hybrid needs --vocab")
            run = qdrant(a.qdrant, f"beir_{a.dataset}", vocab if name == "qdrant-hybrid" else None)
        else:
            sys.exit(f"unknown system {name}")
        run("warm", next(iter(queries.values())), next(iter(vecs.values())))
        nd, r10, r100, lat = [], [], [], []
        for qid, text in queries.items():
            t = time.perf_counter()
            chunks = run(qid, text, vecs[qid])
            lat.append((time.perf_counter() - t) * 1000)
            ranked = to_docs(chunks, owner)
            nd.append(ndcg(ranked, qrels[qid]))
            r10.append(recall(ranked, qrels[qid], 10))
            r100.append(recall(ranked, qrels[qid], 100) if len(chunks) >= DEPTH else None)
        lat.sort()
        row = {
            "ndcg@10": round(statistics.mean(nd), 4),
            "recall@10": round(statistics.mean(r10), 4),
            "recall@100": round(statistics.mean(r100), 4) if None not in r100 else None,
            "p50_ms": round(lat[len(lat) // 2], 1),
            "p95_ms": round(lat[int(len(lat) * 0.95)], 1),
            "queries": len(queries),
        }
        results[name] = row
        print(f"  {name:<20} nDCG@10 {row['ndcg@10']:.4f}  R@10 {row['recall@10']:.4f}  "
              f"R@100 {row['recall@100'] if row['recall@100'] is not None else '  -  '}  "
              f"p50 {row['p50_ms']:.0f} ms  p95 {row['p95_ms']:.0f} ms", flush=True)
        a.out.write_text(json.dumps(results, indent=2), "utf-8")


if __name__ == "__main__":
    main()
