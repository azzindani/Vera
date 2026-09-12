"""One pass: SQLite -> dense + sparse -> Postgres, with the recipe recorded.

! The load-bearing property is that text, dense vector and sparse vector are
produced by ONE run over ONE text. The corpus we inherited had them produced by
separate runs, which is why its vectors could not be reproduced from its text.
Doing all three here makes that class of drift structurally impossible.

! Streaming, batch at a time. Holding 182K x 1024 Python floats would be ~5.8 GB
of object overhead, and a crash at 90% would throw away two hours of GPU. Each
batch is embedded, written and committed, so ingest_progress makes a re-run
continue rather than restart.

Usage:
    python ingest.py --manifest ../../.test/runs/fixture.json --run-id fixture-01
    python ingest.py --manifest ../../.test/runs/spike.json   --run-id spike-01 \
        --fit-on-full-corpus
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from batching import batched  # noqa: E402
from sparse import Bm25Vectorizer  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / ".test" / "ID_REG_DB_2511" / "id_regulations.db"
MODEL_DIR = ROOT / ".test" / "Qwen3-Embedding-0.6B"

TEI = os.environ.get("EMBED_ENDPOINT", "http://localhost:8080")
PG = os.environ.get(
    "DATABASE_URL", "host=localhost port=5432 dbname=vera user=vera password=vera"
)

DENSE_MODEL = "qwen/qwen3-embedding-0.6b"
DENSE_DIM = 1024
DENSE_POOLING = "last-token"
DENSE_DTYPE = "float16"

# Stored and readable, but never indexed. 44K+ corpus rows are the boilerplate
# "Cukup jelas." ("self-explanatory"); embedding them yields thousands of
# near-identical vectors that distort ranking and answer no query.
MIN_INDEXABLE_CHARS = 40

# The source extractor truncated at exactly 32767 chars. Those rows lost
# content, so they are flagged -- a truncated article must never be cited as
# though it were complete.
SOURCE_TRUNCATION_LEN = 32767

ROUNDTRIP_SAMPLE = 24
ROUNDTRIP_MIN_COSINE = 0.999

# ! Rows buffered before a flush. Measured on this schema: per-row execute
# manages 22 rows/s, executemany in blocks of 500 manages 2,268 -- a 101x
# difference that had the writer, not the GPU, setting the pace (the first
# spike ran at 12/s against an embedding rate of 26/s).
#
# The cost of a crash is one unflushed window, about 20 seconds of GPU, and
# ingest_progress is written in the same transaction as the rows it describes
# so a resume can never think work is done that is not.
FLUSH_ROWS = 500

INSERT_CHUNK = """
INSERT INTO chunks (
    id, corpus_id, regulation_type, enacting_body, regulation_number, year,
    about, chapter, article, chunk_no, body, source_url, source_title,
    truncated_at_source, indexable, dense, sparse
) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
ON CONFLICT (id) DO NOTHING
"""

INSERT_PROGRESS = """
INSERT INTO ingest_progress (chunk_id, run_id, corpus_id, stage)
VALUES (%s,%s,%s,'done')
ON CONFLICT (chunk_id) DO UPDATE SET stage = 'done', updated_at = now()
"""


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def embed(texts: list[str]) -> list[list[float]]:
    body = json.dumps({"inputs": texts, "truncate": False}).encode()
    req = urllib.request.Request(
        f"{TEI}/embed", data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)


def cosine(a, b) -> float:
    d = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return d / (na * nb) if na and nb else 0.0


def lit(vec: list[float]) -> str:
    return "[" + ",".join(f"{v:.6g}" for v in vec) + "]"


def source_title(row: dict) -> str:
    """A human-readable document name, composed only from recorded fields."""
    parts = [row["regulation_type"], f"No. {row['regulation_number']}",
             f"Tahun {row['year']}"]
    title = " ".join(p for p in parts if p and p != "None")
    return f"{title} — {row['about']}" if row.get("about") else title


def chunk_values(r: dict, corpus_id: str, dense, sparse_lit):
    return (
        r["global_id"], corpus_id, r["regulation_type"], r["enacting_body"],
        r["regulation_number"],
        int(r["year"]) if str(r["year"]).isdigit() else None,
        r["about"], r["chapter"], r["article"], r["chunk_id"], r["_body"],
        None,  # ! no source_url in this corpus · never invented (invariant 8)
        source_title(r), r["_truncated"], r["_indexable"],
        lit(dense) if dense else None, sparse_lit,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--vectorizer", default=None,
                    help="reuse a fitted BM25 artifact instead of fitting")
    ap.add_argument("--fit-on-full-corpus", action="store_true",
                    help="fit BM25 on all 748K texts, not just this sample")
    ap.add_argument("--max-features", type=int, default=20_000)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    try:
        import psycopg
    except ImportError:
        sys.exit("psycopg not installed · pip install 'psycopg[binary]'")

    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    chunk_ids = manifest["chunk_ids"]
    corpus_id = args.run_id
    print(f"run_id={args.run_id}  chunks={len(chunk_ids):,}", flush=True)

    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    wanted = set(chunk_ids)
    rows = [
        dict(r) for r in con.execute(
            "SELECT global_id, regulation_type, enacting_body, regulation_number,"
            " year, about, chapter, article, content, chunk_id FROM regulations"
        ) if r["global_id"] in wanted
    ]
    print(f"read {len(rows):,} rows from sqlite", flush=True)

    for r in rows:
        body = r["content"] or ""
        r["_body"] = body
        r["_truncated"] = len(body) == SOURCE_TRUNCATION_LEN
        r["_indexable"] = len(body.strip()) >= MIN_INDEXABLE_CHARS

    indexable = [r for r in rows if r["_indexable"]]
    skipped = [r for r in rows if not r["_indexable"]]
    print(f"indexable {len(indexable):,} · skipped {len(skipped):,} "
          f"(below {MIN_INDEXABLE_CHARS} chars)", flush=True)

    # -- sparse -------------------------------------------------------------
    vec_path = manifest_path.with_name(f"{args.run_id}.bm25.json")
    if args.vectorizer:
        vz = Bm25Vectorizer.load(args.vectorizer)
        vocab_sha = sha256_file(Path(args.vectorizer))
        print(f"bm25 loaded from {args.vectorizer} (dim={vz.dim})", flush=True)
    else:
        t0 = time.time()
        if args.fit_on_full_corpus:
            print("fitting bm25 on the FULL corpus (text-only, no GPU)...", flush=True)
            fit_docs = (c for (c,) in con.execute(
                "SELECT content FROM regulations WHERE content IS NOT NULL"))
        else:
            fit_docs = (r["_body"] for r in indexable)
        vz = Bm25Vectorizer.fit(fit_docs, max_features=args.max_features)
        vocab_sha = vz.save(vec_path)
        print(f"bm25 fitted: dim={vz.dim} docs={vz.n_docs:,} avgdl={vz.avgdl:.1f}"
              f" in {time.time() - t0:.1f}s -> {vec_path.name}", flush=True)

    if args.dry_run:
        print("\n--dry-run · stopping before embedding")
        con.close()
        return

    schema = (Path(__file__).parent / "schema.sql").read_text(encoding="utf-8")
    schema = schema.replace("{{DENSE_DIM}}", str(DENSE_DIM))
    schema = schema.replace("{{SPARSE_DIM}}", str(vz.dim))

    gate: list[tuple[dict, list[float]]] = []
    gate_every = max(1, len(indexable) // ROUNDTRIP_SAMPLE)
    t0 = time.time()
    done = 0

    with psycopg.connect(PG, autocommit=False) as pg:
        with pg.cursor() as cur:
            cur.execute(schema)
            cur.execute(
                """
                INSERT INTO corpus_meta (
                    id, run_id, dense_model, dense_dim, dense_pooling,
                    dense_normalize, dense_dtype, dense_instruction,
                    tokenizer_sha256, sparse_scheme, sparse_dim, sparse_k1,
                    sparse_b, sparse_vocab_sha256, sparse_fit_docs,
                    source_db, manifest_sha256, notes
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (id) DO UPDATE SET created_at = now()
                """,
                (corpus_id, args.run_id, DENSE_MODEL, DENSE_DIM, DENSE_POOLING,
                 True, DENSE_DTYPE, None,
                 sha256_file(MODEL_DIR / "tokenizer.json"),
                 "bm25", vz.dim, vz.k1, vz.b, vocab_sha, vz.n_docs,
                 str(DB), sha256_file(manifest_path),
                 f"{len(rows)} chunks · {len(skipped)} non-indexable"),
            )
            pg.commit()

            # Resume: skip what a previous run already finished.
            cur.execute(
                "SELECT chunk_id FROM ingest_progress WHERE corpus_id = %s"
                " AND stage = 'done'", (corpus_id,))
            already = {r[0] for r in cur.fetchall()}
            if already:
                print(f"resuming · {len(already):,} chunks already done", flush=True)
                indexable = [r for r in indexable if r["global_id"] not in already]
                skipped = [r for r in skipped if r["global_id"] not in already]

            # Non-indexable rows: stored and readable, no vectors.
            for i in range(0, len(skipped), FLUSH_ROWS):
                block = skipped[i:i + FLUSH_ROWS]
                cur.executemany(
                    INSERT_CHUNK,
                    [chunk_values(r, corpus_id, None, None) for r in block],
                )
                cur.executemany(
                    INSERT_PROGRESS,
                    [(r["global_id"], args.run_id, corpus_id) for r in block],
                )
                pg.commit()

            pending_rows: list[tuple] = []
            pending_progress: list[tuple] = []

            def flush() -> None:
                if not pending_rows:
                    return
                cur.executemany(INSERT_CHUNK, pending_rows)
                cur.executemany(INSERT_PROGRESS, pending_progress)
                pg.commit()
                pending_rows.clear()
                pending_progress.clear()

            for batch in batched(indexable, text_of=lambda r: r["_body"]):
                vecs = embed([r["_body"] for r in batch])
                for r, v in zip(batch, vecs):
                    if v is None or any(x is None for x in v):
                        sys.exit(f"null vector for {r['global_id']} · refusing")
                    if len(v) != DENSE_DIM:
                        sys.exit(f"dim {len(v)} != {DENSE_DIM} · refusing")
                    sp = vz.to_sparsevec(vz.document(r["_body"]))
                    pending_rows.append(chunk_values(r, corpus_id, v, sp))
                    pending_progress.append(
                        (r["global_id"], args.run_id, corpus_id)
                    )
                    if done % gate_every == 0 and len(gate) < ROUNDTRIP_SAMPLE:
                        gate.append((r, v))
                    done += 1
                if len(pending_rows) >= FLUSH_ROWS:
                    flush()
                el = time.time() - t0
                rate = done / el if el else 0
                eta = (len(indexable) - done) / rate / 60 if rate else 0
                print(f"\r  {done:,}/{len(indexable):,}  {rate:.1f}/s  "
                      f"eta {eta:.0f}m   ", end="", flush=True)

            # ! The tail. Without this the last partial window -- up to 499
            # rows -- is embedded, paid for, and then silently dropped.
            flush()

    el = time.time() - t0
    print(f"\n  loaded {done:,} chunks in {el / 60:.1f}m "
          f"({done / el if el else 0:.1f}/s)", flush=True)

    # -- round-trip gate ----------------------------------------------------
    print("\n--- round-trip gate (EMBEDDING.md §4) ---", flush=True)
    if not gate:
        print("  (nothing new embedded · gate skipped)")
        con.close()
        return
    fresh = []
    for b in batched([g[0]["_body"] for g in gate]):
        fresh += embed(b)
    sims = [cosine(v, f) for (_, v), f in zip(gate, fresh)]
    worst, mean = min(sims), sum(sims) / len(sims)
    print(f"  n={len(sims)}  mean={mean:.6f}  worst={worst:.6f}  "
          f"threshold={ROUNDTRIP_MIN_COSINE}")
    if worst < ROUNDTRIP_MIN_COSINE:
        sys.exit(f"ROUND-TRIP FAILED · worst={worst:.6f} · corpus not trustworthy")
    print("  PASS · query path and corpus share a vector space")
    con.close()


if __name__ == "__main__":
    main()
