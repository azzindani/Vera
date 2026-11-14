"""Build an N-times copy of a Ravel-prepared corpus, laid out the way the engine reads it.

    DATABASE_URL=... python dev_tools/eval/scale_clone.py --factor 3  --target vera5m
    DATABASE_URL=... python dev_tools/eval/scale_clone.py --factor 12 --target vera20m
    DATABASE_URL=... python dev_tools/eval/scale_clone.py --target vera5m --drop

`scale_probe.py` built its replicas for the old schema and the old read path: dense read
from `chunks`, the tsvector arm on. This one copies what the engine reads TODAY, and
nothing else, so a 20M-row replica fits on a workstation:

  copied        every chunk row with its body and metadata (the BM25 arm, snippets, the
                relevance floor and the factors read them), the `chunks_bm25` index rebuilt
                over all of them, `clusters` (k grows with n, ~2,000 rows each), and
                `chunk_vectors` in cluster order (the dense arm's only input)
  NOT copied    `chunks.dense` and `chunks.sparse` (NULL in the copies: with `chunk_vectors`
                and `pg_search` present the engine reads neither), and `tsv` with its RUM
                and GIN indexes (dropped: the text arm is off, `TEXT_WEIGHT=0`)

! **Latency only. Recall from a replica is meaningless.** Every answer exists `factor`
times and every centroid is duplicated, so the router's five clusters are copies of each
other. Rows touched per query are right; what they contain is not.

! Writes a SEPARATE database created with `CREATE DATABASE ... TEMPLATE`, so the real
corpus is untouched and the experiment is one `DROP DATABASE` from gone. TEMPLATE needs no
other session on the source: stop the engine first.
"""

from __future__ import annotations

import argparse
import os
import time

import psycopg

#: Columns written as NULL in the copies, and the generated column that cannot be written.
NOT_COPIED = {"dense", "sparse"}
GENERATED = {"tsv"}

DROP_FIRST = (
    "chunks_bm25", "chunks_tsv_rum", "chunks_tsv_idx", "chunks_cluster_idx",
    "chunks_indexable_idx", "chunks_identifier_idx", "chunks_doc_seq_idx",
    "chunk_vectors_cluster_idx", "chunk_vectors_id_idx",
)
INDEXES = (
    "CREATE INDEX chunks_cluster_idx ON chunks (cluster_id)",
    "CREATE INDEX chunks_indexable_idx ON chunks (indexable) WHERE indexable",
    "CREATE INDEX chunks_identifier_idx ON chunks (regulation_type, regulation_number, year)",
    "CREATE INDEX chunks_doc_seq_idx ON chunks (doc_id, seq)",
    "CREATE INDEX chunk_vectors_cluster_idx ON chunk_vectors (cluster_id)",
    "CREATE INDEX chunks_bm25 ON chunks USING bm25 (id, body) WITH (key_field = 'id')",
)


def admin(dsn: str) -> psycopg.Connection:
    c = psycopg.connect(dsn)
    c.autocommit = True
    return c


def swap_db(dsn: str, name: str) -> str:
    head, _, tail = dsn.partition("dbname=")
    rest = tail.split(" ", 1)
    return f"{head}dbname={name}" + (f" {rest[1]}" if len(rest) > 1 else "")


def step(cur: psycopg.Cursor, label: str, sql: str) -> None:
    t0 = time.monotonic()
    cur.execute(sql)
    print(f"  {label}: {time.monotonic() - t0:.0f}s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--factor", type=int, default=3, help="total copies, original included")
    ap.add_argument("--target", default="vera5m")
    ap.add_argument("--drop", action="store_true", help="delete the target and stop")
    ap.add_argument("--in-place", action="store_true",
                    help="grow DATABASE_URL itself: only on a server that is a disposable copy")
    args = ap.parse_args()

    src = os.environ["DATABASE_URL"]
    with admin(src) as c, c.cursor() as cur:
        name = c.info.dbname
        cur.execute("SELECT count(*), max(id) + 1 FROM clusters")
        k, span = cur.fetchone()
        cur.execute("SELECT count(*) FROM chunks WHERE id NOT LIKE '%#c%'")
        (rows,) = cur.fetchone()
        cur.execute("SELECT count(*) FROM chunks WHERE id LIKE '%#c%'")
        (already,) = cur.fetchone()
    if args.in_place:
        # ! For a server that is itself a disposable copy (a copied volume). With
        # `pg_search` preloaded, `CREATE DATABASE ... TEMPLATE` crashes the server
        # (exit code 2, no message) under both WAL_LOG and FILE_COPY, so on such a server
        # the only copy is below Postgres: copy the volume, start a server on it, run this.
        if already:
            raise SystemExit(f"{name} already holds {already:,} copied rows: start from a "
                             "fresh copy of the volume")
        print(f"IN PLACE on {name}: {rows:,} chunks · {k} clusters -> "
              f"{rows * args.factor:,} chunks, {k * args.factor:,} clusters", flush=True)
        dst = src
    else:
        # ! From `postgres`, ✗ the source: TEMPLATE refuses a source anyone is on.
        with admin(swap_db(src, "postgres")) as c, c.cursor() as cur:
            cur.execute(f'DROP DATABASE IF EXISTS "{args.target}" WITH (FORCE)')
            if args.drop:
                print(f"dropped {args.target}")
                return
            print(f"source {rows:,} chunks · {k} clusters -> {rows * args.factor:,} "
                  f"chunks, {k * args.factor:,} clusters in {args.target}", flush=True)
            step(cur, "template copy",
                 f'CREATE DATABASE "{args.target}" TEMPLATE "{name}" STRATEGY = FILE_COPY')
        dst = swap_db(src, args.target)

    with admin(dst) as c, c.cursor() as cur:
        for idx in DROP_FIRST:
            cur.execute(f"DROP INDEX IF EXISTS {idx}")
        cur.execute("ALTER TABLE chunks DROP COLUMN IF EXISTS tsv")
        cur.execute(
            "SELECT column_name FROM information_schema.columns"
            " WHERE table_name = 'chunks' ORDER BY ordinal_position")
        cols = [r[0] for r in cur.fetchall() if r[0] not in GENERATED]
        target_cols = ", ".join(cols)

        def source(i: int) -> str:
            out = []
            for col in cols:
                if col == "id":
                    out.append(f"id || '#c{i}'")
                elif col == "cluster_id":
                    out.append(f"cluster_id + {i * span}")
                elif col in NOT_COPIED:
                    out.append("NULL")
                else:
                    out.append(col)
            return ", ".join(out)

        # ! The vectors are rebuilt whole, copy by copy in cluster order. Copy i's cluster
        # ids are offset by i * span, so appending the copies one after another keeps the
        # table globally sorted by cluster: no sort over the whole replica is needed.
        cur.execute("ALTER TABLE chunk_vectors RENAME TO chunk_vectors_src")
        cur.execute(
            "CREATE TABLE chunk_vectors (LIKE chunk_vectors_src INCLUDING ALL)")
        cur.execute("ALTER TABLE chunk_vectors ALTER COLUMN dense SET STORAGE PLAIN")
        for i in range(args.factor):
            vid = "id" if i == 0 else f"id || '#c{i}'"
            step(cur, f"vectors {i + 1}/{args.factor}",
                 "INSERT INTO chunk_vectors (cluster_id, id, dense)"
                 f" SELECT cluster_id + {i * span}, {vid}, dense"
                 " FROM chunk_vectors_src ORDER BY cluster_id, id")
        cur.execute("DROP TABLE chunk_vectors_src")

        for i in range(1, args.factor):
            step(cur, f"chunks {i + 1}/{args.factor}",
                 f"INSERT INTO chunks ({target_cols}) SELECT {source(i)} FROM chunks"
                 " WHERE id NOT LIKE '%#c%'")
            cur.execute(
                "INSERT INTO clusters (id, corpus_id, centroid, row_count, generation)"
                f" SELECT id + {i * span}, corpus_id, centroid, row_count, generation"
                f" FROM clusters WHERE id < {span}")

        for sql in INDEXES:
            step(cur, sql.split()[2], sql)
        step(cur, "analyze", "ANALYZE chunks; ANALYZE chunk_vectors; ANALYZE clusters")

        cur.execute("SELECT count(*) FROM chunks")
        (n,) = cur.fetchone()
        cur.execute("SELECT count(*) FROM chunk_vectors")
        (nv,) = cur.fetchone()
        cur.execute("SELECT count(*) FROM clusters")
        (kk,) = cur.fetchone()
        cur.execute("SELECT pg_size_pretty(pg_database_size(current_database())),"
                    " pg_size_pretty(pg_relation_size('chunks_bm25')),"
                    " pg_size_pretty(pg_total_relation_size('chunk_vectors'))")
        size, bm25, vec = cur.fetchone()
    print(f"\n{args.target}: {n:,} chunks · {nv:,} vectors · {kk:,} clusters · {size}"
          f" (bm25 {bm25}, chunk_vectors {vec})")
    print("! Latency only: every answer exists", args.factor, "times.")


if __name__ == "__main__":
    main()
