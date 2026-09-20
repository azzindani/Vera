"""Build a synthetic corpus N times the real one, so scaling stops being arithmetic.

Every claim about 5M or 10M rows in `HARDWARE.md` §6 is an extrapolation from
355,621. The two arms that scan globally are linear in `n` by construction, so the
extrapolation is *probably* right -- and "probably" is what §5.8 exists to forbid.
This builds the corpus and lets the real server answer.

    DATABASE_URL=... python dev_tools/eval/scale_probe.py --factor 14
    DATABASE_URL=... python dev_tools/eval/scale_probe.py --drop

! Writes to a SEPARATE database (default `vera5m`), created with
`CREATE DATABASE ... TEMPLATE`, so the real corpus is untouched and the whole
experiment is one `DROP DATABASE` away from gone.

! **Latency only. Recall from this corpus is meaningless** -- every answer chunk
exists `factor` times, so every arm finds it trivially. Do not run `run.py` or
`e2e.py` against it and quote the number.

What it does and does not reproduce:

  reproduces    row count, so the sparse and text arms grind through a real 5M
                rows; total bytes, so the page cache faces the real working set;
                cluster COUNT, so layer-2's O(k) centroid scan is realistic
  reproduces    cluster SIZE -- copy i gets cluster ids offset by i × 177, so the
                result is ~2,500 clusters of ~2,000 rows, ✗ 177 clusters of
                28,000. A real 5M corpus re-clusters the same way (k ∝ n), so
                this is the shape that matters for the dense arm
  does NOT      vocabulary growth. A real 5M corpus has more distinct terms, so
                IDF and posting-list lengths differ. Term FREQUENCIES scale
                correctly, which is what drives the scan cost being measured
  does NOT      semantic diversity. Centroids are replicated exactly, so the five
                the router picks are copies of one cluster. Rows touched is
                identical, so dense LATENCY is right; routing RECALL is not

! Indexes are dropped before the bulk insert and rebuilt after. Inserting 4.6M
rows into a live RUM index costs far more than rebuilding it once, and the
rebuild is the honest way to measure index size at scale anyway.
"""

from __future__ import annotations

import argparse
import os
import time

import psycopg

# ! tsv is GENERATED ALWAYS and must not appear here; Postgres recomputes it per
# row, which is also what makes the copies' text index genuinely populated.
COLS = (
    "id, corpus_id, regulation_type, enacting_body, regulation_number, year, "
    "about, chapter, article, chunk_no, body, source_url, source_title, "
    "truncated_at_source, indexable, cluster_id, sparse, ingested_at, dense"
)

INDEXES = (
    "CREATE INDEX chunks_cluster_idx ON chunks USING btree (cluster_id)",
    "CREATE INDEX chunks_indexable_idx ON chunks USING btree (indexable) WHERE indexable",
    "CREATE INDEX chunks_identifier_idx ON chunks USING btree "
    "(regulation_type, regulation_number, year)",
    "CREATE INDEX chunks_tsv_rum ON chunks USING rum (tsv)",
    # ! The BM25 index too, or the replica falls back to the sparsevec scan
    # while the real corpus uses pg_search -- and the two scales then measure
    # different engines. Skipped automatically where pg_search is absent.
    "CREATE INDEX chunks_bm25 ON chunks USING bm25 (id, body) WITH (key_field='id')",
)
# ! chunks_bm25 belongs here too. The clone is a file-level copy of a database
# that already has it, so leaving it in place means 4.8M rows are inserted into a
# LIVE Tantivy index -- slow, and it leaves the index bloated in a way that makes
# the size measurement a lie.
DROP_FIRST = ("chunks_cluster_idx", "chunks_indexable_idx",
              "chunks_identifier_idx", "chunks_tsv_rum", "chunks_tsv_idx",
              "chunks_bm25")


def admin(dsn: str) -> psycopg.Connection:
    """! autocommit: CREATE/DROP DATABASE cannot run inside a transaction."""
    c = psycopg.connect(dsn)
    c.autocommit = True
    return c


def swap_db(dsn: str, name: str) -> str:
    head, _, tail = dsn.partition("dbname=")
    rest = tail.split(" ", 1)
    return f"{head}dbname={name}" + (f" {rest[1]}" if len(rest) > 1 else "")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--factor", type=int, default=14,
                    help="total copies of the corpus, including the original")
    ap.add_argument("--target", default="vera5m")
    ap.add_argument("--drop", action="store_true", help="delete it and stop")
    args = ap.parse_args()

    src = os.environ["DATABASE_URL"]
    dst = swap_db(src, args.target)

    with admin(src) as c, c.cursor() as cur:
        cur.execute("SELECT count(*), max(cluster_id) FROM chunks")
        rows, max_cid = cur.fetchone()
        cur.execute(
            "SELECT pg_terminate_backend(pid) FROM pg_stat_activity"
            " WHERE datname = %s AND pid <> pg_backend_pid()", (args.target,))
        cur.execute(f'DROP DATABASE IF EXISTS "{args.target}"')
        if args.drop:
            print(f"dropped {args.target}")
            return
        print(f"source {rows:,} rows · {max_cid + 1} clusters "
              f"-> {rows * args.factor:,} rows in {args.target}", flush=True)

        # ! TEMPLATE is a file-level copy: seconds, ✗ a dump and restore. It
        # needs no other session connected to the source, hence the terminate
        # above -- and it is why this tool insists on its own database.
        t0 = time.monotonic()
        cur.execute(
            "SELECT pg_terminate_backend(pid) FROM pg_stat_activity"
            " WHERE datname = current_database() AND pid <> pg_backend_pid()")
        cur.execute(f'CREATE DATABASE "{args.target}" TEMPLATE {c.info.dbname}')
        print(f"  cloned in {time.monotonic() - t0:.0f}s", flush=True)

    span = max_cid + 1
    with admin(dst) as c, c.cursor() as cur:
        for idx in DROP_FIRST:
            cur.execute(f"DROP INDEX IF EXISTS {idx}")
        print("  indexes dropped", flush=True)

        for i in range(1, args.factor):
            t0 = time.monotonic()
            # ! The ORIGINAL rows only (`id NOT LIKE '%%#c%%'`), or copy 2 would
            # duplicate copy 1 and the factor would be exponential.
            cur.execute(
                f"INSERT INTO chunks ({COLS}) SELECT "
                f"  id || '#c{i}', corpus_id, regulation_type, enacting_body,"
                f"  regulation_number, year, about, chapter, article, chunk_no,"
                f"  body, source_url, source_title, truncated_at_source,"
                f"  indexable, cluster_id + {i * span}, sparse, ingested_at, dense"
                f" FROM chunks WHERE id NOT LIKE '%%#c%%'")
            cur.execute(
                "INSERT INTO clusters (id, corpus_id, centroid, row_count, generation)"
                f" SELECT id + {i * span}, corpus_id, centroid, row_count, generation"
                f" FROM clusters WHERE id < {span}")
            print(f"  copy {i}/{args.factor - 1} in {time.monotonic() - t0:.0f}s",
                  flush=True)

        cur.execute("SELECT EXISTS (SELECT 1 FROM pg_extension"
                    " WHERE extname = 'pg_search')")
        has_bm25 = cur.fetchone()[0]
        for sql in INDEXES:
            if "bm25" in sql and not has_bm25:
                print("  chunks_bm25 SKIPPED - pg_search not installed;"
                      " the lexical arm here will use the sparsevec fallback",
                      flush=True)
                continue
            t0 = time.monotonic()
            cur.execute(sql)
            print(f"  {sql.split()[2]} built in {time.monotonic() - t0:.0f}s",
                  flush=True)
        cur.execute("ANALYZE chunks")

        cur.execute("SELECT count(*) FROM chunks")
        n = cur.fetchone()[0]
        cur.execute("SELECT count(*) FROM clusters")
        k = cur.fetchone()[0]
        cur.execute("SELECT pg_size_pretty(pg_database_size(current_database()))")
        size = cur.fetchone()[0]

    print(f"\n{args.target}: {n:,} rows · {k:,} clusters · {size}")
    print("\nPoint the engine at it and measure:")
    print(f"  DATABASE_URL=...dbname={args.target}  (engine env)")
    print("  VERA_HTTP=http://localhost:8081 python dev_tools/eval/cluster_batch_sweep.py")
    print("\n! Latency only. Recall here is meaningless -- every answer exists "
          f"{args.factor} times.")


if __name__ == "__main__":
    main()
