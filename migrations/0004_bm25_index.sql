-- The lexical arm, as an index instead of a scan.
--
-- The `sparse` arm is BM25 over a `sparsevec` column with NO index: pgvector's
-- `<#>` walks every row. That is affordable at 355K (150 ms) and is the worst
-- scaling in the system at 5M (6,282 ms — 41.9x for 14x the rows, because the
-- 1.6 GB column stops fitting the page cache). `docs/HARDWARE.md` §6.
--
-- `pg_search` is Tantivy — Lucene's algorithms in Rust — as a Postgres index
-- type. Same arm, same job, with posting lists instead of a scan.
--
-- Measured on this corpus, 39 labelled queries (`dev_tools/eval/bm25_compare.py`
-- and `fused_bm25.py`):
--
--     latency      355K  170 ms -> 86 ms      5.1M  6,282 ms -> 1,858 ms
--     overlap      84.4% Jaccard with the arm it replaces
--     fused        dense+bm25+text scores 82.1% @20 and 87.2% @50 —
--                  IDENTICAL to dense+sparse+text, and +2.5 points at @5
--     index        58 MB at 355K, 1,236 MB at 5.1M; builds in 5 s / 95 s
--
-- ! OPTIONAL, and the engine treats it that way — the same contract as
-- migration 0003. `SearchOps::has_bm25` probes for the extension and the index
-- and falls back to the `sparsevec` scan, so a deployment without pg_search
-- still answers. It answers more slowly, never differently.
--
-- ! That is why `BM25_VOCAB` and the `sparse` column are NOT removed. They are
-- the fallback path, exactly as the GIN index backs RUM (`HARDWARE.md` §5).
--
-- ! Apply it only where pg_search is installed. `docker/Dockerfile.db` ships it;
-- a stock `pgvector/pgvector:pg16` does not, and CREATE EXTENSION will fail
-- rather than degrade. That failure is the point: it is visible.
--
--     psql "$DATABASE_URL" -v ON_ERROR_STOP=1 -f migrations/0004_bm25_index.sql
--
-- ! Build it AFTER loading a corpus. Like any index, incremental maintenance
-- during a bulk load costs far more than one build afterwards.

CREATE EXTENSION IF NOT EXISTS pg_search;

-- ! key_field is the primary key the index scores against, and `paradedb.score`
-- takes that same column. Indexing `body` alone is deliberate: the arm ranks
-- text, and every metadata filter the engine applies is already served by a
-- btree.
CREATE INDEX IF NOT EXISTS chunks_bm25
    ON chunks USING bm25 (id, body)
    WITH (key_field = 'id');

ANALYZE chunks;
