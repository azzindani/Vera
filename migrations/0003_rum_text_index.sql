-- Optional accelerator: serve the text arm's ordering from the index.
--
-- The text arm is the most expensive part of a search. It deliberately uses OR
-- semantics — a natural question ANDed together matches nothing here, measured
-- at zero rows for 42 of 44 eval queries — and an OR query over this corpus
-- matches a median of 219,792 rows, 62% of it. A GIN index finds those matches
-- quickly, but `ts_rank` then has to score and sort every one of them.
--
-- RUM keeps the ranking data inside the index, so `ORDER BY tsv <=> tq` becomes
-- a single ordered index scan with no sort at all. Measured over 44 eval
-- queries: p50 750ms -> 404ms, p95 1481ms -> 807ms, with Recall@5 unchanged at
-- 40.9% and the same rank-1 result on 44/44.
--
-- ! OPTIONAL, and the engine treats it that way. RUM is not in
-- `pgvector/pgvector:pg16`, so CI and a fresh clone run without it;
-- `SearchOps::has_rum` probes for it and falls back to `ts_rank`. Both paths
-- return the same rows (98.9% top-20 overlap), so skipping this migration costs
-- latency, never answers. Apply it only where RUM is actually installed:
--
--     apt-get install -y build-essential postgresql-server-dev-16
--     curl -sSL https://github.com/postgrespro/rum/archive/refs/heads/master.tar.gz \
--       | tar xz && cd rum-master && make USE_PGXS=1 && make USE_PGXS=1 install
--
-- ! The GIN index in 0001 stays. It still serves the `tsv @@ tq` match, it is
-- what the fallback path ranks against, and dropping it would strand any
-- deployment that has not built RUM.

CREATE EXTENSION IF NOT EXISTS rum;

-- ~25s and ~134MB on a 355K-chunk corpus (the GIN index is 121MB).
CREATE INDEX IF NOT EXISTS chunks_tsv_rum ON chunks USING rum (tsv rum_tsvector_ops);
