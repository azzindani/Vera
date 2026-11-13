-- 0005 · the corpus's own canary.
--
-- A sentence and the vector the corpus's embedder gave it, recorded by the bundle
-- (`reference.txt`, `reference.npy`) and stamped here by `ravel db prepare` (Ravel's loader).
-- The engine's startup check embeds the sentence through the configured provider, with
-- the corpus's own query prefix, and refuses to serve unless it lands on this vector.
--
-- Why a table and not the chunk-body check it replaces: a corpus whose documents were
-- embedded behind a header cannot be reproduced from a chunk's body alone. A correct
-- provider scores about 0.84 against the stored vector and a wrong one about 0.73, so
-- no cosine bar separates them. A sentence embedded the same way on both sides can be
-- held to 0.98.
--
-- `halfvec` is left without a width on purpose: the width is the corpus's to declare
-- (`corpus_meta.dense_dim`), not this file's.
--
-- Idempotent. A corpus with no row here keeps the body check.

CREATE TABLE IF NOT EXISTS corpus_canary (
    corpus_id TEXT PRIMARY KEY REFERENCES corpus_meta(id) ON DELETE CASCADE,
    text      TEXT NOT NULL,
    dense     halfvec NOT NULL
);
