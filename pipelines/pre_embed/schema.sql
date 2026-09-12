-- Spike schema · rendered by ingest.py with the run's actual dimensions.
--
-- ! This supersedes migrations/0001_init.sql FOR THE SPIKE ONLY. That file is
-- the production target at halfvec(4096); this one is parameterised because the
-- dimension is a property of the corpus, not a constant (see corpus_meta).
--
-- Placeholders: {{DENSE_DIM}} {{SPARSE_DIM}}

CREATE EXTENSION IF NOT EXISTS vector;

-- ---------------------------------------------------------------------------
-- corpus_meta · the recipe, recorded with the data it produced.
--
-- ! This table is the whole lesson of the August/November drift. Vectors whose
-- recipe is not written down cannot be verified, extended, or trusted. The
-- engine validates a query embedding against THIS ROW, not against a constant
-- compiled into the binary -- which is what makes invariant 2 enforceable
-- rather than aspirational.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS corpus_meta (
    id              TEXT PRIMARY KEY,
    run_id          TEXT        NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),

    -- Dense side · every field that changes the vector space.
    dense_model     TEXT    NOT NULL,
    dense_dim       INTEGER NOT NULL,
    dense_pooling   TEXT    NOT NULL,
    dense_normalize BOOLEAN NOT NULL,
    dense_dtype     TEXT    NOT NULL,
    dense_instruction TEXT,
    tokenizer_sha256  TEXT  NOT NULL,

    -- Sparse side · the vocabulary is part of the recipe. Without it a query
    -- cannot be projected into the space at all.
    sparse_scheme   TEXT    NOT NULL,
    sparse_dim      INTEGER NOT NULL,
    sparse_k1       REAL    NOT NULL,
    sparse_b        REAL    NOT NULL,
    sparse_vocab_sha256 TEXT NOT NULL,
    sparse_fit_docs INTEGER NOT NULL,

    source_db       TEXT    NOT NULL,
    manifest_sha256 TEXT    NOT NULL,
    notes           TEXT
);

-- ---------------------------------------------------------------------------
-- chunks
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS chunks (
    id              TEXT PRIMARY KEY,
    corpus_id       TEXT NOT NULL REFERENCES corpus_meta(id),

    regulation_type TEXT,
    enacting_body   TEXT,
    regulation_number TEXT,
    year            INTEGER,
    about           TEXT,
    chapter         TEXT,
    article         TEXT,
    chunk_no        INTEGER,

    body            TEXT NOT NULL,

    -- ! Nullable for the spike. This corpus has no source_url, and inventing
    -- one is forbidden (invariant 8). Results carrying NULL must be reported as
    -- provenance-incomplete rather than dressed up as verifiable.
    source_url      TEXT,
    source_title    TEXT NOT NULL,

    -- Honest flags, set at ingest, never inferred at query time.
    truncated_at_source BOOLEAN NOT NULL DEFAULT FALSE,
    indexable       BOOLEAN NOT NULL DEFAULT TRUE,

    cluster_id      INTEGER,

    dense           halfvec({{DENSE_DIM}}),
    sparse          sparsevec({{SPARSE_DIM}}),

    -- ! 'indonesian', not 'simple'. Indonesian is heavily affixed: dikenakan
    -- and dikenai are one word inflected, and 'simple' indexes them as two
    -- unrelated terms. Verified not to disturb identifier tokenisation.
    tsv tsvector GENERATED ALWAYS AS (to_tsvector('indonesian', body)) STORED,

    ingested_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS chunks_tsv_idx      ON chunks USING GIN (tsv);
CREATE INDEX IF NOT EXISTS chunks_cluster_idx  ON chunks (cluster_id);
CREATE INDEX IF NOT EXISTS chunks_indexable_idx ON chunks (indexable) WHERE indexable;

-- The exact-identifier path (invariant 4) must never depend on routing, so it
-- gets its own btree and searches globally.
CREATE INDEX IF NOT EXISTS chunks_identifier_idx
    ON chunks (regulation_type, regulation_number, year);

-- ---------------------------------------------------------------------------
-- ingest_progress · resumability is a table, not a flag.
--
-- Lets a later run continue the remaining rows without re-embedding what is
-- already done, and makes "which run produced this chunk" answerable.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ingest_progress (
    chunk_id    TEXT PRIMARY KEY,
    run_id      TEXT NOT NULL,
    corpus_id   TEXT NOT NULL,
    stage       TEXT NOT NULL,           -- loaded | dense | sparse | done
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ingest_progress_run_idx ON ingest_progress (run_id, stage);
