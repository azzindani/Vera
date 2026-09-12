//! `store`
//!
//! Postgres and pgvector access · the only crate that talks to the database.
//!
//! ! The query path is **read-only** (`CLAUDE.md` §5.6). Nothing here writes:
//! ingestion and cluster maintenance are offline pipelines that swap versions
//! atomically, and a serving engine that can write is a serving engine that can
//! corrupt a corpus mid-query.
//!
//! ! Vectors cross the wire as **text literals**, cast in SQL. halfvec and
//! sparsevec have no stable Rust binary codec, and a hand-rolled one is a
//! silent-corruption bug waiting to happen. Text is slower per row and
//! obviously correct; the per-cluster scan is disk-bound anyway.

pub mod search;

use std::fmt::Write as _;

pub use search::{ExactHit, Scored, SearchOps};

/// Connection pooling and the corpus contract.
pub use deadpool_postgres::Pool;

#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    #[error("database: {0}")]
    Db(#[from] tokio_postgres::Error),

    #[error("pool: {0}")]
    Pool(String),

    #[error("no corpus_meta row · the database holds no ingested corpus")]
    NoCorpus,

    #[error(
        "corpus was embedded with '{corpus_model}' at {corpus_dim} dims but this \
         engine is configured for '{engine_model}' at {engine_dim} · queries would \
         land in a different vector space · refusing to serve"
    )]
    CorpusMismatch {
        corpus_model: String,
        corpus_dim: i32,
        engine_model: String,
        engine_dim: usize,
    },
}

/// The recipe that produced the corpus · `corpus_meta`.
///
/// ! This row is the authority on the vector space, ✗ a constant compiled into
/// the binary. The corpus we inherited had no such record, which is why its
/// vectors could not be verified, extended, or trusted. Validating the engine
/// against what the corpus *declares* is what makes invariant 2 enforceable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CorpusMeta {
    pub id: String,
    pub dense_model: String,
    pub dense_dim: i32,
    pub dense_pooling: String,
    pub dense_normalize: bool,
    pub sparse_scheme: String,
    pub sparse_dim: i32,
    pub sparse_vocab_sha256: String,
}

impl CorpusMeta {
    /// Refuse to serve unless the engine and the corpus agree.
    ///
    /// # Errors
    /// [`StoreError::CorpusMismatch`] when the model or width differs.
    pub fn ensure_compatible(
        &self,
        engine_model: &str,
        engine_dim: usize,
    ) -> Result<(), StoreError> {
        if self.dense_model != engine_model
            || usize::try_from(self.dense_dim).unwrap_or(0) != engine_dim
        {
            return Err(StoreError::CorpusMismatch {
                corpus_model: self.dense_model.clone(),
                corpus_dim: self.dense_dim,
                engine_model: engine_model.to_owned(),
                engine_dim,
            });
        }
        Ok(())
    }
}

/// Build a bounded connection pool.
///
/// ! `max_size` is the database half of the concurrency budget
/// (`MCP_ENGINE.md` §4). Postgres connections each carry `work_mem`, so an
/// unbounded pool is an OOM vector on the 8 GB target just as surely as an
/// unbounded request queue is.
///
/// `NoTls` because the engine and the database share a host or a private
/// network in every deployment described in `HARDWARE.md`. Exposing Postgres
/// across a public network is a deployment change that must add TLS here.
///
/// # Errors
/// An unparseable connection string, or a pool that cannot be built.
pub fn connect(conn_str: &str, max_size: usize) -> Result<Pool, StoreError> {
    let pg_cfg: tokio_postgres::Config = conn_str.parse()?;
    let mgr = deadpool_postgres::Manager::from_config(
        pg_cfg,
        tokio_postgres::NoTls,
        deadpool_postgres::ManagerConfig {
            recycling_method: deadpool_postgres::RecyclingMethod::Fast,
        },
    );
    Pool::builder(mgr)
        .max_size(max_size)
        .build()
        .map_err(|e| StoreError::Pool(e.to_string()))
}

/// Render a dense vector as a pgvector literal: `[0.1,0.2,…]`.
#[must_use]
pub fn dense_literal(v: &[f32]) -> String {
    let mut s = String::with_capacity(v.len() * 10 + 2);
    s.push('[');
    for (i, x) in v.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        let _ = write!(s, "{x:.6}");
    }
    s.push(']');
    s
}

/// Render a sparse vector as a pgvector literal: `{1:0.5,7:0.25}/20000`.
///
/// ! pgvector sparsevec indices are **1-based**; the vectoriser emits 0-based.
/// Off by one here silently shifts every term to its neighbour's weight, which
/// ranks plausibly and is entirely wrong.
#[must_use]
pub fn sparse_literal(entries: &[(u32, f32)], dim: u32) -> String {
    let mut s = String::with_capacity(entries.len() * 12 + 12);
    s.push('{');
    let mut sorted: Vec<(u32, f32)> = entries.to_vec();
    sorted.sort_by_key(|(i, _)| *i);
    for (n, (i, w)) in sorted.iter().enumerate() {
        if n > 0 {
            s.push(',');
        }
        let _ = write!(s, "{}:{:.6}", i + 1, w);
    }
    let _ = write!(s, "}}/{dim}");
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta() -> CorpusMeta {
        CorpusMeta {
            id: "spike-01".into(),
            dense_model: "qwen/qwen3-embedding-0.6b".into(),
            dense_dim: 1024,
            dense_pooling: "last-token".into(),
            dense_normalize: true,
            sparse_scheme: "bm25".into(),
            sparse_dim: 20000,
            sparse_vocab_sha256: "abc".into(),
        }
    }

    #[test]
    fn a_matching_corpus_and_engine_may_serve() {
        assert!(
            meta()
                .ensure_compatible("qwen/qwen3-embedding-0.6b", 1024)
                .is_ok()
        );
    }

    #[test]
    fn a_different_model_refuses_to_serve() {
        // ! The failure this project exists to prevent: the inherited corpus
        // scored cosine 0.13 against a re-embed and still returned ranked,
        // cited, plausible results.
        let e = meta()
            .ensure_compatible("qwen/qwen3-embedding-8b", 1024)
            .unwrap_err();
        assert!(matches!(e, StoreError::CorpusMismatch { .. }));
        assert!(e.to_string().contains("different vector space"));
    }

    #[test]
    fn a_different_width_refuses_to_serve() {
        let e = meta()
            .ensure_compatible("qwen/qwen3-embedding-0.6b", 4096)
            .unwrap_err();
        assert!(matches!(
            e,
            StoreError::CorpusMismatch {
                corpus_dim: 1024,
                ..
            }
        ));
    }

    #[test]
    fn dense_literals_are_pgvector_shaped() {
        assert_eq!(dense_literal(&[1.0, -0.5]), "[1.000000,-0.500000]");
        assert_eq!(dense_literal(&[]), "[]");
    }

    #[test]
    fn sparse_literals_are_one_based_and_sorted() {
        // ! 0-based in, 1-based out. Index 0 must render as 1.
        let s = sparse_literal(&[(7, 0.25), (0, 0.5)], 20000);
        assert_eq!(s, "{1:0.500000,8:0.250000}/20000");
    }

    #[test]
    fn an_empty_sparse_vector_is_still_well_formed() {
        assert_eq!(sparse_literal(&[], 20000), "{}/20000");
    }
}
