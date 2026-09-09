//! `vera-store`
//!
//! The only crate that talks to a corpus. Everything above it sees the
//! [`ChunkStore`] trait and never a driver type.
//!
//! ! The trait is shaped by the **OOM guarantee**, ✗ by convenience.
//! [`ChunkStore::scan_cluster`] takes a visitor instead of returning a `Vec`,
//! so a caller *cannot* accidentally materialize a whole cluster: rows are
//! handed over one at a time against a reused buffer. Sequential loading stops
//! being a rule someone has to remember and becomes the only thing the API
//! permits (`ARCHITECTURE.md` §4, `MCP_ENGINE.md` §5).
//!
//! Two backends are planned. Postgres + pgvector is the production target
//! (`migrations/`). SQLite is the local one: it is what the current test corpus
//! ships as, and it lets the routing experiment run with no server at all. Both
//! sit behind the same trait, so the engine cannot tell them apart.

pub mod sqlite;

use vera_core::{AnchorStats, Chunk, EmbeddingSpace, Source};

#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    #[error("store backend: {0}")]
    Backend(String),

    #[error("corpus records no embedding space · refusing to guess what it was built with")]
    NoRecordedSpace,

    #[error("corpus metadata is malformed: {0}")]
    Malformed(String),

    #[error(
        "chunk {id} stores a {got}-float vector but the corpus declares {expected} · \
         the corpus is internally inconsistent"
    )]
    VectorWidth {
        id: String,
        expected: usize,
        got: usize,
    },
}

/// A layer-1 domain, with the anchor the query vector is matched against.
#[derive(Debug, Clone, PartialEq)]
pub struct DomainAnchor {
    pub id: String,
    pub description: String,
    pub anchor: Vec<f32>,
    pub row_count: i64,
}

/// A layer-2 k-means centroid · the coarse quantizer.
#[derive(Debug, Clone, PartialEq)]
pub struct Centroid {
    pub id: i32,
    pub domain_id: String,
    pub centroid: Vec<f32>,
    pub row_count: i64,
}

/// One row as seen by a streaming scan.
///
/// ! Borrows. The vector points at a buffer the store reuses for the next row,
/// so a visitor that wants to keep it must copy — which is the point: keeping
/// every row is then a visible allocation, ✗ an accident.
#[derive(Debug)]
pub struct ScannedRow<'a> {
    pub id: &'a str,
    pub vector: &'a [f32],
}

/// A relation an agent can follow from a known chunk.
///
/// ! The variants here are the edges **this corpus schema can actually answer**,
/// ✗ the edges the design anticipates. `MULTI_DOMAIN.md` §5 also calls for
/// `parent`/`children` (hierarchy), `cites`/`cited_by` (graph) and
/// `versions`/`supersedes` (time); none of those have columns yet, so they are
/// deliberately absent rather than stubbed. An edge that exists but returns
/// nothing is indistinguishable from a document with no neighbours — the same
/// silent-emptiness failure the routing bypass exists to prevent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Edge {
    /// Other chunks of the same source document.
    SameDocument,
    /// Other chunks carrying the same canonical identifier.
    SameIdentifier,
}

impl Edge {
    /// Wire name, as advertised by `describe` and accepted by `traverse`.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::SameDocument => "same_document",
            Self::SameIdentifier => "same_identifier",
        }
    }

    /// Parse a wire name.
    #[must_use]
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "same_document" => Some(Self::SameDocument),
            "same_identifier" => Some(Self::SameIdentifier),
            _ => None,
        }
    }

    /// Every edge this build can answer · the closed vocabulary `describe`
    /// publishes so an agent never has to guess one.
    #[must_use]
    pub const fn all() -> &'static [Self] {
        &[Self::SameDocument, Self::SameIdentifier]
    }
}

/// A keyword hit and its BM25 score (higher is better).
#[derive(Debug, Clone, PartialEq)]
pub struct KeywordHit {
    pub id: String,
    pub score: f32,
}

/// Read-only access to a corpus.
///
/// ! Read-only by construction: there is no write method to call. Corpus
/// mutation belongs to the offline pipelines (`CLAUDE.md` §5 rule 6).
pub trait ChunkStore: Send + Sync {
    /// The space this corpus was embedded in, as recorded at ingest.
    ///
    /// # Errors
    /// [`StoreError::NoRecordedSpace`] if ingest recorded nothing — which must
    /// stop startup rather than default to a guess.
    fn corpus_space(&self) -> Result<EmbeddingSpace, StoreError>;

    /// Layer-1 anchors. Small; loaded once and held hot.
    ///
    /// # Errors
    /// Backend failure.
    fn domains(&self) -> Result<Vec<DomainAnchor>, StoreError>;

    /// Layer-2 centroids for one domain. Held hot in engine memory.
    ///
    /// # Errors
    /// Backend failure.
    fn centroids(&self, domain_id: &str) -> Result<Vec<Centroid>, StoreError>;

    /// Stream every row of one cluster through `visit`, in storage order.
    ///
    /// Returns how many rows were scanned. ! The visitor is called against a
    /// reused buffer; nothing accumulates inside the store.
    ///
    /// # Errors
    /// Backend failure, or [`StoreError::VectorWidth`] if a stored vector does
    /// not match the corpus's declared width.
    fn scan_cluster(
        &self,
        cluster_id: i32,
        visit: &mut dyn FnMut(ScannedRow<'_>),
    ) -> Result<usize, StoreError>;

    /// BM25 over one cluster, or the whole corpus when `cluster_id` is `None`.
    ///
    /// ! The `None` case is the routing bypass that makes exact identifiers
    /// un-missable (`LOOPHOLES.md` §1).
    ///
    /// # Errors
    /// Backend failure.
    fn keyword_search(
        &self,
        query: &str,
        cluster_id: Option<i32>,
        limit: usize,
    ) -> Result<Vec<KeywordHit>, StoreError>;

    /// Exact match on the stored `identifier` column, corpus-wide.
    ///
    /// # Errors
    /// Backend failure.
    fn exact_identifier(&self, identifier: &str, limit: usize)
    -> Result<Vec<Chunk>, StoreError>;

    /// Fetch chunks by id, for snippets, `fetch` and provenance.
    ///
    /// # Errors
    /// Backend failure.
    fn chunks_by_id(&self, ids: &[String]) -> Result<Vec<Chunk>, StoreError>;

    /// Provenance bundle for ids · always derived from stored fields.
    ///
    /// # Errors
    /// Backend failure.
    fn provenance(&self, ids: &[String]) -> Result<Vec<(String, Source)>, StoreError> {
        Ok(self
            .chunks_by_id(ids)?
            .into_iter()
            .map(|c| (c.id.clone(), c.source()))
            .collect())
    }

    /// Chunks reachable from `id` along `edge`, excluding `id` itself.
    ///
    /// ! Returns neighbours, ✗ ranked results. Traversal is navigation from a
    /// known point; ranking is `search`'s job, and mixing them would let a
    /// traversal quietly reorder evidence the agent believes it addressed
    /// directly.
    ///
    /// # Errors
    /// Backend failure.
    fn neighbors(
        &self,
        id: &str,
        edge: Edge,
        limit: usize,
    ) -> Result<Vec<Chunk>, StoreError>;

    /// Rows in the largest cluster · the term in the RAM budget.
    ///
    /// # Errors
    /// Backend failure.
    fn largest_cluster_rows(&self) -> Result<usize, StoreError>;

    /// Read a corpus metadata value written at build time.
    ///
    /// # Errors
    /// Backend failure.
    fn meta(&self, key: &str) -> Result<Option<String>, StoreError>;

    /// The anchor geometry this corpus recorded at build time, if any.
    ///
    /// ! Preferred over [`calibrated_domain_threshold`](Self::calibrated_domain_threshold):
    /// the stats let the engine *derive* a threshold under a tunable policy,
    /// where the stored scalar fixes one at build time and can only be changed
    /// by re-ingesting the corpus.
    ///
    /// # Errors
    /// Backend failure.
    fn anchor_stats(&self, domain_id: &str) -> Result<Option<AnchorStats>, StoreError> {
        Ok(self
            .meta(&format!("anchor_stats::{domain_id}"))?
            .and_then(|v| serde_json::from_str(&v).ok()))
    }

    /// The layer-1 threshold calibrated against this corpus's own geometry.
    ///
    /// ! A fixed threshold cannot be right. How close a query sits to a domain
    /// anchor depends on the embedding model's anisotropy — how narrow a cone
    /// its vectors occupy — which varies by model and by corpus and is not
    /// knowable in advance. Guess too low and every off-topic query is forced
    /// into the domain; guess too high and **every** query returns empty, which
    /// the output contract reports as a *successful* empty result (`success:
    /// true`, `confidence: none`). That failure is invisible: it looks exactly
    /// like a corpus that genuinely has no answers.
    ///
    /// So the build measures the actual distribution of row-to-anchor cosines
    /// and records a threshold below essentially all real content. Returns
    /// `None` for a corpus built before calibration existed, leaving the caller
    /// to fall back to its configured value.
    ///
    /// # Errors
    /// Backend failure.
    fn calibrated_domain_threshold(&self, domain_id: &str) -> Result<Option<f32>, StoreError> {
        Ok(self
            .meta(&format!("domain_threshold::{domain_id}"))?
            .and_then(|v| v.parse().ok()))
    }
}

/// Decode a little-endian f32 BLOB into a reusable buffer.
///
/// Writes into `out` rather than allocating, so a scan over a million rows
/// performs one allocation, not a million.
///
/// # Errors
/// [`StoreError::VectorWidth`] if the blob is not exactly `expected_dim` f32s.
pub fn decode_vector_into(
    id: &str,
    blob: &[u8],
    expected_dim: usize,
    out: &mut Vec<f32>,
) -> Result<(), StoreError> {
    if blob.len() != expected_dim * 4 {
        return Err(StoreError::VectorWidth {
            id: id.to_owned(),
            expected: expected_dim,
            got: blob.len() / 4,
        });
    }
    out.clear();
    out.reserve(expected_dim);
    out.extend(
        blob.chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])),
    );
    Ok(())
}

/// Encode f32s as a little-endian BLOB · the ingest side of
/// [`decode_vector_into`].
#[must_use]
pub fn encode_vector(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_vector_survives_a_round_trip() {
        let v = vec![0.0, 1.0, -1.0, 0.5, f32::MIN_POSITIVE];
        let mut out = Vec::new();
        decode_vector_into("c1", &encode_vector(&v), v.len(), &mut out).unwrap();
        assert_eq!(out, v);
    }

    #[test]
    fn a_wrong_width_blob_is_rejected_with_the_offending_id() {
        // ! Names the chunk. A corpus-wide "width mismatch" is unactionable;
        // an id points at the row to re-ingest.
        let err = decode_vector_into("reg::uu-28::c3", &encode_vector(&[1.0; 512]), 1024, &mut Vec::new())
            .unwrap_err();
        match err {
            StoreError::VectorWidth { id, expected, got } => {
                assert_eq!(id, "reg::uu-28::c3");
                assert_eq!((expected, got), (1024, 512));
            }
            other => panic!("{other}"),
        }
    }

    #[test]
    fn decoding_reuses_the_buffer_rather_than_growing_it() {
        // ! The allocation argument for the streaming scan: capacity is taken
        // once and every later row lands in the same memory.
        let mut buf = Vec::new();
        let blob = encode_vector(&[1.0; 256]);
        decode_vector_into("a", &blob, 256, &mut buf).unwrap();
        let cap = buf.capacity();
        let ptr = buf.as_ptr();
        for _ in 0..1_000 {
            decode_vector_into("a", &blob, 256, &mut buf).unwrap();
        }
        assert_eq!(buf.capacity(), cap, "buffer regrew");
        assert_eq!(buf.as_ptr(), ptr, "buffer moved");
    }
}
