//! The read-only query surface: three retrieval arms plus the routing bypass.
//!
//! ! Clusters are scanned **one at a time** (`CLAUDE.md` §7.5). One query per
//! cluster, keeping only that cluster's top-k, so the per-request working set
//! is one cluster regardless of how many are probed. A single
//! `WHERE cluster_id = ANY($1)` would be faster and would make peak RAM a
//! function of `clusters_probed` — which is precisely the OOM guarantee this
//! design trades that speed for.

use deadpool_postgres::Pool;
use tokio_postgres::Row;

use crate::{CorpusMeta, StoreError, dense_literal};

// ! Vector parameters are bound as `$1::text::halfvec`, never `$1::halfvec`.
// The latter makes Postgres infer the PARAMETER type as halfvec, and
// tokio-postgres then refuses to send a Rust string for it (WrongType). Casting
// text -> vector inside the query keeps the wire type text, which is the only
// type both sides agree on.

/// One scored hit from a single arm.
#[derive(Debug, Clone, PartialEq)]
pub struct Scored {
    pub id: String,
    /// Arm-native score. Comparable within an arm, ✗ across arms — which is
    /// why fusion works on ranks (`engine::fusion`).
    pub score: f32,
}

/// A hit from the global exact-identifier path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExactHit {
    pub id: String,
    pub matched_on: String,
}

/// Read-only operations over an ingested corpus.
pub struct SearchOps {
    pool: Pool,
}

impl SearchOps {
    #[must_use]
    pub fn new(pool: Pool) -> Self {
        Self { pool }
    }

    async fn client(&self) -> Result<deadpool_postgres::Object, StoreError> {
        self.pool
            .get()
            .await
            .map_err(|e| StoreError::Pool(e.to_string()))
    }

    /// Load the corpus recipe. Startup calls this before serving anything.
    ///
    /// # Errors
    /// [`StoreError::NoCorpus`] when nothing has been ingested.
    pub async fn corpus_meta(&self) -> Result<CorpusMeta, StoreError> {
        let c = self.client().await?;
        let row = c
            .query_opt(
                "SELECT id, dense_model, dense_dim, dense_pooling, dense_normalize,
                        sparse_scheme, sparse_dim, sparse_vocab_sha256
                 FROM corpus_meta ORDER BY created_at DESC LIMIT 1",
                &[],
            )
            .await?
            .ok_or(StoreError::NoCorpus)?;
        Ok(CorpusMeta {
            id: row.get(0),
            dense_model: row.get(1),
            dense_dim: row.get(2),
            dense_pooling: row.get(3),
            dense_normalize: row.get(4),
            sparse_scheme: row.get(5),
            sparse_dim: row.get(6),
            sparse_vocab_sha256: row.get(7),
        })
    }

    /// Layer-2 centroids, loaded hot at startup and kept in memory.
    ///
    /// # Errors
    /// Database failure.
    pub async fn centroids(&self) -> Result<Vec<(i32, Vec<f32>)>, StoreError> {
        let c = self.client().await?;
        let rows = c
            .query("SELECT id, centroid::text FROM clusters ORDER BY id", &[])
            .await?;
        Ok(rows
            .iter()
            .map(|r| (r.get(0), parse_vector(r.get(1))))
            .collect())
    }

    /// Dense top-k **within one cluster**.
    ///
    /// ! Deliberately single-cluster. See the module note: this signature is
    /// the OOM guarantee, and widening it to take a slice of cluster ids would
    /// quietly dissolve it.
    ///
    /// # Errors
    /// Database failure.
    pub async fn dense_in_cluster(
        &self,
        cluster_id: i32,
        query: &[f32],
        k: i64,
    ) -> Result<Vec<Scored>, StoreError> {
        let c = self.client().await?;
        let rows = c
            .query(
                "SELECT id, 1.0 - (dense <=> $1::text::halfvec) AS sim
                 FROM chunks
                 WHERE indexable AND dense IS NOT NULL AND cluster_id = $2
                 ORDER BY dense <=> $1::text::halfvec
                 LIMIT $3",
                &[&dense_literal(query), &cluster_id, &k],
            )
            .await?;
        Ok(rows.iter().map(scored_f64).collect())
    }

    /// Sparse BM25 top-k over the whole corpus.
    ///
    /// Inner product, ✗ cosine: BM25 weights already encode length
    /// normalisation and cosine would undo it. pgvector's `<#>` returns the
    /// negative inner product, so the sign is flipped back here.
    ///
    /// # Errors
    /// Database failure.
    pub async fn sparse(&self, literal: &str, k: i64) -> Result<Vec<Scored>, StoreError> {
        let c = self.client().await?;
        let rows = c
            .query(
                "SELECT id, -(sparse <#> $1::text::sparsevec) AS score
                 FROM chunks
                 WHERE indexable AND sparse IS NOT NULL
                 ORDER BY sparse <#> $1::text::sparsevec
                 LIMIT $2",
                &[&literal, &k],
            )
            .await?;
        Ok(rows.iter().map(scored_f64).collect())
    }

    /// Stemmed lexical top-k via the Indonesian text search configuration.
    ///
    /// ! OR semantics, ✗ `plainto_tsquery`. plainto ANDs every term, so a
    /// natural question — "siapa yang berwenang menetapkan kelas jalan
    /// provinsi" — demands one chunk containing all seven lexemes and matched
    /// **nothing** on the whole eval set. OR restores recall.
    ///
    /// ! This arm is a recall net, ✗ a precision instrument. `ts_rank` is
    /// term-frequency only with no IDF, so "yang" weighs as much as
    /// "provinsi" — which is why `CLAUDE.md` §3's "Postgres full-text BM25" is
    /// really the `sparse` arm, where IDF is computed properly. Measured at
    /// 0.0% Recall@5 on its own; weight accordingly.
    ///
    /// # Errors
    /// Database failure.
    pub async fn text(&self, query: &str, k: i64) -> Result<Vec<Scored>, StoreError> {
        let c = self.client().await?;
        let rows = c
            .query(
                "WITH q AS (
                     SELECT array_to_string(
                         tsvector_to_array(to_tsvector('indonesian', $1)), ' | '
                     )::tsquery AS tq
                 )
                 SELECT id, ts_rank(tsv, q.tq) AS score
                 FROM chunks, q
                 WHERE indexable AND tsv @@ q.tq
                 ORDER BY score DESC
                 LIMIT $2",
                &[&query, &k],
            )
            .await?;
        Ok(rows.iter().map(scored_f32).collect())
    }

    /// Global exact-identifier lookup · **bypasses routing entirely**.
    ///
    /// ! Invariant 4. Semantic routing measured 95.0% recall on the spike
    /// corpus, so 5% of the time the right document sits in a cluster that was
    /// never probed. A known regulation number must never be lost that way, so
    /// this scans globally and is never gated by cluster selection
    /// (`LOOPHOLES.md` §1).
    ///
    /// # Errors
    /// Database failure.
    pub async fn exact_identifier(
        &self,
        regulation_number: &str,
        year: Option<i32>,
        k: i64,
    ) -> Result<Vec<ExactHit>, StoreError> {
        let c = self.client().await?;
        let rows = c
            .query(
                "SELECT id, regulation_type, regulation_number, year
                 FROM chunks
                 WHERE regulation_number = $1 AND ($2::int IS NULL OR year = $2)
                 ORDER BY year DESC NULLS LAST, chunk_no
                 LIMIT $3",
                &[&regulation_number, &year, &k],
            )
            .await?;
        Ok(rows
            .iter()
            .map(|r| {
                let rtype: Option<String> = r.get(1);
                let num: Option<String> = r.get(2);
                let yr: Option<i32> = r.get(3);
                ExactHit {
                    id: r.get(0),
                    matched_on: format!(
                        "{} {}/{}",
                        rtype.unwrap_or_default(),
                        num.unwrap_or_default(),
                        yr.map_or_else(|| "?".to_owned(), |y| y.to_string())
                    )
                    .trim()
                    .to_owned(),
                }
            })
            .collect())
    }

    /// Fetch chunks by id, for snippets, `read_chunk` and `get_provenance`.
    ///
    /// # Errors
    /// Database failure.
    pub async fn chunks_by_id(&self, ids: &[String]) -> Result<Vec<ChunkRow>, StoreError> {
        if ids.is_empty() {
            return Ok(Vec::new());
        }
        let c = self.client().await?;
        let rows = c
            .query(
                "SELECT id, body, source_title, source_url, chapter, article,
                        regulation_type, regulation_number, year,
                        truncated_at_source, cluster_id
                 FROM chunks WHERE id = ANY($1)",
                &[&ids],
            )
            .await?;
        Ok(rows.iter().map(ChunkRow::from_row).collect())
    }
}

/// A stored chunk as the engine sees it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChunkRow {
    pub id: String,
    pub body: String,
    pub source_title: String,
    /// ! `None` when ingestion recorded no URL. Never synthesised at query
    /// time (invariant 8) — a result with no URL is reported as
    /// provenance-incomplete, ✗ dressed up as verifiable.
    pub source_url: Option<String>,
    pub chapter: Option<String>,
    pub article: Option<String>,
    pub regulation_type: Option<String>,
    pub regulation_number: Option<String>,
    pub year: Option<i32>,
    pub truncated_at_source: bool,
    pub cluster_id: Option<i32>,
}

impl ChunkRow {
    fn from_row(r: &Row) -> Self {
        Self {
            id: r.get(0),
            body: r.get(1),
            source_title: r.get(2),
            source_url: r.get(3),
            chapter: r.get(4),
            article: r.get(5),
            regulation_type: r.get(6),
            regulation_number: r.get(7),
            year: r.get(8),
            truncated_at_source: r.get(9),
            cluster_id: r.get(10),
        }
    }

    /// Whether this result can actually be double-checked by a human.
    #[must_use]
    pub fn provenance_complete(&self) -> bool {
        self.source_url.is_some()
    }
}

#[allow(clippy::cast_possible_truncation)]
fn scored_f64(r: &Row) -> Scored {
    Scored {
        id: r.get(0),
        score: r.get::<_, f64>(1) as f32,
    }
}

fn scored_f32(r: &Row) -> Scored {
    Scored {
        id: r.get(0),
        score: r.get(1),
    }
}

/// Parse a pgvector text literal back into floats.
fn parse_vector(text: &str) -> Vec<f32> {
    text.trim_matches(['[', ']'])
        .split(',')
        .filter(|s| !s.is_empty())
        .filter_map(|s| s.trim().parse().ok())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vectors_round_trip_through_their_text_form() {
        let v = vec![0.25_f32, -0.5, 1.0];
        let parsed = parse_vector(&dense_literal(&v));
        assert_eq!(parsed.len(), 3);
        for (a, b) in v.iter().zip(&parsed) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
    }

    #[test]
    fn parsing_tolerates_whitespace_and_empty_input() {
        assert_eq!(parse_vector("[1.0, 2.0]").len(), 2);
        assert!(parse_vector("[]").is_empty());
        assert!(parse_vector("").is_empty());
    }

    #[test]
    fn a_chunk_without_a_url_is_not_provenance_complete() {
        // The spike corpus has no source_url at all · every result must admit
        // that rather than imply a link a human could open.
        let row = ChunkRow {
            id: "c1".into(),
            body: "x".into(),
            source_title: "UU No. 28 Tahun 2007".into(),
            source_url: None,
            chapter: None,
            article: Some("Pasal 9".into()),
            regulation_type: Some("UNDANG-UNDANG".into()),
            regulation_number: Some("28".into()),
            year: Some(2007),
            truncated_at_source: false,
            cluster_id: Some(3),
        };
        assert!(!row.provenance_complete());
        let with_url = ChunkRow {
            source_url: Some("https://x".into()),
            ..row
        };
        assert!(with_url.provenance_complete());
    }
}
