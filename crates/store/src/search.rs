//! The read-only query surface: three retrieval arms plus the routing bypass.
//!
//! ! **Nothing here returns a vector**, and that — ✗ any loop shape — is why the
//! engine's resident set is flat. Every arm selects `(id, score)` under a
//! `LIMIT`, so Postgres performs the scan and a few hundred short rows cross the
//! wire. Measured engine peak RSS is 14 / 12 / 14 MB at `CLUSTER_BATCH` 1 / 2 / 5
//! (`docs/HARDWARE.md` §6a). Adding a column that returns `dense` would undo the
//! guarantee that the loop below is often credited with.
//!
//! ! Clusters are scanned in **batches of `CLUSTER_BATCH`** (`CLAUDE.md` §7.5).
//! This is a **latency** knob: one statement over five clusters is 3× faster than
//! five statements (81 ms against 268 ms), worth −182 ms end to end. It is
//! configuration rather than a constant because hardcoding it would hardcode a
//! limit, which §7.12 forbids — ✗ because it trades memory for speed. It does
//! not; that was arithmetic this module used to assert and the measurement
//! refuted.
//!
//! ! The cluster-sized working set is real and belongs to **Postgres** — median
//! 4.2 MB, worst 23.4 MB of pages per cluster — bounded by `shared_buffers`,
//! `work_mem` and the database container's limit. No setting in this crate
//! governs it.

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

/// Which regulation a candidate belongs to · the key sibling expansion walks.
///
/// ! All three parts, ✗ number and year. `chunks_identifier_idx` is
/// `(regulation_type, regulation_number, year)` and the tier is the leading
/// column, so this is both the correct key and the fast one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SiblingKey<'a> {
    pub regulation_type: &'a str,
    pub regulation_number: &'a str,
    pub year: Option<i32>,
}

impl<'a> SiblingKey<'a> {
    /// `None` when the row does not carry enough identity to walk from.
    #[must_use]
    pub fn of(row: &'a ChunkRow) -> Option<Self> {
        Some(Self {
            regulation_type: row.regulation_type.as_deref()?,
            regulation_number: row.regulation_number.as_deref()?,
            year: row.year,
        })
    }
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
    /// Whether this corpus has the RUM index. Probed once, on first text
    /// search, then cached for the life of the process.
    rum: std::sync::OnceLock<bool>,
    /// Whether this corpus has the `pg_search` BM25 index (migration 0004).
    /// Same contract as `rum`: probed once, cached, and its absence costs
    /// latency rather than answers.
    bm25: std::sync::OnceLock<bool>,
}

impl SearchOps {
    #[must_use]
    pub fn new(pool: Pool) -> Self {
        Self {
            pool,
            rum: std::sync::OnceLock::new(),
            bm25: std::sync::OnceLock::new(),
        }
    }

    /// Whether `ORDER BY tsv <=> tq` can be served from an index here.
    ///
    /// ! Probed, ✗ assumed. RUM is an accelerator, not a requirement: it is
    /// built from source rather than shipped in `pgvector/pgvector:pg16`, so
    /// CI and a fresh clone run without it. Both paths return the same rows —
    /// measured 98.9% top-20 overlap and the same rank-1 on 44/44 eval
    /// queries — so falling back costs latency, ✗ answers.
    async fn has_rum(&self) -> Result<bool, StoreError> {
        if let Some(v) = self.rum.get() {
            return Ok(*v);
        }
        let c = self.client().await?;
        let row = c
            .query_one(
                "SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'rum')
                    AND EXISTS (SELECT 1 FROM pg_class WHERE relname = 'chunks_tsv_rum')",
                &[],
            )
            .await?;
        let found: bool = row.get(0);
        let _ = self.rum.set(found);
        Ok(found)
    }

    /// Whether `pg_search` and its index are present (migration 0004).
    ///
    /// ! The same optional-extension contract as `has_rum`, and for the same
    /// reason: `pg_search` is not in stock Postgres, so CI and a fresh clone run
    /// without it. Both paths are BM25 over the same text and return
    /// substantially the same rows — measured 84.4% Jaccard at depth 100, and
    /// fused recall identical at @20 and @50 (`dev_tools/eval/fused_bm25.py`) —
    /// so falling back costs latency, ✗ answers.
    ///
    /// ! It costs a LOT of latency: 1,858 ms against 6,282 ms at 5M rows, and
    /// the fallback is the worst-scaling arm in the system (41.9× for 14× the
    /// rows, `docs/HARDWARE.md` §6). This probe is why that is a deployment
    /// choice rather than a code change.
    async fn has_bm25(&self) -> Result<bool, StoreError> {
        if let Some(v) = self.bm25.get() {
            return Ok(*v);
        }
        let c = self.client().await?;
        let row = c
            .query_one(
                "SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'pg_search')
                    AND EXISTS (SELECT 1 FROM pg_class WHERE relname = 'chunks_bm25')",
                &[],
            )
            .await?;
        let found: bool = row.get(0);
        let _ = self.bm25.set(found);
        Ok(found)
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
                // ! `to_jsonb(corpus_meta) ->> ...` rather than naming the column.
                // Every corpus loaded before Ravel added `scoring_vocabulary` has a
                // table without it, and naming a missing column is an error, not a
                // NULL. Through `to_jsonb` the key is simply absent and the result is
                // NULL -- so one query serves both schemas with no probe and no
                // version flag.
                "SELECT id, dense_model, dense_dim, dense_pooling, dense_normalize,
                        sparse_scheme, sparse_dim, sparse_vocab_sha256,
                        dense_instruction,
                        to_jsonb(corpus_meta) ->> 'scoring_vocabulary'
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
            dense_instruction: row.get(8),
            scoring_vocabulary: row.get(9),
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

    /// Dense top-k within **one batch of clusters**.
    ///
    /// ! The caller decides the batch size. Passing every probed cluster in one
    /// call is **allowed and measured safe** for this process — engine RSS does
    /// not move with the window — but it widens the statement Postgres plans,
    /// and that side is unmeasured. The pipeline chunks by `CLUSTER_BATCH` so
    /// the width stays a stated choice rather than a consequence of
    /// `clusters_probed`.
    ///
    /// ! `k` is a budget for the WHOLE window, ✗ per cluster, because one
    /// `LIMIT` cannot be per-partition. A caller batching `m` clusters must
    /// therefore ask for `m × k` or the window silently returns a fraction of
    /// what the same clusters yield one at a time — batching would quietly
    /// change results, which is the one thing it must not do.
    ///
    /// # Errors
    /// Database failure.
    pub async fn dense_in_clusters(
        &self,
        cluster_ids: &[i32],
        query: &[f32],
        k: i64,
    ) -> Result<Vec<Scored>, StoreError> {
        if cluster_ids.is_empty() {
            return Ok(Vec::new());
        }
        let c = self.client().await?;
        let rows = c
            .query(
                "SELECT id, 1.0 - (dense <=> $1::text::halfvec) AS sim
                 FROM chunks
                 WHERE indexable AND dense IS NOT NULL AND cluster_id = ANY($2)
                 ORDER BY dense <=> $1::text::halfvec
                 LIMIT $3",
                &[&dense_literal(query), &cluster_ids, &(k * OVERFETCH)],
            )
            .await?;
        Ok(settle(rows.iter().map(scored_f64).collect(), k))
    }

    /// One cluster · the `CLUSTER_BATCH=1` case, kept for callers that mean it.
    ///
    /// # Errors
    /// Database failure.
    pub async fn dense_in_cluster(
        &self,
        cluster_id: i32,
        query: &[f32],
        k: i64,
    ) -> Result<Vec<Scored>, StoreError> {
        self.dense_in_clusters(&[cluster_id], query, k).await
    }

    /// BM25 top-k over the whole corpus · the lexical arm.
    ///
    /// Two implementations of one arm, chosen by what the database has:
    ///
    /// - **`pg_search`** (migration 0004): Tantivy's BM25 served from a real
    ///   index. Posting lists, ✗ a scan.
    /// - **`sparsevec`**: BM25 weights precomputed by the corpus compiler and
    ///   compared with pgvector's `<#>`. No index exists for this operator, so
    ///   it walks every row.
    ///
    /// Inner product, ✗ cosine, on the fallback path: BM25 weights already
    /// encode length normalisation and cosine would undo it. pgvector's `<#>`
    /// returns the negative inner product, so the sign is flipped back.
    ///
    /// ! Both arguments are always supplied because the choice is made here,
    /// ✗ by the caller. `literal` is cheap to build from a short query and is
    /// discarded unused when the index is present; making the caller probe
    /// first would leak a storage detail into the pipeline.
    ///
    /// # Errors
    /// Database failure.
    pub async fn sparse(
        &self,
        query: &str,
        literal: &str,
        k: i64,
    ) -> Result<Vec<Scored>, StoreError> {
        let c = self.client().await?;
        if self.has_bm25().await? {
            // ! `id @@@ ...` with `key_field = 'id'`, and `paradedb.score` takes
            // that same column. Scoring a different column returns NULL rather
            // than failing, which would rank everything equally and look like a
            // bad corpus.
            let rows = c
                .query(
                    "SELECT id, paradedb.score(id) AS score
                     FROM chunks
                     WHERE indexable AND id @@@ paradedb.match('body', $1)
                     ORDER BY paradedb.score(id) DESC
                     LIMIT $2",
                    &[&query, &(k * OVERFETCH)],
                )
                .await?;
            return Ok(settle(rows.iter().map(scored_f32).collect(), k));
        }
        let rows = c
            .query(
                "SELECT id, -(sparse <#> $1::text::sparsevec) AS score
                 FROM chunks
                 WHERE indexable AND sparse IS NOT NULL
                 ORDER BY sparse <#> $1::text::sparsevec
                 LIMIT $2",
                &[&literal, &(k * OVERFETCH)],
            )
            .await?;
        Ok(settle(rows.iter().map(scored_f64).collect(), k))
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
    /// 40.9% Recall@5 on spike-02; it was 4.5% before the corpus was chunked.
    ///
    /// # Errors
    /// Database failure.
    pub async fn text(&self, query: &str, k: i64) -> Result<Vec<Scored>, StoreError> {
        // ! `ts_rank` cannot be served from a GIN index: the index finds the
        // matches, then every one of them is scored and sorted. An OR query
        // over this corpus matches a median of 220K rows (62%), so that sort
        // is the single most expensive thing a search does. RUM stores the
        // ranking data in the index, turning the whole arm into one ordered
        // index scan — measured 750ms -> 404ms at p50, same Recall@5.
        let sql = if self.has_rum().await? {
            "WITH q AS (
                 SELECT array_to_string(
                     tsvector_to_array(to_tsvector('indonesian', $1)), ' | '
                 )::tsquery AS tq
             )
             SELECT id, (1.0 / (1.0 + (tsv <=> q.tq)))::real AS score
             FROM chunks, q
             WHERE indexable AND tsv @@ q.tq
             ORDER BY tsv <=> q.tq
             LIMIT $2"
        } else {
            "WITH q AS (
                 SELECT array_to_string(
                     tsvector_to_array(to_tsvector('indonesian', $1)), ' | '
                 )::tsquery AS tq
             )
             SELECT id, ts_rank(tsv, q.tq) AS score
             FROM chunks, q
             WHERE indexable AND tsv @@ q.tq
             ORDER BY score DESC
             LIMIT $2"
        };
        let c = self.client().await?;
        let rows = c.query(sql, &[&query, &(k * OVERFETCH)]).await?;
        Ok(settle(rows.iter().map(scored_f32).collect(), k))
    }

    /// Other chunks of the same regulation · the sibling expansion of
    /// `docs/SCORING.md` §7.
    ///
    /// ! Keyed on `(regulation_type, regulation_number, year)`, which is
    /// exactly `chunks_identifier_idx`. Number and year alone are NOT unique
    /// in this corpus — 60/2014 matches 12 regulations — so dropping the tier
    /// would pull in siblings from a different law entirely.
    ///
    /// ! `exclude` is the pool. A sibling already retrieved is not an
    /// expansion, and admitting it twice would let one chunk hold two slots.
    ///
    /// # Errors
    /// Database failure.
    pub async fn siblings(
        &self,
        key: &SiblingKey<'_>,
        exclude: &[String],
        k: i64,
    ) -> Result<Vec<ChunkRow>, StoreError> {
        let c = self.client().await?;
        let rows = c
            .query(
                "SELECT id, body, source_title, source_url, chapter, article,
                        regulation_type, regulation_number, year, about,
                        truncated_at_source, cluster_id
                 FROM chunks
                 WHERE indexable
                   AND regulation_type = $1
                   AND regulation_number = $2
                   AND ($3::int IS NULL OR year = $3)
                   AND NOT (id = ANY($4))
                 ORDER BY chunk_no, id
                 LIMIT $5",
                &[
                    &key.regulation_type,
                    &key.regulation_number,
                    &key.year,
                    &exclude,
                    &k,
                ],
            )
            .await?;
        Ok(rows.iter().map(ChunkRow::from_row).collect())
    }

    /// Global exact-identifier lookup · **bypasses routing entirely**.
    ///
    /// ! Invariant 4. Semantic routing measured 95.0% recall on the spike
    /// corpus, so 5% of the time the right document sits in a cluster that was
    /// never probed. A known regulation number must never be lost that way, so
    /// this scans globally and is never gated by cluster selection
    /// (`docs/FAILURE_MODES.md` §1).
    ///
    /// # Errors
    /// Database failure.
    pub async fn exact_identifier(
        &self,
        regulation_number: &str,
        year: Option<i32>,
        reg_type: Option<&str>,
        k: i64,
    ) -> Result<Vec<ExactHit>, StoreError> {
        let c = self.client().await?;
        // ! `regulation_type` filters when the query named a tier. Number and
        // year are NOT a unique reference in this corpus — 60/2014 alone
        // matches 12 regulations, nine of them PERATURAN BUPATI — so dropping
        // the tier the user typed answers a different question. Filtering
        // strictly is right: returning nothing here still leaves the semantic
        // arms, while returning the wrong tier looks authoritative.
        let rows = c
            .query(
                "SELECT id, regulation_type, regulation_number, year
                 FROM chunks
                 WHERE regulation_number = $1
                   AND ($2::int IS NULL OR year = $2)
                   AND ($3::text IS NULL OR regulation_type = $3)
                 -- ! `id` last, ✗ decoration: chunk_no repeats across the
                 -- regulations a number+year can match, and a tie here picks a
                 -- different CLAUSE each run. This result set is small and
                 -- already filtered, so the sort is free.
                 ORDER BY year DESC NULLS LAST, chunk_no, id
                 LIMIT $4",
                &[&regulation_number, &year, &reg_type, &k],
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

    /// One deterministic chunk with its stored vector, for the startup canary.
    ///
    /// ! This is what makes invariant 2 enforceable rather than declarative.
    /// Comparing the configured model NAME against `corpus_meta` only proves
    /// two strings match; it cannot notice an endpoint serving different
    /// weights under the same name. Re-embedding a chunk's own text and
    /// comparing against the vector ingestion stored for it tests the actual
    /// vector space, using data the corpus already contains.
    ///
    /// Picks a mid-length body: long enough to be distinctive, short enough
    /// that no provider truncates it differently than ingestion did.
    ///
    /// # Errors
    /// Database failure.
    pub async fn canary_sample(&self) -> Result<(String, String, Vec<f32>), StoreError> {
        let c = self.client().await?;
        let row = c
            .query_opt(
                "SELECT id, body, dense::text FROM chunks
                 WHERE dense IS NOT NULL AND indexable
                   AND length(body) BETWEEN 200 AND 500
                 ORDER BY id LIMIT 1",
                &[],
            )
            .await?
            .ok_or(StoreError::NoCanary)?;
        let raw: String = row.get(2);
        let v = raw
            .trim_matches(['[', ']'])
            .split(',')
            .filter_map(|x| x.trim().parse::<f32>().ok())
            .collect();
        Ok((row.get(0), row.get(1), v))
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
                // ! `about` is selected because the topical factor needs it
                // (`docs/SCORING.md` §2) and ships at weight 0.25. Omitting it
                // made that weight silently inert: the factor was fitted
                // against real subject lines and evaluated against NULL.
                "SELECT id, body, source_title, source_url, chapter, article,
                        regulation_type, regulation_number, year, about,
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
    /// What the instrument is about · the topical factor's input.
    pub about: Option<String>,
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
            about: r.get(9),
            truncated_at_source: r.get(10),
            cluster_id: r.get(11),
        }
    }

    /// Whether this result can actually be double-checked by a human.
    #[must_use]
    pub fn provenance_complete(&self) -> bool {
        self.source_url.is_some()
    }
}

/// How much deeper than `k` each arm reads before it settles ties itself.
///
/// ! `ORDER BY <distance> LIMIT k` is **not deterministic**. Rows at equal
/// distance come back in any order, and a tie group straddling the limit
/// yields a different SET on each run. Measured on spike-02: 7 of the 44 eval
/// queries returned different results across three runs of ONE server process,
/// which moved Recall@5 by 2.3 points between two identical evaluations.
///
/// The obvious fix — appending `, id` to the ORDER BY — costs **8×**: it
/// defeats the RUM index's early termination and sorts all 286,199 matching
/// rows (765ms → 6,069ms, measured with EXPLAIN ANALYZE). Reading deeper does
/// not, because RUM's cost is in the scan setup rather than the depth (922ms
/// at LIMIT 20, 944ms at LIMIT 60). So each arm over-reads and [`settle`]
/// applies the tie-break in memory.
///
/// ! This shrinks the window rather than closing it: a tie group straddling
/// `OVERFETCH × k` is still resolved by the database. `docs/EVAL.md` §4
/// records the residual, measured rather than assumed.
const OVERFETCH: i64 = 3;
// ! Below 1 every arm would silently read LESS than it was asked for, and the
// recall loss would present as a ranking bug. Checked at compile time, because
// this is a constant and a runtime assertion over one would never fail.
const _: () = assert!(OVERFETCH >= 1, "OVERFETCH must never shrink an arm");

/// Order by score, break ties on id, cut to `k`.
///
/// ! The same rule `engine::fusion` uses, for the same reason. Both halves
/// have to obey it: a deterministic fusion over a non-deterministic pool is
/// still non-deterministic.
fn settle(mut rows: Vec<Scored>, k: i64) -> Vec<Scored> {
    rows.sort_by(|a, b| b.score.total_cmp(&a.score).then_with(|| a.id.cmp(&b.id)));
    rows.truncate(usize::try_from(k).unwrap_or(usize::MAX));
    rows
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
            about: Some("PAJAK DAERAH DAN RETRIBUSI DAERAH".into()),
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

    fn s(id: &str, score: f32) -> Scored {
        Scored {
            id: id.into(),
            score,
        }
    }

    #[test]
    fn settle_breaks_ties_on_id_rather_than_on_arrival_order() {
        // ! The defect this exists for: `ORDER BY <distance> LIMIT k` returns
        // tied rows in whatever order the plan produced, and that order is not
        // stable across runs. Measured on spike-02, 7 of 44 eval queries
        // returned different results across three runs of one process, moving
        // Recall@5 by 2.3 points between two identical evaluations.
        let forward = settle(vec![s("b", 1.0), s("a", 1.0), s("c", 0.5)], 3);
        let reverse = settle(vec![s("a", 1.0), s("b", 1.0), s("c", 0.5)], 3);
        let ids: Vec<&str> = forward.iter().map(|x| x.id.as_str()).collect();
        assert_eq!(ids, ["a", "b", "c"], "ties must order on id");
        assert_eq!(forward, reverse, "input order must not reach the output");
    }

    #[test]
    fn settle_picks_the_same_members_out_of_a_tie_group_at_the_cut() {
        // The case that changes the ANSWER rather than the order: more rows
        // tie at the cut than fit through it. Over-reading is what puts the
        // whole tie group in front of this function; the tie-break is what
        // makes the choice among them repeatable.
        let group = |first: &str| {
            vec![
                s("keep", 2.0),
                s(first, 1.0),
                s("tie-b", 1.0),
                s("tie-c", 1.0),
            ]
        };
        let a = settle(group("tie-a"), 2);
        let b = settle(
            vec![
                s("tie-c", 1.0),
                s("tie-b", 1.0),
                s("keep", 2.0),
                s("tie-a", 1.0),
            ],
            2,
        );
        assert_eq!(a, b);
        assert_eq!(a[1].id, "tie-a", "the lowest id wins the last slot");
    }

    #[test]
    fn settle_keeps_the_arms_score_order() {
        // Tie-breaking must not reorder anything the arm actually separated.
        let out = settle(vec![s("low", 0.1), s("high", 0.9), s("mid", 0.5)], 3);
        let ids: Vec<&str> = out.iter().map(|x| x.id.as_str()).collect();
        assert_eq!(ids, ["high", "mid", "low"]);
    }
}
