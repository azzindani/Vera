//! Corpus construction: rows + vectors → a routed, searchable store.
//!
//! One entry point, [`build_corpus`], used by both the synthetic benchmark
//! fixture and the real ingest. Keeping them on the same path means the numbers
//! the benchmark reports describe the code that will actually serve.
//!
//! ! Offline only. Nothing here runs while serving (`CLAUDE.md` §5 rule 6).

use rusqlite::params;
use vera_core::{AnchorStats, CorpusProfile, EmbeddingSpace, ProfileBuilder};
use vera_store::{StoreError, encode_vector, sqlite::SqliteStore};

use crate::{KMeans, KMeansConfig, Matrix, kmeans, kmeans::domain_anchor};

/// One row on its way into the corpus.
#[derive(Debug, Clone)]
pub struct IngestRow {
    pub id: String,
    pub body: String,
    pub source_title: String,
    pub source_url: String,
    pub locator_page: Option<i32>,
    pub locator_section: Option<String>,
    pub heading_path: Option<String>,
    /// Canonical regulation id, e.g. `UU 28/2007` · powers the routing bypass.
    pub identifier: Option<String>,
    /// Digest of the source document · `LOOPHOLES.md` §8.
    pub source_hash: Option<String>,
}

/// What a build produced · reported so an operator can sanity-check the index
/// before trusting a single query against it.
#[derive(Debug, Clone)]
pub struct BuildReport {
    pub rows: usize,
    pub clusters: usize,
    pub iterations: usize,
    /// Mean cosine of a row to its own centroid.
    ///
    /// ! The number that predicts whether routing can work. Near 1.0 means
    /// tight, findable clusters; near 0 means the corpus has no cluster
    /// structure at this `k` and routing will prune away real answers.
    pub mean_similarity: f32,
    pub smallest_cluster: usize,
    pub largest_cluster: usize,
    pub build_seconds: f64,
    /// Distribution of cosine(row, domain anchor) across the corpus.
    pub anchor: AnchorStats,
    /// Lexical shape of the corpus · what decides whether a keyword measurement
    /// taken here means anything anywhere else (`vera_core::profile`).
    pub profile: CorpusProfile,
    /// What split-on-size changed, if `max_cluster_rows` was set.
    pub split: crate::split::SplitReport,
}

/// Measure a corpus against its layer-1 anchor.
///
/// Lives here because it needs [`Matrix`]; the [`AnchorStats`] type itself is in
/// `vera-core` so the query path can read it back without depending on the
/// offline indexer.
///
/// ! The `threshold` field is filled in under the default margin for reference
/// only — the engine recomputes it from these stats at load time, so retuning
/// the policy never requires re-ingesting the corpus.
#[must_use]
pub fn measure_anchor(vectors: &Matrix, anchor: &[f32]) -> AnchorStats {
    let mut sims: Vec<f32> = vectors
        .iter_rows()
        .map(|row| crate::kmeans::dot(row, anchor))
        .collect();
    if sims.is_empty() {
        return AnchorStats {
            min: 0.0,
            p1: 0.0,
            p5: 0.0,
            p50: 0.0,
            p95: 0.0,
            max: 0.0,
            mean: 0.0,
            threshold: 0.0,
        };
    }
    sims.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let at = |p: f64| -> f32 {
        #[allow(
            clippy::cast_precision_loss,
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss
        )]
        let i = ((p * sims.len() as f64).ceil() as usize).saturating_sub(1);
        sims[i.min(sims.len() - 1)]
    };
    #[allow(clippy::cast_precision_loss)]
    let mean = sims.iter().sum::<f32>() / sims.len() as f32;
    let stats = AnchorStats {
        min: sims[0],
        p1: at(0.01),
        p5: at(0.05),
        p50: at(0.50),
        p95: at(0.95),
        max: sims[sims.len() - 1],
        mean,
        threshold: 0.0,
    };
    AnchorStats {
        threshold: stats.threshold_at(1.0),
        ..stats
    }
}

#[derive(Debug, thiserror::Error)]
pub enum BuildError {
    #[error(transparent)]
    Store(#[from] StoreError),
    #[error("backend: {0}")]
    Backend(String),
    #[error("{rows} rows but {vectors} vectors · every row needs exactly one vector")]
    Mismatched { rows: usize, vectors: usize },
    #[error(transparent)]
    Preflight(#[from] crate::preflight::PreflightError),
    #[error("row {index} has {got} dimensions, corpus declares {expected}")]
    Width {
        index: usize,
        expected: usize,
        got: usize,
    },
}

impl From<rusqlite::Error> for BuildError {
    fn from(e: rusqlite::Error) -> Self {
        Self::Backend(e.to_string())
    }
}

/// Cluster `vectors`, then write rows, centroids and the domain anchor.
///
/// # Errors
/// Width or count mismatches, or a backend failure.
///
/// # Panics
/// Never on well-formed input; the width checks run before any write.
pub fn build_corpus(
    path: impl AsRef<std::path::Path>,
    space: &EmbeddingSpace,
    domain_id: &str,
    domain_description: &str,
    rows: &[IngestRow],
    vectors: &Matrix,
    cfg: &KMeansConfig,
) -> Result<BuildReport, BuildError> {
    let started = std::time::Instant::now();

    if rows.len() != vectors.rows() {
        return Err(BuildError::Mismatched {
            rows: rows.len(),
            vectors: vectors.rows(),
        });
    }
    if vectors.dim() != space.dim {
        return Err(BuildError::Width {
            index: 0,
            expected: space.dim,
            got: vectors.dim(),
        });
    }

    // ! Before the k-means, not just before the write. Clustering 100M vectors
    // is hours of work, and finishing it only to discover the disk cannot hold
    // the result wastes all of it (`LOOPHOLES.md` §10).
    let mean_body_bytes = if rows.is_empty() {
        0
    } else {
        rows.iter().map(|r| r.body.len()).sum::<usize>() / rows.len()
    };
    crate::preflight::require_free_space(
        path.as_ref(),
        crate::preflight::estimated_bytes(rows.len(), space.dim, mean_body_bytes),
    )?;

    let km: KMeans = kmeans(vectors, cfg);
    // ! Tier-2 split-on-size, run offline where nothing is live
    // (`CLUSTER_MAINTENANCE.md` §2, `crate::split`). k = √N minimises query cost
    // and says nothing about the *largest* cluster, which is what sets the
    // per-request RAM ceiling — so the two bounds are enforced separately.
    let (km, split) = match cfg.max_cluster_rows {
        Some(cap) => crate::split::split_oversized(vectors, km, cap, cfg, 16),
        None => (
            km,
            crate::split::SplitReport {
                splits: 0,
                clusters_before: 0,
                clusters_after: 0,
                largest_before: 0,
                largest_after: 0,
            },
        ),
    };
    let anchor = domain_anchor(vectors);
    let anchor_stats = measure_anchor(vectors, &anchor);

    // ! Profiled here, once, while the bodies are already in hand. Measuring it
    // later means a second full pass over the corpus, and measuring it *never*
    // is how a benchmark ends up reporting a full scan as an index lookup
    // (`vera_core::profile`).
    let mut profiler = ProfileBuilder::new();
    for row in rows {
        profiler.observe(&row.body, &row.source_url, row.identifier.as_deref());
    }
    let profile = profiler.finish();

    let store = SqliteStore::create(path, space)?;

    store.with_connection(|conn| -> Result<(), BuildError> {
        conn.execute("DELETE FROM chunks", [])?;
        conn.execute("DELETE FROM clusters", [])?;
        conn.execute("DELETE FROM domains", [])?;

        // ! Recorded so the engine reads a measured threshold rather than a
        // guessed one. See measure_anchor and AnchorStats::threshold_at.
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?1, ?2)",
            params![
                format!("domain_threshold::{domain_id}"),
                anchor_stats.threshold.to_string()
            ],
        )?;
        // ! The full distribution, ✗ only the derived number. Retuning the
        // threshold policy must not require re-ingesting the corpus.
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?1, ?2)",
            params![
                format!("anchor_stats::{domain_id}"),
                serde_json::to_string(&anchor_stats)
                    .map_err(|e| BuildError::Backend(e.to_string()))?
            ],
        )?;
        // ! Stored with the corpus, ✗ printed and forgotten. The caveat that
        // qualifies every keyword number has to travel with the data, or it ends
        // up living in a document that the next measurement does not read.
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?1, ?2)",
            params![
                format!("corpus_profile::{domain_id}"),
                serde_json::to_string(&profile).map_err(|e| BuildError::Backend(e.to_string()))?
            ],
        )?;

        conn.execute(
            "INSERT INTO domains (id, description, anchor, row_count) VALUES (?1,?2,?3,?4)",
            params![
                domain_id,
                domain_description,
                encode_vector(&anchor),
                i64::try_from(rows.len()).unwrap_or(i64::MAX)
            ],
        )?;

        for (i, centroid) in km.centroids.iter_rows().enumerate() {
            conn.execute(
                "INSERT INTO clusters (id, domain_id, centroid, row_count, generation) \
                 VALUES (?1,?2,?3,?4,1)",
                params![
                    i as i64,
                    domain_id,
                    encode_vector(centroid),
                    i64::try_from(km.sizes[i]).unwrap_or(0)
                ],
            )?;
        }

        // ! One transaction for the whole load. Per-row autocommit costs an
        // fsync each and turns a 200K-row build from seconds into ~an hour.
        conn.execute_batch("BEGIN")?;
        {
            let mut stmt = conn.prepare(
                "INSERT INTO chunks (rowid, id, domain_id, cluster_id, body, embedding, \
                 source_title, source_url, locator_page, locator_section, heading_path, \
                 identifier, source_hash) \
                 VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13)",
            )?;
            for (i, row) in rows.iter().enumerate() {
                stmt.execute(params![
                    i as i64 + 1,
                    row.id,
                    domain_id,
                    i64::from(km.assignments[i]),
                    row.body,
                    encode_vector(vectors.row(i)),
                    row.source_title,
                    row.source_url,
                    row.locator_page,
                    row.locator_section,
                    row.heading_path,
                    row.identifier,
                    row.source_hash,
                ])?;
            }
        }
        conn.execute_batch("COMMIT")?;

        // External-content FTS5 indexes nothing until told to · without this the
        // keyword half of every hybrid search silently returns zero rows.
        conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')", [])?;
        conn.execute_batch("ANALYZE")?;
        Ok(())
    })?;

    Ok(BuildReport {
        rows: rows.len(),
        clusters: km.centroids.rows(),
        iterations: km.iterations,
        mean_similarity: km.mean_similarity,
        smallest_cluster: km.sizes.iter().copied().min().unwrap_or(0),
        largest_cluster: km.sizes.iter().copied().max().unwrap_or(0),
        build_seconds: started.elapsed().as_secs_f64(),
        anchor: anchor_stats,
        profile,
        split,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use vera_store::ChunkStore;

    fn tiny() -> (Vec<IngestRow>, Matrix) {
        let mut m = Matrix::new(4);
        let mut rows = Vec::new();
        for i in 0..40 {
            let mut v = vec![0.0; 4];
            v[i % 4] = 1.0;
            m.push(&v);
            rows.push(IngestRow {
                id: format!("c{i}"),
                body: format!("ketentuan pajak nomor {i} tentang sanksi"),
                source_title: format!("Doc {i}"),
                source_url: format!("https://example/{i}.pdf"),
                locator_page: Some(i as i32),
                locator_section: Some(format!("Pasal {i}")),
                heading_path: None,
                identifier: (i == 7).then(|| "UU 28/2007".to_owned()),
                source_hash: None,
            });
        }
        (rows, m)
    }

    fn space() -> EmbeddingSpace {
        EmbeddingSpace {
            model_id: "test/tiny".into(),
            dim: 4,
            normalized: true,
            query_instruction: String::new(),
            validated_providers: Vec::new(),
        }
    }

    #[test]
    fn a_built_corpus_is_immediately_searchable_end_to_end() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("c.db");
        let (rows, vectors) = tiny();
        let report = build_corpus(
            &path,
            &space(),
            "reg",
            "test",
            &rows,
            &vectors,
            &KMeansConfig { k: 4, ..Default::default() },
        )
        .unwrap();
        assert_eq!(report.rows, 40);
        assert_eq!(report.clusters, 4);
        assert_eq!(report.smallest_cluster + report.largest_cluster, 20);

        let store = SqliteStore::open(&path).unwrap();
        assert_eq!(store.domains().unwrap().len(), 1);
        assert_eq!(store.centroids("reg").unwrap().len(), 4);
        // ! The FTS rebuild: without it this returns nothing and the hybrid
        // engine quietly degrades to dense-only.
        assert!(!store.keyword_search("sanksi", None, 10).unwrap().is_empty());
        assert_eq!(store.exact_identifier("UU 28/2007", 10).unwrap().len(), 1);
    }

    #[test]
    fn every_row_lands_in_exactly_one_cluster() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("c.db");
        let (rows, vectors) = tiny();
        build_corpus(&path, &space(), "reg", "t", &rows, &vectors,
            &KMeansConfig { k: 4, ..Default::default() }).unwrap();

        let store = SqliteStore::open(&path).unwrap();
        let mut total = 0;
        for c in store.centroids("reg").unwrap() {
            total += store.scan_cluster(c.id, &mut |_| {}).unwrap();
        }
        assert_eq!(total, 40, "rows were lost or duplicated across clusters");
    }

    #[test]
    fn a_max_cluster_cap_bounds_the_per_request_ram_ceiling() {
        // ! `METRICS.md` §4: the ceiling is set by the LARGEST cluster, not the
        // mean, so cluster-size skew is a memory risk and not only a latency
        // one. k=1 puts all 40 rows in one cluster; the cap must break it up.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("c.db");
        let (rows, vectors) = tiny();
        let report = build_corpus(
            &path,
            &space(),
            "reg",
            "t",
            &rows,
            &vectors,
            &KMeansConfig {
                k: 1,
                max_cluster_rows: Some(12),
                ..Default::default()
            },
        )
        .unwrap();
        assert!(report.split.splits > 0, "{:?}", report.split);
        assert!(report.largest_cluster <= 12, "largest {}", report.largest_cluster);

        // The split index must still be complete and searchable.
        let store = SqliteStore::open(&path).unwrap();
        let mut total = 0;
        for c in store.centroids("reg").unwrap() {
            total += store.scan_cluster(c.id, &mut |_| {}).unwrap();
        }
        assert_eq!(total, 40, "rows were lost or duplicated by the split");
    }

    #[test]
    fn without_a_cap_the_build_is_unchanged() {
        let dir = tempfile::tempdir().unwrap();
        let (rows, vectors) = tiny();
        let report = build_corpus(
            dir.path().join("c.db"),
            &space(),
            "reg",
            "t",
            &rows,
            &vectors,
            &KMeansConfig { k: 4, ..Default::default() },
        )
        .unwrap();
        assert_eq!(report.split.splits, 0);
        assert_eq!(report.clusters, 4);
    }

    #[test]
    fn a_row_vector_count_mismatch_is_refused_before_any_write() {
        let dir = tempfile::tempdir().unwrap();
        let (rows, mut vectors) = tiny();
        vectors.push(&[1.0, 0.0, 0.0, 0.0]);
        let err = build_corpus(
            dir.path().join("c.db"),
            &space(),
            "reg",
            "t",
            &rows,
            &vectors,
            &KMeansConfig::default(),
        )
        .unwrap_err();
        assert!(matches!(err, BuildError::Mismatched { rows: 40, vectors: 41 }));
    }

    #[test]
    fn a_corpus_whose_vectors_are_the_wrong_width_is_refused() {
        // ! The 1024-vs-4096 mistake, caught at build time rather than at the
        // first query.
        let dir = tempfile::tempdir().unwrap();
        let (rows, vectors) = tiny();
        let mut wrong = space();
        wrong.dim = 1024;
        let err = build_corpus(
            dir.path().join("c.db"),
            &wrong,
            "reg",
            "t",
            &rows,
            &vectors,
            &KMeansConfig::default(),
        )
        .unwrap_err();
        assert!(matches!(err, BuildError::Width { expected: 1024, got: 4, .. }));
    }
}
