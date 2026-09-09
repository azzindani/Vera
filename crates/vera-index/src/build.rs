//! Corpus construction: rows + vectors → a routed, searchable store.
//!
//! One entry point, [`build_corpus`], used by both the synthetic benchmark
//! fixture and the real ingest. Keeping them on the same path means the numbers
//! the benchmark reports describe the code that will actually serve.
//!
//! ! Offline only. Nothing here runs while serving (`CLAUDE.md` §5 rule 6).

use vera_core::{AnchorStats, CorpusProfile, EmbeddingSpace};
use vera_store::StoreError;

use crate::{KMeansConfig, Matrix};

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

/// Measure a corpus against its layer-1 anchor, **exactly**.
///
/// ! No longer on the build path — [`crate::stream`] uses a fixed-width
/// histogram, because holding one `f32` per row costs 400 MB at 100M rows purely
/// to read six percentiles off it. This stays as the *reference*: it sorts every
/// value and is therefore exact, and `stream`'s tests check the histogram
/// against it. An approximation nobody can compare to an exact answer is an
/// approximation nobody can bound.
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
/// ! A thin adapter over [`crate::stream::build_corpus_streaming`], ✗ a second
/// implementation. Two construction paths would mean the benchmark measures code
/// that is not what ingest runs — and the whole reason the fixture and the real
/// import share `build_corpus` is that the numbers must describe the serving
/// index. Callers that already hold the corpus in memory (the fixture, tests)
/// use this; callers that cannot (ingest at scale) use the streaming entry
/// point directly.
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
    // ! Checked here rather than in the streaming builder: a `RowSource` yields
    // a row and its vector together, so the two counts cannot disagree. They can
    // only disagree when the caller holds two parallel collections, which is
    // exactly this entry point.
    if rows.len() != vectors.rows() {
        return Err(BuildError::Mismatched {
            rows: rows.len(),
            vectors: vectors.rows(),
        });
    }
    crate::stream::build_corpus_streaming(
        path,
        space,
        domain_id,
        domain_description,
        &mut crate::stream::SliceSource { rows, vectors },
        cfg,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use vera_store::{ChunkStore, sqlite::SqliteStore};

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
