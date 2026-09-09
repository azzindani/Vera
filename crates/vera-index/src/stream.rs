//! Streaming corpus construction · build an index without holding the corpus.
//!
//! ! The in-memory build held **four** things that grow with the corpus, and
//! every one of them is fatal at the design point (100M × 4096):
//!
//! | held | at 100M × 4096 | replaced by |
//! |---|---|---|
//! | `Vec<IngestRow>` — bodies + provenance | ~100 GB | nothing · rows are written as they arrive |
//! | `Matrix` — every vector | **1.6 TB** | a bounded training sample |
//! | `Vec<f32>` — one anchor cosine per row, sorted for percentiles | 400 MB | a fixed-width histogram |
//! | one SQL transaction over every insert | a ~1.6 TB WAL | batched commits + a completeness marker |
//!
//! What remains is `O(train_sample + vocabulary + 1 row)`, and the first term is
//! a dial.
//!
//! ! **The corpus is streamed twice, ✗ once.** k-means has to finish before any
//! row can be assigned, and the domain anchor has to be known before any
//! row-to-anchor cosine can be measured — so one pass collects (sample, anchor
//! sum, lexical profile, count) and the second assigns and writes. There is no
//! single-pass version that is not a guess: sampling the anchor as well would
//! make the layer-1 threshold, which `CLAUDE.md` §8 finding 1 shows is the
//! difference between a working corpus and one that silently rejects 97% of
//! queries, depend on a random subset.
//!
//! ! **Training on a sample is the standard IVF construction**, not a shortcut
//! taken here: FAISS trains a coarse quantizer on ~30–256 vectors per centroid
//! and then adds the full corpus. Assignment is still exact — every row is
//! compared against every centroid in pass 2. Only the *centroid positions* come
//! from a sample, and they are a summary of the corpus's shape, which a large
//! sample estimates well. Leaving `train_sample` unset trains on everything and
//! reproduces the in-memory build exactly.

use rusqlite::params;
use vera_core::{AnchorStats, EmbeddingSpace, ProfileBuilder};
use vera_store::{encode_vector, sqlite::SqliteStore};

use crate::build::{BuildError, BuildReport, IngestRow};
use crate::kmeans::{KMeans, dot};
use crate::{KMeansConfig, Matrix, Rng, kmeans};

/// Rows to insert between commits.
///
/// ! A compromise, and the reasoning is worth keeping. One transaction over the
/// whole load is *atomic* — a crash leaves no corpus at all — but at 100M rows
/// its WAL is the size of the corpus, so the guarantee costs more disk than the
/// result. Per-row autocommit costs an fsync each and turns a 200K-row build
/// into about an hour. Batching gives up atomicity, so the build marks the
/// corpus incomplete until it finishes ([`BUILD_STATE_KEY`]) rather than leaving
/// a half-written index that opens and answers queries.
const COMMIT_EVERY: usize = 50_000;

/// Meta key recording whether a build finished.
///
/// ! `LOOPHOLES.md` §10's "never partially apply", enforced for the case
/// batching introduces. A corpus interrupted mid-load has a valid schema, a
/// plausible row count and a working FTS table for the rows it did write — it
/// answers queries and silently omits the rest, which is the failure this
/// project is organised around. The marker makes that state refuse to open.
pub const BUILD_STATE_KEY: &str = "build_state";

/// A replayable source of rows and their vectors.
///
/// ! **Replayable**, and that is a real requirement rather than a convenience:
/// [`build_corpus_streaming`] calls [`stream`](Self::stream) twice and the two
/// passes must see the same rows in the same order, or a row's assignment will
/// belong to a different row. A source that cannot be replayed (a pipe, a
/// network response) has to be staged to disk first.
pub trait RowSource {
    /// Row count if the source knows it · used to preflight disk **before** the
    /// first pass rather than after it.
    fn rows_hint(&self) -> Option<usize> {
        None
    }

    /// Present every row exactly once, in a stable order.
    ///
    /// # Errors
    /// Whatever the source raises, mapped to [`BuildError`].
    fn stream(
        &mut self,
        visit: &mut dyn FnMut(&IngestRow, &[f32]) -> Result<(), BuildError>,
    ) -> Result<(), BuildError>;
}

/// A [`RowSource`] over slices already in memory.
///
/// Lets the in-memory [`crate::build_corpus`] delegate here, so the benchmark
/// fixture and the real ingest keep running the **same** construction code.
pub struct SliceSource<'a> {
    pub rows: &'a [IngestRow],
    pub vectors: &'a Matrix,
}

impl RowSource for SliceSource<'_> {
    fn rows_hint(&self) -> Option<usize> {
        Some(self.rows.len())
    }

    fn stream(
        &mut self,
        visit: &mut dyn FnMut(&IngestRow, &[f32]) -> Result<(), BuildError>,
    ) -> Result<(), BuildError> {
        for (i, row) in self.rows.iter().enumerate() {
            visit(row, self.vectors.row(i))?;
        }
        Ok(())
    }
}

// ── bounded accumulators ─────────────────────────────────────────────────────

/// Uniform sample of fixed capacity, drawn in one pass · Vitter's Algorithm R.
///
/// ! Uniform matters. Taking the *first* `capacity` rows instead would train the
/// quantizer on whatever the source happened to order first — by document, by
/// date, by id — and the centroids would describe that prefix rather than the
/// corpus. On a regulation corpus ordered by year that is a quantizer for the
/// 1990s.
struct Reservoir {
    capacity: usize,
    seen: usize,
    sample: Matrix,
    rng: Rng,
}

impl Reservoir {
    fn new(capacity: usize, dim: usize, seed: u64) -> Self {
        Self {
            capacity,
            seen: 0,
            sample: Matrix::with_capacity(dim, capacity.min(1_000_000)),
            rng: Rng::new(seed),
        }
    }

    fn offer(&mut self, v: &[f32]) {
        self.seen += 1;
        if self.sample.rows() < self.capacity {
            self.sample.push(v);
            return;
        }
        // Replace with probability capacity/seen, keeping the sample uniform.
        let j = self.rng.below(self.seen);
        if j < self.capacity {
            self.sample.set_row(j, v);
        }
    }
}

/// Bins used by [`CosineHistogram`] across `[-1, 1]`.
///
/// 8192 bins is a resolution of 2/8192 ≈ 0.00024. The layer-1 threshold is
/// `p1 − margin·(p50 − p1)`, and the measured `p50 − p1` spread is ~0.03, so the
/// quantisation is under 1% of the quantity being derived.
const HISTOGRAM_BINS: usize = 8192;

/// Row-to-anchor cosine distribution, in constant space.
///
/// ! Replaces sorting one `f32` per row. At 100M that array is 400 MB held only
/// to read six percentiles off it. `min`, `max` and `mean` stay **exact** — they
/// need no ordering — so only the percentiles are quantised, and only to the bin
/// width above.
#[derive(Debug)]
struct CosineHistogram {
    bins: Vec<u64>,
    min: f32,
    max: f32,
    sum: f64,
    count: u64,
}

impl CosineHistogram {
    fn new() -> Self {
        Self {
            bins: vec![0; HISTOGRAM_BINS],
            min: f32::INFINITY,
            max: f32::NEG_INFINITY,
            sum: 0.0,
            count: 0,
        }
    }

    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::cast_precision_loss)]
    fn observe(&mut self, cosine: f32) {
        let c = cosine.clamp(-1.0, 1.0);
        self.min = self.min.min(c);
        self.max = self.max.max(c);
        self.sum += f64::from(c);
        self.count += 1;
        let idx = (((c + 1.0) * 0.5) * HISTOGRAM_BINS as f32) as usize;
        self.bins[idx.min(HISTOGRAM_BINS - 1)] += 1;
    }

    #[allow(clippy::cast_precision_loss)]
    fn percentile(&self, p: f64) -> f32 {
        if self.count == 0 {
            return 0.0;
        }
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let target = ((p * self.count as f64).ceil() as u64).max(1);
        let mut cumulative = 0u64;
        for (i, n) in self.bins.iter().enumerate() {
            cumulative += n;
            if cumulative >= target {
                // Upper edge of the bin · the smallest value at least this many
                // observations are below, which is what a percentile is.
                let edge = ((i + 1) as f32 / HISTOGRAM_BINS as f32) * 2.0 - 1.0;
                return edge.clamp(self.min, self.max);
            }
        }
        self.max
    }

    fn finish(&self) -> AnchorStats {
        if self.count == 0 {
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
        #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
        let mean = (self.sum / self.count as f64) as f32;
        let stats = AnchorStats {
            min: self.min,
            p1: self.percentile(0.01),
            p5: self.percentile(0.05),
            p50: self.percentile(0.50),
            p95: self.percentile(0.95),
            max: self.max,
            mean,
            threshold: 0.0,
        };
        AnchorStats {
            threshold: stats.threshold_at(1.0),
            ..stats
        }
    }
}

// ── the build ────────────────────────────────────────────────────────────────

/// Build a corpus from a streaming source.
///
/// # Errors
/// Width mismatches, a failed disk preflight, or a backend failure.
///
/// # Panics
/// Never on well-formed input.
#[allow(clippy::too_many_lines)]
pub fn build_corpus_streaming<S: RowSource>(
    path: impl AsRef<std::path::Path>,
    space: &EmbeddingSpace,
    domain_id: &str,
    domain_description: &str,
    source: &mut S,
    cfg: &KMeansConfig,
) -> Result<BuildReport, BuildError> {
    let started = std::time::Instant::now();
    let dim = space.dim;
    let path = path.as_ref();

    // Early refusal when the source knows its size · cheaper than discovering it
    // after a full pass (`LOOPHOLES.md` §10).
    if let Some(hint) = source.rows_hint() {
        crate::preflight::require_free_space(
            path,
            crate::preflight::estimated_bytes(hint, dim, 512),
        )?;
    }

    // ── Pass 1 · sample, anchor, profile, count ──────────────────────────────
    // ! Nothing accumulated here grows with the corpus except the vocabulary,
    // which grows with the *language* and flattens.
    let train_capacity = cfg.train_sample.unwrap_or(usize::MAX);
    let mut reservoir = Reservoir::new(train_capacity, dim, cfg.seed ^ 0x5A3D_1177);
    let mut anchor_sum = vec![0f64; dim];
    let mut profiler = ProfileBuilder::new();
    let mut rows = 0usize;
    let mut body_bytes = 0usize;

    source.stream(&mut |row, vector| {
        if vector.len() != dim {
            return Err(BuildError::Width {
                index: rows,
                expected: dim,
                got: vector.len(),
            });
        }
        rows += 1;
        body_bytes += row.body.len();
        for (acc, x) in anchor_sum.iter_mut().zip(vector) {
            *acc += f64::from(*x);
        }
        reservoir.offer(vector);
        profiler.observe(&row.body, &row.source_url, row.identifier.as_deref());
        Ok(())
    })?;

    if rows == 0 {
        return Err(BuildError::Backend("source produced no rows".to_owned()));
    }
    let profile = profiler.finish();

    // Definitive preflight, now that the size is known — still before the
    // k-means and before any write.
    crate::preflight::require_free_space(
        path,
        crate::preflight::estimated_bytes(rows, dim, body_bytes / rows),
    )?;

    // The domain anchor is the corpus mean direction · exact, from a running
    // sum, ✗ estimated from the sample.
    #[allow(clippy::cast_possible_truncation)]
    let mut anchor: Vec<f32> = anchor_sum.iter().map(|x| *x as f32).collect();
    let norm = dot(&anchor, &anchor).sqrt();
    if norm > f32::EPSILON {
        for x in &mut anchor {
            *x /= norm;
        }
    }

    // ── Train the quantizer on the sample ────────────────────────────────────
    let train = reservoir.sample;
    let trained: KMeans = kmeans(&train, cfg);

    // Split-on-size against a **scaled** cap: the sample holds `train.rows()` of
    // `rows`, so a cluster allowed `max_cluster_rows` in the corpus may hold only
    // that fraction here. Splitting on the unscaled cap would never fire.
    let (trained, split) = match cfg.max_cluster_rows {
        Some(cap) if train.rows() > 0 => {
            let scaled = (cap.saturating_mul(train.rows()) / rows).max(1);
            crate::split::split_oversized(&train, trained, scaled, cfg, 16)
        }
        _ => (
            trained,
            crate::split::SplitReport {
                splits: 0,
                clusters_before: 0,
                clusters_after: 0,
                largest_before: 0,
                largest_after: 0,
            },
        ),
    };
    let centroids = trained.centroids;
    let cluster_count = centroids.rows();
    drop(train);

    // ── Pass 2 · assign, write, measure ──────────────────────────────────────
    let store = SqliteStore::create(path, space)?;
    let mut sizes = vec![0usize; cluster_count];
    let mut histogram = CosineHistogram::new();
    let mut similarity_sum = 0f64;

    store.with_connection(|conn| -> Result<(), BuildError> {
        conn.execute("DELETE FROM chunks", [])?;
        conn.execute("DELETE FROM clusters", [])?;
        conn.execute("DELETE FROM domains", [])?;
        // ! Marked before the first insert. Everything between here and the
        // final mark is a corpus that must not be served.
        set_meta(conn, BUILD_STATE_KEY, "in_progress")?;

        conn.execute(
            "INSERT INTO domains (id, description, anchor, row_count) VALUES (?1,?2,?3,?4)",
            params![
                domain_id,
                domain_description,
                encode_vector(&anchor),
                i64::try_from(rows).unwrap_or(i64::MAX)
            ],
        )?;

        let mut written = 0usize;
        conn.execute_batch("BEGIN")?;
        {
            let mut stmt = conn.prepare(
                "INSERT INTO chunks (rowid, id, domain_id, cluster_id, body, embedding, \
                 source_title, source_url, locator_page, locator_section, heading_path, \
                 identifier, source_hash) \
                 VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13)",
            )?;

            source.stream(&mut |row, vector| {
                // ! Assignment is exact — every row against every centroid.
                // Sampling decided where the centroids *are*; it does not decide
                // which one a row belongs to.
                let mut best = 0usize;
                let mut best_score = f32::NEG_INFINITY;
                for (c, centroid) in centroids.iter_rows().enumerate() {
                    let s = dot(vector, centroid);
                    if s > best_score {
                        best_score = s;
                        best = c;
                    }
                }
                sizes[best] += 1;
                similarity_sum += f64::from(best_score);
                histogram.observe(dot(vector, &anchor));

                written += 1;
                stmt.execute(params![
                    i64::try_from(written).unwrap_or(i64::MAX),
                    row.id,
                    domain_id,
                    i64::try_from(best).unwrap_or(0),
                    row.body,
                    encode_vector(vector),
                    row.source_title,
                    row.source_url,
                    row.locator_page,
                    row.locator_section,
                    row.heading_path,
                    row.identifier,
                    row.source_hash,
                ])?;

                // ! Safe to commit with `stmt` still prepared: `execute` runs the
                // statement to completion and resets it, so nothing is mid-scan.
                // A statement left iterating across a COMMIT would hold the read
                // transaction open and the batching would buy nothing.
                if written % COMMIT_EVERY == 0 {
                    conn.execute_batch("COMMIT; BEGIN")?;
                }
                Ok(())
            })?;
        }
        conn.execute_batch("COMMIT")?;

        for (i, centroid) in centroids.iter_rows().enumerate() {
            conn.execute(
                "INSERT INTO clusters (id, domain_id, centroid, row_count, generation) \
                 VALUES (?1,?2,?3,?4,1)",
                params![
                    i as i64,
                    domain_id,
                    encode_vector(centroid),
                    i64::try_from(sizes[i]).unwrap_or(0)
                ],
            )?;
        }

        let anchor_stats = histogram.finish();
        set_meta(
            conn,
            &format!("domain_threshold::{domain_id}"),
            &anchor_stats.threshold.to_string(),
        )?;
        set_meta(
            conn,
            &format!("anchor_stats::{domain_id}"),
            &serde_json::to_string(&anchor_stats)
                .map_err(|e| BuildError::Backend(e.to_string()))?,
        )?;
        set_meta(
            conn,
            &format!("corpus_profile::{domain_id}"),
            &serde_json::to_string(&profile).map_err(|e| BuildError::Backend(e.to_string()))?,
        )?;

        // External-content FTS5 indexes nothing until told to · without this the
        // keyword half of every hybrid search silently returns zero rows.
        conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')", [])?;
        conn.execute_batch("ANALYZE")?;
        // ! Last, after the FTS rebuild. A corpus whose keyword half is empty is
        // exactly as broken as one missing rows, and just as quiet about it.
        set_meta(conn, BUILD_STATE_KEY, "complete")?;
        Ok(())
    })?;

    let anchor_stats = histogram.finish();
    #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
    let mean_similarity = (similarity_sum / rows as f64) as f32;

    Ok(BuildReport {
        rows,
        clusters: cluster_count,
        iterations: trained.iterations,
        mean_similarity,
        smallest_cluster: sizes.iter().copied().min().unwrap_or(0),
        largest_cluster: sizes.iter().copied().max().unwrap_or(0),
        build_seconds: started.elapsed().as_secs_f64(),
        anchor: anchor_stats,
        profile,
        split,
    })
}

fn set_meta(conn: &rusqlite::Connection, key: &str, value: &str) -> Result<(), BuildError> {
    conn.execute(
        "INSERT OR REPLACE INTO meta (key, value) VALUES (?1, ?2)",
        params![key, value],
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use vera_store::ChunkStore;

    fn space(dim: usize) -> EmbeddingSpace {
        EmbeddingSpace {
            model_id: "test/tiny".into(),
            dim,
            normalized: true,
            query_instruction: String::new(),
            validated_providers: Vec::new(),
        }
    }

    /// `n` rows across 4 orthogonal directions.
    fn corpus(n: usize) -> (Vec<IngestRow>, Matrix) {
        let mut m = Matrix::new(4);
        let mut rows = Vec::new();
        for i in 0..n {
            let mut v = vec![0.0; 4];
            v[i % 4] = 1.0;
            m.push(&v);
            rows.push(IngestRow {
                id: format!("c{i}"),
                body: format!("ketentuan pajak nomor {i} tentang sanksi"),
                source_title: format!("Doc {i}"),
                source_url: format!("https://example/{}.pdf", i % 7),
                locator_page: Some(i as i32),
                locator_section: Some(format!("Pasal {i}")),
                heading_path: None,
                identifier: (i % 10 == 0).then(|| format!("UU {i}/2007")),
                source_hash: None,
            });
        }
        (rows, m)
    }

    #[test]
    fn a_streamed_build_is_immediately_searchable() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("c.db");
        let (rows, vectors) = corpus(80);
        let report = build_corpus_streaming(
            &path,
            &space(4),
            "reg",
            "test",
            &mut SliceSource { rows: &rows, vectors: &vectors },
            &KMeansConfig { k: 4, ..Default::default() },
        )
        .unwrap();
        assert_eq!(report.rows, 80);
        assert_eq!(report.clusters, 4);

        let store = SqliteStore::open(&path).unwrap();
        assert_eq!(store.domains().unwrap()[0].row_count, 80);
        assert!(!store.keyword_search("sanksi", None, 10).unwrap().is_empty());
        assert_eq!(store.exact_identifier("UU 0/2007", 10).unwrap().len(), 1);
        // Every row landed in exactly one cluster.
        let mut total = 0;
        for c in store.centroids("reg").unwrap() {
            total += store.scan_cluster(c.id, &mut |_| {}).unwrap();
        }
        assert_eq!(total, 80);
    }

    #[test]
    fn training_on_a_sample_still_assigns_every_row_exactly() {
        // ! The claim that makes sampling safe. Only the centroid *positions*
        // come from a sample; assignment compares every row against every
        // centroid, so no row is placed by extrapolation.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("c.db");
        let (rows, vectors) = corpus(4_000);
        let report = build_corpus_streaming(
            &path,
            &space(4),
            "reg",
            "t",
            &mut SliceSource { rows: &rows, vectors: &vectors },
            &KMeansConfig { k: 4, train_sample: Some(200), ..Default::default() },
        )
        .unwrap();
        assert_eq!(report.rows, 4_000);
        // The four directions are cleanly separable, so a 200-row sample finds
        // the same four centres and every row sits on top of one.
        assert!(report.mean_similarity > 0.99, "{}", report.mean_similarity);
        assert_eq!(report.largest_cluster, 1_000);
        assert_eq!(report.smallest_cluster, 1_000);
    }

    #[test]
    fn the_sample_is_uniform_rather_than_the_first_rows() {
        // ! Ordered sources are the normal case — by document, by date, by id.
        // Taking a prefix would train the quantizer on whichever slice came
        // first. Here the first 500 rows are all one direction; a prefix sample
        // would see one cluster and miss the other three entirely.
        let mut m = Matrix::new(4);
        let mut rows = Vec::new();
        for i in 0..2_000usize {
            let mut v = vec![0.0; 4];
            // First 500 rows are direction 0; the rest spread over 1..4.
            v[if i < 500 { 0 } else { 1 + (i % 3) }] = 1.0;
            m.push(&v);
            rows.push(IngestRow {
                id: format!("c{i}"),
                body: "isi".into(),
                source_title: "t".into(),
                source_url: "u".into(),
                locator_page: None,
                locator_section: None,
                heading_path: None,
                identifier: None,
                source_hash: None,
            });
        }
        let dir = tempfile::tempdir().unwrap();
        let report = build_corpus_streaming(
            dir.path().join("c.db"),
            &space(4),
            "reg",
            "t",
            &mut SliceSource { rows: &rows, vectors: &m },
            &KMeansConfig { k: 4, train_sample: Some(300), ..Default::default() },
        )
        .unwrap();
        assert!(
            report.mean_similarity > 0.99,
            "the sample missed a direction: {}",
            report.mean_similarity
        );
    }

    #[test]
    fn the_histogram_matches_an_exact_percentile_within_its_resolution() {
        // ! The one approximation the streaming build introduces. It must be
        // small relative to `p50 − p1`, which is what the layer-1 threshold is
        // derived from (`AnchorStats::threshold_at`).
        let mut h = CosineHistogram::new();
        let values: Vec<f32> = (0..10_000).map(|i| 0.70 + (i as f32) * 0.00003).collect();
        for v in &values {
            h.observe(*v);
        }
        let stats = h.finish();
        let mut sorted = values.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let exact = |p: f64| sorted[((p * sorted.len() as f64).ceil() as usize - 1).min(9_999)];

        for (got, want) in [
            (stats.p1, exact(0.01)),
            (stats.p50, exact(0.50)),
            (stats.p95, exact(0.95)),
        ] {
            assert!((got - want).abs() < 0.0005, "{got} vs {want}");
        }
        // min, max and mean need no ordering, so they stay exact.
        assert!((stats.min - sorted[0]).abs() < 1e-6);
        assert!((stats.max - sorted[9_999]).abs() < 1e-6);
    }

    #[test]
    fn the_histogram_tracks_the_exact_reference_on_a_realistic_distribution() {
        // ! Checked against `measure_anchor`, which sorts every value — the code
        // that used to run on the build path. The point is that the streaming
        // approximation is *bounded*, not merely plausible: `p50 − p1` is what
        // the layer-1 threshold is derived from, and an error comparable to that
        // spread would move the threshold enough to reject real queries
        // (CLAUDE.md §8 findings 1 and 7).
        let mut rng = Rng::new(4);
        let dim = 64;
        // A corpus in a narrow cone, as real embeddings are.
        let mut base: Vec<f32> = (0..dim).map(|_| rng.unit() - 0.5).collect();
        let n = dot(&base, &base).sqrt();
        for x in &mut base { *x /= n; }
        let mut m = Matrix::new(dim);
        for _ in 0..20_000 {
            let mut v: Vec<f32> = base
                .iter()
                .map(|b| b * 0.85 + (rng.unit() - 0.5) * 0.15)
                .collect();
            let n = dot(&v, &v).sqrt();
            for x in &mut v { *x /= n; }
            m.push(&v);
        }
        let anchor = crate::kmeans::domain_anchor(&m);
        let exact = crate::build::measure_anchor(&m, &anchor);

        let mut h = CosineHistogram::new();
        for row in m.iter_rows() {
            h.observe(dot(row, &anchor));
        }
        let streamed = h.finish();

        let spread = exact.p50 - exact.p1;
        assert!(spread > 0.0, "degenerate fixture");
        for (got, want, name) in [
            (streamed.p1, exact.p1, "p1"),
            (streamed.p50, exact.p50, "p50"),
            (streamed.p95, exact.p95, "p95"),
        ] {
            assert!(
                (got - want).abs() < spread * 0.1,
                "{name}: {got} vs {want} · error must stay well under the {spread} spread"
            );
        }
        // The threshold is what actually matters downstream.
        assert!(
            (streamed.threshold - exact.threshold).abs() < spread * 0.2,
            "threshold {} vs {}",
            streamed.threshold,
            exact.threshold
        );
    }

    #[test]
    fn an_empty_histogram_reports_zeroes_rather_than_infinities() {
        // `min` starts at +inf; finishing without observations must not leak it
        // into a threshold.
        let stats = CosineHistogram::new().finish();
        assert!(stats.min.abs() < f32::EPSILON, "{}", stats.min);
        assert!(stats.max.abs() < f32::EPSILON, "{}", stats.max);
        assert!(stats.threshold.abs() < f32::EPSILON, "{}", stats.threshold);
    }

    #[test]
    fn a_reservoir_smaller_than_the_stream_stays_at_capacity() {
        let mut r = Reservoir::new(50, 4, 1);
        for i in 0..5_000 {
            r.offer(&[i as f32, 0.0, 0.0, 0.0]);
        }
        assert_eq!(r.sample.rows(), 50);
        assert_eq!(r.seen, 5_000);
    }

    #[test]
    fn a_reservoir_larger_than_the_stream_keeps_everything() {
        let mut r = Reservoir::new(500, 4, 1);
        for i in 0..30 {
            r.offer(&[i as f32, 0.0, 0.0, 0.0]);
        }
        assert_eq!(r.sample.rows(), 30);
    }

    #[test]
    fn a_wrong_width_vector_is_refused_before_any_write() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("c.db");
        let (rows, vectors) = corpus(8);
        let err = build_corpus_streaming(
            &path,
            &space(1024),
            "reg",
            "t",
            &mut SliceSource { rows: &rows, vectors: &vectors },
            &KMeansConfig { k: 2, ..Default::default() },
        )
        .unwrap_err();
        assert!(matches!(err, BuildError::Width { expected: 1024, got: 4, .. }));
        assert!(!path.exists(), "a rejected build must leave no corpus behind");
    }

    #[test]
    fn a_source_with_no_rows_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let err = build_corpus_streaming(
            dir.path().join("c.db"),
            &space(4),
            "reg",
            "t",
            &mut SliceSource { rows: &[], vectors: &Matrix::new(4) },
            &KMeansConfig::default(),
        )
        .unwrap_err();
        assert!(matches!(err, BuildError::Backend(_)), "{err}");
    }

    #[test]
    fn a_finished_build_is_marked_complete() {
        // ! The marker LOOPHOLES.md §10 needs once inserts are batched: a
        // corpus interrupted mid-load has a valid schema and a working FTS
        // table for the rows it did write, so nothing else distinguishes it.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("c.db");
        let (rows, vectors) = corpus(40);
        build_corpus_streaming(
            &path,
            &space(4),
            "reg",
            "t",
            &mut SliceSource { rows: &rows, vectors: &vectors },
            &KMeansConfig { k: 2, ..Default::default() },
        )
        .unwrap();
        let store = SqliteStore::open(&path).unwrap();
        assert_eq!(store.meta(BUILD_STATE_KEY).unwrap().as_deref(), Some("complete"));
    }

    #[test]
    fn many_rows_cross_several_commit_batches_without_losing_any() {
        // COMMIT_EVERY is 50_000, so drive past one boundary with a narrow dim.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("c.db");
        let (rows, vectors) = corpus(COMMIT_EVERY + 137);
        let report = build_corpus_streaming(
            &path,
            &space(4),
            "reg",
            "t",
            &mut SliceSource { rows: &rows, vectors: &vectors },
            &KMeansConfig { k: 4, train_sample: Some(1_000), ..Default::default() },
        )
        .unwrap();
        assert_eq!(report.rows, COMMIT_EVERY + 137);

        let store = SqliteStore::open(&path).unwrap();
        let mut total = 0;
        for c in store.centroids("reg").unwrap() {
            total += store.scan_cluster(c.id, &mut |_| {}).unwrap();
        }
        assert_eq!(total, COMMIT_EVERY + 137, "a commit boundary dropped rows");
        assert!(
            !store.keyword_search("sanksi", None, 5).unwrap().is_empty(),
            "the FTS rebuild must cover every batch"
        );
    }
}
