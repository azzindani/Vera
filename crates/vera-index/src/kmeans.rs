//! Spherical k-means · the layer-2 coarse quantizer.
//!
//! **Spherical**, ✗ Euclidean: the corpus is L2-normalized, retrieval ranks by
//! cosine, and clustering must optimize the same geometry the search uses.
//! Clustering by Euclidean distance and then searching by cosine puts the
//! routing and the ranking in disagreement, which shows up as unexplained
//! recall loss that no amount of `clusters_probed` tuning fixes.
//!
//! On unit vectors cosine similarity is a plain dot product, so assignment is a
//! matrix-vector product per point and the update is a mean followed by a
//! renormalize.

use rayon::prelude::*;

use crate::{Matrix, Rng};

#[derive(Debug, Clone)]
pub struct KMeansConfig {
    /// Number of clusters. `ARCHITECTURE.md` targets ~10K rows per cluster, so
    /// a good default is `rows / 10_000`, floored at 1.
    pub k: usize,
    pub max_iters: usize,
    /// Stop when this fraction of points changes assignment in an iteration.
    pub tolerance: f32,
    pub seed: u64,
    /// Cap on the sample k-means++ initializes from.
    ///
    /// ! Seeding from a sample rather than the full corpus. Full k-means++ costs
    /// `k` sequential passes over every row — at 750K rows and k≈870 that is as
    /// expensive as the entire Lloyd loop, for an initialization. A 50K sample
    /// picks near-identical seeds at a fraction of the cost.
    pub sample_for_init: usize,
}

impl Default for KMeansConfig {
    fn default() -> Self {
        Self {
            k: 256,
            max_iters: 25,
            tolerance: 0.001,
            seed: 0x5EED,
            sample_for_init: 50_000,
        }
    }
}

impl KMeansConfig {
    /// The cluster count the architecture implies for a corpus of `rows`.
    #[must_use]
    pub const fn clusters_for(rows: usize, rows_per_cluster: usize) -> usize {
        let k = rows / rows_per_cluster;
        if k == 0 { 1 } else { k }
    }
}

#[derive(Debug, Clone)]
pub struct KMeans {
    /// `k × dim`, unit-normalized.
    pub centroids: Matrix,
    /// Cluster index per input row.
    pub assignments: Vec<u32>,
    pub iterations: usize,
    /// Mean cosine of each point to its centroid · higher is tighter.
    ///
    /// The single number that says whether routing can work at all: if points
    /// sit no closer to their own centroid than to any other, no amount of
    /// probing will find them.
    pub mean_similarity: f32,
    /// Rows per cluster, by cluster index.
    pub sizes: Vec<usize>,
}

/// Dot product. Autovectorizes to AVX-512 on this target under `--release`;
/// written plainly rather than with intrinsics so it stays portable.
#[must_use]
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn normalize(v: &mut [f32]) {
    let n = dot(v, v).sqrt();
    if n > f32::EPSILON {
        for x in v.iter_mut() {
            *x /= n;
        }
    }
}

/// Assign one point to its nearest centroid, returning `(index, similarity)`.
fn nearest(point: &[f32], centroids: &Matrix) -> (u32, f32) {
    let mut best = 0u32;
    let mut best_sim = f32::NEG_INFINITY;
    for (i, c) in centroids.iter_rows().enumerate() {
        let s = dot(point, c);
        if s > best_sim {
            best_sim = s;
            best = i as u32;
        }
    }
    (best, best_sim)
}

/// k-means++ seeding over a (possibly sampled) subset.
fn seed_centroids(data: &Matrix, cfg: &KMeansConfig, rng: &mut Rng) -> Matrix {
    let dim = data.dim();
    let n = data.rows();
    let sample: Vec<usize> = if n <= cfg.sample_for_init {
        (0..n).collect()
    } else {
        (0..cfg.sample_for_init).map(|_| rng.below(n)).collect()
    };

    let mut centroids = Matrix::with_capacity(dim, cfg.k);
    centroids.push(data.row(sample[rng.below(sample.len())]));

    // Squared cosine *distance* to the closest chosen centroid, per sample
    // point. Kept incrementally: each new centroid only lowers it.
    let mut closest: Vec<f32> = sample
        .iter()
        .map(|&i| {
            let s = dot(data.row(i), centroids.row(0));
            (1.0 - s).max(0.0).powi(2)
        })
        .collect();

    while centroids.rows() < cfg.k {
        let total: f32 = closest.iter().sum();
        let pick = if total <= f32::EPSILON {
            // Degenerate (all points identical to a centroid) · fall back to
            // uniform so the loop always terminates with k centroids.
            sample[rng.below(sample.len())]
        } else {
            let target = rng.unit() * total;
            let mut acc = 0.0;
            let mut chosen = sample[sample.len() - 1];
            for (idx, &d) in closest.iter().enumerate() {
                acc += d;
                if acc >= target {
                    chosen = sample[idx];
                    break;
                }
            }
            chosen
        };
        centroids.push(data.row(pick));
        let new = centroids.rows() - 1;
        for (slot, &i) in closest.iter_mut().zip(&sample) {
            let s = dot(data.row(i), centroids.row(new));
            *slot = slot.min((1.0 - s).max(0.0).powi(2));
        }
    }
    centroids
}

/// Cluster `data` into `cfg.k` unit centroids.
///
/// Deterministic for a given seed and input.
#[must_use]
pub fn kmeans(data: &Matrix, cfg: &KMeansConfig) -> KMeans {
    let dim = data.dim();
    let n = data.rows();
    let k = cfg.k.clamp(1, n.max(1));

    if n == 0 {
        return KMeans {
            centroids: Matrix::new(dim),
            assignments: Vec::new(),
            iterations: 0,
            mean_similarity: 0.0,
            sizes: Vec::new(),
        };
    }

    let mut rng = Rng::new(cfg.seed);
    let mut centroids = seed_centroids(
        data,
        &KMeansConfig {
            k,
            ..cfg.clone()
        },
        &mut rng,
    );

    let mut assignments = vec![u32::MAX; n];
    let mut iterations = 0;
    let mut mean_similarity = 0.0;

    for iter in 0..cfg.max_iters {
        iterations = iter + 1;

        // ── Assign ───────────────────────────────────────────────────────────
        // Parallel over points: the dominant cost, and embarrassingly parallel
        // because centroids are read-only within an iteration.
        let assigned: Vec<(u32, f32)> = (0..n)
            .into_par_iter()
            .map(|i| nearest(data.row(i), &centroids))
            .collect();

        let changed = assigned
            .iter()
            .zip(&assignments)
            .filter(|((new, _), old)| new != *old)
            .count();
        for (slot, (new, _)) in assignments.iter_mut().zip(&assigned) {
            *slot = *new;
        }
        #[allow(clippy::cast_precision_loss)]
        {
            mean_similarity =
                assigned.iter().map(|(_, s)| *s).sum::<f32>() / n as f32;
        }

        // ── Update ───────────────────────────────────────────────────────────
        let mut sums = vec![0.0f32; k * dim];
        let mut counts = vec![0usize; k];
        for (i, (c, _)) in assigned.iter().enumerate() {
            let c = *c as usize;
            counts[c] += 1;
            let base = c * dim;
            for (slot, x) in sums[base..base + dim].iter_mut().zip(data.row(i)) {
                *slot += x;
            }
        }

        let mut next = Matrix::with_capacity(dim, k);
        for c in 0..k {
            if counts[c] == 0 {
                // ! An empty cluster is re-seeded to the **worst-fit** point in
                // the corpus, ✗ a random one. The worst-fit point is by
                // definition the one routing serves least well, so this both
                // fills the cluster and targets the corpus's weakest coverage.
                let worst = assigned
                    .par_iter()
                    .enumerate()
                    .min_by(|(_, (_, a)), (_, (_, b))| {
                        a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map_or(0, |(i, _)| i);
                next.push(data.row(worst));
                continue;
            }
            #[allow(clippy::cast_precision_loss)]
            let inv = 1.0 / counts[c] as f32;
            let base = c * dim;
            let mut row: Vec<f32> = sums[base..base + dim].iter().map(|x| x * inv).collect();
            normalize(&mut row);
            next.push(&row);
        }
        centroids = next;

        #[allow(clippy::cast_precision_loss)]
        let churn = changed as f32 / n as f32;
        if churn <= cfg.tolerance {
            break;
        }
    }

    let mut sizes = vec![0usize; k];
    for a in &assignments {
        sizes[*a as usize] += 1;
    }

    KMeans {
        centroids,
        assignments,
        iterations,
        mean_similarity,
        sizes,
    }
}

/// The layer-1 anchor for a set of vectors · the normalized mean.
///
/// ! Derived from the corpus, ✗ hand-written. An anchor is "what this knowledge
/// base looks like in vector space", and the only honest source for that is the
/// corpus itself.
#[must_use]
pub fn domain_anchor(data: &Matrix) -> Vec<f32> {
    let dim = data.dim();
    let mut sum = vec![0.0f32; dim];
    for row in data.iter_rows() {
        for (slot, x) in sum.iter_mut().zip(row) {
            *slot += x;
        }
    }
    if data.rows() > 0 {
        #[allow(clippy::cast_precision_loss)]
        let inv = 1.0 / data.rows() as f32;
        for x in &mut sum {
            *x *= inv;
        }
    }
    normalize(&mut sum);
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Three well-separated blobs on the unit sphere in 8 dims.
    fn blobs(per_blob: usize) -> (Matrix, Vec<usize>) {
        let mut rng = Rng::new(11);
        let mut m = Matrix::new(8);
        let mut truth = Vec::new();
        let centers = [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ];
        for (label, c) in centers.iter().enumerate() {
            for _ in 0..per_blob {
                let mut v: Vec<f32> = c
                    .iter()
                    .map(|x| x + (rng.unit() - 0.5) * 0.15)
                    .collect();
                normalize(&mut v);
                m.push(&v);
                truth.push(label);
            }
        }
        (m, truth)
    }

    #[test]
    fn it_recovers_well_separated_blobs() {
        let (data, truth) = blobs(60);
        let km = kmeans(
            &data,
            &KMeansConfig {
                k: 3,
                seed: 1,
                ..Default::default()
            },
        );
        assert_eq!(km.centroids.rows(), 3);
        // Points sharing a true label must share a cluster.
        for i in 0..truth.len() {
            for j in 0..truth.len() {
                if truth[i] == truth[j] {
                    assert_eq!(
                        km.assignments[i], km.assignments[j],
                        "rows {i} and {j} are the same blob but landed apart"
                    );
                }
            }
        }
        assert!(km.mean_similarity > 0.95, "{}", km.mean_similarity);
    }

    #[test]
    fn centroids_come_back_unit_length() {
        // ! Routing compares the query against these with a dot product on the
        // assumption they are unit. A non-unit centroid would rank by magnitude.
        let (data, _) = blobs(40);
        let km = kmeans(&data, &KMeansConfig { k: 4, ..Default::default() });
        for c in km.centroids.iter_rows() {
            let n = dot(c, c).sqrt();
            assert!((n - 1.0).abs() < 1e-4, "centroid norm {n}");
        }
    }

    #[test]
    fn the_same_seed_produces_the_same_index() {
        // ! Reproducibility: an eval delta must be attributable to a code
        // change, not to a reshuffled index.
        let (data, _) = blobs(50);
        let cfg = KMeansConfig { k: 5, seed: 99, ..Default::default() };
        let a = kmeans(&data, &cfg);
        let b = kmeans(&data, &cfg);
        assert_eq!(a.assignments, b.assignments);
        assert_eq!(a.centroids, b.centroids);
    }

    #[test]
    fn every_point_is_assigned_and_sizes_add_up() {
        let (data, _) = blobs(30);
        let km = kmeans(&data, &KMeansConfig { k: 6, ..Default::default() });
        assert_eq!(km.assignments.len(), data.rows());
        assert!(km.assignments.iter().all(|&a| (a as usize) < 6));
        assert_eq!(km.sizes.iter().sum::<usize>(), data.rows());
    }

    #[test]
    fn asking_for_more_clusters_than_points_is_clamped() {
        let mut m = Matrix::new(4);
        for i in 0..3 {
            let mut v = vec![0.0; 4];
            v[i] = 1.0;
            m.push(&v);
        }
        let km = kmeans(&m, &KMeansConfig { k: 50, ..Default::default() });
        assert_eq!(km.centroids.rows(), 3);
    }

    #[test]
    fn an_empty_corpus_yields_an_empty_index_rather_than_panicking() {
        let km = kmeans(&Matrix::new(16), &KMeansConfig::default());
        assert_eq!(km.centroids.rows(), 0);
        assert!(km.assignments.is_empty());
    }

    #[test]
    fn identical_points_do_not_hang_the_seeding() {
        // ! Degenerate case: every squared distance is 0, so the weighted pick
        // has nothing to weight. Must still terminate with k centroids.
        let mut m = Matrix::new(4);
        for _ in 0..100 {
            m.push(&[1.0, 0.0, 0.0, 0.0]);
        }
        let km = kmeans(&m, &KMeansConfig { k: 5, ..Default::default() });
        assert_eq!(km.centroids.rows(), 5);
    }

    #[test]
    fn the_cluster_count_heuristic_never_returns_zero() {
        assert_eq!(KMeansConfig::clusters_for(200_000, 10_000), 20);
        assert_eq!(KMeansConfig::clusters_for(500, 10_000), 1, "tiny corpus");
    }

    #[test]
    fn the_domain_anchor_is_the_unit_mean_of_the_corpus() {
        let mut m = Matrix::new(2);
        m.push(&[1.0, 0.0]);
        m.push(&[0.0, 1.0]);
        let a = domain_anchor(&m);
        let expect = 1.0 / 2.0f32.sqrt();
        assert!((a[0] - expect).abs() < 1e-5 && (a[1] - expect).abs() < 1e-5, "{a:?}");
    }

    #[test]
    fn an_anchor_over_nothing_is_zero_rather_than_nan() {
        assert!(domain_anchor(&Matrix::new(4)).iter().all(|x| *x == 0.0));
    }
}
