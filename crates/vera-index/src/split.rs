//! Split-on-size · the Tier-2 maintenance algorithm, applied at **build time**.
//!
//! `CLUSTER_MAINTENANCE.md` §2 Tier 2: when a cluster exceeds its row cap, run
//! local k-means (k=2) on **only that cluster's members** and replace its
//! centroid with two. Bounds cluster size without touching any other cluster.
//!
//! ! Two things make an oversized cluster worse than a cosmetic imbalance, and
//! they are the two ceilings the whole design rests on:
//!
//! - **The RAM ceiling is set by the largest cluster, not the average**
//!   (`METRICS.md` §4). `per_request_ceiling = largest_cluster × bytes_per_vector`,
//!   so one fat cluster raises the peak-RAM bound for every request, including
//!   those that never probe it.
//! - **Worst-case probe latency is the largest cluster too.** A p99 is made of
//!   the queries that route into it.
//!
//! Measured at √N on the benchmark corpus, sizes ran p10 148 / p50 395 / p90 981
//! / max 1085 against a mean of 447 — a 2.4× spread from empty-cluster reseeding
//! alone (`CLAUDE.md` §8 finding 8). `METRICS.md` §4 wants max ≤ 2× mean.
//!
//! ! **Build time, ✗ online.** The same algorithm run against a live corpus
//! would have to move rows between clusters while queries read them. The schema
//! stores one `cluster_id` per chunk, so a live split either mutates rows in
//! place — forbidden by `CLAUDE.md` §7 rule 9 — or needs a per-generation
//! assignment table, which is a schema change, and `MULTI_DOMAIN.md` §12 says
//! schema and storage layout are **one decision made once** because both are
//! paid for in re-ingest. Online Tier 2 therefore waits for that decision. This
//! module is its algorithm, proven offline, where nothing is live and there is
//! nothing to swap.

use crate::kmeans::{KMeans, dot};
use crate::{KMeansConfig, Matrix, kmeans};

/// What splitting changed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SplitReport {
    pub splits: usize,
    pub clusters_before: usize,
    pub clusters_after: usize,
    pub largest_before: usize,
    pub largest_after: usize,
}

/// Recursively split every cluster larger than `max_rows`.
///
/// Returns the updated clustering and a report. A `max_rows` of 0 or one that no
/// cluster exceeds is a no-op, and the input is returned unchanged.
///
/// ! Recursive because one split need not be enough: halving a cluster of 25,000
/// against a cap of 10,000 leaves two of 12,500. The recursion is bounded by
/// `max_passes` rather than by "until every cluster fits", because a cluster of
/// near-identical vectors can refuse to divide — k=2 on 500 copies of one point
/// puts everything in one half — and an unbounded loop would spin on it forever.
/// Hitting the bound is reported by `largest_after` still exceeding the cap, ✗
/// swallowed.
#[must_use]
pub fn split_oversized(
    data: &Matrix,
    km: KMeans,
    max_rows: usize,
    cfg: &KMeansConfig,
    max_passes: usize,
) -> (KMeans, SplitReport) {
    let clusters_before = km.centroids.rows();
    let largest_before = km.sizes.iter().copied().max().unwrap_or(0);
    let mut report = SplitReport {
        splits: 0,
        clusters_before,
        clusters_after: clusters_before,
        largest_before,
        largest_after: largest_before,
    };
    if max_rows == 0 || data.rows() == 0 {
        return (km, report);
    }

    let mut assignments = km.assignments;
    let mut centroids = km.centroids;

    for _ in 0..max_passes {
        let sizes = tally(&assignments, centroids.rows());
        let oversized: Vec<usize> = sizes
            .iter()
            .enumerate()
            .filter(|(_, n)| **n > max_rows)
            .map(|(i, _)| i)
            .collect();
        if oversized.is_empty() {
            break;
        }

        let mut progressed = false;
        for cluster in oversized {
            let members: Vec<usize> = assignments
                .iter()
                .enumerate()
                .filter(|(_, c)| **c as usize == cluster)
                .map(|(i, _)| i)
                .collect();
            if members.len() < 2 {
                continue;
            }

            // Local k=2 over this cluster's members only · never touches another
            // cluster, which is what makes the operation bounded.
            let mut sub = Matrix::with_capacity(data.dim(), members.len());
            for i in &members {
                sub.push(data.row(*i));
            }
            let halves = kmeans(
                &sub,
                &KMeansConfig {
                    k: 2,
                    // Seeded from the cluster id so a rebuild is reproducible.
                    seed: cfg.seed ^ (cluster as u64).wrapping_mul(0x9E37_79B9),
                    ..cfg.clone()
                },
            );
            if halves.centroids.rows() < 2 || halves.sizes.iter().any(|n| *n == 0) {
                // Degenerate: the members do not divide. Leave it and report the
                // cap as unmet rather than looping.
                continue;
            }

            // ! The first half **replaces** the original centroid, the second is
            // appended. Reusing the id keeps every untouched row's assignment
            // valid, so a split is O(members), ✗ O(corpus).
            let new_id = centroids.rows();
            centroids.set_row(cluster, halves.centroids.row(0));
            centroids.push(halves.centroids.row(1));
            for (slot, i) in members.iter().enumerate() {
                if halves.assignments[slot] == 1 {
                    assignments[*i] = u32::try_from(new_id).unwrap_or(u32::MAX);
                }
            }
            report.splits += 1;
            progressed = true;
        }
        if !progressed {
            break;
        }
    }

    let sizes = tally(&assignments, centroids.rows());
    report.clusters_after = centroids.rows();
    report.largest_after = sizes.iter().copied().max().unwrap_or(0);

    let mean_similarity = mean_similarity(data, &centroids, &assignments);
    (
        KMeans {
            centroids,
            assignments,
            iterations: km.iterations,
            mean_similarity,
            sizes,
        },
        report,
    )
}

fn tally(assignments: &[u32], k: usize) -> Vec<usize> {
    let mut sizes = vec![0usize; k];
    for c in assignments {
        if let Some(slot) = sizes.get_mut(*c as usize) {
            *slot += 1;
        }
    }
    sizes
}

/// Mean cosine of each point to its own centroid, recomputed after splitting.
///
/// ! Recomputed rather than carried over. Splitting moves points to nearer
/// centroids, so the pre-split figure understates the result — and this number
/// is the one that predicts whether routing can work at all (`KMeans`).
fn mean_similarity(data: &Matrix, centroids: &Matrix, assignments: &[u32]) -> f32 {
    if data.rows() == 0 {
        return 0.0;
    }
    let total: f32 = (0..data.rows())
        .map(|i| {
            let c = assignments[i] as usize;
            if c < centroids.rows() {
                dot(data.row(i), centroids.row(c))
            } else {
                0.0
            }
        })
        .sum();
    #[allow(clippy::cast_precision_loss)]
    {
        total / data.rows() as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Rng;

    /// `groups` tight blobs of `per` points each, in `dim` dimensions.
    fn blobs(groups: usize, per: usize, dim: usize) -> Matrix {
        let mut rng = Rng::new(9);
        let mut m = Matrix::with_capacity(dim, groups * per);
        for g in 0..groups {
            for _ in 0..per {
                let mut v = vec![0.0f32; dim];
                v[g % dim] = 1.0;
                for x in v.iter_mut() {
                    *x += (rng.unit() - 0.5) * 0.05;
                }
                let n = dot(&v, &v).sqrt();
                for x in v.iter_mut() {
                    *x /= n;
                }
                m.push(&v);
            }
        }
        m
    }

    fn cfg() -> KMeansConfig {
        KMeansConfig { k: 2, max_iters: 20, ..KMeansConfig::default() }
    }

    #[test]
    fn an_oversized_cluster_is_halved_and_the_cap_is_met() {
        // One blob of 400 clustered into k=2 · both halves are still large, so
        // the recursion has to run more than once to reach a cap of 60.
        let data = blobs(2, 200, 8);
        let km = kmeans(&data, &KMeansConfig { k: 2, ..cfg() });
        let (split, report) = split_oversized(&data, km, 60, &cfg(), 8);
        assert!(report.splits > 0, "{report:?}");
        assert!(
            report.largest_after <= 60,
            "cap not met: {report:?} sizes {:?}",
            split.sizes
        );
        assert!(report.clusters_after > report.clusters_before);
    }

    #[test]
    fn every_row_still_belongs_to_exactly_one_cluster() {
        // ! The invariant a split can most easily break: rows lost or duplicated
        // between the replaced centroid and the appended one.
        let data = blobs(3, 150, 8);
        let km = kmeans(&data, &KMeansConfig { k: 3, ..cfg() });
        let (split, _) = split_oversized(&data, km, 40, &cfg(), 8);
        assert_eq!(split.sizes.iter().sum::<usize>(), data.rows());
        assert_eq!(split.assignments.len(), data.rows());
        assert!(
            split.assignments.iter().all(|c| (*c as usize) < split.centroids.rows()),
            "an assignment points past the centroid list"
        );
    }

    #[test]
    fn splitting_never_loosens_the_clustering() {
        // Points move to nearer centroids, so tightness can only improve. If it
        // fell, the split had put rows on the wrong side.
        let data = blobs(2, 200, 8);
        let km = kmeans(&data, &KMeansConfig { k: 2, ..cfg() });
        let before = km.mean_similarity;
        let (split, _) = split_oversized(&data, km, 50, &cfg(), 8);
        assert!(
            split.mean_similarity >= before - 1e-5,
            "{} fell below {before}",
            split.mean_similarity
        );
    }

    #[test]
    fn a_cap_nothing_exceeds_is_a_no_op() {
        let data = blobs(2, 50, 8);
        let km = kmeans(&data, &KMeansConfig { k: 4, ..cfg() });
        let ids = km.assignments.clone();
        let (split, report) = split_oversized(&data, km, 10_000, &cfg(), 8);
        assert_eq!(report.splits, 0);
        assert_eq!(split.assignments, ids, "assignments changed on a no-op");
        assert_eq!(report.clusters_before, report.clusters_after);
    }

    #[test]
    fn a_zero_cap_is_a_no_op_rather_than_an_infinite_split() {
        // ! 0 means "no cap". Read as a literal maximum it would demand clusters
        // of zero rows, which cannot terminate.
        let data = blobs(2, 40, 8);
        let km = kmeans(&data, &KMeansConfig { k: 2, ..cfg() });
        let (_, report) = split_oversized(&data, km, 0, &cfg(), 8);
        assert_eq!(report.splits, 0);
    }

    #[test]
    fn identical_vectors_that_cannot_divide_stop_rather_than_spinning() {
        // ! The termination hazard. k=2 over 300 copies of one point puts every
        // member in one half, so the cap can never be met — the pass budget is
        // what stops it, and the unmet cap is reported rather than hidden.
        let mut data = Matrix::with_capacity(4, 300);
        for _ in 0..300 {
            data.push(&[1.0, 0.0, 0.0, 0.0]);
        }
        let km = kmeans(&data, &KMeansConfig { k: 1, ..cfg() });
        let (_, report) = split_oversized(&data, km, 10, &cfg(), 4);
        assert!(
            report.largest_after > 10,
            "the cap cannot be met here; the report must say so: {report:?}"
        );
    }

    #[test]
    fn splitting_is_reproducible_for_a_seed() {
        // A rebuild must produce the same index, or two replicas disagree about
        // which cluster holds what.
        let data = blobs(2, 200, 8);
        let a = split_oversized(&data, kmeans(&data, &KMeansConfig { k: 2, ..cfg() }), 60, &cfg(), 8);
        let b = split_oversized(&data, kmeans(&data, &KMeansConfig { k: 2, ..cfg() }), 60, &cfg(), 8);
        assert_eq!(a.0.assignments, b.0.assignments);
        assert_eq!(a.1, b.1);
    }
}
