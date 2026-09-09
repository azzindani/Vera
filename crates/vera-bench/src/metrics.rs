//! Latency percentiles and retrieval quality.
//!
//! ! Percentiles, ✗ means. A mean latency hides exactly the behavior that
//! matters under the concurrency ceiling: a p99 several times the p50 means
//! some requests are starving, and averaging makes that invisible.

use std::collections::{HashMap, HashSet};
use std::time::Duration;

/// A sorted sample of durations, queryable by percentile.
#[derive(Debug, Clone)]
pub struct Latencies(Vec<f64>);

impl Latencies {
    /// Collect and sort. Milliseconds throughout.
    #[must_use]
    pub fn from_durations(samples: &[Duration]) -> Self {
        let mut ms: Vec<f64> = samples.iter().map(Duration::as_secs_f64).map(|s| s * 1e3).collect();
        ms.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        Self(ms)
    }

    /// Nearest-rank percentile, `p` in `[0, 100]`.
    ///
    /// Nearest-rank rather than interpolated: with the small sample counts a
    /// benchmark run produces, interpolation invents a value between two
    /// observations and reports it as if it were measured.
    #[must_use]
    pub fn percentile(&self, p: f64) -> f64 {
        if self.0.is_empty() {
            return 0.0;
        }
        #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let rank = ((p / 100.0) * self.0.len() as f64).ceil() as usize;
        self.0[rank.saturating_sub(1).min(self.0.len() - 1)]
    }

    #[must_use]
    pub fn p50(&self) -> f64 {
        self.percentile(50.0)
    }
    #[must_use]
    pub fn p95(&self) -> f64 {
        self.percentile(95.0)
    }
    #[must_use]
    pub fn p99(&self) -> f64 {
        self.percentile(99.0)
    }
    #[must_use]
    pub fn max(&self) -> f64 {
        self.0.last().copied().unwrap_or(0.0)
    }
    #[must_use]
    pub fn mean(&self) -> f64 {
        if self.0.is_empty() {
            return 0.0;
        }
        #[allow(clippy::cast_precision_loss)]
        {
            self.0.iter().sum::<f64>() / self.0.len() as f64
        }
    }
    #[must_use]
    #[allow(dead_code, reason = "part of the metrics surface; exercised by tests")]
    pub fn len(&self) -> usize {
        self.0.len()
    }
    #[must_use]
    #[allow(dead_code, reason = "part of the metrics surface; exercised by tests")]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

/// Fraction of the exhaustive top-k that routed search also found.
///
/// ! This is the number routing is *bought* with. Latency without it is
/// meaningless — probing zero clusters is infinitely fast and returns nothing.
/// The baseline is a full scan of the same corpus with the same scoring, so any
/// gap is attributable to routing alone.
#[must_use]
pub fn recall_at_k(routed: &[String], baseline: &[String], k: usize) -> f32 {
    let truth: HashSet<&String> = baseline.iter().take(k).collect();
    if truth.is_empty() {
        // Nothing to recall · counts as perfect, not as zero, or an empty
        // baseline would drag the average down for no reason.
        return 1.0;
    }
    let found = routed.iter().take(k).filter(|id| truth.contains(id)).count();
    #[allow(clippy::cast_precision_loss)]
    {
        found as f32 / truth.len() as f32
    }
}

/// Fraction of the true top-k that landed in a cluster the query actually
/// probed · `EVAL.md` §3.
///
/// ! The diagnostic metric, and the one that says *where* to fix a low
/// recall@k. If routing recall is high but recall@k is low, the answers were
/// reachable and fusion or the candidate cap dropped them. If routing recall
/// itself is low, no amount of fusion tuning helps — the clustering is wrong or
/// `clusters_probed` is too small. Without it, a recall number says something
/// is broken but not which half.
#[must_use]
pub fn routing_recall(
    baseline: &[String],
    cluster_of: &HashMap<String, i32>,
    probed: &HashSet<i32>,
    k: usize,
) -> f32 {
    let truth: Vec<&String> = baseline.iter().take(k).collect();
    if truth.is_empty() {
        return 1.0;
    }
    let reachable = truth
        .iter()
        .filter(|id| cluster_of.get(**id).is_some_and(|c| probed.contains(c)))
        .count();
    #[allow(clippy::cast_precision_loss)]
    {
        reachable as f32 / truth.len() as f32
    }
}

/// Reciprocal rank of the first baseline result present in `routed` · `EVAL.md` §3.
#[must_use]
pub fn reciprocal_rank(routed: &[String], baseline: &[String]) -> f32 {
    let Some(target) = baseline.first() else {
        return 1.0;
    };
    routed
        .iter()
        .position(|id| id == target)
        .map_or(0.0, |i| 1.0 / (i as f32 + 1.0))
}

/// Whether the single best baseline result survived routing.
///
/// Complements recall@k: losing rank 1 matters more than losing rank 10, and an
/// aggregate recall figure hides which one went missing.
#[must_use]
pub fn top1_hit(routed: &[String], baseline: &[String]) -> bool {
    match (routed.first(), baseline.first()) {
        (Some(a), Some(b)) => a == b,
        (_, None) => true,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ids(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| (*s).to_owned()).collect()
    }

    #[test]
    fn percentiles_come_from_observed_values_only() {
        let l = Latencies::from_durations(
            &(1..=100).map(|i| Duration::from_millis(i)).collect::<Vec<_>>(),
        );
        assert!((l.p50() - 50.0).abs() < 1e-6);
        assert!((l.p95() - 95.0).abs() < 1e-6);
        assert!((l.p99() - 99.0).abs() < 1e-6);
        assert!((l.max() - 100.0).abs() < 1e-6);
    }

    #[test]
    fn a_single_sample_is_every_percentile() {
        let l = Latencies::from_durations(&[Duration::from_millis(7)]);
        assert!((l.p50() - 7.0).abs() < 1e-6);
        assert!((l.p99() - 7.0).abs() < 1e-6);
    }

    #[test]
    fn an_empty_sample_reports_zero_rather_than_panicking() {
        let l = Latencies::from_durations(&[]);
        assert_eq!(l.p50(), 0.0);
        assert!(l.is_empty());
    }

    #[test]
    fn unsorted_input_is_sorted_before_measuring() {
        let l = Latencies::from_durations(&[
            Duration::from_millis(50),
            Duration::from_millis(1),
            Duration::from_millis(10),
        ]);
        assert!((l.p50() - 10.0).abs() < 1e-6);
    }

    #[test]
    fn recall_counts_the_baseline_results_that_survived_routing() {
        let baseline = ids(&["a", "b", "c", "d"]);
        assert!((recall_at_k(&ids(&["a", "b", "c", "d"]), &baseline, 4) - 1.0).abs() < 1e-6);
        assert!((recall_at_k(&ids(&["a", "b", "x", "y"]), &baseline, 4) - 0.5).abs() < 1e-6);
        assert!((recall_at_k(&ids(&["x", "y"]), &baseline, 4)).abs() < 1e-6);
    }

    #[test]
    fn recall_ignores_order_within_k() {
        // ! Recall is a set measure. Reordering within the top-k is a ranking
        // question, and conflating the two hides which one actually regressed.
        let baseline = ids(&["a", "b", "c"]);
        assert!((recall_at_k(&ids(&["c", "b", "a"]), &baseline, 3) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn an_empty_baseline_is_perfect_recall_not_zero() {
        assert!((recall_at_k(&ids(&["a"]), &[], 10) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn routing_recall_separates_a_routing_miss_from_a_ranking_miss() {
        // ! The whole point of the metric. Same recall@k of 0.5, two different
        // causes — and only this number tells them apart.
        let baseline = ids(&["a", "b"]);
        let cluster_of: HashMap<String, i32> =
            [("a".to_owned(), 1), ("b".to_owned(), 2)].into_iter().collect();

        // `b` sat in an unprobed cluster · a routing failure.
        let probed: HashSet<i32> = [1].into_iter().collect();
        assert!((routing_recall(&baseline, &cluster_of, &probed, 2) - 0.5).abs() < 1e-6);

        // Both clusters probed · anything lost after this is ranking, not routing.
        let probed_both: HashSet<i32> = [1, 2].into_iter().collect();
        assert!((routing_recall(&baseline, &cluster_of, &probed_both, 2) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn routing_recall_counts_an_unknown_chunk_as_unreachable() {
        let baseline = ids(&["ghost"]);
        assert!(
            routing_recall(&baseline, &HashMap::new(), &HashSet::new(), 1).abs() < 1e-6
        );
    }

    #[test]
    fn reciprocal_rank_rewards_finding_the_best_answer_early() {
        let baseline = ids(&["a"]);
        assert!((reciprocal_rank(&ids(&["a", "b"]), &baseline) - 1.0).abs() < 1e-6);
        assert!((reciprocal_rank(&ids(&["b", "a"]), &baseline) - 0.5).abs() < 1e-6);
        assert!(reciprocal_rank(&ids(&["x", "y"]), &baseline).abs() < 1e-6);
    }

    #[test]
    fn top1_tracks_the_single_most_important_result() {
        assert!(top1_hit(&ids(&["a", "z"]), &ids(&["a", "b"])));
        assert!(!top1_hit(&ids(&["z", "a"]), &ids(&["a", "b"])));
        assert!(!top1_hit(&[], &ids(&["a"])), "found nothing at all");
        assert!(top1_hit(&[], &[]), "nothing to find");
    }
}
