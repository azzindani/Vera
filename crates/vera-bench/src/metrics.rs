//! Latency percentiles and retrieval quality.
//!
//! ! Percentiles, ✗ means. A mean latency hides exactly the behavior that
//! matters under the concurrency ceiling: a p99 several times the p50 means
//! some requests are starving, and averaging makes that invisible.

use std::collections::HashSet;
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
    fn top1_tracks_the_single_most_important_result() {
        assert!(top1_hit(&ids(&["a", "z"]), &ids(&["a", "b"])));
        assert!(!top1_hit(&ids(&["z", "a"]), &ids(&["a", "b"])));
        assert!(!top1_hit(&[], &ids(&["a"])), "found nothing at all");
        assert!(top1_hit(&[], &[]), "nothing to find");
    }
}
