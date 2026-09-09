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

/// Why a true result did not come back · `METRICS.md` §3.1.
///
/// ! The causes are dialled by **different knobs**, so an undecomposed recall
/// figure cannot say which one to turn:
///
/// | cause | dial |
/// |---|---|
/// | routing | `clusters_probed` |
/// | cap | `per_cluster_top_k`, `candidate_cap` |
/// | fusion | `rrf_k`, weights |
/// | by design | none · this one is not a defect |
///
/// The cap column is the one that did not exist before, and its absence was the
/// dangerous part: `recall@k` scores against a baseline that applies the *same*
/// cap, and `route/D` counts a probed-then-discarded row as a routing success.
/// A cap set too low therefore showed up as **nothing at all** — the same shape
/// of silent failure as the layer-1 over-rejection, and `EVAL.md` §4 lists
/// "per-cluster top-k · recall@k vs candidate-cap misses" as a dial evidence
/// must set, which was not possible until this existed.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RecallLoss {
    /// Truth items the search returned.
    pub found: usize,
    /// Lost because no probed cluster contained them.
    pub routing: usize,
    /// In a probed cluster, but cut before fusion ever saw them.
    pub cap: usize,
    /// Reached fusion, ranked out — and the exhaustive search **did** return
    /// them, so having fewer candidates is what cost the rank.
    pub fusion: usize,
    /// Reached fusion, ranked out — and the exhaustive search dropped them too.
    ///
    /// ! Not a loss. This is RRF preferring a keyword hit over a dense one, at
    /// every probe count including "probe everything", which is the hybrid
    /// design working as specified. It is counted and named rather than folded
    /// into `fusion` because on a real corpus it is **large**, and a metric that
    /// reported it as loss would argue for retuning `rrf_k` to fix nothing.
    pub by_design: usize,
}

impl RecallLoss {
    #[must_use]
    pub const fn truth(&self) -> usize {
        self.found + self.routing + self.cap + self.fusion + self.by_design
    }

    /// Accumulate another query's attribution.
    pub const fn add(&mut self, other: Self) {
        self.found += other.found;
        self.routing += other.routing;
        self.cap += other.cap;
        self.fusion += other.fusion;
        self.by_design += other.by_design;
    }

    /// Each cause as a share of the truth set · what the report prints.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn shares(&self) -> [f32; 5] {
        let total = self.truth();
        if total == 0 {
            return [1.0, 0.0, 0.0, 0.0, 0.0];
        }
        let n = total as f32;
        [
            self.found as f32 / n,
            self.routing as f32 / n,
            self.cap as f32 / n,
            self.fusion as f32 / n,
            self.by_design as f32 / n,
        ]
    }
}

/// Attribute every miss in the **uncapped dense truth** to the stage that
/// dropped it.
///
/// ! Two references, because the causes are not comparable against one.
///
/// `dense_truth` is an exhaustive scan keeping a global top-k with **no
/// per-cluster cap** — the only reference under which cap loss is visible at
/// all. Against the *exhaustive fused* result it is structurally invisible:
/// that baseline applies the same `per_cluster_top_k` to the same clusters, so
/// any row it keeps the routed run keeps too, and the column reads 0.0% for
/// every setting of the dial. Measured that way, tightening the cap from 50 to 1
/// moved *routing* and left *cap* at zero — a metric that cannot move when its
/// own dial moves, which is precisely the failure `METRICS.md` §3.1 describes
/// and this function exists to fix.
///
/// `exhaustive_fused` is what the same engine returns probing everything, and it
/// splits the rows that reached fusion and lost. If the exhaustive run returned
/// the row, having fewer candidates cost it the rank — real loss, dialled by
/// `rrf_k`. If the exhaustive run dropped it too, RRF prefers a keyword hit
/// there regardless of probing: `by_design`, ✗ a loss. Without that split, a
/// hybrid engine reads as two-thirds "fusion loss" while its end-to-end recall
/// is 98%.
///
/// ! Order is physical, not arbitrary. `found` first: a row the caller got is
/// not a loss however it arrived, so one routing missed and BM25 recovered is a
/// success. Then whatever **reached fusion**, because arriving there is direct
/// evidence that neither routing nor the cap dropped it — testing routing first
/// would blame a global keyword hit that merely ranked low on `clusters_probed`,
/// which cannot move it. Then **routing**, since a row in no probed cluster was
/// never subject to a per-cluster cap. What remains is the **cap**: probed,
/// scanned, and cut before fusion saw it.
#[must_use]
pub fn attribute_recall_loss(
    dense_truth: &[String],
    returned: &[String],
    exhaustive_fused: &[String],
    reached_fusion: &dyn Fn(&str) -> bool,
    cluster_of: &HashMap<String, i32>,
    probed: &HashSet<i32>,
    k: usize,
) -> RecallLoss {
    let got: HashSet<&String> = returned.iter().take(k).collect();
    let ideal: HashSet<&String> = exhaustive_fused.iter().take(k).collect();
    let mut loss = RecallLoss::default();
    for id in dense_truth.iter().take(k) {
        if got.contains(id) {
            loss.found += 1;
        } else if reached_fusion(id) {
            if ideal.contains(id) {
                loss.fusion += 1;
            } else {
                loss.by_design += 1;
            }
        } else if !cluster_of.get(id).is_some_and(|c| probed.contains(c)) {
            loss.routing += 1;
        } else {
            loss.cap += 1;
        }
    }
    loss
}

/// Normalized discounted cumulative gain over the top-k · `EVAL.md` §3.
///
/// Binary relevance: an item is relevant if the exhaustive baseline placed it in
/// its own top-k.
///
/// ! Complements recall@k rather than restating it. Recall is a **set** measure
/// and cannot tell "the right answer came back at rank 1" from "it came back at
/// rank 10" — both score identically while being very different products for a
/// caller that reads the first result and stops. nDCG is the discount recall
/// deliberately lacks.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn ndcg_at_k(routed: &[String], baseline: &[String], k: usize) -> f32 {
    let truth: HashSet<&String> = baseline.iter().take(k).collect();
    if truth.is_empty() {
        return 1.0;
    }
    let discount = |rank: usize| 1.0 / ((rank as f64) + 2.0).log2();

    let dcg: f64 = routed
        .iter()
        .take(k)
        .enumerate()
        .filter(|(_, id)| truth.contains(id))
        .map(|(i, _)| discount(i))
        .sum();
    // Ideal: every relevant item packed into the highest ranks.
    let ideal: f64 = (0..truth.len().min(k)).map(discount).sum();
    if ideal <= f64::EPSILON {
        return 1.0;
    }
    #[allow(clippy::cast_possible_truncation)]
    {
        (dcg / ideal) as f32
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

    fn clusters(pairs: &[(&str, i32)]) -> HashMap<String, i32> {
        pairs.iter().map(|(id, c)| ((*id).to_owned(), *c)).collect()
    }

    #[test]
    fn candidate_cap_loss_is_distinguished_from_a_routing_miss() {
        // ! The measurement METRICS.md §3.1 says did not exist. Both rows are
        // missing from the results and both score identically under recall@k;
        // they need opposite dials. `a` was never probed → clusters_probed.
        // `b` was probed and cut by per_cluster_top_k → the candidate cap.
        let truth = ids(&["a", "b", "c"]);
        let returned = ids(&["c"]);
        let cluster_of = clusters(&[("a", 1), ("b", 2), ("c", 2)]);
        let probed: HashSet<i32> = [2].into_iter().collect();
        // `b` never reached fusion; `c` did.
        let reached = |id: &str| id == "c";

        let loss = attribute_recall_loss(
            &truth, &returned, &truth, &reached, &cluster_of, &probed, 10,
        );
        assert_eq!(
            loss,
            RecallLoss { found: 1, routing: 1, cap: 1, fusion: 0, by_design: 0 }
        );
    }

    #[test]
    fn the_cap_column_moves_when_the_cap_moves() {
        // ! The regression that caught the first version of this metric. Scored
        // against the exhaustive *fused* result the cap column was structurally
        // pinned at zero — that baseline applies the same per-cluster cap, so
        // both sides drop the same row — and tightening the cap showed up as a
        // routing loss instead. Against uncapped dense truth the row is visible.
        let truth = ids(&["a"]);
        let cluster_of = clusters(&[("a", 1)]);
        let probed: HashSet<i32> = [1].into_iter().collect();

        // Cap generous: `a` survives the scan, reaches fusion, comes back.
        let generous = attribute_recall_loss(
            &truth, &ids(&["a"]), &truth, &|_| true, &cluster_of, &probed, 10,
        );
        assert_eq!(generous.cap, 0);

        // Cap tight: same cluster probed, `a` cut before fusion.
        let tight = attribute_recall_loss(
            &truth, &[], &truth, &|_| false, &cluster_of, &probed, 10,
        );
        assert_eq!(tight.cap, 1, "the cap must be observable: {tight:?}");
        assert_eq!(tight.routing, 0, "the cluster was probed");
    }

    #[test]
    fn a_row_the_exhaustive_search_also_drops_is_by_design_not_loss() {
        // ! On a hybrid engine most of the dense top-k is legitimately outranked
        // by keyword hits at *every* probe count. Counting that as fusion loss
        // reads as two-thirds failure while end-to-end recall is 98%, and would
        // argue for retuning rrf_k to fix nothing.
        let truth = ids(&["a"]);
        let cluster_of = clusters(&[("a", 1)]);
        let probed: HashSet<i32> = [1].into_iter().collect();
        let loss = attribute_recall_loss(
            &truth,
            &[],
            &ids(&["something-else"]),
            &|_| true,
            &cluster_of,
            &probed,
            10,
        );
        assert_eq!(loss, RecallLoss { found: 0, routing: 0, cap: 0, fusion: 0, by_design: 1 });
    }

    #[test]
    fn a_row_only_the_exhaustive_search_returned_is_real_fusion_loss() {
        // Probed, survived the cap, reached fusion, and the unrouted version of
        // the same search *did* return it · fewer candidates cost it the rank.
        let truth = ids(&["a"]);
        let cluster_of = clusters(&[("a", 1)]);
        let probed: HashSet<i32> = [1].into_iter().collect();
        let loss = attribute_recall_loss(
            &truth, &[], &truth, &|_| true, &cluster_of, &probed, 10,
        );
        assert_eq!(loss, RecallLoss { found: 0, routing: 0, cap: 0, fusion: 1, by_design: 0 });
    }

    #[test]
    fn a_row_routing_missed_but_bm25_recovered_counts_as_found() {
        // ! Why `found` is tested first. The global keyword net exists precisely
        // to rescue rows routing missed (LOOPHOLES.md §1); scoring that rescue
        // as a routing failure would argue for raising clusters_probed to fix a
        // problem that is already solved.
        let truth = ids(&["a"]);
        let cluster_of = clusters(&[("a", 9)]);
        let probed: HashSet<i32> = [1].into_iter().collect();
        let loss = attribute_recall_loss(
            &truth, &ids(&["a"]), &truth, &|_| true, &cluster_of, &probed, 10,
        );
        assert_eq!(loss, RecallLoss { found: 1, routing: 0, cap: 0, fusion: 0, by_design: 0 });
    }

    #[test]
    fn an_unprobed_row_is_never_blamed_on_the_cap() {
        // ! The ordering rule. A row in no probed cluster cannot have been cut
        // by a per-cluster cap that never ran on it, even though it also never
        // reached fusion — so the routing test must come before the cap.
        let truth = ids(&["a"]);
        let cluster_of = clusters(&[("a", 7)]);
        let loss = attribute_recall_loss(
            &truth, &[], &truth, &|_| false, &cluster_of, &HashSet::new(), 10,
        );
        assert_eq!(loss, RecallLoss { found: 0, routing: 1, cap: 0, fusion: 0, by_design: 0 });
    }

    #[test]
    fn a_global_keyword_hit_that_ranked_out_is_never_blamed_on_routing() {
        // ! The other half of the ordering rule, and the reason reaching fusion
        // is tested before routing. This row sat in no probed cluster, but the
        // global BM25 query found it anyway — so it *did* reach fusion and lost
        // on rank. Blaming `clusters_probed` would argue for probing more
        // clusters to recover a row that probing cannot affect.
        let truth = ids(&["a"]);
        let cluster_of = clusters(&[("a", 7)]);
        let loss = attribute_recall_loss(
            &truth, &[], &truth, &|_| true, &cluster_of, &HashSet::new(), 10,
        );
        assert_eq!(loss, RecallLoss { found: 0, routing: 0, cap: 0, fusion: 1, by_design: 0 });
    }

    #[test]
    fn the_causes_always_partition_the_truth_set() {
        let truth = ids(&["a", "b", "c", "d"]);
        let cluster_of = clusters(&[("a", 1), ("b", 1), ("c", 2), ("d", 2)]);
        let probed: HashSet<i32> = [1].into_iter().collect();
        let loss = attribute_recall_loss(
            &truth,
            &ids(&["a"]),
            &truth,
            &|id| id == "b",
            &cluster_of,
            &probed,
            10,
        );
        assert_eq!(loss.truth(), 4, "every truth item must be attributed exactly once");
        assert!((loss.shares().iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn losses_accumulate_across_queries() {
        let mut total = RecallLoss::default();
        total.add(RecallLoss { found: 8, routing: 1, cap: 1, fusion: 0, by_design: 0 });
        total.add(RecallLoss { found: 9, routing: 0, cap: 0, fusion: 1, by_design: 0 });
        assert_eq!(
            total,
            RecallLoss { found: 17, routing: 1, cap: 1, fusion: 1, by_design: 0 }
        );
        assert_eq!(total.truth(), 20);
    }

    #[test]
    fn ndcg_rewards_the_ranking_recall_is_blind_to() {
        // ! The complaint against recall as a lone quality number: these two
        // result lists have identical recall and are not equally good.
        let baseline = ids(&["a", "b"]);
        let early = ndcg_at_k(&ids(&["a", "b", "x", "y"]), &baseline, 4);
        let late = ndcg_at_k(&ids(&["x", "y", "a", "b"]), &baseline, 4);
        assert!((recall_at_k(&ids(&["a", "b", "x", "y"]), &baseline, 4)
            - recall_at_k(&ids(&["x", "y", "a", "b"]), &baseline, 4))
        .abs()
            < 1e-6);
        assert!(early > late, "{early} vs {late}");
        assert!((early - 1.0).abs() < 1e-6, "a perfect ordering is 1.0");
    }

    #[test]
    fn ndcg_is_zero_when_nothing_relevant_came_back() {
        assert!(ndcg_at_k(&ids(&["x", "y"]), &ids(&["a"]), 10).abs() < 1e-6);
    }

    #[test]
    fn ndcg_over_an_empty_baseline_is_perfect_not_zero() {
        // Same convention as recall@k · an empty truth set must not drag the
        // average down for queries that had nothing to find.
        assert!((ndcg_at_k(&ids(&["a"]), &[], 10) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn top1_tracks_the_single_most_important_result() {
        assert!(top1_hit(&ids(&["a", "z"]), &ids(&["a", "b"])));
        assert!(!top1_hit(&ids(&["z", "a"]), &ids(&["a", "b"])));
        assert!(!top1_hit(&[], &ids(&["a"])), "found nothing at all");
        assert!(top1_hit(&[], &[]), "nothing to find");
    }
}
