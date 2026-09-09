//! Reciprocal Rank Fusion.
//!
//! ! Model-free by design. A cross-encoder reranker would score better on paper
//! and would put a model back on the query path, which is the one thing the
//! architecture spends everything else to avoid (`CLAUDE.md` §5 rule 1).
//!
//! RRF scores a document by `Σ 1/(k + rank)` over the lists it appears in. It
//! consumes **ranks, ✗ scores**, which is exactly why it works here: a cosine
//! in [-1,1] and a BM25 in [0,∞) have no common scale, and any attempt to
//! normalize them into one introduces a tuning constant per corpus. Ranks need
//! no such constant.

use std::collections::HashMap;

use crate::topk::Scored;

/// One input list to the fusion, already ranked best-first.
#[derive(Debug, Clone)]
pub struct RankedList<'a> {
    pub label: &'static str,
    pub items: &'a [Scored],
}

/// A fused result, carrying enough detail for `search(dry_run)` and for the
/// per-modality scores the output contract publishes.
#[derive(Debug, Clone, PartialEq)]
pub struct Fused {
    pub id: String,
    pub score: f32,
    /// Raw score from each list that contained this id, by label.
    pub components: HashMap<&'static str, f32>,
    /// 1-based rank in each list that contained this id.
    pub ranks: HashMap<&'static str, usize>,
}

/// Fuse ranked lists into one ordered result set.
///
/// `k` damps the advantage of the top rank; 60 is the constant from the
/// original RRF paper. Ties break on id so the output is deterministic — an
/// unstable order would make the eval harness flap for no reason.
#[must_use]
pub fn reciprocal_rank_fusion(lists: &[RankedList<'_>], k: f32) -> Vec<Fused> {
    let mut acc: HashMap<String, Fused> = HashMap::new();

    for list in lists {
        for (i, item) in list.items.iter().enumerate() {
            let rank = i + 1;
            #[allow(clippy::cast_precision_loss)]
            let contribution = 1.0 / (k + rank as f32);
            let entry = acc.entry(item.id.clone()).or_insert_with(|| Fused {
                id: item.id.clone(),
                score: 0.0,
                components: HashMap::new(),
                ranks: HashMap::new(),
            });
            entry.score += contribution;
            // ! First occurrence wins. A document can surface in several probed
            // clusters; its best rank is the one that reflects how well it
            // actually matched, and double-counting it would let a duplicate
            // outrank a better unique document.
            entry.components.entry(list.label).or_insert(item.score);
            entry.ranks.entry(list.label).or_insert(rank);
        }
    }

    let mut out: Vec<Fused> = acc.into_values().collect();
    out.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.id.cmp(&b.id))
    });
    out
}

/// Merge per-cluster ranked lists into one corpus-wide ranked list.
///
/// Used to turn the sequential leaf scan's N per-cluster lists into the single
/// dense list RRF consumes. Re-sorts by raw score, which is valid **within** a
/// modality — every cluster's dense scores are cosines in the same space.
#[must_use]
pub fn merge_by_score(lists: Vec<Vec<Scored>>, limit: usize) -> Vec<Scored> {
    let mut all: Vec<Scored> = lists.into_iter().flatten().collect();
    all.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.id.cmp(&b.id))
    });
    // ! Set-based, ✗ `Vec::dedup_by`, which only removes *consecutive* equals.
    // After sorting by score a document's two copies are almost never adjacent
    // — a better-scoring different id sits between them — so dedup_by would let
    // duplicates through and they would then be double-counted by RRF.
    let mut seen = std::collections::HashSet::with_capacity(all.len());
    all.retain(|s| seen.insert(s.id.clone()));
    all.truncate(limit);
    all
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scored(pairs: &[(&str, f32)]) -> Vec<Scored> {
        pairs
            .iter()
            .map(|(id, s)| Scored {
                id: (*id).to_owned(),
                score: *s,
            })
            .collect()
    }

    #[test]
    fn a_document_ranked_well_by_both_halves_beats_one_ranked_well_by_either() {
        // ! The entire reason for hybrid search: agreement between an
        // independent dense and keyword signal is stronger evidence than a top
        // rank in one of them.
        let dense = scored(&[("only-dense", 0.99), ("both", 0.80)]);
        let keyword = scored(&[("only-keyword", 12.0), ("both", 9.0)]);
        let fused = reciprocal_rank_fusion(
            &[
                RankedList {
                    label: "dense",
                    items: &dense,
                },
                RankedList {
                    label: "bm25",
                    items: &keyword,
                },
            ],
            60.0,
        );
        assert_eq!(fused[0].id, "both");
        assert_eq!(fused[0].ranks["dense"], 2);
        assert_eq!(fused[0].ranks["bm25"], 2);
    }

    #[test]
    fn fusion_uses_ranks_so_incommensurable_scales_never_meet() {
        // BM25 in the thousands must not swamp a cosine in [0,1]. Under any
        // score-additive scheme "keyword-only" would win; under RRF it cannot.
        let dense = scored(&[("a", 0.9)]);
        let keyword = scored(&[("b", 5_000.0)]);
        let fused = reciprocal_rank_fusion(
            &[
                RankedList {
                    label: "dense",
                    items: &dense,
                },
                RankedList {
                    label: "bm25",
                    items: &keyword,
                },
            ],
            60.0,
        );
        assert!(
            (fused[0].score - fused[1].score).abs() < f32::EPSILON,
            "rank-1 in either list must contribute equally: {fused:?}"
        );
    }

    #[test]
    fn raw_component_scores_survive_for_the_output_contract() {
        let dense = scored(&[("a", 0.83)]);
        let fused = reciprocal_rank_fusion(
            &[RankedList {
                label: "dense",
                items: &dense,
            }],
            60.0,
        );
        assert!((fused[0].components["dense"] - 0.83).abs() < 1e-6);
        assert!(!fused[0].components.contains_key("bm25"));
    }

    #[test]
    fn a_document_in_two_probed_clusters_is_not_double_counted() {
        // Same list label appearing twice for one id · keep the better rank.
        let first = scored(&[("dup", 0.9)]);
        let second = scored(&[("other", 0.8), ("dup", 0.4)]);
        let fused = reciprocal_rank_fusion(
            &[
                RankedList {
                    label: "dense",
                    items: &first,
                },
                RankedList {
                    label: "dense",
                    items: &second,
                },
            ],
            60.0,
        );
        let dup = fused.iter().find(|f| f.id == "dup").unwrap();
        assert_eq!(dup.ranks["dense"], 1, "kept the first (better) rank");
        assert!((dup.components["dense"] - 0.9).abs() < 1e-6);
    }

    #[test]
    fn the_order_is_deterministic_when_scores_tie() {
        // ! The eval harness diffs rankings; a HashMap-ordered tie-break would
        // make it flap between runs for no real reason.
        let a = scored(&[("z", 1.0), ("y", 1.0), ("x", 1.0)]);
        let first = reciprocal_rank_fusion(
            &[RankedList {
                label: "dense",
                items: &a,
            }],
            60.0,
        );
        for _ in 0..20 {
            let again = reciprocal_rank_fusion(
                &[RankedList {
                    label: "dense",
                    items: &a,
                }],
                60.0,
            );
            assert_eq!(
                first.iter().map(|f| &f.id).collect::<Vec<_>>(),
                again.iter().map(|f| &f.id).collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn fusing_nothing_yields_nothing() {
        assert!(reciprocal_rank_fusion(&[], 60.0).is_empty());
    }

    #[test]
    fn merging_cluster_lists_dedupes_and_truncates() {
        let merged = merge_by_score(
            vec![
                scored(&[("a", 0.9), ("b", 0.5)]),
                scored(&[("c", 0.8), ("a", 0.7)]),
            ],
            3,
        );
        let ids: Vec<_> = merged.iter().map(|s| s.id.as_str()).collect();
        assert_eq!(ids, ["a", "c", "b"], "best score per id, best-first");
    }

    #[test]
    fn a_duplicate_separated_by_a_better_document_is_still_deduped() {
        // ! Regression: the sorted order here is a(0.9), c(0.8), a(0.7) — the
        // two copies of `a` are not adjacent, so a consecutive-only dedup keeps
        // both and RRF then counts the same document twice.
        let merged = merge_by_score(
            vec![scored(&[("a", 0.9)]), scored(&[("c", 0.8), ("a", 0.7)])],
            10,
        );
        let ids: Vec<_> = merged.iter().map(|s| s.id.as_str()).collect();
        assert_eq!(ids, ["a", "c"]);
        assert!((merged[0].score - 0.9).abs() < 1e-6, "kept the better score");
    }
}
