//! Reciprocal Rank Fusion · combining the retrieval arms.
//!
//! ! RRF fuses **ranks**, ✗ scores. That is the whole point: cosine distance,
//! BM25 inner product and `ts_rank` live on incomparable scales, and any
//! attempt to normalise them into a shared range bakes in a calibration that
//! drifts the moment the corpus changes. Ranks need no calibration.
//!
//! Measured on the 182K-row spike corpus, the arms are lopsided — the lexical
//! arms far outrank dense on subject-line queries. Weights exist so that can be
//! corrected from config once a *paraphrase-based* eval set exists, ✗ tuned by
//! hand against a biased one (`EVAL.md` §2).

use std::collections::HashMap;

/// Damping constant. 60 is the value from the original RRF paper and the
/// de-facto default; it flattens the contribution of the very top ranks so one
/// confident arm cannot monopolise the fused ordering.
pub const DEFAULT_K: f32 = 60.0;

/// One arm's contribution: an ordered list of ids, best first.
#[derive(Debug, Clone)]
pub struct Arm<'a> {
    pub name: &'static str,
    pub ids: &'a [String],
    /// Relative influence. 1.0 is neutral; 0.0 disables the arm entirely.
    pub weight: f32,
}

/// A fused result: the id, its combined score, and which arms found it.
#[derive(Debug, Clone, PartialEq)]
pub struct Fused {
    pub id: String,
    pub score: f32,
    /// `(arm name, 1-based rank)` for every arm that returned this id.
    /// Published so `explain_routing` can show *why* something ranked.
    pub contributions: Vec<(&'static str, usize)>,
}

/// Fuse ranked lists into one ordering.
///
/// Ties break on id so the output is deterministic — an unstable sort here
/// would make golden-file tests flap for no reason.
#[must_use]
pub fn reciprocal_rank_fusion(arms: &[Arm<'_>], k: f32) -> Vec<Fused> {
    let mut scores: HashMap<&str, f32> = HashMap::new();
    let mut where_found: HashMap<&str, Vec<(&'static str, usize)>> = HashMap::new();

    for arm in arms {
        if arm.weight == 0.0 {
            continue;
        }
        for (i, id) in arm.ids.iter().enumerate() {
            let rank = i + 1;
            #[allow(clippy::cast_precision_loss)]
            let contribution = arm.weight / (k + rank as f32);
            *scores.entry(id).or_insert(0.0) += contribution;
            where_found.entry(id).or_default().push((arm.name, rank));
        }
    }

    let mut out: Vec<Fused> = scores
        .into_iter()
        .map(|(id, score)| Fused {
            id: id.to_owned(),
            score,
            contributions: where_found.remove(id).unwrap_or_default(),
        })
        .collect();

    out.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.id.cmp(&b.id))
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ids(xs: &[&str]) -> Vec<String> {
        xs.iter().map(|s| (*s).to_owned()).collect()
    }

    #[test]
    fn an_item_ranked_well_by_two_arms_beats_one_ranked_well_by_one() {
        let a = ids(&["x", "y"]);
        let b = ids(&["x", "z"]);
        let fused = reciprocal_rank_fusion(
            &[
                Arm {
                    name: "dense",
                    ids: &a,
                    weight: 1.0,
                },
                Arm {
                    name: "sparse",
                    ids: &b,
                    weight: 1.0,
                },
            ],
            DEFAULT_K,
        );
        assert_eq!(fused[0].id, "x");
        assert_eq!(fused[0].contributions.len(), 2, "found by both arms");
    }

    #[test]
    fn fusing_identical_lists_preserves_their_order() {
        let a = ids(&["a", "b", "c"]);
        let fused = reciprocal_rank_fusion(
            &[
                Arm {
                    name: "one",
                    ids: &a,
                    weight: 1.0,
                },
                Arm {
                    name: "two",
                    ids: &a,
                    weight: 1.0,
                },
            ],
            DEFAULT_K,
        );
        assert_eq!(
            fused.iter().map(|f| f.id.as_str()).collect::<Vec<_>>(),
            ["a", "b", "c"]
        );
    }

    #[test]
    fn a_zero_weight_arm_contributes_nothing() {
        // ! The spike measured dense at 26.7% Recall@5 against sparse at 100%.
        // Disabling an arm must be a config change, ✗ a code change.
        let good = ids(&["right"]);
        let noise = ids(&["wrong", "wrong2"]);
        let fused = reciprocal_rank_fusion(
            &[
                Arm {
                    name: "sparse",
                    ids: &good,
                    weight: 1.0,
                },
                Arm {
                    name: "dense",
                    ids: &noise,
                    weight: 0.0,
                },
            ],
            DEFAULT_K,
        );
        assert_eq!(fused.len(), 1);
        assert_eq!(fused[0].id, "right");
    }

    #[test]
    fn weights_shift_the_ordering_without_touching_the_arms() {
        let dense = ids(&["d"]);
        let sparse = ids(&["s"]);
        let weighted = |wd: f32, ws: f32| {
            reciprocal_rank_fusion(
                &[
                    Arm {
                        name: "dense",
                        ids: &dense,
                        weight: wd,
                    },
                    Arm {
                        name: "sparse",
                        ids: &sparse,
                        weight: ws,
                    },
                ],
                DEFAULT_K,
            )[0]
            .id
            .clone()
        };
        assert_eq!(weighted(2.0, 1.0), "d");
        assert_eq!(weighted(1.0, 2.0), "s");
    }

    #[test]
    fn output_is_deterministic_when_scores_tie() {
        let a = ids(&["b"]);
        let b = ids(&["a"]);
        let run = || {
            reciprocal_rank_fusion(
                &[
                    Arm {
                        name: "one",
                        ids: &a,
                        weight: 1.0,
                    },
                    Arm {
                        name: "two",
                        ids: &b,
                        weight: 1.0,
                    },
                ],
                DEFAULT_K,
            )
        };
        // Equal scores · id breaks the tie, and it breaks it the same way twice.
        assert_eq!(run()[0].id, "a");
        assert_eq!(run(), run());
    }

    #[test]
    fn contributions_record_the_rank_each_arm_gave() {
        let dense = ids(&["p", "q"]);
        let sparse = ids(&["q", "p"]);
        let fused = reciprocal_rank_fusion(
            &[
                Arm {
                    name: "dense",
                    ids: &dense,
                    weight: 1.0,
                },
                Arm {
                    name: "sparse",
                    ids: &sparse,
                    weight: 1.0,
                },
            ],
            DEFAULT_K,
        );
        let p = fused.iter().find(|f| f.id == "p").unwrap();
        assert!(p.contributions.contains(&("dense", 1)));
        assert!(p.contributions.contains(&("sparse", 2)));
    }

    #[test]
    fn no_arms_yields_no_results_rather_than_panicking() {
        assert!(reciprocal_rank_fusion(&[], DEFAULT_K).is_empty());
    }
}
