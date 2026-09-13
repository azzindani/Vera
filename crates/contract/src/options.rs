//! Per-request search options · `docs/TOOL_SURFACE.md`.
//!
//! What the **caller** may steer, as opposed to what the engine keeps. The line
//! is not about trust: it is about what the caller can possibly know. Latency
//! budget and context size are properties of the caller's situation; factor
//! weights and cluster probe width are properties of the corpus, fitted against
//! the eval set, and an agent has no fitting signal for them.
//!
//! Three rules hold for everything here, and [`SearchOptions::resolve`] is where
//! the first two are enforced:
//!
//! 1. **A caller may narrow a server limit, never widen it.** Every numeric
//!    field clamps to the configured ceiling. A request is a preference, ✗ an
//!    override.
//! 2. **Anything that changed the ranking is echoed back** ([`AppliedOptions`]).
//!    A ranking that cannot be reproduced is not evidence, and invariant 15 is
//!    unenforceable if no two calls are comparable.
//! 3. **Every field is optional and every default is the measured one.** A
//!    caller that passes nothing gets exactly the behaviour `EVAL.md` scored.

use serde::{Deserialize, Serialize};

/// Which retrieval arms run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Mode {
    /// All three arms, fused by RRF. The measured default.
    #[default]
    Hybrid,
    /// Sparse (BM25) + text (`tsvector`) only.
    ///
    /// The two arms that carry this corpus: 38.6% and 40.9% Recall@5 alone
    /// (`EVAL.md` §4). Cheaper than `Hybrid` by one embedding call.
    Keyword,
    /// The dense arm alone.
    ///
    /// ! **Measurably non-contributing on this corpus.** Dense scores 0.0%
    /// Recall@5 alone and ships at weight 0.0 (`EMBEDDING.md` §5). The mode
    /// exists so the argument surface need not be reshaped when the embedding
    /// space is fixed, and a response produced under it carries an explicit
    /// hint saying so — offering a capability we measured as absent, silently,
    /// is the worse option.
    Semantic,
}

impl Mode {
    /// Whether this mode is known to contribute nothing on the current corpus.
    /// Drives the hint on the response, ✗ a refusal: the caller asked.
    #[must_use]
    pub const fn is_measured_empty(self) -> bool {
        matches!(self, Self::Semantic)
    }
}

/// A fitted intent for the factor layer.
///
/// ! The agent selects **intent**; Vera keeps the **numbers**
/// (`TOOL_SURFACE.md` §5). Fitting these weights took 625 combinations against
/// 40 labelled cases, and the in-sample peak flattered itself by ten points. An
/// agent has no way to run that fit, so a raw weight vector from a caller is a
/// guess that also makes the same query return different rankings on different
/// calls.
///
/// ! A profile is fitted, ✗ chosen. One that does not beat [`Balanced`] on the
/// query shapes it targets does not ship — and until it is fitted it is not
/// listed in the tool schema, because an unfitted profile is a raw weight
/// vector with a friendly name.
///
/// [`Balanced`]: Profile::Balanced
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Profile {
    /// The general case · `authority 0.5 · structural 0.25 · completeness 0.25`,
    /// fitted by `dev_tools/eval/fit_factors.py` (+7.5 points leave-one-out).
    #[default]
    Balanced,
}

/// Explicit factor weights · **experimental**.
///
/// ! Reachable for experimentation, never the recommended path. A response
/// produced with these set carries `experimental: true`, and the effective
/// weights are echoed back whether or not the caller passed any.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FactorWeights {
    pub authority: f32,
    pub structural: f32,
    pub temporal: f32,
    pub completeness: f32,
    pub topical: f32,
}

/// What the caller asked for. Every field optional; `None` means "use the
/// measured default".
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SearchOptions {
    pub mode: Option<Mode>,
    /// Results returned. Clamped to the server's `TOP_K`.
    pub top_k: Option<usize>,
    /// Candidates ranked before the cut. Clamped to the server's
    /// `CANDIDATE_POOL`, and floored at the effective `top_k` — a pool smaller
    /// than the answer silently caps the reply (`SCORING.md` §7).
    pub candidate_pool: Option<usize>,
    pub profile: Option<Profile>,
    pub factor_weights: Option<FactorWeights>,
}

/// The server's ceilings · what a caller may narrow toward.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Ceilings {
    pub top_k: usize,
    pub candidate_pool: usize,
}

/// What the engine actually used · echoed on every response.
///
/// ! Present even when the caller passed nothing. "The defaults were used" is
/// itself the reproducibility record, and a field that appears only sometimes
/// is one a client learns to ignore.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AppliedOptions {
    pub mode: Mode,
    pub top_k: usize,
    pub candidate_pool: usize,
    pub profile: Profile,
    pub factor_weights: FactorWeights,
    /// Set when the caller supplied raw `factor_weights` rather than a profile.
    pub experimental: bool,
    /// Every place the request was narrowed to a server ceiling, in the caller's
    /// words. Empty when nothing was clamped.
    ///
    /// ! Silent clamping is the failure this prevents: a caller that asks for
    /// 500 results and receives 10 must be able to see *why* without reading
    /// the server's configuration.
    pub clamped: Vec<String>,
}

impl SearchOptions {
    /// Resolve a request against the server's ceilings and defaults.
    ///
    /// Narrows only. A caller asking for more than the server allows gets the
    /// server's value and a line in [`AppliedOptions::clamped`] saying so.
    #[must_use]
    pub fn resolve(&self, ceilings: Ceilings, default_weights: FactorWeights) -> AppliedOptions {
        let mut clamped = Vec::new();

        let top_k = match self.top_k {
            Some(0) => {
                clamped.push("top_k 0 raised to 1".to_owned());
                1
            }
            Some(n) if n > ceilings.top_k => {
                clamped.push(format!(
                    "top_k {n} narrowed to the server ceiling {}",
                    ceilings.top_k
                ));
                ceilings.top_k
            }
            Some(n) => n,
            None => ceilings.top_k,
        };

        let mut candidate_pool = match self.candidate_pool {
            Some(n) if n > ceilings.candidate_pool => {
                clamped.push(format!(
                    "candidate_pool {n} narrowed to the server ceiling {}",
                    ceilings.candidate_pool
                ));
                ceilings.candidate_pool
            }
            Some(n) => n,
            None => ceilings.candidate_pool,
        };

        // ! Floored at top_k, ✗ merely validated. A pool smaller than the answer
        // caps the reply below what the caller asked for, which is a wrong
        // answer rather than a slow one.
        if candidate_pool < top_k {
            clamped.push(format!(
                "candidate_pool {candidate_pool} raised to top_k {top_k}"
            ));
            candidate_pool = top_k;
        }

        let experimental = self.factor_weights.is_some();
        let profile = self.profile.unwrap_or_default();
        let factor_weights = self.factor_weights.unwrap_or(match profile {
            Profile::Balanced => default_weights,
        });

        AppliedOptions {
            mode: self.mode.unwrap_or_default(),
            top_k,
            candidate_pool,
            profile,
            factor_weights,
            experimental,
            clamped,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CEIL: Ceilings = Ceilings {
        top_k: 10,
        candidate_pool: 60,
    };
    const W: FactorWeights = FactorWeights {
        authority: 0.5,
        structural: 0.25,
        temporal: 0.0,
        completeness: 0.25,
        topical: 0.0,
    };

    #[test]
    fn passing_nothing_yields_the_measured_defaults() {
        let a = SearchOptions::default().resolve(CEIL, W);
        assert_eq!(a.mode, Mode::Hybrid);
        assert_eq!(a.top_k, 10);
        assert_eq!(a.candidate_pool, 60);
        assert_eq!(a.profile, Profile::Balanced);
        assert_eq!(a.factor_weights, W);
        assert!(!a.experimental);
        assert!(a.clamped.is_empty());
    }

    #[test]
    fn a_caller_may_narrow_a_limit() {
        let a = SearchOptions {
            top_k: Some(3),
            ..SearchOptions::default()
        }
        .resolve(CEIL, W);
        assert_eq!(a.top_k, 3);
        assert!(a.clamped.is_empty(), "narrowing is not clamping");
    }

    #[test]
    fn a_caller_may_not_widen_one() {
        let a = SearchOptions {
            top_k: Some(500),
            candidate_pool: Some(100_000),
            ..SearchOptions::default()
        }
        .resolve(CEIL, W);
        assert_eq!(a.top_k, 10);
        assert_eq!(a.candidate_pool, 60);
        assert_eq!(a.clamped.len(), 2, "{:?}", a.clamped);
    }

    #[test]
    fn every_clamp_is_reported_not_silent() {
        // A caller asking for 500 and receiving 10 must see why without
        // reading the server's configuration.
        let a = SearchOptions {
            top_k: Some(500),
            ..SearchOptions::default()
        }
        .resolve(CEIL, W);
        assert!(a.clamped[0].contains("500"), "{:?}", a.clamped);
        assert!(a.clamped[0].contains("10"), "{:?}", a.clamped);
    }

    #[test]
    fn a_pool_smaller_than_the_answer_is_raised_to_it() {
        let a = SearchOptions {
            top_k: Some(10),
            candidate_pool: Some(2),
            ..SearchOptions::default()
        }
        .resolve(CEIL, W);
        assert_eq!(a.candidate_pool, 10);
        assert!(!a.clamped.is_empty());
    }

    #[test]
    fn zero_results_is_raised_rather_than_returning_nothing() {
        let a = SearchOptions {
            top_k: Some(0),
            ..SearchOptions::default()
        }
        .resolve(CEIL, W);
        assert_eq!(a.top_k, 1);
    }

    #[test]
    fn raw_weights_are_marked_experimental_and_echoed() {
        let mine = FactorWeights {
            authority: 1.0,
            ..W
        };
        let a = SearchOptions {
            factor_weights: Some(mine),
            ..SearchOptions::default()
        }
        .resolve(CEIL, W);
        assert!(a.experimental);
        assert_eq!(
            a.factor_weights, mine,
            "the effective weights must be echoed"
        );
    }

    #[test]
    fn the_semantic_mode_declares_itself_non_contributing() {
        // Dense is 0.0% Recall@5 on this corpus. The caller may still ask.
        assert!(Mode::Semantic.is_measured_empty());
        assert!(!Mode::Hybrid.is_measured_empty());
        assert!(!Mode::Keyword.is_measured_empty());
    }

    #[test]
    fn unknown_arguments_are_rejected_rather_than_ignored() {
        // An agent that misspells a knob must be told, not silently served the
        // default -- that is a wrong answer wearing a correct one's clothes.
        let err = serde_json::from_str::<SearchOptions>(r#"{"top_kk": 5}"#);
        assert!(err.is_err());
    }

    #[test]
    fn options_round_trip_through_json() {
        let o = SearchOptions {
            mode: Some(Mode::Keyword),
            top_k: Some(5),
            ..SearchOptions::default()
        };
        let s = serde_json::to_string(&o).expect("serialize");
        assert_eq!(serde_json::from_str::<SearchOptions>(&s).expect("parse"), o);
    }
}
