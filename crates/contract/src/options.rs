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

use crate::algorithms::FactorEntry;

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

// ! `Profile` was an enum here with one variant, `Balanced`. It is now a
// registry key — see [`crate::algorithms`]. Adding a way to rank meant editing
// this file, `tools.rs` and `main.rs` and shipping a new binary, which is the
// same mistake as compiling in a model width (invariant 2). A ranking strategy
// is a property of the corpus, fitted against its eval set, so it is declared
// beside the corpus.
//
// What did **not** change is the rule the enum existed to enforce: the agent
// selects **intent**, Vera keeps the **numbers** (`TOOL_SURFACE.md` §5).
// Fitting took 3,125 combinations against 40 labelled cases and the in-sample
// peak flattered itself by five points; an agent has no way to run that fit.
// The registry keeps the numbers server-side and offers only fitted names.

/// Explicit factor weights · **experimental**.
///
/// ! Reachable for experimentation, never the recommended path. A response
/// produced with these set carries `experimental: true`, and the effective
/// weights are echoed back whether or not the caller passed any.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FactorWeights {
    /// Minimum share of the query's content terms a candidate must contain ·
    /// **a floor, ✗ a weight** (`docs/SCORING.md` §3). Worth more than every
    /// weight below combined.
    pub relevance_floor: f32,
    pub authority: f32,
    pub structural: f32,
    pub temporal: f32,
    pub completeness: f32,
    pub topical: f32,
}

/// Which neighbours may be pulled in beside what retrieval found.
///
/// ! An expansion admits chunks **no arm retrieved**. That is a different kind
/// of result, so it is opt-in and marked on the wire (`SearchResult::
/// expanded_from`), ✗ folded silently into the ranking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Expansion {
    /// Other chunks of the same regulation · `docs/SCORING.md` §7.
    Siblings,
}

/// What the caller asked for. Every field optional; `None` means "use the
/// measured default".
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SearchOptions {
    pub mode: Option<Mode>,
    /// Results returned. Clamped to the server's `TOP_K`.
    pub top_k: Option<usize>,
    /// Candidates ranked before the cut. Clamped to the server's
    /// `CANDIDATE_POOL`, and floored at the effective `top_k` — a pool smaller
    /// than the answer silently caps the reply (`SCORING.md` §7).
    pub candidate_pool: Option<usize>,
    /// A key into the algorithm registry. `None` means
    /// [`DEFAULT_ALGORITHM`](crate::algorithms::DEFAULT_ALGORITHM).
    ///
    /// ! An unknown name is an **error**, ✗ a fallback to the default. A caller
    /// that misspells `sanction` and is silently served `balanced` gets a wrong
    /// answer wearing a correct one's clothes, and its ranking is then
    /// unreproducible.
    pub profile: Option<String>,
    pub factor_weights: Option<FactorWeights>,
    /// Empty by default · see [`Expansion`].
    pub expand: Option<Vec<Expansion>>,
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
    /// The registry key that ranked this request.
    pub profile: String,
    pub factor_weights: FactorWeights,
    /// The scoring function that actually ran.
    ///
    /// ! Echoed in full, because with assembly the weights alone no longer say
    /// what happened: two algorithms can share a weight vector and read
    /// different variables through different shapes. A ranking that cannot be
    /// reproduced from the response is not evidence.
    pub composition: Vec<FactorEntry>,
    pub expand: Vec<Expansion>,
    /// Set when the caller supplied raw `factor_weights`, **or** named an
    /// algorithm carrying no `fitted` block.
    ///
    /// ! Both cases are the same claim to a caller — "nobody measured this
    /// ranking" — so they set the same flag rather than two the client has to
    /// learn to check separately.
    pub experimental: bool,
    /// Every place the request was narrowed to a server ceiling, in the caller's
    /// words. Empty when nothing was clamped.
    ///
    /// ! Silent clamping is the failure this prevents: a caller that asks for
    /// 500 results and receives 10 must be able to see *why* without reading
    /// the server's configuration.
    pub clamped: Vec<String>,
}

/// A request that cannot be served as written.
///
/// ! One variant, deliberately. Every *numeric* overreach narrows silently-
/// but-reported into [`AppliedOptions::clamped`]; only naming something that
/// does not exist is fatal, because there is no nearest sensible value for it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResolveError {
    /// The caller named an algorithm the registry does not define.
    UnknownProfile {
        asked: String,
        /// Every name the registry does define · the `hint` the caller needs to
        /// correct itself without reading the server's configuration.
        available: Vec<String>,
    },
}

impl std::fmt::Display for ResolveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownProfile { asked, available } => write!(
                f,
                "no algorithm named `{asked}` · this engine serves: {}",
                available.join(", ")
            ),
        }
    }
}

impl std::error::Error for ResolveError {}

impl SearchOptions {
    /// Resolve a request against the server's ceilings and its registry.
    ///
    /// Narrows only. A caller asking for more than the server allows gets the
    /// server's value and a line in [`AppliedOptions::clamped`] saying so.
    ///
    /// # Errors
    /// [`ResolveError::UnknownProfile`] when the caller names an algorithm the
    /// registry does not define. Not a fallback: see [`SearchOptions::profile`].
    pub fn resolve(
        &self,
        ceilings: Ceilings,
        registry: &crate::algorithms::Registry,
    ) -> Result<AppliedOptions, ResolveError> {
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

        let name = self
            .profile
            .clone()
            .unwrap_or_else(|| crate::algorithms::DEFAULT_ALGORITHM.to_owned());
        let algo = registry
            .get(&name)
            .ok_or_else(|| ResolveError::UnknownProfile {
                asked: name.clone(),
                available: registry.names().into_iter().map(str::to_owned).collect(),
            })?;

        // Raw weights win over the algorithm's, and say so. An unfitted
        // algorithm makes the same claim, so both raise the one flag.
        let experimental = self.factor_weights.is_some() || !algo.is_fitted();
        let factor_weights = self.factor_weights.unwrap_or(algo.factors);
        // ! Raw weights override an assembled composition too, and reduce to the
        // five-term form. A caller passing `factor_weights` is asking for the
        // shorthand; silently keeping the algorithm's extra factors would make
        // the response's own echo wrong about what ranked it.
        let composition = match (&algo.composition, self.factor_weights) {
            (Some(entries), None) => entries.clone(),
            _ => crate::algorithms::classic_composition(&factor_weights),
        };

        // ! The algorithm's own expansions are a floor, ✗ a replacement. An
        // algorithm fitted WITH siblings admitted is a different measurement
        // from the same weights without them, so a caller that names it and
        // passes no `expand` must get what was fitted.
        let mut expand = algo.expand.clone();
        expand.extend(self.expand.clone().unwrap_or_default());
        // ! Sorted before dedup. `dedup` only removes ADJACENT duplicates, so
        // `[siblings, citations, siblings]` kept the walk twice — and asking
        // for the same expansion twice must cost what asking once costs.
        expand.sort_unstable_by_key(|e| format!("{e:?}"));
        expand.dedup();

        Ok(AppliedOptions {
            mode: self.mode.unwrap_or_default(),
            top_k,
            candidate_pool,
            profile: name,
            factor_weights,
            composition,
            expand,
            experimental,
            clamped,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithms::{Algorithm, Fitted, Registry};

    const CEIL: Ceilings = Ceilings {
        top_k: 10,
        candidate_pool: 60,
    };
    const W: FactorWeights = FactorWeights {
        relevance_floor: 0.3,
        authority: 0.5,
        structural: 0.25,
        temporal: 0.0,
        completeness: 0.0,
        topical: 0.25,
    };

    fn reg() -> Registry {
        Registry::builtin(W)
    }

    /// A registry with a second, unfitted algorithm · what the per-query-type
    /// work of `SCORING.md` §4 looks like before it is fitted.
    fn reg_plus() -> Registry {
        let mut r = Registry::builtin(W);
        r.algorithms.insert(
            "sanction".to_owned(),
            Algorithm {
                note: None,
                description: "Penalty and prohibition questions.".to_owned(),
                fitted: None,
                factors: FactorWeights {
                    structural: 0.5,
                    ..W
                },
                composition: None,
                expand: vec![Expansion::Siblings],
            },
        );
        r
    }

    fn resolve(o: &SearchOptions) -> AppliedOptions {
        o.resolve(CEIL, &reg()).expect("resolves")
    }

    #[test]
    fn passing_nothing_yields_the_measured_defaults() {
        let a = resolve(&SearchOptions::default());
        assert_eq!(a.mode, Mode::Hybrid);
        assert_eq!(a.top_k, 10);
        assert_eq!(a.candidate_pool, 60);
        assert_eq!(a.profile, crate::algorithms::DEFAULT_ALGORITHM);
        assert_eq!(a.factor_weights, W);
        assert!(!a.experimental);
        assert!(a.clamped.is_empty());
    }

    #[test]
    fn a_caller_may_narrow_a_limit() {
        let a = resolve(&SearchOptions {
            top_k: Some(3),
            ..SearchOptions::default()
        });
        assert_eq!(a.top_k, 3);
        assert!(a.clamped.is_empty(), "narrowing is not clamping");
    }

    #[test]
    fn a_caller_may_not_widen_one() {
        let a = resolve(&SearchOptions {
            top_k: Some(500),
            candidate_pool: Some(100_000),
            ..SearchOptions::default()
        });
        assert_eq!(a.top_k, 10);
        assert_eq!(a.candidate_pool, 60);
        assert_eq!(a.clamped.len(), 2, "{:?}", a.clamped);
    }

    #[test]
    fn every_clamp_is_reported_not_silent() {
        // A caller asking for 500 and receiving 10 must see why without
        // reading the server's configuration.
        let a = resolve(&SearchOptions {
            top_k: Some(500),
            ..SearchOptions::default()
        });
        assert!(a.clamped[0].contains("500"), "{:?}", a.clamped);
        assert!(a.clamped[0].contains("10"), "{:?}", a.clamped);
    }

    #[test]
    fn a_pool_smaller_than_the_answer_is_raised_to_it() {
        let a = resolve(&SearchOptions {
            top_k: Some(10),
            candidate_pool: Some(2),
            ..SearchOptions::default()
        });
        assert_eq!(a.candidate_pool, 10);
        assert!(!a.clamped.is_empty());
    }

    #[test]
    fn zero_results_is_raised_rather_than_returning_nothing() {
        let a = resolve(&SearchOptions {
            top_k: Some(0),
            ..SearchOptions::default()
        });
        assert_eq!(a.top_k, 1);
    }

    #[test]
    fn raw_weights_are_marked_experimental_and_echoed() {
        let mine = FactorWeights {
            authority: 1.0,
            ..W
        };
        let a = resolve(&SearchOptions {
            factor_weights: Some(mine),
            ..SearchOptions::default()
        });
        assert!(a.experimental);
        assert_eq!(a.factor_weights, mine, "the effective weights must be echoed");
    }

    #[test]
    fn a_named_algorithm_supplies_its_own_weights() {
        let a = SearchOptions {
            profile: Some("sanction".to_owned()),
            ..SearchOptions::default()
        }
        .resolve(CEIL, &reg_plus())
        .expect("resolves");
        assert_eq!(a.profile, "sanction");
        assert!((a.factor_weights.structural - 0.5).abs() < 1e-6);
    }

    #[test]
    fn an_unfitted_algorithm_is_experimental_even_without_raw_weights() {
        // ! Same claim to the caller as raw weights -- "nobody measured this
        // ranking" -- so it raises the same flag rather than a second one.
        let a = SearchOptions {
            profile: Some("sanction".to_owned()),
            ..SearchOptions::default()
        }
        .resolve(CEIL, &reg_plus())
        .expect("resolves");
        assert!(a.experimental);
    }

    #[test]
    fn an_algorithms_own_expansions_survive_a_caller_passing_none() {
        // An algorithm fitted WITH siblings admitted is a different measurement
        // from the same weights without them.
        let a = SearchOptions {
            profile: Some("sanction".to_owned()),
            ..SearchOptions::default()
        }
        .resolve(CEIL, &reg_plus())
        .expect("resolves");
        assert_eq!(a.expand, vec![Expansion::Siblings]);
    }

    #[test]
    fn asking_for_an_expansion_twice_costs_what_asking_once_costs() {
        // ! `dedup` alone only removes ADJACENT duplicates. The algorithm
        // contributes `siblings` and so does the caller, and they are not
        // adjacent once more expansions exist.
        let a = SearchOptions {
            profile: Some("sanction".to_owned()),
            expand: Some(vec![Expansion::Siblings, Expansion::Siblings]),
            ..SearchOptions::default()
        }
        .resolve(CEIL, &reg_plus())
        .expect("resolves");
        assert_eq!(a.expand, vec![Expansion::Siblings]);
    }

    #[test]
    fn an_unknown_algorithm_is_refused_rather_than_silently_defaulted() {
        // ! The failure this prevents: a caller misspells `sanction`, is served
        // `balanced`, and reports a ranking it cannot reproduce.
        let err = SearchOptions {
            profile: Some("sanctions".to_owned()),
            ..SearchOptions::default()
        }
        .resolve(CEIL, &reg_plus())
        .expect_err("a misspelled profile must not fall back");
        match err {
            ResolveError::UnknownProfile { ref available, .. } => {
                assert!(available.contains(&"sanction".to_owned()), "{available:?}");
            }
        }
        // The message alone has to be enough to correct the call.
        let msg = err.to_string();
        assert!(msg.contains("sanctions"), "{msg}");
        assert!(msg.contains("balanced"), "{msg}");
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
            profile: Some("balanced".to_owned()),
            ..SearchOptions::default()
        };
        let s = serde_json::to_string(&o).expect("serialize");
        assert_eq!(serde_json::from_str::<SearchOptions>(&s).expect("parse"), o);
    }

    #[test]
    fn a_fitted_block_is_what_makes_an_algorithm_offerable() {
        let mut r = reg_plus();
        assert_eq!(r.offered(), vec!["balanced"]);
        r.algorithms.get_mut("sanction").expect("sanction").fitted = Some(Fitted {
            recall_at_5: 0.60,
            recall_at_10: None,
            mrr: None,
            n: 44,
            harness: "dev_tools/eval/e2e.py".to_owned(),
            note: None,
        });
        assert_eq!(r.offered(), vec!["balanced", "sanction"]);
    }
}
