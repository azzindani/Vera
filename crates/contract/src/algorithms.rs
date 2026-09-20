//! The algorithm registry · named scoring configurations, declared in a file
//! rather than compiled in.
//!
//! [`Profile`] used to be a Rust enum with one variant. Adding a second way to
//! rank meant editing `contract`, `tools.rs` and `main.rs` and shipping a new
//! binary — which is the same mistake as compiling in a model width (invariant
//! 2). A ranking strategy is a property of the **corpus**, fitted against its
//! eval set, so it belongs beside the corpus, not inside the engine.
//!
//! This module is the type and the validator. Reading the file is `mcp`'s job;
//! `contract` holds zero I/O by design.
//!
//! # The three rules, enforced at load
//!
//! 1. **`1 + Σw` may not exceed [`MAX_PRIOR_BOUND`].** This is invariant 9 as
//!    arithmetic: the fused pool's own relevance spans 2.56× median but only
//!    **1.31× at its narrowest** across the 40 eval cases, so a prior bounded
//!    above 2.00× can reorder a pool on metadata alone — authority without
//!    relevance, the failure the whole factor layer is built around. The same
//!    bound already guards the shipped weights in
//!    `engine::factors::tests::the_shipped_weights_stay_inside_the_pool_spread`;
//!    here it guards every algorithm anyone adds later.
//!
//! 2. **An algorithm with no `fitted` block is `experimental`.** It loads, it
//!    can be called by name, and it is **excluded from the tool schema's
//!    `enum`** — the agent is never *offered* a weight vector nobody measured.
//!    That is the existing `tools::tests::only_fitted_profiles_are_offered`
//!    rule, moved from a hardcoded list to a property of the data.
//!
//! 3. **An unknown key fails the load, and an unknown profile name fails the
//!    request.** Neither falls back to the default. A caller that misspells
//!    `sanction` and silently receives `balanced` gets a wrong answer wearing a
//!    correct one's clothes, and the ranking is then unreproducible.
//!
//! # What is deliberately NOT here
//!
//! No viewpoints, no consensus, no in-engine effort tiers. `docs/SCORING.md`
//! §5-§6 designed all three, borrowed from `06_ID_Legal`, and they are dropped
//! rather than built.
//!
//! ! **Consensus cannot be made to work under invariant 9.** The prior is
//! bounded at [`MAX_PRIOR_BOUND`] precisely so relevance dominates metadata.
//! Several weight vectors over one pool therefore produce near-identical
//! orderings -- they can only reorder candidates whose relevance is already
//! close. Either viewpoints disagree enough to be informative, which means the
//! prior out-spans the pool and invariant 9 is broken, or they agree trivially
//! and agreement carries no signal. The two requirements are the same dial
//! turned opposite ways.
//!
//! ! It is also **unfittable on this eval set**. `EVAL.md` §5 records that at
//! n=40 a legally-correct hierarchy ordering and an inverted one score
//! identically; an agreement statistic is a finer signal than the one the set
//! already cannot resolve. Under invariant 8 that makes it unshippable, ✗
//! merely unbuilt.
//!
//! What replaces all three is the **caller**. An agent that judges a ranking
//! unfit calls again naming a different algorithm, or raw weights. It sees the
//! results and the query intent, which is strictly more than rank agreement
//! between weight vectors can observe, and it is what CLAUDE.md §2 already
//! assigns it. Escalation is a second short request rather than one long one,
//! which also dissolves the conflict `SCORING.md` §6 recorded as unresolved:
//! a 30s tier-3 request would hold one of `MAX_CONCURRENCY`'s four permits for
//! half a minute, and a retry holds none.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::options::{Expansion, FactorWeights};

/// The largest `1 + Σw` any algorithm may declare.
///
/// ! Not a round number and not the median pool spread. 2.00× is the
/// constraint the shipped fit was run under (`dev_tools/eval/fit_factors.py`),
/// and the reason it is *below* the 2.56× median is that the median is not the
/// binding case: the narrowest pool measured spans 1.31×, and a bound chosen
/// against the median would out-span the pool in every case below it.
///
/// The unconstrained best fit — `authority=1.0 structural=0.25
/// completeness=1.0 topical=0.5`, bound 3.75× — out-spans the pool in 40 of 40
/// cases. In situ it has the best Recall@5 ever measured here (56.8%) and an
/// MRR *below* factors-off. Recall@5 on 40 cases cannot see that, which is
/// exactly why this is a load-time check and not a reviewer's judgement.
pub const MAX_PRIOR_BOUND: f32 = 2.00;

/// The algorithm every registry must define · the default when a caller names
/// none.
pub const DEFAULT_ALGORITHM: &str = "balanced";

/// What an algorithm scored, and on what.
///
/// ! Required for an algorithm to be *offered*. Invariant 15 says never claim a
/// number you did not measure; this is that rule given a place to live, so an
/// unmeasured algorithm is structurally distinguishable from a measured one
/// rather than distinguishable by whoever reads the file.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Fitted {
    pub recall_at_5: f32,
    #[serde(default)]
    pub recall_at_10: Option<f32>,
    #[serde(default)]
    pub mrr: Option<f32>,
    /// Labelled cases the figure came from. Small `n` is not disqualifying —
    /// hiding it is.
    pub n: u32,
    /// The command that produced it, e.g. `dev_tools/eval/e2e.py`.
    pub harness: String,
    /// Anything that qualifies the number: a corpus version, a known staleness.
    #[serde(default)]
    pub note: Option<String>,
}


/// How one assembled factor turns a stored value into a number in `0.0..=1.0`.
///
/// ! A closed set, ✗ an expression language, and the reason is
/// [`MAX_PRIOR_BOUND`]. `Σw` bounds the prior only if every term is at most 1;
/// given a formula as text you cannot compute that by inspection, and invariant
/// 9 stops being arithmetic. The grammar is constrained so the assembly can be
/// free.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum TransformSpec {
    /// The corpus's authority ladder · `tier / authority_scale`.
    Authority,
    /// The corpus's structural ladder · annex versus operative text.
    Structural,
    /// Recency decaying with age · `years / (years + age)`. Absolute, so two
    /// identical candidates score the same whatever else was retrieved.
    HalfLife { years: f32 },
    /// Position within the pool's own span of this variable · 0.5 when flat.
    Range,
    /// Saturating growth · `1 - e^(-x/at)`. Past `at`, more is not more.
    Saturate { at: f32 },
    /// Share of the query's content terms this text contains.
    MatchShare,
    /// 1.0 when the value is present and non-empty, otherwise absent.
    Present,
    /// 1.0 when the value is at least `min`, else 0.0.
    AtLeast { min: f32 },
}

/// One factor in an assembled scoring function.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FactorEntry {
    /// Published beside its contribution in the response. A factor whose
    /// contribution cannot be seen is one that cannot be debugged, and assembly
    /// makes that failure much easier to reach than five compiled functions did.
    pub name: String,
    /// Which stored value to read.
    ///
    /// A bare column — `regulation_type`, `article`, `chapter`, `about`, `body`,
    /// `body_len`, `year` — or `number:<key>` / `text:<key>` for an enrichment
    /// value the engine has never heard of. The second form is the point: a
    /// corpus that gains `citation_in_degree` gets a factor over it by adding
    /// four lines here, ✗ by shipping a binary.
    pub variable: String,
    pub transform: TransformSpec,
    pub weight: f32,
}

/// A named way to rank.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Algorithm {
    /// A note about this algorithm, ignored by the engine · distinct from
    /// `description`, which is shown to the agent.
    #[serde(rename = "_note", default, skip_serializing_if = "Option::is_none")]
    pub note: Option<serde_json::Value>,
    /// One line, shown to the agent by `list_algorithms`. It is how the caller
    /// chooses, so it describes *when to use this*, ✗ what the numbers are.
    pub description: String,
    /// Absent ⇒ experimental ⇒ not offered in the tool schema.
    #[serde(default)]
    pub fitted: Option<Fitted>,
    pub factors: FactorWeights,
    /// An assembled scoring function · **wins over `factors` when present**.
    ///
    /// ! `factors` is the five-term shorthand and stays the default, because
    /// every fitted weight in this repository was measured against it and
    /// `Composition::classic` reproduces it term for term. `composition` is the
    /// general form: any variable, any bounded shape, any number of terms.
    ///
    /// The combination rule is **not** configurable and is not meant to be.
    /// `relevance × (1 + Σ wᵢ·fᵢ)` is what encodes "factors rank, but only after
    /// relevance"; it is the one part of the formula with nothing to gain from
    /// being declared and everything to lose (invariant 9).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub composition: Option<Vec<FactorEntry>>,
    /// Expansions this algorithm turns on by default. Still reported per
    /// result via `expanded_from`; an algorithm cannot make an expanded chunk
    /// look retrieved.
    #[serde(default)]
    pub expand: Vec<Expansion>,
}

impl Algorithm {
    /// `1 + Σw` · the most a candidate's metadata can multiply its relevance by.
    ///
    /// ! Reads whichever form this algorithm actually uses. A bound computed
    /// from `factors` while the engine ranks on `composition` is a check on
    /// something that does not run.
    #[must_use]
    pub fn prior_bound(&self) -> f32 {
        1.0 + self.composition.as_ref().map_or_else(
            || weight_sum(&self.factors),
            |c| c.iter().map(|f| f.weight).sum(),
        )
    }

    /// Offered to the agent in the tool schema?
    #[must_use]
    pub const fn is_fitted(&self) -> bool {
        self.fitted.is_some()
    }
}

/// The five-term shorthand, written out as an assembled composition.
///
/// ! The bridge that keeps one code path. `factors` is what every fitted weight
/// in this repository was measured against, and the engine now ranks on a
/// composition — so the shorthand has to *become* one rather than being scored
/// by a second, parallel implementation that could drift from it.
/// `engine::Composition::classic` is asserted to match the compiled scorer term
/// for term; this is the same list on the wire side.
#[must_use]
pub fn classic_composition(w: &FactorWeights) -> Vec<FactorEntry> {
    vec![
        FactorEntry {
            name: "authority".to_owned(),
            variable: "regulation_type".to_owned(),
            transform: TransformSpec::Authority,
            weight: w.authority,
        },
        FactorEntry {
            name: "structural".to_owned(),
            variable: "article".to_owned(),
            transform: TransformSpec::Structural,
            weight: w.structural,
        },
        FactorEntry {
            name: "completeness".to_owned(),
            variable: "body_len".to_owned(),
            transform: TransformSpec::Saturate { at: 400.0 },
            weight: w.completeness,
        },
        FactorEntry {
            name: "temporal".to_owned(),
            variable: "year".to_owned(),
            transform: TransformSpec::Range,
            weight: w.temporal,
        },
        FactorEntry {
            name: "topical".to_owned(),
            variable: "about".to_owned(),
            transform: TransformSpec::MatchShare,
            weight: w.topical,
        },
    ]
}

/// `Σw`, excluding the floor · the floor is a threshold, ✗ a weight, and adding
/// it here would bound the prior by a number that never multiplies anything.
#[must_use]
pub fn weight_sum(w: &FactorWeights) -> f32 {
    w.authority + w.structural + w.temporal + w.completeness + w.topical
}

/// Every algorithm this engine can be asked for.
///
/// `BTreeMap`, ✗ `HashMap`: `list_algorithms` returns these to an agent, and an
/// introspection surface whose order changes between calls is one a client
/// learns to distrust.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Registry {
    /// The file's own notes, ignored by the engine.
    ///
    /// ! JSON has no comments and this repository annotates everything. A
    /// config file that cannot say *why* it holds a value is one whose values
    /// get copied into the next deployment without their reasons — and every
    /// number here has a reason that a reader needs.
    #[serde(rename = "_comment", default, skip_serializing_if = "Option::is_none")]
    pub comment: Option<serde_json::Value>,
    pub algorithms: BTreeMap<String, Algorithm>,
}

/// Why a registry was refused. Every variant names the algorithm and the
/// number, because a startup failure that does not say which line of the file
/// is wrong costs an operator the same hour every time.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RegistryError {
    /// The file did not parse. Carries serde's message, which already names the
    /// offending key.
    Malformed(String),
    /// No `balanced`. It is the default for every request that names nothing,
    /// so a registry without it cannot serve.
    MissingDefault,
    /// A name that cannot be a JSON-schema enum token.
    BadName { name: String, why: String },
    /// Invariant 9, as arithmetic.
    PriorTooWide {
        name: String,
        /// Hundredths, so the error stays `Eq` and prints exactly.
        bound_centi: u32,
    },
    /// A floor outside `0.0..=1.0`, or a weight that is negative or not finite.
    BadWeight {
        name: String,
        field: &'static str,
        why: String,
    },
}

impl std::fmt::Display for RegistryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Malformed(m) => write!(f, "algorithm registry did not parse · {m}"),
            Self::MissingDefault => write!(
                f,
                "algorithm registry defines no `{DEFAULT_ALGORITHM}` · it is the default for \
                 every request that names no profile, so the registry cannot serve without it"
            ),
            Self::BadName { name, why } => {
                write!(f, "algorithm name `{name}` is unusable · {why}")
            }
            Self::PriorTooWide { name, bound_centi } => write!(
                f,
                "algorithm `{name}` has a prior bound of {}.{:02}x, above the {MAX_PRIOR_BOUND:.2}x \
                 the fit was constrained to · a prior wider than the pool's own relevance spread \
                 can reorder the pool on metadata alone (docs/SCORING.md §3, invariant 9)",
                bound_centi / 100,
                bound_centi % 100
            ),
            Self::BadWeight { name, field, why } => {
                write!(f, "algorithm `{name}` field `{field}` · {why}")
            }
        }
    }
}

impl std::error::Error for RegistryError {}

impl Registry {
    /// The registry used when no file is configured · exactly today's
    /// behaviour, under the name every caller already gets by default.
    ///
    /// ! Takes the weights rather than owning them. The fitted numbers live in
    /// `engine::Weights::FITTED` and reach here through the environment
    /// (invariant 12), so `contract` never carries a second copy that can drift
    /// from the one the fitter produced.
    #[must_use]
    pub fn builtin(default: FactorWeights) -> Self {
        let mut algorithms = BTreeMap::new();
        algorithms.insert(
            DEFAULT_ALGORITHM.to_owned(),
            Algorithm {
                note: None,
                description: "The general case · the measured default.".to_owned(),
                fitted: Some(Fitted {
                    recall_at_5: 0.545,
                    recall_at_10: Some(0.591),
                    mrr: Some(0.454),
                    n: 44,
                    harness: "dev_tools/eval/e2e.py".to_owned(),
                    note: Some(
                        "Fitted before the 2026-09-19 re-embed, on a corpus where the dense arm \
                         contributed 0.0%. Needs refitting (docs/SCORING.md §9)."
                            .to_owned(),
                    ),
                }),
                factors: default,
                composition: None,
                expand: Vec::new(),
            },
        );
        Self {
            comment: None,
            algorithms,
        }
    }

    /// Parse and validate. Both, always — a `Registry` that exists has been
    /// checked, so nothing downstream has to wonder.
    ///
    /// # Errors
    /// [`RegistryError`] naming the algorithm and the offending number.
    pub fn parse(json: &str) -> Result<Self, RegistryError> {
        let me: Self =
            serde_json::from_str(json).map_err(|e| RegistryError::Malformed(e.to_string()))?;
        me.validate()?;
        Ok(me)
    }

    /// The rules in the module docs, applied to every algorithm and every
    /// viewpoint inside it.
    ///
    /// # Errors
    /// The first violation found, in a stable order — `BTreeMap` iteration is
    /// sorted, so the same bad file always names the same algorithm first.
    pub fn validate(&self) -> Result<(), RegistryError> {
        if !self.algorithms.contains_key(DEFAULT_ALGORITHM) {
            return Err(RegistryError::MissingDefault);
        }
        for (name, algo) in &self.algorithms {
            check_name(name)?;
            if let Some(entries) = &algo.composition {
                check_composition(name, algo.factors.relevance_floor, entries)?;
            } else {
                check_weights(name, &algo.factors)?;
            }
            // ! The bound is checked against whichever form runs, so this stays
            // one call rather than two branches that could disagree.
            if algo.prior_bound() > MAX_PRIOR_BOUND + f32::EPSILON {
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                return Err(RegistryError::PriorTooWide {
                    name: name.clone(),
                    bound_centi: (algo.prior_bound() * 100.0).round() as u32,
                });
            }

        }
        Ok(())
    }

    /// Look up by name.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&Algorithm> {
        self.algorithms.get(name)
    }

    /// Names offered in the tool schema · fitted only, sorted.
    ///
    /// ! An unfitted algorithm is reachable by name and absent from this list.
    /// Offering it would be advertising a weight vector with a friendly name.
    #[must_use]
    pub fn offered(&self) -> Vec<&str> {
        self.algorithms
            .iter()
            .filter(|(_, a)| a.is_fitted())
            .map(|(n, _)| n.as_str())
            .collect()
    }

    /// Every name, fitted or not · what an unknown-profile error lists so the
    /// caller can correct itself without reading the server's configuration.
    #[must_use]
    pub fn names(&self) -> Vec<&str> {
        self.algorithms.keys().map(String::as_str).collect()
    }
}

/// A name has to survive being a JSON-schema `enum` token and an MCP argument,
/// so it is restricted at load rather than discovered to be a problem later.
fn check_name(name: &str) -> Result<(), RegistryError> {
    if name.is_empty() {
        return Err(RegistryError::BadName {
            name: name.to_owned(),
            why: "empty".to_owned(),
        });
    }
    if !name
        .chars()
        .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
    {
        return Err(RegistryError::BadName {
            name: name.to_owned(),
            why: "only lowercase ascii, digits and `_` · it appears verbatim in the tool schema"
                .to_owned(),
        });
    }
    Ok(())
}

fn check_weights(name: &str, w: &FactorWeights) -> Result<(), RegistryError> {
    let bad = |field: &'static str, why: String| RegistryError::BadWeight {
        name: name.to_owned(),
        field,
        why,
    };
    for (field, v) in [
        ("authority", w.authority),
        ("structural", w.structural),
        ("temporal", w.temporal),
        ("completeness", w.completeness),
        ("topical", w.topical),
    ] {
        if !v.is_finite() {
            return Err(bad(field, format!("{v} is not a finite number")));
        }
        if v < 0.0 {
            // ! Negative weights are refused rather than clamped. A negative
            // weight inverts a factor -- "rank annexes above articles" -- and
            // that is a legible thing to want; it is just not something to
            // acquire by typing a minus sign into a config file and never
            // measuring it. Declare the inverted factor instead.
            return Err(bad(field, format!("{v} is negative · see the note in source")));
        }
    }
    if !w.relevance_floor.is_finite() || !(0.0..=1.0).contains(&w.relevance_floor) {
        return Err(bad(
            "relevance_floor",
            format!(
                "{} is outside 0.0..=1.0 · it is a share of the query's content terms",
                w.relevance_floor
            ),
        ));
    }
    Ok(())
}

// ! `check_bound` used to live here and is gone. The bound is now checked
// through `Algorithm::prior_bound`, which reads whichever form the algorithm
// actually uses -- a bound computed from `factors` while the engine ranks on
// `composition` is a check on something that does not run.

/// Known bare column names · anything else must carry a `number:`/`text:` prefix.
///
/// ! Checked at load. A misspelled `regulaton_type` would otherwise resolve to
/// nothing on every row, and a factor that silently contributes nothing is
/// indistinguishable from one measured and found not to help — which is the
/// exact bug this project shipped once with `topical` against NULL.
const COLUMNS: &[&str] = &[
    "regulation_type",
    "article",
    "chapter",
    "about",
    "body",
    "body_len",
    "year",
];

fn check_composition(
    name: &str,
    floor: f32,
    entries: &[FactorEntry],
) -> Result<(), RegistryError> {
    let bad = |field: &'static str, why: String| RegistryError::BadWeight {
        name: name.to_owned(),
        field,
        why,
    };
    if !floor.is_finite() || !(0.0..=1.0).contains(&floor) {
        return Err(bad(
            "relevance_floor",
            format!("{floor} is outside 0.0..=1.0 · it is a share of the query's content terms"),
        ));
    }
    if entries.is_empty() {
        return Err(bad(
            "composition",
            "declared but empty · omit it to use the five-term `factors` form".to_owned(),
        ));
    }
    let mut seen: Vec<&str> = Vec::new();
    for e in entries {
        if e.name.trim().is_empty() {
            return Err(bad("composition", "a factor with no name".to_owned()));
        }
        // ! Names are published per result as contributions, so a duplicate
        // makes the explanation ambiguous exactly where it is needed most.
        if seen.contains(&e.name.as_str()) {
            return Err(bad(
                "composition",
                format!("`{}` declared twice · contributions are reported by name", e.name),
            ));
        }
        seen.push(&e.name);

        if !e.weight.is_finite() || e.weight < 0.0 {
            return Err(bad(
                "composition",
                format!("`{}` has weight {} · must be finite and non-negative", e.name, e.weight),
            ));
        }
        let v = e.variable.trim();
        let known = COLUMNS.contains(&v)
            || v.strip_prefix("number:").is_some_and(|k| !k.is_empty())
            || v.strip_prefix("text:").is_some_and(|k| !k.is_empty());
        if !known {
            return Err(bad(
                "composition",
                format!(
                    "`{}` reads `{v}`, which is neither a known column ({}) nor prefixed \
                     `number:` / `text:` for an enrichment value",
                    e.name,
                    COLUMNS.join(", ")
                ),
            ));
        }
        // ! A shape parameter that cannot work. `saturate` at 0 divides by zero
        // and `half_life` at 0 makes every document infinitely old; both would
        // produce a plausible-looking number rather than an error.
        let shape = match &e.transform {
            TransformSpec::Saturate { at } => Some(("saturate.at", *at)),
            TransformSpec::HalfLife { years } => Some(("half_life.years", *years)),
            TransformSpec::AtLeast { min } => {
                if min.is_finite() {
                    None
                } else {
                    Some(("at_least.min", *min))
                }
            }
            _ => None,
        };
        if let Some((what, value)) = shape
            && (!value.is_finite() || value <= 0.0)
        {
            return Err(bad(
                "composition",
                format!("`{}` declares {what} = {value} · must be finite and positive", e.name),
            ));
        }
    }
    Ok(())
}


#[cfg(test)]
mod tests {
    use super::*;

    const W: FactorWeights = FactorWeights {
        relevance_floor: 0.3,
        authority: 0.5,
        structural: 0.25,
        temporal: 0.0,
        completeness: 0.0,
        topical: 0.25,
    };

    fn registry(body: &str) -> Result<Registry, RegistryError> {
        Registry::parse(body)
    }

    const MINIMAL: &str = r#"{"algorithms":{"balanced":{
        "description":"d",
        "factors":{"relevance_floor":0.3,"authority":0.5,"structural":0.25,
                   "temporal":0.0,"completeness":0.0,"topical":0.25}}}}"#;

    #[test]
    fn the_builtin_registry_is_todays_behaviour_under_todays_name() {
        let r = Registry::builtin(W);
        assert_eq!(r.names(), vec![DEFAULT_ALGORITHM]);
        assert_eq!(r.get(DEFAULT_ALGORITHM).expect("default").factors, W);
        r.validate().expect("the builtin registry must be valid");
    }

    #[test]
    fn a_minimal_file_parses_and_validates() {
        let r = registry(MINIMAL).expect("valid");
        assert_eq!(r.get("balanced").expect("balanced").factors, W);
    }

    #[test]
    fn a_registry_without_the_default_cannot_serve() {
        let body = MINIMAL.replace("balanced", "sanction");
        assert_eq!(registry(&body), Err(RegistryError::MissingDefault));
    }

    #[test]
    fn a_prior_wider_than_the_pool_fails_the_load_not_a_review() {
        // ! The unconstrained best fit, which has the best Recall@5 ever
        // measured here and an MRR below factors-off. It must not be loadable
        // by writing it into a file, because the metric that would catch it is
        // not the metric anyone looks at.
        let body = MINIMAL.replace(
            r#""authority":0.5,"structural":0.25,
                   "temporal":0.0,"completeness":0.0,"topical":0.25"#,
            r#""authority":1.0,"structural":0.25,
                   "temporal":0.0,"completeness":1.0,"topical":0.5"#,
        );
        let err = registry(&body).expect_err("3.75x must not load");
        assert!(
            matches!(err, RegistryError::PriorTooWide { bound_centi: 375, .. }),
            "{err:?}"
        );
        // The operator must be able to act on the message without the source.
        let msg = err.to_string();
        assert!(msg.contains("3.75x"), "{msg}");
        assert!(msg.contains("2.00x"), "{msg}");
    }

    #[test]
    fn the_shipped_weights_sit_exactly_on_the_bound() {
        // Σw = 1.0, bound 2.00x. If this ever fails, either the fit moved or
        // MAX_PRIOR_BOUND did, and both are things to notice.
        let a = Registry::builtin(W);
        let bound = a.get(DEFAULT_ALGORITHM).expect("default").prior_bound();
        assert!((bound - 2.00).abs() < 1e-6, "{bound}");
    }

    #[test]
    fn an_unfitted_algorithm_loads_but_is_not_offered() {
        // Rule 2: reachable by name, absent from the schema. The agent is never
        // offered a weight vector nobody measured.
        let body = MINIMAL.replace(
            r#""balanced":{
        "description":"d","#,
            r#""balanced":{
        "description":"d","fitted":{"recall_at_5":0.545,"n":44,"harness":"e2e.py"},"#,
        );
        let body = body.replace(
            r#""topical":0.25}}}}"#,
            r#""topical":0.25}},
            "guess":{"description":"unmeasured","factors":{
              "relevance_floor":0.3,"authority":0.2,"structural":0.0,
              "temporal":0.0,"completeness":0.0,"topical":0.0}}}}"#,
        );
        let r = registry(&body).expect("valid");
        assert_eq!(r.names(), vec!["balanced", "guess"]);
        assert_eq!(r.offered(), vec!["balanced"], "unfitted must not be offered");
        assert!(r.get("guess").is_some(), "still reachable by name");
    }

    #[test]
    fn a_name_that_cannot_be_a_schema_token_is_refused() {
        for bad in ["Balanced", "by authority", "auth-weighted", ""] {
            let body = MINIMAL.replace("balanced", bad);
            // MissingDefault also fires; either way it does not load.
            assert!(registry(&body).is_err(), "`{bad}` must not load");
        }
    }

    #[test]
    fn an_unknown_key_fails_the_load_rather_than_being_ignored() {
        // A typo in a config file that silently does nothing is the same class
        // of defect as `topical` being fitted against NULL for every candidate.
        let body = MINIMAL.replace(r#""description":"d""#, r#""description":"d","weights":{}"#);
        assert!(matches!(registry(&body), Err(RegistryError::Malformed(_))));
    }

    #[test]
    fn a_floor_outside_its_own_units_is_refused() {
        let body = MINIMAL.replace(r#""relevance_floor":0.3,"authority":0.5"#, r#""relevance_floor":30.0,"authority":0.5"#);
        let err = registry(&body).expect_err("30.0 is not a share");
        assert!(matches!(
            err,
            RegistryError::BadWeight {
                field: "relevance_floor",
                ..
            }
        ));
    }

    #[test]
    fn the_registry_round_trips() {
        let r = Registry::builtin(W);
        let s = serde_json::to_string(&r).expect("serialize");
        assert_eq!(Registry::parse(&s).expect("reparse"), r);
    }
}
