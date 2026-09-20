//! Loading the corpus's own scoring vocabulary · the wire shape of
//! [`engine::Vocabulary`].
//!
//! ! Two types on purpose, exactly as `engine::Weights` and
//! `contract::FactorWeights` are two types. `engine` carries **zero
//! dependencies** — no serde, no I/O — because that is what makes the scoring
//! logic testable without a database and replicable without this binary. So
//! the deserialisable mirror lives here and is mapped across.
//!
//! # Why there is a file at all
//!
//! `authority`, `structural` and the term splitter used to compile in three
//! Indonesian tables. Vera is not a legal engine — Indonesian regulation is the
//! corpus it was built against first, the same way `Qwen3-Embedding` is the
//! model it was built against first. Invariant 2 already answers that case for
//! the vector space: the corpus declares, the engine matches, and a canary
//! refuses to serve if they disagree. Scoring vocabulary is the same claim with
//! no answer, and this is the answer.
//!
//! ! `VOCABULARY_PATH` is a **staging post, ✗ the destination.** The right home
//! is `corpus_meta`, beside the model and pooling the corpus already declares,
//! written by Ravel from the profile that already contains the table
//! (`registry/profiles/id_regulation@1.0.yaml`, key `identity.authority`).
//! That needs a schema change in a repository Vera does not own, so the file
//! exists to make the capability real today. It reads the same data either way.

use serde::Deserialize;

/// The JSON shape · mirrors [`engine::Vocabulary`].
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VocabularyFile {
    /// Free-text notes. JSON has no comments and this repository annotates
    /// everything; a config file that cannot say why it holds what it holds is
    /// a config file whose values get copied without their reasons.
    // ! Deserialised and never read, on purpose. Their job is to make the key
    // LEGAL under `deny_unknown_fields`: without them a file that explains
    // itself fails to load, and a config file nobody may annotate is one whose
    // values get copied into the next corpus without their reasons.
    #[allow(dead_code)]
    #[serde(rename = "_comment", default)]
    pub comment: Option<serde_json::Value>,
    /// A note attached to one section rather than to the file.
    #[allow(dead_code)]
    #[serde(rename = "_note", default)]
    pub note: Option<serde_json::Value>,
    /// Label → tier. Higher binds harder.
    #[serde(default)]
    pub authority: std::collections::BTreeMap<String, u8>,
    /// What `authority` normalises against · the top of the **declared** scale,
    /// ✗ the top present in this corpus.
    #[serde(default = "default_max_tier")]
    pub max_tier: f32,
    /// Ordered, first match wins.
    #[serde(default)]
    pub structural: Vec<StructuralRuleFile>,
    #[serde(default = "default_structural")]
    pub structural_default: f32,
    #[serde(default)]
    pub stopwords: Vec<String>,
    #[serde(default = "default_min_term")]
    pub min_term_chars: usize,
}

const fn default_max_tier() -> f32 {
    10.0
}
const fn default_structural() -> f32 {
    0.5
}
const fn default_min_term() -> usize {
    3
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuralRuleFile {
    /// Why this rule exists · carried per rule, because a ladder's ordering is
    /// the part a later reader is most likely to 'tidy' without knowing what it
    /// was measured against.
    #[allow(dead_code)]
    #[serde(rename = "_note", default)]
    pub note: Option<serde_json::Value>,
    pub label: String,
    /// `article` · `chapter` · `either`.
    #[serde(default = "default_field")]
    pub field: String,
    /// `true` matches with `starts_with`, `false` with `contains`.
    #[serde(default)]
    pub prefix: bool,
    pub score: f32,
}

fn default_field() -> String {
    "either".to_owned()
}

/// Why a vocabulary file was refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VocabError {
    Malformed(String),
    /// A `field` that is not one of the three the engine understands.
    ///
    /// ! Refused rather than defaulted. A rule silently widened from `article`
    /// to `either` matches chunks its author never intended, and the ranking
    /// change is invisible.
    BadField { label: String, field: String },
    /// A score outside `0.0..=1.0`, or a non-finite `max_tier`.
    BadScore { label: String, why: String },
}

impl std::fmt::Display for VocabError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Malformed(m) => write!(f, "did not parse · {m}"),
            Self::BadField { label, field } => write!(
                f,
                "structural rule `{label}` has field `{field}` · must be one of \
                 article, chapter, either"
            ),
            Self::BadScore { label, why } => write!(f, "`{label}` · {why}"),
        }
    }
}

impl std::error::Error for VocabError {}

impl VocabularyFile {
    /// Parse and validate, then map onto the engine's type.
    ///
    /// # Errors
    /// [`VocabError`] naming the rule that is wrong.
    pub fn load(json: &str) -> Result<engine::Vocabulary, VocabError> {
        let me: Self =
            serde_json::from_str(json).map_err(|e| VocabError::Malformed(e.to_string()))?;

        if !me.max_tier.is_finite() || me.max_tier <= 0.0 {
            return Err(VocabError::BadScore {
                label: "max_tier".to_owned(),
                why: format!("{} is not a positive finite number", me.max_tier),
            });
        }
        if !(0.0..=1.0).contains(&me.structural_default) {
            return Err(VocabError::BadScore {
                label: "structural_default".to_owned(),
                why: format!("{} is outside 0.0..=1.0", me.structural_default),
            });
        }

        let mut structural = Vec::with_capacity(me.structural.len());
        for r in &me.structural {
            let field = match r.field.as_str() {
                "article" => engine::Field::Article,
                "chapter" => engine::Field::Chapter,
                "either" => engine::Field::Either,
                other => {
                    return Err(VocabError::BadField {
                        label: r.label.clone(),
                        field: other.to_owned(),
                    });
                }
            };
            if !r.score.is_finite() || !(0.0..=1.0).contains(&r.score) {
                return Err(VocabError::BadScore {
                    label: r.label.clone(),
                    why: format!("score {} is outside 0.0..=1.0", r.score),
                });
            }
            structural.push(engine::StructuralRule {
                // Uppercased once here, ✗ per candidate on every request.
                label: r.label.trim().to_uppercase(),
                field,
                prefix: r.prefix,
                score: r.score,
            });
        }

        Ok(engine::Vocabulary {
            authority: me
                .authority
                .iter()
                .map(|(k, v)| (k.trim().to_uppercase(), *v))
                .collect(),
            max_tier: me.max_tier,
            structural,
            structural_default: me.structural_default,
            // ! Lowercased. `content_terms` lowercases each token before
            // comparing, so an uppercase entry here would never match and would
            // fail silently -- the whole class of bug this module exists for.
            stopwords: me.stopwords.iter().map(|s| s.trim().to_lowercase()).collect(),
            min_term_chars: me.min_term_chars,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_corpus_that_is_not_indonesian_can_declare_its_own_ladder() {
        // ! The point of the whole module. An RFC corpus has a source hierarchy
        // and a body/appendix split, and neither is spelled in Indonesian.
        let v = VocabularyFile::load(
            r#"{
              "authority": {"INTERNET STANDARD": 5, "PROPOSED STANDARD": 3,
                            "INFORMATIONAL": 1},
              "max_tier": 5,
              "structural": [
                {"label": "APPENDIX", "field": "either", "score": 0.0},
                {"label": "SECTION", "field": "article", "prefix": true, "score": 1.0}
              ],
              "structural_default": 0.5,
              "stopwords": ["the", "and", "for"],
              "min_term_chars": 2
            }"#,
        )
        .expect("valid");

        assert_eq!(v.tier("Internet Standard"), Some(5));
        assert_eq!(v.tier("PROPOSED STANDARD"), Some(3));
        assert_eq!(v.tier("UNDANG-UNDANG"), None, "no Indonesian leaks in");
        assert!(!v.is_builtin());

        let appendix = engine::Facets {
            article: Some("Appendix B"),
            ..engine::Facets::default()
        };
        let section = engine::Facets {
            article: Some("Section 4.2"),
            ..engine::Facets::default()
        };
        assert!(engine::factors::structural(&section, &v) > engine::factors::structural(&appendix, &v));
    }

    #[test]
    fn authority_normalises_against_the_declared_scale_not_this_corpus() {
        // max_tier 5 with a top tier of 5 means the most binding instrument
        // scores 1.0 -- the same as UU does against the Indonesian 10.
        let v = VocabularyFile::load(
            r#"{"authority": {"INTERNET STANDARD": 5}, "max_tier": 5}"#,
        )
        .expect("valid");
        let f = engine::Facets {
            regulation_type: Some("INTERNET STANDARD"),
            ..engine::Facets::default()
        };
        assert!((engine::factors::authority(&f, &v) - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn an_unrecognised_field_is_refused_rather_than_widened() {
        let err = VocabularyFile::load(
            r#"{"structural": [{"label": "X", "field": "heading", "score": 0.5}]}"#,
        )
        .expect_err("must not load");
        assert!(matches!(err, VocabError::BadField { .. }), "{err:?}");
        assert!(err.to_string().contains("heading"), "{err}");
    }

    #[test]
    fn an_unknown_key_kills_the_load() {
        let err = VocabularyFile::load(r#"{"authorty": {"X": 1}}"#).expect_err("typo");
        assert!(matches!(err, VocabError::Malformed(_)));
    }

    #[test]
    fn a_score_outside_its_own_units_is_refused() {
        let err = VocabularyFile::load(
            r#"{"structural": [{"label": "X", "field": "either", "score": 8.0}]}"#,
        )
        .expect_err("8.0 is not a factor score");
        assert!(matches!(err, VocabError::BadScore { .. }));
    }

    #[test]
    fn stopwords_are_lowercased_because_the_splitter_compares_lowercase() {
        // ! An uppercase entry would never match and would fail SILENTLY --
        // the exact failure mode this module exists to remove.
        let v = VocabularyFile::load(r#"{"stopwords": ["THE", "And"]}"#).expect("valid");
        assert_eq!(v.stopwords, vec!["the".to_owned(), "and".to_owned()]);
    }

    #[test]
    fn an_empty_file_is_a_vocabulary_that_scores_nothing() {
        // ! Legal, and it must be, because a corpus may genuinely have no
        // hierarchy. It is also exactly the state `missing_for` exists to
        // report: every factor is inert and no weight can act.
        let v = VocabularyFile::load("{}").expect("valid");
        assert!(v.authority.is_empty() && v.structural.is_empty());
        let w = engine::Weights::FITTED;
        assert_eq!(v.missing_for(&w), vec!["authority", "structural"]);
    }

    #[test]
    fn the_builtin_vocabulary_reports_itself_as_builtin() {
        assert!(engine::Vocabulary::id_regulation().is_builtin());
    }
}
