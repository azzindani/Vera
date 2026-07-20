//! The `search_knowledge` response contract · `OUTPUT_CONTRACT.md` §2.
//!
//! ! Vera returns **evidence**, the agent writes the prose. Nothing here holds a
//! summary field, and nothing here may ever hold one: summarizing needs an LLM,
//! and putting one in the engine breaks statelessness and adds a model
//! dependency the design exists to avoid (`LOOPHOLES.md` §2).

use serde::{Deserialize, Serialize};

/// How much the engine trusts this result set · `OUTPUT_CONTRACT.md` §4.
///
/// `None` is not "an error" — it is the honest answer when no domain anchor
/// matched. A confidently wrong domain is worse than "nothing matched".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Confidence {
    High,
    Medium,
    Low,
    None,
}

/// Where a result came from, precisely enough for a human to check it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Source {
    pub title: String,
    /// ! The **original** source a human clicks · ✗ an internal path.
    pub url: String,
    pub locator: Locator,
}

/// The most precise address available · clause > section > page.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct Locator {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub page: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub section: Option<String>,
}

impl Locator {
    /// Human-readable tail of a citation line, e.g. `Pasal 9 ayat (3), p.14`.
    /// Empty when ingest recorded neither — ✗ a fabricated placeholder.
    #[must_use]
    pub fn render(&self) -> String {
        match (&self.section, self.page) {
            (Some(s), Some(p)) => format!("{s}, p.{p}"),
            (Some(s), None) => s.clone(),
            (None, Some(p)) => format!("p.{p}"),
            (None, None) => String::new(),
        }
    }
}

/// Per-modality scores, published so the agent can see *why* something ranked.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ComponentScores {
    pub dense: f32,
    pub bm25: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SearchResult {
    pub id: String,
    /// Bounded preview · ✗ the full body. Full text comes from `read_chunk`.
    pub snippet: String,
    /// Fused RRF score.
    pub score: f32,
    pub scores: ComponentScores,
    pub source: Source,
}

/// A hit from the global exact-identifier path that bypasses routing.
///
/// Reported separately so the agent can tell "this exact regulation exists"
/// from "semantic routing surfaced this".
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExactMatch {
    pub id: String,
    pub matched_on: String,
}

/// A ready-to-render citation line, numbered from 1.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Citation {
    pub index: usize,
    pub text: String,
}

/// Compact, de-duplicated material the agent uses to write its summary.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct SummaryPayload {
    pub snippets: Vec<String>,
    pub sources: Vec<String>,
    pub coverage: String,
}

/// The full `search_knowledge` response.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SearchResponse {
    pub success: bool,
    pub op: &'static str,
    pub query: String,
    /// `None` when no anchor matched above threshold · ✗ a guessed domain.
    pub detected_domain: Option<String>,
    pub domain_confidence: f32,
    pub clusters_probed: usize,
    pub results: Vec<SearchResult>,
    pub citation_block: Vec<Citation>,
    pub summary_payload: SummaryPayload,
    pub exact_matches: Vec<ExactMatch>,
    pub confidence: Confidence,
    pub progress: Vec<String>,
    pub token_estimate: usize,
    pub truncated: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hint: Option<String>,
}

impl SearchResponse {
    /// The honest empty answer: no domain anchor matched above threshold.
    ///
    /// ! `success: true` — the engine worked correctly and found nothing. This
    /// is deliberately distinct from an error (`OUTPUT_CONTRACT.md` §4).
    #[must_use]
    pub fn no_matching_domain(query: impl Into<String>, progress: Vec<String>) -> Self {
        let mut out = Self {
            success: true,
            op: "search_knowledge",
            query: query.into(),
            detected_domain: None,
            domain_confidence: 0.0,
            clusters_probed: 0,
            results: Vec::new(),
            citation_block: Vec::new(),
            summary_payload: SummaryPayload::default(),
            exact_matches: Vec::new(),
            confidence: Confidence::None,
            progress,
            token_estimate: 0,
            truncated: false,
            hint: Some(
                "query matched no known knowledge base · widen the query, or check \
                 list_domains for what this engine covers"
                    .into(),
            ),
        };
        out.token_estimate = out.estimate_tokens();
        out
    }

    /// `len(str(response)) / 4` · the agent budgets its own context with this.
    #[must_use]
    pub fn estimate_tokens(&self) -> usize {
        serde_json::to_string(self).map_or(0, |s| s.len() / 4)
    }
}

/// Build the numbered citation block from ranked results.
///
/// One entry per result, in rank order, so `[1]` in `summary_payload.sources`
/// always addresses `results[0]`.
#[must_use]
pub fn citation_block(results: &[SearchResult]) -> Vec<Citation> {
    results
        .iter()
        .enumerate()
        .map(|(i, r)| {
            let locator = r.source.locator.render();
            let middle = if locator.is_empty() {
                String::new()
            } else {
                format!(", {locator}")
            };
            Citation {
                index: i + 1,
                text: format!(
                    "[{}] {}{} — {}",
                    i + 1,
                    r.source.title,
                    middle,
                    r.source.url
                ),
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn result(title: &str, page: Option<i32>, section: Option<&str>) -> SearchResult {
        SearchResult {
            id: "c1".into(),
            snippet: "Wajib Pajak…".into(),
            score: 0.871,
            scores: ComponentScores {
                dense: 0.83,
                bm25: 0.61,
            },
            source: Source {
                title: title.into(),
                url: "https://peraturan.example/uu-28-2007.pdf".into(),
                locator: Locator {
                    page,
                    section: section.map(Into::into),
                },
            },
        }
    }

    #[test]
    fn locator_prefers_the_most_precise_address_available() {
        assert_eq!(
            Locator {
                page: Some(14),
                section: Some("Pasal 9 ayat (3)".into())
            }
            .render(),
            "Pasal 9 ayat (3), p.14"
        );
        assert_eq!(
            Locator {
                page: Some(14),
                section: None
            }
            .render(),
            "p.14"
        );
        // ! Nothing recorded → empty, ✗ a placeholder like "p.?" that would
        // read as provenance a human could act on.
        assert_eq!(Locator::default().render(), "");
    }

    #[test]
    fn citations_are_numbered_from_one_in_rank_order() {
        let block = citation_block(&[
            result("UU No. 28 Tahun 2007", Some(14), Some("Pasal 9 ayat (3)")),
            result("PP No. 74 Tahun 2011", Some(3), None),
        ]);
        assert_eq!(block[0].index, 1);
        assert!(
            block[0]
                .text
                .starts_with("[1] UU No. 28 Tahun 2007, Pasal 9 ayat (3), p.14 — https://")
        );
        assert_eq!(block[1].index, 2);
        assert!(block[1].text.contains("[2] PP No. 74 Tahun 2011, p.3 — "));
    }

    #[test]
    fn a_citation_without_a_locator_still_renders_a_usable_line() {
        let block = citation_block(&[result("Untitled Source", None, None)]);
        assert_eq!(
            block[0].text,
            "[1] Untitled Source — https://peraturan.example/uu-28-2007.pdf"
        );
    }

    #[test]
    fn no_matching_domain_succeeds_with_nothing_rather_than_guessing() {
        let r = SearchResponse::no_matching_domain("ketentuan sanksi", vec!["embed".into()]);
        assert!(
            r.success,
            "engine worked correctly · it simply found nothing"
        );
        assert!(r.detected_domain.is_none());
        assert_eq!(r.confidence, Confidence::None);
        assert!(r.results.is_empty());
        assert!(r.hint.is_some(), "must tell the agent what to do next");
        assert!(r.token_estimate > 0);
    }

    #[test]
    fn the_response_carries_no_summary_field() {
        // ! Guards the core invariant: Vera returns evidence, the agent writes
        // prose. A `summary` on the wire would mean an LLM crept into the engine.
        let json = serde_json::to_string(&SearchResponse::no_matching_domain("q", vec![])).unwrap();
        for banned in ["\"summary\":", "\"answer\":", "\"completion\":"] {
            assert!(!json.contains(banned), "{banned} must never be serialized");
        }
    }
}
