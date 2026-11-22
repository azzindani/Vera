//! Query-side BM25 vectorisation.
//!
//! ! Documents are vectorised offline by `dev_tools/pre_embed/sparse.py`. This
//! is only the query half — and the two halves must tokenise **identically**.
//! The vocabulary artifact records the tokeniser it was built with, and loading
//! refuses on mismatch, because a query tokenised differently than the corpus
//! maps terms onto the wrong indices and scores plausibly against the wrong
//! documents.
//!
//! The corpus we inherited had tfidf vectors whose vocabulary was never saved —
//! 20,000 opaque integers with no term map, unusable for scoring any query.
//! That is why the vocabulary travels with the corpus and is hashed into
//! `corpus_meta`.

use std::collections::HashMap;
use std::path::Path;

/// The tokeniser the Python side uses: `(?u)\b\w\w+\b`, lowercased.
///
/// Reimplemented rather than pulled in via a regex crate so the rule is
/// visible: runs of alphanumeric-or-underscore, two characters or longer.
#[must_use]
pub fn tokenize(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for ch in text.chars() {
        if ch.is_alphanumeric() || ch == '_' {
            cur.extend(ch.to_lowercase());
        } else if cur.chars().count() >= 2 {
            out.push(std::mem::take(&mut cur));
        } else {
            cur.clear();
        }
    }
    if cur.chars().count() >= 2 {
        out.push(cur);
    }
    out
}

/// Kana, CJK ideographs and half-width katakana: the ranges Ravel's `_CJK` names.
fn is_cjk(ch: char) -> bool {
    matches!(ch,
        '\u{3040}'..='\u{30ff}' | '\u{3400}'..='\u{4dbf}' | '\u{4e00}'..='\u{9fff}'
        | '\u{f900}'..='\u{faff}' | '\u{ff66}'..='\u{ff9f}')
}

/// `tokenize`, except that a run of CJK characters becomes its overlapping character
/// bigrams, and a lone CJK character is kept as itself. Ravel's `WORDS_CJK`.
///
/// ! Japanese and Chinese are written without spaces: under `tokenize` a whole clause is
/// one term no query repeats. Text with no CJK tokenises exactly as under `tokenize`.
#[must_use]
pub fn tokenize_cjk(text: &str) -> Vec<String> {
    fn flush(out: &mut Vec<String>, cur: &mut Vec<char>, cjk: bool) {
        if cjk {
            if cur.len() == 1 {
                out.push(cur.iter().collect());
            } else {
                out.extend(cur.windows(2).map(|w| w.iter().collect::<String>()));
            }
        } else if cur.len() >= 2 {
            out.push(cur.iter().collect());
        }
        cur.clear();
    }
    let mut out = Vec::new();
    let mut cur: Vec<char> = Vec::new();
    let mut cur_cjk = false;
    for ch in text.chars() {
        if ch.is_alphanumeric() || ch == '_' {
            let cjk = is_cjk(ch);
            if !cur.is_empty() && cjk != cur_cjk {
                flush(&mut out, &mut cur, cur_cjk);
            }
            cur_cjk = cjk;
            cur.extend(ch.to_lowercase());
        } else if !cur.is_empty() {
            flush(&mut out, &mut cur, cur_cjk);
        }
    }
    if !cur.is_empty() {
        flush(&mut out, &mut cur, cur_cjk);
    }
    out
}

#[derive(Debug, thiserror::Error)]
pub enum Bm25Error {
    #[error("reading vocabulary: {0}")]
    Io(#[from] std::io::Error),

    #[error("parsing vocabulary: {0}")]
    Parse(#[from] serde_json::Error),

    #[error(
        "vocabulary was built with tokeniser {artifact:?} but this engine uses \
         {engine:?} · query terms would map to the wrong indices · refusing"
    )]
    TokenizerDrift { artifact: String, engine: String },
}

/// The token patterns this implementation reproduces, by the name an artifact records.
/// Compared against the artifact on load: `WORDS` is `tokenize`, `WORDS_CJK` is
/// `tokenize_cjk` (Ravel's `enrich/sparse.py`).
const ENGINE_TOKEN_PATTERN: &str = r"(?u)\b\w\w+\b";
const ENGINE_TOKEN_PATTERN_CJK: &str = r"(?u)\b\w\w+\b|cjk-bigrams";

#[derive(Debug, serde::Deserialize)]
struct Artifact {
    dim: u32,
    token_pattern: String,
    vocab: HashMap<String, u32>,
    /// Per-term inverse document frequency, as fitted over the corpus.
    ///
    /// ! Kept, not discarded. It is what lets the engine weigh a query term by
    /// how much it actually distinguishes documents: matching "yang" says
    /// nothing, matching "retribusi" says a great deal.
    idf: HashMap<String, f32>,
}

/// A loaded vocabulary, sufficient to project a query into the sparse space.
#[derive(Debug, Clone)]
pub struct QueryVectorizer {
    vocab: HashMap<String, u32>,
    idf: HashMap<String, f32>,
    /// IDF charged to a term the corpus has never seen.
    ///
    /// An unknown word is the strongest evidence the corpus cannot answer the
    /// question, so it is charged the most any known term could be worth.
    unknown_idf: f32,
    dim: u32,
    /// The artifact was fitted with `tokenize_cjk`.
    cjk: bool,
}

impl QueryVectorizer {
    /// Load the artifact written alongside the corpus.
    ///
    /// # Errors
    /// Unreadable or unparseable file, or a tokeniser that does not match.
    pub fn load(path: &Path) -> Result<Self, Bm25Error> {
        let raw = std::fs::read_to_string(path)?;
        let art: Artifact = serde_json::from_str(&raw)?;
        let cjk = match art.token_pattern.as_str() {
            ENGINE_TOKEN_PATTERN => false,
            ENGINE_TOKEN_PATTERN_CJK => true,
            _ => {
                return Err(Bm25Error::TokenizerDrift {
                    artifact: art.token_pattern,
                    engine: format!("{ENGINE_TOKEN_PATTERN} or {ENGINE_TOKEN_PATTERN_CJK}"),
                });
            }
        };
        let unknown_idf = art.idf.values().copied().fold(0.0_f32, f32::max);
        Ok(Self {
            vocab: art.vocab,
            idf: art.idf,
            unknown_idf,
            dim: art.dim,
            cjk,
        })
    }

    /// `text` tokenised as the corpus was.
    #[must_use]
    pub fn tokens(&self, text: &str) -> Vec<String> {
        if self.cjk {
            tokenize_cjk(text)
        } else {
            tokenize(text)
        }
    }

    #[must_use]
    pub fn dim(&self) -> u32 {
        self.dim
    }

    /// How much of `query`, by IDF mass, the given texts actually account for.
    ///
    /// ! This is the lexical half of the domain gate, and it asks the corpus
    /// rather than guessing from the query alone. Word PRESENCE does not
    /// separate domains: this corpus's 20,000-term vocabulary covers ordinary
    /// Indonesian, so "keran" and "dapur" are both in it. What an out-of-domain
    /// question lacks is any single retrieved document where its distinctive
    /// terms appear TOGETHER.
    ///
    /// Returns 0.0 for an empty query, which the caller must treat as "no
    /// evidence" rather than "certainly out of domain".
    #[must_use]
    pub fn evidence(&self, query: &str, texts: &[String]) -> f32 {
        let mut terms: Vec<String> = self.tokens(query);
        terms.sort_unstable();
        terms.dedup();
        if terms.is_empty() {
            return 0.0;
        }

        let weight = |t: &String| self.idf.get(t).copied().unwrap_or(self.unknown_idf);
        let total: f32 = terms.iter().map(weight).sum();
        if total <= 0.0 {
            return 0.0;
        }

        let mut pooled: Vec<String> = texts.iter().flat_map(|t| self.tokens(t)).collect();
        pooled.sort_unstable();
        pooled.dedup();

        let matched: f32 = terms
            .iter()
            .filter(|t| pooled.binary_search(t).is_ok())
            .map(weight)
            .sum();
        matched / total
    }

    /// The query with its low-information terms removed · what the `text` arm is sent.
    ///
    /// Keeps a term when its corpus IDF is at least `min_idf`, or when the corpus has
    /// never seen it (an unknown word is charged the maximum IDF, as in `evidence`).
    /// `min_idf <= 0.0`, or a query that would be left empty, returns the query
    /// unchanged -- pruning must never turn a question into no question.
    ///
    /// ! Why it exists: the `text` arm ORs every term, so a function word matches
    /// most of the corpus and the arm's cost is the MATCH COUNT, ✗ the ranking. At
    /// 5.0M rows it took 6.2 s; keeping IDF >= 2 took 1.4 s with no hit-rate loss on
    /// the 44 labelled queries (`docs/operations/HARDWARE.md` §8.3). Keeping only the 3-4
    /// rarest terms lost half the hits, so this is a floor on IDF, ✗ a term count.
    #[must_use]
    pub fn prune(&self, text: &str, min_idf: f32) -> String {
        if min_idf <= 0.0 {
            return text.to_owned();
        }
        let mut seen: Vec<String> = Vec::new();
        for t in self.tokens(text) {
            if self.idf.get(&t).is_none_or(|v| *v >= min_idf) && !seen.contains(&t) {
                seen.push(t);
            }
        }
        if seen.is_empty() {
            text.to_owned()
        } else {
            seen.join(" ")
        }
    }

    /// Project a query into `(index, weight)` pairs.
    ///
    /// Presence-weighted, so the dot product against a BM25-weighted document
    /// vector *is* the BM25 score. Out-of-vocabulary terms are dropped — they
    /// contribute nothing to any document by definition.
    #[must_use]
    pub fn query(&self, text: &str) -> Vec<(u32, f32)> {
        let mut seen: Vec<(u32, f32)> = Vec::new();
        let mut indices: Vec<u32> = self
            .tokens(text)
            .iter()
            .filter_map(|t| self.vocab.get(t).copied())
            .collect();
        indices.sort_unstable();
        indices.dedup();
        for i in indices {
            seen.push((i, 1.0));
        }
        seen
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenisation_lowercases_and_drops_single_characters() {
        // "a" is one char and must be dropped, matching \w\w+.
        assert_eq!(tokenize("Sanksi a Denda"), ["sanksi", "denda"]);
    }

    #[test]
    fn punctuation_and_digits_split_the_way_the_python_side_splits_them() {
        // "UU 28/2007" -> uu, 28, 2007 · the slash is a separator, and each
        // number survives as its own token.
        assert_eq!(tokenize("UU 28/2007"), ["uu", "28", "2007"]);
        assert_eq!(tokenize("Pasal 9 ayat (3)"), ["pasal", "ayat"]);
    }

    #[test]
    fn indonesian_text_tokenises_without_loss() {
        assert_eq!(
            tokenize("Wajib Pajak yang terlambat"),
            ["wajib", "pajak", "yang", "terlambat"]
        );
    }

    #[test]
    fn empty_and_symbol_only_input_yield_no_tokens() {
        assert!(tokenize("").is_empty());
        assert!(tokenize("--- ... ///").is_empty());
    }

    fn vz() -> QueryVectorizer {
        let mut vocab = HashMap::new();
        vocab.insert("sanksi".to_owned(), 5);
        vocab.insert("denda".to_owned(), 2);
        // "yang" is the shape of a term that distinguishes nothing; the two
        // content words carry real weight.
        vocab.insert("yang".to_owned(), 9);
        let mut idf = HashMap::new();
        idf.insert("sanksi".to_owned(), 6.0);
        idf.insert("denda".to_owned(), 6.0);
        idf.insert("yang".to_owned(), 0.5);
        QueryVectorizer {
            vocab,
            idf,
            unknown_idf: 9.0,
            dim: 20000,
            cjk: false,
        }
    }

    #[test]
    fn cjk_runs_become_bigrams_as_ravel_tokenises_them() {
        // ! Mirrors Ravel's `test_cjk_runs_become_bigrams...`: the two halves must agree.
        assert_eq!(
            tokenize_cjk("博物館法第二条"),
            ["博物", "物館", "館法", "法第", "第二", "二条"]
        );
        assert_eq!(tokenize_cjk("法"), ["法"]);
        assert_eq!(
            tokenize_cjk("博物館法第二条 の定義 BGB § 7 Wohnsitz"),
            [
                "博物", "物館", "館法", "法第", "第二", "二条", "の定", "定義", "bgb", "wohnsitz"
            ]
        );
        for text in ["Wajib Pajak a 28/2007", "C-46, s. 718.2 Purpose of Act"] {
            assert_eq!(tokenize_cjk(text), tokenize(text));
        }
    }

    #[test]
    fn evidence_is_the_share_of_a_query_the_documents_account_for() {
        let v = vz();
        let docs = vec!["ketentuan sanksi dan denda yang berlaku".to_owned()];
        assert!(
            (v.evidence("sanksi denda", &docs) - 1.0).abs() < 1e-6,
            "every term found"
        );
        assert!(
            v.evidence("sanksi denda", &["tidak ada apa pun".to_owned()])
                .abs()
                < 1e-6,
            "no term found"
        );
    }

    #[test]
    fn an_unknown_word_counts_against_the_query_more_than_a_common_one() {
        let v = vz();
        let docs = vec!["sanksi denda yang berlaku".to_owned()];
        // "yang" missing from the documents barely matters; a word the corpus
        // has never seen is the strongest evidence it cannot answer.
        let common_missing = v.evidence("sanksi denda yang", &docs);
        let unknown_missing = v.evidence("sanksi denda kwacamole", &docs);
        assert!(
            unknown_missing < common_missing,
            "unknown {unknown_missing} should hurt more than common {common_missing}"
        );
    }

    #[test]
    fn evidence_pools_across_documents() {
        // A question answered across two clauses must not be punished for it.
        let v = vz();
        let split = vec!["hanya sanksi".to_owned(), "hanya denda".to_owned()];
        assert!((v.evidence("sanksi denda", &split) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn an_empty_query_yields_no_evidence_rather_than_full_confidence() {
        assert!(vz().evidence("", &["apa pun".to_owned()]).abs() < 1e-6);
    }

    #[test]
    fn pruning_keeps_rare_and_unknown_terms_and_drops_common_ones() {
        let v = vz(); // idf: sanksi 6.0, denda 6.0, yang 0.5
        assert_eq!(
            v.prune("Sanksi denda yang berlaku", 2.0),
            "sanksi denda berlaku"
        );
        assert_eq!(
            v.prune("yang yang", 2.0),
            "yang yang",
            "never prune a query to nothing"
        );
        assert_eq!(
            v.prune("sanksi yang", 0.0),
            "sanksi yang",
            "0.0 disables pruning"
        );
        assert_eq!(v.prune("denda denda", 2.0), "denda", "repeats collapse");
    }

    #[test]
    fn a_query_maps_known_terms_and_drops_the_rest() {
        let v = vz().query("sanksi denda kwacamole");
        assert_eq!(v, [(2, 1.0), (5, 1.0)], "sorted, presence-weighted");
    }

    #[test]
    fn repeated_terms_appear_once() {
        // Presence weighting · a term repeated in the query must not double
        // its own contribution.
        assert_eq!(vz().query("denda denda denda").len(), 1);
    }

    #[test]
    fn a_query_of_only_unknown_terms_is_empty_rather_than_an_error() {
        // A stopword-only query produces this, and it must not blow up.
        assert!(vz().query("zzz qqq").is_empty());
    }
}
