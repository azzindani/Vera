//! Query-side BM25 vectorisation.
//!
//! ! Documents are vectorised offline by `pipelines/pre_embed/sparse.py`. This
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

/// The token pattern this implementation reproduces. Compared against the
/// artifact on load.
const ENGINE_TOKEN_PATTERN: &str = r"(?u)\b\w\w+\b";

#[derive(Debug, serde::Deserialize)]
struct Artifact {
    dim: u32,
    token_pattern: String,
    vocab: HashMap<String, u32>,
}

/// A loaded vocabulary, sufficient to project a query into the sparse space.
#[derive(Debug, Clone)]
pub struct QueryVectorizer {
    vocab: HashMap<String, u32>,
    dim: u32,
}

impl QueryVectorizer {
    /// Load the artifact written alongside the corpus.
    ///
    /// # Errors
    /// Unreadable or unparseable file, or a tokeniser that does not match.
    pub fn load(path: &Path) -> Result<Self, Bm25Error> {
        let raw = std::fs::read_to_string(path)?;
        let art: Artifact = serde_json::from_str(&raw)?;
        if art.token_pattern != ENGINE_TOKEN_PATTERN {
            return Err(Bm25Error::TokenizerDrift {
                artifact: art.token_pattern,
                engine: ENGINE_TOKEN_PATTERN.to_owned(),
            });
        }
        Ok(Self {
            vocab: art.vocab,
            dim: art.dim,
        })
    }

    #[must_use]
    pub fn dim(&self) -> u32 {
        self.dim
    }

    /// Project a query into `(index, weight)` pairs.
    ///
    /// Presence-weighted, so the dot product against a BM25-weighted document
    /// vector *is* the BM25 score. Out-of-vocabulary terms are dropped — they
    /// contribute nothing to any document by definition.
    #[must_use]
    pub fn query(&self, text: &str) -> Vec<(u32, f32)> {
        let mut seen: Vec<(u32, f32)> = Vec::new();
        let mut indices: Vec<u32> = tokenize(text)
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
        QueryVectorizer { vocab, dim: 20000 }
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
