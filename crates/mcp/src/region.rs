//! The region a free-text question names · the scope of the region arm.
//!
//! ! Why an arm exists for this (`ROADMAP.md` Stage 3): hundreds of regions enact
//! near-identical clauses, so a question naming one ("retribusi reklame di Kabupaten
//! Bangka Tengah") is answered by that region's regulation and by no other. The global
//! arms rarely find it. Measured on the 1.7M corpus, 604 cases: 137 name the answer's
//! region and 54 have it in the top 5; in 73 of the 83 misses the named region IS in the
//! top 10, with the wrong clause, because the region's name matches its every chunk.
//!
//! ! The region list is the CORPUS's (`corpus_regions`, migration 0007), ✗ compiled in.
//! What is compiled in is how Indonesian names a place, as in `identifier.rs`: the
//! words that announce one (`MARKERS`).

use std::collections::HashMap;

/// Words that announce a place name. A one-word region counts only after one of these.
///
/// ! Without the rule, short region names collide with ordinary words: `BATU` is in
/// *batu bara*, `PATI` and `BURU` are words. A name of two or more words ("Bangka
/// Tengah") is specific enough to match alone. Measured on the eval set: 136 of 137
/// region-naming questions found and 24 false scopes without the rule, 133 and 20 with.
const MARKERS: &[&str] = &[
    "KABUPATEN",
    "KAB",
    "KOTA",
    "PROVINSI",
    "PROV",
    "PEMKAB",
    "PEMKOT",
    "PEMPROV",
    "DI",
    "DAERAH",
    "ISTIMEWA",
    "WILAYAH",
    "BUPATI",
    "WALIKOTA",
    "GUBERNUR",
    "PERDA",
    "PERBUP",
    "PERWALI",
    "PERGUB",
    "QANUN",
];

fn tokens(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(str::to_uppercase)
        .collect()
}

/// The regions of one corpus, keyed by their normalised words.
#[derive(Debug, Default)]
pub struct Regions {
    /// Normalised words → every stored spelling. The column carries variants of one
    /// name (`SUMATERA SELATAN`, `SUMATERA SElATAN`), and the arm filters on the
    /// stored value, so a match scopes to all of them.
    by_words: HashMap<Vec<String>, Vec<String>>,
    longest: usize,
}

/// What a question named.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Scope {
    /// Stored `enacting_body` values to search within.
    pub bodies: Vec<String>,
    /// The question without the region's words · what the arm matches on.
    ///
    /// ! Stripped, because inside the scope the name matches every chunk that
    /// mentions it — definitions ("Daerah adalah Kabupaten Bangka Tengah") before the
    /// clause asked about. Measured: the answer in the scoped top 5 for 37 of 81
    /// misses stripped, 26 of 81 with the name left in.
    pub text: String,
}

impl Regions {
    pub fn new(stored: impl IntoIterator<Item = String>) -> Self {
        let mut by_words: HashMap<Vec<String>, Vec<String>> = HashMap::new();
        for body in stored {
            let words = tokens(&body);
            if words.is_empty() {
                continue;
            }
            let spellings = by_words.entry(words).or_default();
            if !spellings.contains(&body) {
                spellings.push(body);
            }
        }
        let longest = by_words.keys().map(Vec::len).max().unwrap_or(0);
        Self { by_words, longest }
    }

    pub fn is_empty(&self) -> bool {
        self.by_words.is_empty()
    }

    /// The regions `query` names, longest name first at each position.
    ///
    /// ! Longest first, so "Daerah Istimewa Yogyakarta" is the province and not also
    /// the city "Yogyakarta" inside it.
    pub fn detect(&self, query: &str) -> Option<Scope> {
        let words = tokens(query);
        let mut bodies: Vec<String> = Vec::new();
        let mut keep = vec![true; words.len()];
        let mut i = 0;
        while i < words.len() {
            let mut matched = 0;
            for n in (1..=self.longest.min(words.len() - i)).rev() {
                let Some(spellings) = self.by_words.get(&words[i..i + n]) else {
                    continue;
                };
                if n == 1 && !(i > 0 && MARKERS.contains(&words[i - 1].as_str())) {
                    continue;
                }
                for s in spellings {
                    if !bodies.contains(s) {
                        bodies.push(s.clone());
                    }
                }
                matched = n;
                break;
            }
            if matched > 0 {
                keep[i..i + matched].iter_mut().for_each(|k| *k = false);
                i += matched;
            } else {
                i += 1;
            }
        }
        if bodies.is_empty() {
            return None;
        }
        let text = words
            .iter()
            .zip(&keep)
            .filter(|(_, k)| **k)
            .map(|(w, _)| w.to_lowercase())
            .collect::<Vec<_>>()
            .join(" ");
        Some(Scope { bodies, text })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn corpus() -> Regions {
        Regions::new(
            [
                "BANGKA TENGAH",
                "BATU",
                "YOGYAKARTA",
                "DAERAH ISTIMEWA YOGYAKARTA",
                "SUMATERA SELATAN",
                "SUMATERA SElATAN",
            ]
            .map(String::from),
        )
    }

    #[test]
    fn a_multi_word_region_is_found_anywhere_and_stripped_from_the_text() {
        let s = corpus()
            .detect("Standar reklame di Kabupaten Bangka Tengah?")
            .unwrap();
        assert_eq!(s.bodies, vec!["BANGKA TENGAH"]);
        assert_eq!(s.text, "standar reklame di kabupaten");
    }

    #[test]
    fn a_one_word_region_needs_a_marker_before_it() {
        assert!(corpus().detect("tarif royalti batu bara").is_none());
        let s = corpus().detect("retribusi pasar di Kota Batu").unwrap();
        assert_eq!(s.bodies, vec!["BATU"]);
    }

    #[test]
    fn the_longest_name_wins_at_a_position() {
        let s = corpus()
            .detect("pajak hotel Daerah Istimewa Yogyakarta")
            .unwrap();
        assert_eq!(s.bodies, vec!["DAERAH ISTIMEWA YOGYAKARTA"]);
    }

    #[test]
    fn every_stored_spelling_of_a_name_is_in_scope() {
        let s = corpus().detect("zoonosis di Sumatera Selatan").unwrap();
        assert_eq!(s.bodies.len(), 2);
    }

    #[test]
    fn no_region_no_scope() {
        assert!(corpus().detect("apa sanksi pidana korupsi").is_none());
        assert!(Regions::new(Vec::new()).is_empty());
    }
}
