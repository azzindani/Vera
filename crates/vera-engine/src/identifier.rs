//! Exact-identifier extraction · the routing bypass.
//!
//! ! `LOOPHOLES.md` §1 / `CLAUDE.md` §7 rule 4. Semantic routing is a
//! probabilistic accelerator: a query naming a specific regulation can route to
//! the wrong clusters and return **nothing**, with no error and no signal that
//! anything was missed. Silent zero-recall on "UU 28/2007" is the worst failure
//! this engine can produce, because the user cannot tell it from "no such
//! regulation exists".
//!
//! So identifiers are pulled out of the query text here and searched
//! **globally**, outside the routing layers entirely. Routing accelerates; this
//! path guarantees.
//!
//! Hand-written rather than a regex: the grammar is small and closed, and the
//! normalization (many spellings → one canonical form) is the actual work.

/// Document types that prefix an Indonesian regulation number.
///
/// Long forms first so `undang-undang` is not shadowed by a prefix match.
const DOC_TYPES: &[(&str, &str)] = &[
    ("undang-undang dasar", "UUD"),
    ("undang-undang", "UU"),
    ("peraturan pemerintah", "PP"),
    ("peraturan presiden", "PERPRES"),
    ("peraturan menteri", "PERMEN"),
    ("peraturan daerah", "PERDA"),
    ("keputusan presiden", "KEPPRES"),
    ("instruksi presiden", "INPRES"),
    ("uud", "UUD"),
    ("uu", "UU"),
    ("perppu", "PERPPU"),
    ("perpres", "PERPRES"),
    ("permenkeu", "PERMENKEU"),
    ("permen", "PERMEN"),
    ("keppres", "KEPPRES"),
    ("inpres", "INPRES"),
    ("perda", "PERDA"),
    ("pmk", "PMK"),
    ("pp", "PP"),
];

/// Words that sit between the type and the number and carry no information.
const FILLER: &[&str] = &["no", "nomor", "number", "tahun", "th", "year"];

/// A regulation identifier found in query text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Identifier {
    /// Canonical form, e.g. `UU 28/2007` · what the corpus stores.
    pub canonical: String,
    /// The span as the user actually wrote it, for `explain_routing`.
    pub matched_on: String,
}

/// Extract every regulation identifier in `query`.
///
/// Recognizes `UU 28/2007`, `UU No. 28 Tahun 2007`, `Undang-Undang Nomor 28
/// Tahun 2007` and the same shapes for other document types, normalizing all of
/// them to `TYPE NUMBER/YEAR`.
///
/// ! Deliberately conservative. A false positive costs one extra global BM25
/// query; a false negative is the silent zero-recall this whole path exists to
/// prevent — so the bar for *rejecting* a candidate is high, not low.
#[must_use]
pub fn extract(query: &str) -> Vec<Identifier> {
    let tokens = tokenize(query);
    let mut out: Vec<Identifier> = Vec::new();
    let mut i = 0usize;

    while i < tokens.len() {
        let Some((consumed, doc_type)) = match_doc_type(&tokens[i..]) else {
            i += 1;
            continue;
        };
        let mut j = i + consumed;

        // Skip "No." / "Nomor" and friends.
        while j < tokens.len() && FILLER.contains(&tokens[j].lower.as_str()) {
            j += 1;
        }
        let Some(number) = tokens.get(j).and_then(|t| numeric(&t.lower)) else {
            i += 1;
            continue;
        };
        j += 1;

        // Skip "Tahun" / "/" between number and year.
        while j < tokens.len() && FILLER.contains(&tokens[j].lower.as_str()) {
            j += 1;
        }
        let Some(year) = tokens.get(j).and_then(|t| year(&t.lower)) else {
            i += 1;
            continue;
        };

        let start = tokens[i].start;
        let end = tokens[j].end;
        out.push(Identifier {
            canonical: format!("{doc_type} {number}/{year}"),
            matched_on: query[start..end].to_owned(),
        });
        i = j + 1;
    }

    // ! De-duplicate on the canonical form: a query naming the same regulation
    // twice must not cost two identical global scans. Set-based, ✗ adjacent-only,
    // so "UU 28/2007 ... PP 74/2011 ... UU No. 28 Tahun 2007" collapses too.
    let mut seen = std::collections::HashSet::new();
    out.retain(|id| seen.insert(id.canonical.clone()));
    out
}

struct Token {
    lower: String,
    start: usize,
    end: usize,
}

/// Split on anything that is not alphanumeric or a hyphen, keeping byte spans
/// into the **original** string so the match can be quoted back verbatim.
///
/// Hyphens stay inside tokens so `undang-undang` is one word.
///
/// ! Lowercases per token, ✗ the whole string up front. `str::to_lowercase` is
/// not length-preserving for every input (`İ` becomes two chars), so byte
/// offsets taken from a lowercased copy can land mid-character in the original
/// and panic on slicing. Tokenizing the original keeps every span valid.
fn tokenize(original: &str) -> Vec<Token> {
    let mut out = Vec::new();
    let mut start = None;
    for (i, c) in original.char_indices() {
        let is_word = c.is_alphanumeric() || c == '-';
        match (is_word, start) {
            (true, None) => start = Some(i),
            (false, Some(s)) => {
                out.push(Token {
                    lower: original[s..i].to_lowercase(),
                    start: s,
                    end: i,
                });
                start = None;
            }
            _ => {}
        }
    }
    if let Some(s) = start {
        out.push(Token {
            lower: original[s..].to_lowercase(),
            start: s,
            end: original.len(),
        });
    }
    out
}

/// Match a document type at the head of `tokens`, returning how many tokens it
/// consumed and its canonical abbreviation.
fn match_doc_type(tokens: &[Token]) -> Option<(usize, &'static str)> {
    for (spelling, canonical) in DOC_TYPES {
        let words: Vec<&str> = spelling.split(' ').collect();
        if tokens.len() >= words.len()
            && words
                .iter()
                .zip(tokens)
                .all(|(w, t)| t.lower.as_str() == *w)
        {
            return Some((words.len(), canonical));
        }
    }
    None
}

/// A regulation number: 1–4 digits, no leading zero games.
fn numeric(token: &str) -> Option<&str> {
    let ok = !token.is_empty()
        && token.len() <= 4
        && token.chars().all(|c| c.is_ascii_digit());
    ok.then_some(token)
}

/// A plausible four-digit year. Bounded so `UU 28 1234` is not read as one.
fn year(token: &str) -> Option<&str> {
    let ok = token.len() == 4
        && token.chars().all(|c| c.is_ascii_digit())
        && token.parse::<u32>().is_ok_and(|y| (1945..=2100).contains(&y));
    ok.then_some(token)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn canonicals(q: &str) -> Vec<String> {
        extract(q).into_iter().map(|i| i.canonical).collect()
    }

    #[test]
    fn the_slash_form_is_recognized() {
        assert_eq!(canonicals("apa isi UU 28/2007"), ["UU 28/2007"]);
    }

    #[test]
    fn the_verbose_form_normalizes_to_the_same_canonical() {
        // ! All three spellings must collapse to one lookup key, or a corpus
        // storing one form silently fails to match a query using another.
        for q in [
            "UU No. 28 Tahun 2007",
            "Undang-Undang Nomor 28 Tahun 2007",
            "uu nomor 28 tahun 2007",
        ] {
            assert_eq!(canonicals(q), ["UU 28/2007"], "failed on {q}");
        }
    }

    #[test]
    fn other_document_types_are_recognized() {
        assert_eq!(canonicals("PP No. 74 Tahun 2011"), ["PP 74/2011"]);
        assert_eq!(canonicals("Perpres 16/2018"), ["PERPRES 16/2018"]);
        assert_eq!(
            canonicals("Peraturan Pemerintah Nomor 74 Tahun 2011"),
            ["PP 74/2011"]
        );
    }

    #[test]
    fn several_identifiers_in_one_query_are_all_extracted() {
        assert_eq!(
            canonicals("bandingkan UU 28/2007 dengan PP 74/2011"),
            ["UU 28/2007", "PP 74/2011"]
        );
    }

    #[test]
    fn the_same_regulation_named_twice_costs_one_lookup() {
        assert_eq!(
            canonicals("UU 28/2007 dan juga UU No. 28 Tahun 2007"),
            ["UU 28/2007"]
        );
        // Non-adjacent duplicate · an adjacent-only dedup would miss this.
        assert_eq!(
            canonicals("UU 28/2007, PP 74/2011, lalu Undang-Undang Nomor 28 Tahun 2007"),
            ["UU 28/2007", "PP 74/2011"]
        );
    }

    #[test]
    fn the_original_spelling_is_preserved_for_explain_routing() {
        let found = extract("lihat Undang-Undang Nomor 28 Tahun 2007 pasal 9");
        assert_eq!(found[0].matched_on, "Undang-Undang Nomor 28 Tahun 2007");
        assert_eq!(found[0].canonical, "UU 28/2007");
    }

    #[test]
    fn a_purely_conceptual_query_yields_nothing_to_bypass_with() {
        // Nothing to look up globally · the query is routed normally.
        assert!(extract("apa sanksi keterlambatan pelaporan pajak").is_empty());
        assert!(extract("").is_empty());
    }

    #[test]
    fn a_section_reference_is_not_a_document_identifier() {
        // ! "Pasal 9" addresses a place *inside* a document; treating it as a
        // document id would send a meaningless global scan on every query.
        assert!(extract("pasal 9 ayat 3").is_empty());
    }

    #[test]
    fn a_type_without_a_usable_number_and_year_is_rejected() {
        assert!(extract("undang-undang perpajakan").is_empty());
        assert!(extract("UU tentang pajak").is_empty());
        // ! Out-of-range year: a bare number pair is not a citation.
        assert!(extract("UU 28 1234").is_empty());
    }

    #[test]
    fn matching_survives_multibyte_text_around_the_identifier() {
        // Byte spans are sliced out of the original string · an off-by-one here
        // would panic on a char boundary.
        let found = extract("peraturan “khusus” — UU 28/2007 — berlaku");
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].matched_on, "UU 28/2007");
    }

    #[test]
    fn a_character_whose_lowercase_is_longer_does_not_corrupt_spans() {
        // ! `İ`.to_lowercase() is two chars, so offsets from a lowercased copy
        // would drift and could slice mid-character. Regression guard: this
        // must return the right span, and above all must not panic.
        let found = extract("İSTANBUL peraturan UU 28/2007 berlaku");
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].matched_on, "UU 28/2007");
    }

    #[test]
    fn extraction_is_case_insensitive_but_reports_canonical_uppercase() {
        assert_eq!(canonicals("uu 28/2007"), ["UU 28/2007"]);
        assert_eq!(canonicals("Uu No 28 Tahun 2007"), ["UU 28/2007"]);
    }
}
