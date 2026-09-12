//! Pulling regulation identifiers out of a free-text query.
//!
//! ! This feeds the routing bypass (`CLAUDE.md` §7.4). Routing measured 95.0%
//! recall on the spike corpus, so roughly one query in twenty has its answer in
//! a cluster that was never probed. When a user names a regulation outright,
//! that must never be how it gets lost — so anything recognised here is looked
//! up globally, independent of cluster selection.
//!
//! Recognises how Indonesian regulations are actually cited:
//!
//! ```text
//! UU 28/2007              number/year
//! PP No. 26 Tahun 2009    number + year, spelled out
//! Nomor 26 Tahun 2009     the same, without a type prefix
//! ```

/// A regulation reference found in a query.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Identifier {
    pub number: String,
    pub year: Option<i32>,
}

fn is_year(s: &str) -> bool {
    s.len() == 4 && s.chars().all(|c| c.is_ascii_digit()) && s.starts_with(['1', '2'])
}

/// Extract every regulation reference in the query, in order of appearance.
///
/// Deliberately permissive about surrounding words and punctuation, and
/// deliberately strict about shape: a bare number is *not* an identifier, or
/// every query containing a digit would trigger a global scan.
#[must_use]
pub fn extract(query: &str) -> Vec<Identifier> {
    let mut out: Vec<Identifier> = Vec::new();

    // Normalise separators so "No.", "Nomor" and "/" all become word breaks,
    // then walk the token stream.
    let lowered = query.to_lowercase();
    let tokens: Vec<String> = lowered
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(str::to_owned)
        .collect();

    // Pass 1 · `28/2007` survives normalisation as adjacent [28, 2007].
    // Pass 2 · `nomor 26 tahun 2009` as [nomor, 26, tahun, 2009].
    let mut i = 0;
    while i < tokens.len() {
        let t = &tokens[i];

        // "tahun YYYY" preceded by a number somewhere in the last few tokens.
        if t == "tahun" && i + 1 < tokens.len() && is_year(&tokens[i + 1]) {
            if let Some(num) = tokens[..i]
                .iter()
                .rev()
                .take(3)
                .find(|c| c.chars().all(|ch| ch.is_ascii_digit()) && !is_year(c))
            {
                push_unique(
                    &mut out,
                    Identifier {
                        number: num.clone(),
                        year: tokens[i + 1].parse().ok(),
                    },
                );
                i += 2;
                continue;
            }
        }

        // A non-year number immediately followed by a year: `28 2007`.
        if t.chars().all(|c| c.is_ascii_digit())
            && !is_year(t)
            && i + 1 < tokens.len()
            && is_year(&tokens[i + 1])
        {
            push_unique(
                &mut out,
                Identifier {
                    number: t.clone(),
                    year: tokens[i + 1].parse().ok(),
                },
            );
            i += 2;
            continue;
        }

        i += 1;
    }

    out
}

fn push_unique(out: &mut Vec<Identifier>, id: Identifier) {
    if !out.contains(&id) {
        out.push(id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn one(q: &str) -> Identifier {
        let found = extract(q);
        assert_eq!(
            found.len(),
            1,
            "expected exactly one in {q:?}, got {found:?}"
        );
        found.into_iter().next().unwrap()
    }

    #[test]
    fn slash_form_is_recognised() {
        assert_eq!(
            one("UU 28/2007"),
            Identifier {
                number: "28".into(),
                year: Some(2007)
            }
        );
    }

    #[test]
    fn spelled_out_form_is_recognised() {
        assert_eq!(
            one("Peraturan Pemerintah No. 26 Tahun 2009"),
            Identifier {
                number: "26".into(),
                year: Some(2009)
            }
        );
    }

    #[test]
    fn nomor_without_a_type_prefix_is_recognised() {
        assert_eq!(
            one("nomor 26 tahun 2009"),
            Identifier {
                number: "26".into(),
                year: Some(2009)
            }
        );
    }

    #[test]
    fn a_conceptual_query_yields_no_identifier() {
        // ! The common case. A query with no reference must not trigger a
        // global scan, or the bypass becomes the default path.
        assert!(extract("sanksi administrasi berupa denda di bidang cukai").is_empty());
        assert!(extract("apa itu wajib pajak").is_empty());
    }

    #[test]
    fn a_bare_number_is_not_an_identifier() {
        // "pasal 9" is a locator, ✗ a regulation reference.
        assert!(extract("pasal 9").is_empty());
        assert!(extract("26").is_empty());
    }

    #[test]
    fn a_bare_year_is_not_an_identifier() {
        assert!(extract("peraturan tahun 2009").is_empty());
    }

    #[test]
    fn several_references_in_one_query_are_all_found() {
        let ids = extract("bandingkan UU 28/2007 dengan PP 26 tahun 2009");
        assert_eq!(ids.len(), 2);
        assert_eq!(ids[0].number, "28");
        assert_eq!(ids[1].number, "26");
    }

    #[test]
    fn the_same_reference_written_twice_is_reported_once() {
        assert_eq!(extract("UU 28/2007 dan juga 28 tahun 2007").len(), 1);
    }

    #[test]
    fn mixed_case_and_punctuation_do_not_matter() {
        assert_eq!(one("UU No.28/2007 tentang pajak").number, "28");
    }
}
