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
    /// The tier the user named, as stored in `chunks.regulation_type`.
    ///
    /// ! Not decoration. Number and year alone are NOT a unique reference:
    /// 12 distinct regulations share 60/2014 and nine of them are PERATURAN
    /// BUPATI, so an engine that drops the "PP" the user typed answers a
    /// different question confidently. `None` means the query named no tier
    /// and every tier is a legitimate hit.
    pub reg_type: Option<&'static str>,
}

/// How Indonesian regulations are abbreviated in practice, longest first so
/// "peraturan presiden" is matched before the bare "presiden" of another form.
///
/// ! "pp" and "perpres" must never collide: both begin with "p" and they are
/// different tiers of law.
const TYPE_WORDS: &[(&[&str], &str)] = &[
    (&["undang", "undang"], "UNDANG-UNDANG"),
    (&["uu"], "UNDANG-UNDANG"),
    (&["peraturan", "pemerintah"], "PERATURAN PEMERINTAH"),
    (&["pp"], "PERATURAN PEMERINTAH"),
    (&["peraturan", "presiden"], "PERATURAN PRESIDEN"),
    (&["perpres"], "PERATURAN PRESIDEN"),
    (&["instruksi", "presiden"], "INSTRUKSI PRESIDEN"),
    (&["inpres"], "INSTRUKSI PRESIDEN"),
    (&["peraturan", "gubernur"], "PERATURAN GUBERNUR"),
    (&["pergub"], "PERATURAN GUBERNUR"),
    (&["peraturan", "bupati"], "PERATURAN BUPATI"),
    (&["perbup"], "PERATURAN BUPATI"),
    (&["peraturan", "walikota"], "PERATURAN WALIKOTA"),
    (&["peraturan", "wali", "kota"], "PERATURAN WALIKOTA"),
    (&["perwali"], "PERATURAN WALIKOTA"),
    (
        &["peraturan", "daerah", "provinsi"],
        "PERATURAN DAERAH PROVINSI",
    ),
    (
        &["peraturan", "daerah", "kabupaten"],
        "PERATURAN DAERAH KABUPATEN",
    ),
    (&["peraturan", "daerah", "kota"], "PERATURAN DAERAH KOTA"),
];

/// The regulation tier named in the tokens before a number, if any.
///
/// Looks back a short distance only: "UU 28/2007 tentang pajak" names a tier,
/// but the word "pemerintah" three sentences earlier does not.
fn type_before(tokens: &[String], upto: usize) -> Option<&'static str> {
    let lo = upto.saturating_sub(4);
    let window = &tokens[lo..upto];
    let mut best: Option<(usize, &'static str)> = None;
    for (words, canonical) in TYPE_WORDS {
        for start in 0..window.len() {
            if window.len() - start >= words.len()
                && window[start..start + words.len()]
                    .iter()
                    .zip(*words)
                    .all(|(t, w)| t == w)
            {
                // Nearest to the number wins: in "peraturan pemerintah
                // pengganti undang-undang 1/2020" the last tier word is the
                // operative one.
                let end = start + words.len();
                if best.is_none_or(|(b, _)| end >= b) {
                    best = Some((end, canonical));
                }
            }
        }
    }
    best.map(|(_, c)| c)
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
                        reg_type: type_before(&tokens, i),
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
                    reg_type: type_before(&tokens, i),
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
    // ! Identity is number + year. A restatement that omits the tier ("UU
    // 28/2007 dan juga 28 tahun 2007") is the SAME regulation, so it must not
    // become a second reference — and when one mention carries the tier and
    // another does not, the tier is kept: the user did say it once.
    if let Some(existing) = out
        .iter_mut()
        .find(|e| e.number == id.number && e.year == id.year)
    {
        if existing.reg_type.is_none() {
            existing.reg_type = id.reg_type;
        }
        return;
    }
    out.push(id);
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
                year: Some(2007),
                reg_type: Some("UNDANG-UNDANG"),
            }
        );
    }

    #[test]
    fn spelled_out_form_is_recognised() {
        assert_eq!(
            one("Peraturan Pemerintah No. 26 Tahun 2009"),
            Identifier {
                number: "26".into(),
                year: Some(2009),
                reg_type: Some("PERATURAN PEMERINTAH"),
            }
        );
    }

    #[test]
    fn nomor_without_a_type_prefix_is_recognised() {
        assert_eq!(
            one("nomor 26 tahun 2009"),
            Identifier {
                number: "26".into(),
                year: Some(2009),
                // No tier named · every tier is a legitimate hit.
                reg_type: None,
            }
        );
    }

    #[test]
    fn the_tier_the_user_typed_is_kept() {
        // ! The bug this guards: 60/2014 matches 12 regulations in the spike
        // corpus, nine of them PERATURAN BUPATI. Dropping "PP" made the engine
        // answer about a regency rule when asked about a government regulation.
        assert_eq!(one("PP 60/2014").reg_type, Some("PERATURAN PEMERINTAH"));
        assert_eq!(one("UU No. 17 Tahun 2013").reg_type, Some("UNDANG-UNDANG"));
        assert_eq!(one("Perpres 76/2016").reg_type, Some("PERATURAN PRESIDEN"));
        assert_eq!(one("perbup 27/2013").reg_type, Some("PERATURAN BUPATI"));
        assert_eq!(
            one("Peraturan Daerah Kabupaten 1/2010").reg_type,
            Some("PERATURAN DAERAH KABUPATEN")
        );
    }

    #[test]
    fn pp_and_perpres_are_never_confused() {
        // Both start with "p" and they are different tiers of law.
        assert_eq!(one("PP 76/2016").reg_type, Some("PERATURAN PEMERINTAH"));
        assert_eq!(
            one("Peraturan Presiden 76 Tahun 2016").reg_type,
            Some("PERATURAN PRESIDEN")
        );
    }

    #[test]
    fn a_tier_word_far_from_the_number_is_not_claimed() {
        // The word has to be part of the citation, not merely somewhere in
        // the sentence, or every query mentioning government claims a tier.
        let found = one("apakah pemerintah daerah wajib melapor sesuai aturan nomor 12 tahun 2011");
        assert_eq!(found.reg_type, None, "{found:?}");
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
        let ids = extract("UU 28/2007 dan juga 28 tahun 2007");
        assert_eq!(ids.len(), 1);
        // The tier survives even though the second mention omitted it.
        assert_eq!(ids[0].reg_type, Some("UNDANG-UNDANG"));
    }

    #[test]
    fn a_tier_stated_only_on_the_second_mention_is_still_kept() {
        let ids = extract("aturan 28 tahun 2007, maksud saya UU 28/2007");
        assert_eq!(ids.len(), 1);
        assert_eq!(ids[0].reg_type, Some("UNDANG-UNDANG"));
    }

    #[test]
    fn mixed_case_and_punctuation_do_not_matter() {
        assert_eq!(one("UU No.28/2007 tentang pajak").number, "28");
    }
}
