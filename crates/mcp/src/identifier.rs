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
//! Perda Kabupaten Sleman Nomor 5 Tahun 2014     a regional tier and its region
//! Pasal 14 PP 26/2009     the article as well (`ROADMAP.md` Stage 1)
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
    /// The article the query names ("Pasal 14", "Pasal 178G", "Pasal I"), if any.
    pub article: Option<String>,
    /// For a regional tier, the region words between the tier and the number
    /// ("Perbup **Sleman** 12/2019"), uppercased, level words dropped, common
    /// abbreviations expanded. Empty when none was named.
    ///
    /// ! Without it a regional reference is not unique: PERDA KABUPATEN 1/2016 is
    /// dozens of regencies' regulations, and the engine would answer for whichever
    /// one ranked first.
    pub region: Vec<String>,
}

/// The tiers enacted per region, so a region may sit between tier and number.
const REGIONAL: &[&str] = &[
    "PERATURAN DAERAH",
    "PERATURAN GUBERNUR",
    "PERATURAN BUPATI",
    "PERATURAN WALIKOTA",
    "QANUN",
];

/// Every spelling of a tier as `chunks.regulation_type` stores it.
///
/// ! The column carries OCR variants ("PERATURANBUPATI", "PERATURAN WALI KOTA") and
/// the identifier index leads with the type, so the lookup names each spelling
/// rather than normalising the column, which no index could serve. A generic tier
/// ("Perda", "Qanun") stands for every level of it.
#[must_use]
pub fn type_variants(canonical: &str) -> Vec<String> {
    let base: Vec<&str> = match canonical {
        "PERATURAN DAERAH" => vec![
            "PERATURAN DAERAH",
            "PERATURAN DAERAH PROVINSI",
            "PERATURAN DAERAH KABUPATEN",
            "PERATURAN DAERAH KOTA",
        ],
        "QANUN" => vec!["QANUN", "QANUN ACEH", "QANUN KABUPATEN", "QANUN KOTA"],
        "PERATURAN WALIKOTA" => vec!["PERATURAN WALIKOTA", "PERATURAN WALI KOTA"],
        other => vec![other],
    };
    let mut out: Vec<String> = Vec::new();
    for b in base {
        for v in [b.to_owned(), b.replace(' ', "")] {
            if !out.contains(&v) {
                out.push(v);
            }
        }
    }
    out
}

/// Words between a regional tier and its number that are not the region's name.
const NOT_REGION: &[&str] = &[
    "nomor",
    "no",
    "nr",
    "kabupaten",
    "kab",
    "kota",
    "provinsi",
    "prov",
    "pemerintah",
];

/// Province abbreviations as people write them, expanded to the words a title carries.
const REGION_ALIASES: &[(&str, &[&str])] = &[
    ("dki", &[]),
    ("diy", &["YOGYAKARTA"]),
    ("jabar", &["JAWA", "BARAT"]),
    ("jateng", &["JAWA", "TENGAH"]),
    ("jatim", &["JAWA", "TIMUR"]),
    ("kalbar", &["KALIMANTAN", "BARAT"]),
    ("kalteng", &["KALIMANTAN", "TENGAH"]),
    ("kalsel", &["KALIMANTAN", "SELATAN"]),
    ("kaltim", &["KALIMANTAN", "TIMUR"]),
    ("sulsel", &["SULAWESI", "SELATAN"]),
    ("sulut", &["SULAWESI", "UTARA"]),
    ("sulteng", &["SULAWESI", "TENGAH"]),
    ("sumut", &["SUMATERA", "UTARA"]),
    ("sumbar", &["SUMATERA", "BARAT"]),
    ("sumsel", &["SUMATERA", "SELATAN"]),
    ("ntb", &["NUSA", "TENGGARA", "BARAT"]),
    ("ntt", &["NUSA", "TENGGARA", "TIMUR"]),
];

/// Words that end a citation when walking back from the number: a sentence, ✗ a name.
const SENTENCE_WORDS: &[&str] = &[
    "yang",
    "tentang",
    "dan",
    "atau",
    "di",
    "dalam",
    "dari",
    "ke",
    "untuk",
    "pada",
    "sesuai",
    "menurut",
    "berdasarkan",
    "aturan",
    "mengatur",
    "apa",
    "bagaimana",
];

fn region_words(tokens: &[String]) -> Vec<String> {
    let mut out = Vec::new();
    for t in tokens {
        if NOT_REGION.contains(&t.as_str()) {
            continue;
        }
        if let Some((_, words)) = REGION_ALIASES.iter().find(|(a, _)| a == t) {
            out.extend(words.iter().map(|w| (*w).to_owned()));
        } else {
            out.push(t.to_uppercase());
        }
    }
    out
}

fn is_roman(s: &str) -> bool {
    matches!(
        s,
        "i" | "ii" | "iii" | "iv" | "v" | "vi" | "vii" | "viii" | "ix" | "x"
    )
}

/// The article named in the query: `pasal 14`, `pasal 178g`, `pasal ii`.
fn article_in(tokens: &[String]) -> Option<String> {
    tokens.windows(2).find_map(|w| {
        if w[0] != "pasal" {
            return None;
        }
        let t = &w[1];
        let digits: String = t.chars().take_while(char::is_ascii_digit).collect();
        let rest = &t[digits.len()..];
        if !digits.is_empty() && rest.len() <= 1 && rest.chars().all(|c| c.is_ascii_alphabetic()) {
            Some(format!("Pasal {digits}{}", rest.to_uppercase()))
        } else if is_roman(t) {
            Some(format!("Pasal {}", t.to_uppercase()))
        } else {
            None
        }
    })
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
    (&["perwalkot"], "PERATURAN WALIKOTA"),
    (&["perwal"], "PERATURAN WALIKOTA"),
    (&["keputusan", "presiden"], "KEPUTUSAN PRESIDEN"),
    (&["keppres"], "KEPUTUSAN PRESIDEN"),
    (&["kepres"], "KEPUTUSAN PRESIDEN"),
    // ! After "undang undang": at an equal end the later entry wins, and "peraturan
    // pemerintah pengganti undang-undang" is a Perpu, ✗ a law.
    (
        &["pengganti", "undang", "undang"],
        "PERATURAN PEMERINTAH PENGGANTI UNDANG-UNDANG",
    ),
    (&["perpu"], "PERATURAN PEMERINTAH PENGGANTI UNDANG-UNDANG"),
    (&["perppu"], "PERATURAN PEMERINTAH PENGGANTI UNDANG-UNDANG"),
    (&["peraturan", "daerah"], "PERATURAN DAERAH"),
    (&["perda"], "PERATURAN DAERAH"),
    (
        &["peraturan", "daerah", "provinsi"],
        "PERATURAN DAERAH PROVINSI",
    ),
    (&["perda", "provinsi"], "PERATURAN DAERAH PROVINSI"),
    (
        &["peraturan", "daerah", "kabupaten"],
        "PERATURAN DAERAH KABUPATEN",
    ),
    (&["perda", "kabupaten"], "PERATURAN DAERAH KABUPATEN"),
    (&["perda", "kab"], "PERATURAN DAERAH KABUPATEN"),
    (&["peraturan", "daerah", "kota"], "PERATURAN DAERAH KOTA"),
    (&["perda", "kota"], "PERATURAN DAERAH KOTA"),
    (&["qanun"], "QANUN"),
    (&["qanun", "kabupaten"], "QANUN KABUPATEN"),
    (&["qanun", "kota"], "QANUN KOTA"),
    (&["qanun", "aceh"], "QANUN ACEH"),
];

/// The regulation tier named in the tokens before a number, if any.
///
/// Looks back a short distance only: "UU 28/2007 tentang pajak" names a tier,
/// but the word "pemerintah" three sentences earlier does not.
/// The tier before a number, and for a regional tier the region words between them.
fn tier_and_region(tokens: &[String], upto: usize) -> (Option<&'static str>, Vec<String>) {
    match tier_before(tokens, upto) {
        Some((canonical, end)) if REGIONAL.iter().any(|r| canonical.starts_with(r)) => {
            (Some(canonical), region_words(&tokens[end..upto]))
        }
        Some((canonical, _)) => (Some(canonical), Vec::new()),
        None => (None, Vec::new()),
    }
}

/// A national tier must sit within 4 tokens of the number, as it always had to. A
/// regional tier may sit up to 8 back, because its region comes between ("Peraturan
/// Daerah Kabupaten Kotawaringin Timur Nomor 5"), but only across words that can be
/// part of a name: a sentence word ("yang", "tentang") ends the citation.
fn tier_before(tokens: &[String], upto: usize) -> Option<(&'static str, usize)> {
    let near = scan_tier(tokens, upto.saturating_sub(4), upto);
    if near.is_some() {
        return near;
    }
    let mut lo = upto;
    while lo > 0 && upto - lo < 8 && !SENTENCE_WORDS.contains(&tokens[lo - 1].as_str()) {
        lo -= 1;
    }
    scan_tier(tokens, lo, upto).filter(|(c, _)| REGIONAL.iter().any(|r| c.starts_with(r)))
}

/// The tier phrase in `tokens[lo..upto]` that ends nearest the number, with its end.
fn scan_tier(tokens: &[String], lo: usize, upto: usize) -> Option<(&'static str, usize)> {
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
    best.map(|(end, c)| (c, lo + end))
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
                // The tier and region sit before the NUMBER, ✗ before "tahun".
                let at = tokens[..i].iter().rposition(|c| c == num).unwrap_or(i);
                let (reg_type, region) = tier_and_region(&tokens, at);
                push_unique(
                    &mut out,
                    Identifier {
                        number: num.clone(),
                        year: tokens[i + 1].parse().ok(),
                        reg_type,
                        article: None,
                        region,
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
            let (reg_type, region) = tier_and_region(&tokens, i);
            push_unique(
                &mut out,
                Identifier {
                    number: t.clone(),
                    year: tokens[i + 1].parse().ok(),
                    reg_type,
                    article: None,
                    region,
                },
            );
            i += 2;
            continue;
        }

        i += 1;
    }

    // ! One article named in the query belongs to the regulation cited with it. With
    // several regulations and one article it is attached to each: the lookup finds it
    // only in the regulation that has it.
    if let Some(article) = article_in(&tokens) {
        for id in &mut out {
            id.article = Some(article.clone());
        }
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
        if existing.region.is_empty() {
            existing.region = id.region;
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
                article: None,
                region: vec![],
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
                article: None,
                region: vec![],
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
                article: None,
                region: vec![],
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

    #[test]
    fn the_named_article_is_kept() {
        assert_eq!(
            one("Pasal 14 PP 26/2009").article.as_deref(),
            Some("Pasal 14")
        );
        assert_eq!(
            one("menurut UU 10/2016 pasal 178g, apa sanksinya")
                .article
                .as_deref(),
            Some("Pasal 178G")
        );
        assert_eq!(
            one("bunyi pasal II UU 11/2020").article.as_deref(),
            Some("Pasal II")
        );
        assert_eq!(one("UU 28/2007").article, None);
    }

    #[test]
    fn a_regional_citation_keeps_its_tier_and_region() {
        let id = one("Bunyi Pasal 52 Perda Kabupaten Kotawaringin Timur Nomor 5 Tahun 2014");
        assert_eq!(id.reg_type, Some("PERATURAN DAERAH KABUPATEN"));
        assert_eq!(id.region, vec!["KOTAWARINGIN", "TIMUR"]);
        assert_eq!(id.article.as_deref(), Some("Pasal 52"));

        let id = one("Pergub DKI Jakarta No. 18 Tahun 2015");
        assert_eq!(id.reg_type, Some("PERATURAN GUBERNUR"));
        assert_eq!(id.region, vec!["JAKARTA"]);

        let id = one("Perbup Sleman 12/2019");
        assert_eq!(id.reg_type, Some("PERATURAN BUPATI"));
        assert_eq!(id.region, vec!["SLEMAN"]);

        let id = one("Peraturan Daerah Kota Ternate Nomor 10 Tahun 2010");
        assert_eq!(id.reg_type, Some("PERATURAN DAERAH KOTA"));
        assert_eq!(id.region, vec!["TERNATE"]);
    }

    #[test]
    fn a_national_tier_carries_no_region() {
        assert!(one("UU 28/2007").region.is_empty());
        assert!(
            one("Peraturan Pemerintah No. 26 Tahun 2009")
                .region
                .is_empty()
        );
    }

    #[test]
    fn the_newer_tier_words_are_recognised() {
        assert_eq!(one("Keppres 12/1999").reg_type, Some("KEPUTUSAN PRESIDEN"));
        assert_eq!(
            one("Perpu 1/2020").reg_type,
            Some("PERATURAN PEMERINTAH PENGGANTI UNDANG-UNDANG")
        );
        assert_eq!(
            one("Peraturan Pemerintah Pengganti Undang-Undang Nomor 1 Tahun 2020").reg_type,
            Some("PERATURAN PEMERINTAH PENGGANTI UNDANG-UNDANG")
        );
        assert_eq!(one("perda 3/2011").reg_type, Some("PERATURAN DAERAH"));
        assert_eq!(one("perwali 7/2018").reg_type, Some("PERATURAN WALIKOTA"));
    }

    #[test]
    fn a_sentence_word_ends_a_regional_citation() {
        // "perda" four words back, across "yang mengatur", is not this number's tier.
        let id = one("perda yang mengatur pajak nomor 12 tahun 2011");
        assert_eq!(id.reg_type, None, "{id:?}");
    }

    #[test]
    fn every_stored_spelling_of_a_tier_is_looked_up() {
        let v = type_variants("PERATURAN BUPATI");
        assert!(v.contains(&"PERATURAN BUPATI".to_owned()));
        assert!(v.contains(&"PERATURANBUPATI".to_owned()));
        assert!(type_variants("PERATURAN WALIKOTA").contains(&"PERATURAN WALI KOTA".to_owned()));
        assert_eq!(type_variants("PERATURAN DAERAH").len(), 8);
    }
}
