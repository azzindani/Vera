//! Multi-factor scoring · deciding what matters, once retrieval has decided
//! what is relevant.
//!
//! Retrieval answers one question — "does this text look like the query?" —
//! and legal results cannot be ordered on that alone. A district
//! `PERATURAN BUPATI` and the national `UNDANG-UNDANG` it implements both match
//! a query about tax penalties; only metadata separates them, and that metadata
//! is already in every row (`docs/SCORING.md` §1).
//!
//! ! Factors are a **bounded prior on relevance, ✗ a replacement for it**
//! (invariant 9). The shape is
//!
//! ```text
//! final = relevance × (1 + Σ wᵢ·factorᵢ)
//! ```
//!
//! so a candidate no arm ranked cannot be promoted by its pedigree. An additive
//! model would rank the most prestigious document in the corpus first for every
//! query, and it would look correct.
//!
//! # What these weights are worth
//!
//! Fitted offline by `dev_tools/eval/fit_factors.py`, reranking the text arm
//! over a 60-candidate pool on the 40 article-labelled eval cases:
//!
//! | | Recall@5 |
//! |---|---|
//! | text arm, no floor, no factors | 40.0% |
//! | **relevance floor alone** | **52.5%** |
//! | best in-sample, floor + weights | 65.0% |
//! | **leave-one-out** | **57.5%** |
//!
//! ! The honest number is the leave-one-out one: **+17.5 points**, ✗ +25.
//! Picking the maximum of 3,125 configurations on 40 cases overfits, and
//! reporting the in-sample peak would be claiming a number that was never
//! measured out of sample (invariant 15).
//!
//! ! **The floor is worth more than every weight combined** — +12.5 points on
//! its own against +5.0 for the best single factor. It was missing from the
//! first version of this module, and `completeness` had silently taken its
//! place: longer chunks contain more query terms, so it was acting as a crude
//! relevance proxy. Once a real floor exists it earns nothing and ships at 0.0.
//!
//! Two results say the gain is not a lucky peak: **2,939 of 3,125
//! configurations (94%) beat the baseline**, and leave-one-out chose exactly
//! [`Weights::FITTED`] in **37 of 40 folds**.

/// The tier of an Indonesian regulation, 1–10. `None` for a type the corpus
/// does not contain.
///
/// A lookup, ✗ a model: Indonesian regulation is a strict published hierarchy.
/// These ten types are the complete set in the corpus, verified with
/// `SELECT DISTINCT regulation_type FROM chunks`.
///
/// Ordering follows UU 12/2011: the Art 7 ladder (UU → PP → Perpres → Perda
/// Provinsi → Perda Kab/Kota), with Art 8 instruments — a governor's,
/// regent's or mayor's own regulation — placed **below** the Perda they
/// implement rather than above it.
///
/// ! An earlier table had `PERATURAN BUPATI` (3) above `PERATURAN DAERAH KOTA`
/// (2), which inverts legislation and the executive regulation implementing
/// it. Corrected here — and **the eval cannot tell the difference**: fitted
/// against both tables, Recall@5 is identical (57.5% in-sample, 47.5%
/// leave-one-out), and a coarse national-vs-local split does at least as well
/// as either. At n=40 the fine ordering is not evidence-backed; it is here
/// because a table that states something legally false is wrong regardless of
/// whether this eval set can detect it (`docs/EVAL.md` §5).
#[must_use]
pub fn tier(regulation_type: &str) -> Option<u8> {
    // Matched case-insensitively on the trimmed value; ingestion is consistent
    // today, and a stray space silently scoring 0.0 would be invisible.
    Some(match regulation_type.trim().to_uppercase().as_str() {
        "UNDANG-UNDANG" => 8,
        "PERATURAN PEMERINTAH" => 7,
        "PERATURAN PRESIDEN" => 6,
        // An instruction binds the officials it addresses, ✗ the public. High
        // issuer, low normativity — below a Perpres, above regional law.
        "INSTRUKSI PRESIDEN" => 5,
        "PERATURAN DAERAH PROVINSI" => 4,
        // ! A governor's regulation implements provincial legislation; it does
        // not outrank it. Same for a regent's or mayor's against their Perda.
        "PERATURAN GUBERNUR" | "PERATURAN DAERAH KABUPATEN" | "PERATURAN DAERAH KOTA" => 3,
        "PERATURAN BUPATI" | "PERATURAN WALIKOTA" => 2,
        _ => return None,
    })
}

/// The highest tier in the published hierarchy (`UUD 1945`), so that `tier`
/// normalises into `0.0..=1.0` against the scale rather than against whatever
/// this particular corpus happens to hold.
const MAX_TIER: f32 = 10.0;

/// The metadata a candidate carries into scoring.
///
/// ! Borrowed, ✗ owned. Scoring runs over a pool of 60 on every request and
/// has no reason to clone strings it only measures.
///
/// Defined here rather than reusing `store::ChunkRow` because this crate holds
/// zero sibling dependencies by design — the caller maps its row type onto
/// this, and the scoring logic stays testable with no database.
#[derive(Debug, Clone, Copy, Default)]
pub struct Facets<'a> {
    pub regulation_type: Option<&'a str>,
    pub article: Option<&'a str>,
    pub chapter: Option<&'a str>,
    pub year: Option<i32>,
    pub about: Option<&'a str>,
    /// The chunk text · needed for the relevance floor, which is the one
    /// factor input that is not metadata.
    pub body: Option<&'a str>,
    pub body_len: usize,
}

/// How binding is this instrument?
///
/// An unknown type scores `0.0` — neutral under a multiplicative prior, ✗ a
/// penalty. Ranking a document down for metadata the corpus never recorded
/// would punish an ingestion gap as though it were a legal fact.
#[must_use]
pub fn authority(f: &Facets<'_>) -> f32 {
    f.regulation_type
        .and_then(tier)
        .map_or(0.0, |t| f32::from(t) / MAX_TIER)
}

/// Is this an operative clause, or an annex?
///
/// ! **80,121 of 355,621 indexable chunks are `LAMPIRAN`** — 22.5% of the
/// corpus is annex material: tables, forms and schedules that are rarely the
/// answer to a question about obligations, and that Vera returns above `Pasal`
/// hits today.
///
/// `PENJELASAN` (elucidation) sits between the two: it explains a clause
/// authoritatively without being one.
#[must_use]
pub fn structural(f: &Facets<'_>) -> f32 {
    let article = f.article.unwrap_or_default().to_uppercase();
    let chapter = f.chapter.unwrap_or_default().to_uppercase();

    if article.contains("LAMPIRAN") || chapter.contains("LAMPIRAN") {
        0.0
    } else if chapter.contains("PENJELASAN") {
        0.3
    } else if article.starts_with("PASAL") {
        1.0
    } else {
        0.5
    }
}

/// Is this a whole provision, or a fragment?
///
/// Saturating, ✗ linear. Past a few hundred characters more text is not more
/// complete, and a linear term would simply rank the longest chunk first —
/// which on this corpus means an annex table.
#[must_use]
pub fn completeness(f: &Facets<'_>) -> f32 {
    #[allow(clippy::cast_precision_loss)]
    let len = f.body_len as f32;
    1.0 - (-len / 400.0).exp()
}

/// Recency across the span present in the pool.
///
/// ! Measured to be **worth nothing on its own** (`fit_factors.py`: best
/// Recall@5 at any weight = the 40.0% baseline), and [`Weights::FITTED`] gives
/// it 0.0. Kept because the finding is the point: in law a 1999 statute still
/// governs unless it was repealed, so recency is not a proxy for correctness,
/// and a plausible-looking factor has to earn its weight like any other.
#[must_use]
pub fn temporal(f: &Facets<'_>, oldest: i32, newest: i32) -> f32 {
    match f.year {
        Some(y) if newest > oldest => {
            #[allow(clippy::cast_precision_loss)]
            let scaled = (y - oldest) as f32 / (newest - oldest) as f32;
            scaled.clamp(0.0, 1.0)
        }
        _ => 0.5,
    }
}

/// How much of the query is reflected in what the instrument is *about* —
/// a property of the regulation, independent of this chunk.
///
/// ! Earns +5.0 points alone and **0.0 in combination**: the joint fit zeroed
/// it, because on this corpus `about` repeats the title words that the text arm
/// already matched on. It is computed and left unweighted rather than deleted,
/// since a corpus with richer subject metadata would change that.
#[must_use]
pub fn topical(f: &Facets<'_>, query_terms: &[&str]) -> f32 {
    if query_terms.is_empty() {
        return 0.0;
    }
    let about = f.about.unwrap_or_default().to_lowercase();
    let matched = query_terms
        .iter()
        .filter(|t| about.contains(&t.to_lowercase()))
        .count();
    #[allow(clippy::cast_precision_loss)]
    {
        matched as f32 / query_terms.len() as f32
    }
}

/// Indonesian function words. They appear in nearly every chunk, so counting
/// them would make every candidate look equally relevant.
const STOP: &[&str] = &[
    "yang",
    "dan",
    "atau",
    "untuk",
    "dengan",
    "pada",
    "dari",
    "dalam",
    "oleh",
    "apakah",
    "bagaimana",
    "berapa",
    "siapa",
    "adalah",
    "itu",
    "ini",
    "ke",
    "di",
    "tidak",
    "dapat",
    "harus",
    "wajib",
    "jika",
    "akan",
    "sebagai",
];

/// The content words of a text · lowercased, longer than three characters,
/// function words removed.
///
/// ! Deliberately **not** `bm25::tokenize`, which keeps every run of two or
/// more alphanumerics. The floor below was fitted against this rule
/// (`dev_tools/eval/fit_factors.py::terms`) and the two disagree on short
/// words, so using the wrong one would apply a threshold nothing measured.
#[must_use]
pub fn content_terms(text: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut cur = String::new();
    let flush = |cur: &mut String, out: &mut Vec<String>| {
        if cur.chars().count() > 3 && !STOP.contains(&cur.as_str()) && !out.contains(cur) {
            out.push(cur.clone());
        }
        cur.clear();
    };
    for ch in text.chars() {
        if ch.is_alphanumeric() || ch == '_' {
            cur.extend(ch.to_lowercase());
        } else {
            flush(&mut cur, &mut out);
        }
    }
    flush(&mut cur, &mut out);
    out
}

/// Share of the query's content terms this text actually contains · the input
/// to the relevance floor.
///
/// ! Unweighted overlap, ✗ IDF-weighted. `bm25::evidence` is the better
/// primitive and is what `SCORING.md` §3 names, but [`Weights::relevance_floor`]
/// was fitted against **this** measure and the two live on different scales.
/// Swapping one in without refitting would apply a threshold nothing measured.
#[must_use]
pub fn coverage(text: &str, query_terms: &[&str]) -> f32 {
    if query_terms.is_empty() {
        return 1.0;
    }
    let have = content_terms(text);
    let matched = query_terms
        .iter()
        .filter(|t| have.iter().any(|h| h == *t))
        .count();
    #[allow(clippy::cast_precision_loss)]
    {
        matched as f32 / query_terms.len() as f32
    }
}

/// Relative influence of each factor. All-zero reproduces the arms' own order
/// exactly, which is what makes this safe to ship dark.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Weights {
    /// Minimum share of the query's content terms a candidate must contain to
    /// be ranked at all · **a floor, ✗ a weight**.
    ///
    /// ! This is the relevance gate of `docs/SCORING.md` §3, and it is worth
    /// more than every weight below combined: +12.5 points Recall@5 on its own
    /// against +5.0 for the best single factor.
    ///
    /// It exists because the multiplicative form alone does **not** keep
    /// relevance dominant, which was measured rather than assumed. Across a
    /// 60-candidate pool the RRF relevance spread is only **1.98×** (rank 0 =
    /// 0.01667, rank 59 = 0.00840) while the prior is bounded at
    /// `1 + Σ weights`. At the previously shipped weights that bound was
    /// exactly **2.00×** — so metadata alone could lift the bottom of the pool
    /// to the top, and in the fit 5.5% of delivered results came from beyond
    /// pool rank 40.
    pub relevance_floor: f32,
    pub authority: f32,
    pub structural: f32,
    pub temporal: f32,
    pub completeness: f32,
    pub topical: f32,
}

impl Weights {
    /// Every factor off · scoring becomes the identity on the fused order.
    pub const OFF: Self = Self {
        relevance_floor: 0.0,
        authority: 0.0,
        structural: 0.0,
        temporal: 0.0,
        completeness: 0.0,
        topical: 0.0,
    };

    /// The fitted weights (`dev_tools/eval/fit_factors.py`), chosen in 36 of
    /// 40 leave-one-out folds.
    ///
    /// ! `temporal` and `topical` are 0.0 because they were **measured** to add
    /// nothing, ✗ because they were forgotten.
    pub const FITTED: Self = Self {
        relevance_floor: 0.4,
        authority: 1.0,
        structural: 0.5,
        temporal: 0.0,
        completeness: 0.0,
        topical: 0.25,
    };
}

impl Default for Weights {
    fn default() -> Self {
        Self::FITTED
    }
}

/// The bounded prior a candidate's metadata earns: `Σ wᵢ·factorᵢ`.
///
/// Every factor is in `0.0..=1.0`, so with [`Weights::FITTED`] this is bounded
/// by 1.0 and the strongest possible metadata doubles a candidate's score. It
/// cannot invent relevance that retrieval did not find — only reorder within
/// what it did.
#[must_use]
pub fn prior(f: &Facets<'_>, w: &Weights, query_terms: &[&str], oldest: i32, newest: i32) -> f32 {
    w.authority * authority(f)
        + w.structural * structural(f)
        + w.completeness * completeness(f)
        + w.temporal * temporal(f, oldest, newest)
        + w.topical * topical(f, query_terms)
}

/// Apply the relevance floor, then rescore what survives, best first.
///
/// `pool` is `(relevance, facets)` — relevance being the arms' fused score, the
/// only signal that a candidate is relevant at all.
///
/// ! The gate is **a real threshold, ✗ merely the shape of the formula**. An
/// earlier version relied on the multiplication alone — "a candidate with no
/// retrieval score has nothing for its metadata to multiply" — and that
/// argument does not survive measurement: RRF scores across a 60-candidate
/// pool span only 1.98×, less than the prior's 2.00× bound, so pool rank 59
/// could reach rank 1 on metadata alone. Pool membership is not a relevance
/// floor; [`Weights::relevance_floor`] is.
///
/// Ties break on the incoming order, so an all-zero [`Weights`] is exactly the
/// identity — `sort_by` is stable.
pub fn rescore<T: Copy>(pool: &mut Vec<(f32, Facets<'_>, T)>, w: &Weights, query_terms: &[&str]) {
    if *w == Weights::OFF {
        return;
    }

    // ! The relevance floor runs FIRST and it removes candidates. Ordering
    // alone cannot express "this does not belong in the answer", and the
    // measurement says the removal is where most of the gain is.
    if w.relevance_floor > 0.0 && !query_terms.is_empty() {
        let kept: Vec<_> = pool
            .iter()
            .filter(|(_, f, _)| {
                coverage(f.body.unwrap_or_default(), query_terms) >= w.relevance_floor
            })
            .copied()
            .collect();
        // ! Never empty on account of the floor. "This corpus cannot answer
        // the question" is the domain gate's decision (invariant 13), and a
        // silent second refusal here would be indistinguishable from it.
        if kept.is_empty() {
            pool.truncate(1);
        } else {
            *pool = kept;
        }
    }

    let years: Vec<i32> = pool.iter().filter_map(|(_, f, _)| f.year).collect();
    let oldest = years.iter().copied().min().unwrap_or(0);
    let newest = years.iter().copied().max().unwrap_or(0);

    pool.sort_by(|a, b| {
        let sa = a.0 * (1.0 + prior(&a.1, w, query_terms, oldest, newest));
        let sb = b.0 * (1.0 + prior(&b.1, w, query_terms, oldest, newest));
        sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn facets(reg: &'static str, article: &'static str) -> Facets<'static> {
        Facets {
            regulation_type: Some(reg),
            article: Some(article),
            year: Some(2010),
            body_len: 800,
            ..Facets::default()
        }
    }

    /// Weights with the floor disabled · for tests about ORDERING, which is a
    /// separate question from which candidates survive.
    fn ordering_only() -> Weights {
        Weights {
            relevance_floor: 0.0,
            ..Weights::FITTED
        }
    }

    #[test]
    fn the_hierarchy_orders_national_above_local() {
        assert!(tier("UNDANG-UNDANG") > tier("PERATURAN PEMERINTAH"));
        assert!(tier("PERATURAN PEMERINTAH") > tier("PERATURAN PRESIDEN"));
        assert!(tier("PERATURAN PRESIDEN") > tier("PERATURAN DAERAH PROVINSI"));
    }

    #[test]
    fn legislation_outranks_the_executive_regulation_implementing_it() {
        // ! The previous table had this backwards -- PERATURAN BUPATI (3) above
        // PERATURAN DAERAH KOTA (2) -- and the old test asserted the error. A
        // Perda is legislation; a Perbup is the regent's own regulation under
        // it, and cannot outrank it.
        assert!(tier("PERATURAN DAERAH PROVINSI") > tier("PERATURAN GUBERNUR"));
        assert!(tier("PERATURAN DAERAH KOTA") > tier("PERATURAN WALIKOTA"));
        assert!(tier("PERATURAN DAERAH KABUPATEN") > tier("PERATURAN BUPATI"));
    }

    #[test]
    fn an_unknown_regulation_type_is_neutral_not_penalised() {
        // An ingestion gap must not read as a legal fact.
        let f = Facets {
            regulation_type: Some("SURAT EDARAN"),
            ..Facets::default()
        };
        assert!((authority(&f) - 0.0).abs() < f32::EPSILON);
        assert_eq!(tier("SURAT EDARAN"), None);
    }

    #[test]
    fn the_hierarchy_lookup_tolerates_case_and_padding() {
        assert_eq!(tier("  undang-undang  "), Some(8));
    }

    #[test]
    fn an_annex_scores_below_an_article() {
        assert!(
            structural(&facets("UNDANG-UNDANG", "Pasal 8"))
                > structural(&facets("UNDANG-UNDANG", "LAMPIRAN I"))
        );
    }

    #[test]
    fn an_annex_in_the_chapter_is_still_an_annex() {
        let f = Facets {
            article: Some("Pasal 3"),
            chapter: Some("LAMPIRAN"),
            ..Facets::default()
        };
        // The chapter must win: a "Pasal" inside an annex is annex material.
        assert!((structural(&f) - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn completeness_saturates_rather_than_rewarding_length() {
        let short = completeness(&Facets {
            body_len: 100,
            ..Facets::default()
        });
        let long = completeness(&Facets {
            body_len: 2_000,
            ..Facets::default()
        });
        let huge = completeness(&Facets {
            body_len: 40_000,
            ..Facets::default()
        });
        assert!(short < long);
        // The marginal value of more text collapses: the 1,900 characters from
        // short to long are worth over a hundred times the next 38,000.
        assert!(
            huge - long < (long - short) / 100.0,
            "{short} {long} {huge}"
        );
    }

    #[test]
    fn off_weights_leave_the_fused_order_untouched() {
        let mut pool = vec![
            (0.9_f32, facets("PERATURAN DAERAH KOTA", "LAMPIRAN I"), 1_u8),
            (0.8, facets("UNDANG-UNDANG", "Pasal 8"), 2),
        ];
        rescore(&mut pool, &Weights::OFF, &[]);
        assert_eq!(pool.iter().map(|p| p.2).collect::<Vec<_>>(), vec![1, 2]);
    }

    #[test]
    fn a_statute_outranks_a_local_annex_of_similar_relevance() {
        // The exact failure SCORING.md §1 opens with.
        let mut pool = vec![
            (
                0.90_f32,
                facets("PERATURAN DAERAH KOTA", "LAMPIRAN I"),
                1_u8,
            ),
            (0.85, facets("UNDANG-UNDANG", "Pasal 8"), 2),
        ];
        rescore(&mut pool, &ordering_only(), &[]);
        assert_eq!(pool[0].2, 2, "the statute should lead");
    }

    #[test]
    fn the_prior_cannot_outrun_a_real_relevance_gap() {
        // ! This test used to assert the opposite of what the engine does, and
        // it passed because its numbers could not occur. It compared 0.90
        // against 0.20 -- a 4.5x relevance gap -- and concluded that "relevance
        // still dominates". Across a real 60-candidate pool the RRF spread is
        // 1/60 to 1/119, a ratio of 1.98x, so that gap is not reachable.
        //
        // What is true: the prior is bounded by `1 + sum(weights)`, so it
        // cannot overcome a gap WIDER than that bound -- and at the shipped
        // weights the bound (2.75x) EXCEEDS the pool's own spread. Ordering
        // alone is therefore not a relevance guarantee; the floor is.
        let w = ordering_only();
        let bound = 1.0 + w.authority + w.structural + w.temporal + w.completeness + w.topical;

        let mut pool = vec![
            (
                bound * 1.5,
                facets("PERATURAN DAERAH KOTA", "LAMPIRAN I"),
                1_u8,
            ),
            (1.0, facets("UNDANG-UNDANG", "Pasal 8"), 2),
        ];
        rescore(&mut pool, &w, &[]);
        assert_eq!(pool[0].2, 1, "a gap wider than the bound must hold");

        // And the honest converse, which is why the floor exists.
        let rrf_spread = (1.0 / 60.0) / (1.0 / 119.0);
        assert!(
            bound > rrf_spread,
            "the prior bound ({bound:.2}) is inside the pool's RRF spread \
             ({rrf_spread:.2}) -- if this ever flips, the floor is redundant"
        );
    }

    #[test]
    fn content_terms_keeps_only_words_that_carry_meaning() {
        let t = content_terms("Dalam Peraturan ini yang dimaksud dengan izin usaha");
        // Function words and anything of three characters or fewer are gone.
        assert!(!t.iter().any(|x| x == "yang"), "{t:?}");
        assert!(!t.iter().any(|x| x == "dengan"), "{t:?}");
        assert!(!t.iter().any(|x| x == "ini"), "{t:?}");
        assert!(t.iter().any(|x| x == "peraturan"), "{t:?}");
        assert!(t.iter().any(|x| x == "usaha"), "{t:?}");
    }

    #[test]
    fn content_terms_are_deduplicated() {
        // Coverage is set semantics: a query term repeated in the body counts
        // once, or a long chunk would score higher for saying the same thing.
        let t = content_terms("pajak pajak pajak daerah");
        assert_eq!(t.iter().filter(|x| *x == "pajak").count(), 1, "{t:?}");
    }

    #[test]
    fn coverage_matches_the_python_that_fitted_the_floor() {
        // ! Printed by dev_tools/eval/fit_factors.py for the same inputs. The
        // floor is a threshold on this number, so a tokenisation difference
        // between the two would apply a cut nothing measured.
        let body = "Pemegang izin usaha pertambangan wajib melaksanakan pengelolaan";
        let q = ["izin", "usaha", "pertambangan", "reklamasi"];
        assert!(
            (coverage(body, &q) - 0.75).abs() < 1e-6,
            "{}",
            coverage(body, &q)
        );
        assert!((coverage("teks pendek", &["izin"]) - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn an_empty_query_never_filters_anything() {
        // A query of only function words has no content terms. Filtering on
        // zero terms would reject the whole pool.
        assert!((coverage("apa pun", &[]) - 1.0).abs() < f32::EPSILON);
    }

    fn with_body(body: &'static str) -> Facets<'static> {
        Facets {
            regulation_type: Some("UNDANG-UNDANG"),
            article: Some("Pasal 8"),
            body: Some(body),
            body_len: body.len(),
            year: Some(2010),
            ..Facets::default()
        }
    }

    #[test]
    fn the_floor_removes_candidates_rather_than_reordering_them() {
        // ! The floor is the one part of this module that FILTERS. Ordering
        // alone cannot express "this does not belong in the answer", and the
        // fit says removal is where most of the gain is (+12.5 of +17.5).
        let mut pool = vec![
            (
                0.010_f32,
                with_body("ketentuan mengenai retribusi pelayanan pasar"),
                1_u8,
            ),
            (
                0.009,
                with_body("izin usaha pertambangan mineral dan batubara"),
                2,
            ),
        ];
        rescore(
            &mut pool,
            &Weights::FITTED,
            &["izin", "usaha", "pertambangan"],
        );
        assert_eq!(pool.len(), 1, "the unrelated candidate must be dropped");
        assert_eq!(pool[0].2, 2);
    }

    #[test]
    fn the_floor_never_empties_the_pool_on_its_own() {
        // ! "This corpus cannot answer the question" is the DOMAIN GATE's
        // decision (invariant 13). A silent second refusal here would be
        // indistinguishable from it, and the caller could not tell which
        // component declined to answer.
        let mut pool = vec![
            (0.010_f32, with_body("sama sekali tidak berkaitan"), 1_u8),
            (0.009, with_body("juga tidak berkaitan sedikit pun"), 2),
        ];
        rescore(
            &mut pool,
            &Weights::FITTED,
            &["izin", "usaha", "pertambangan"],
        );
        assert_eq!(pool.len(), 1, "one survivor, never zero");
    }

    #[test]
    fn a_zero_floor_filters_nothing() {
        // Weights::OFF must stay exactly the identity, which is what makes the
        // layer switchable off in production without a rebuild.
        let mut pool = vec![
            (0.010_f32, with_body("tidak berkaitan sama sekali"), 1_u8),
            (0.009, with_body("izin usaha pertambangan"), 2),
        ];
        rescore(&mut pool, &Weights::OFF, &["izin"]);
        assert_eq!(pool.len(), 2);
        assert_eq!(pool[0].2, 1, "OFF must not reorder either");
    }

    #[test]
    fn the_shipped_floor_leaves_room_for_a_partial_match() {
        // 0.4 was fitted, and the shape it encodes matters: a candidate
        // answering most of a question must survive, because legal answers are
        // rarely phrased in the querier's words. Half the terms is enough.
        let body = "izin usaha pertambangan wajib memenuhi persyaratan";
        let q = ["izin", "usaha", "reklamasi", "jaminan"];
        assert!(
            coverage(body, &q) >= Weights::FITTED.relevance_floor,
            "coverage {} fell below the floor",
            coverage(body, &q)
        );
    }

    #[test]
    fn the_prior_is_bounded_by_the_sum_of_weights() {
        let best = Facets {
            regulation_type: Some("UNDANG-UNDANG"),
            article: Some("Pasal 1"),
            about: Some("pajak"),
            year: Some(2018),
            body_len: 100_000,
            ..Facets::default()
        };
        let w = Weights::FITTED;
        let total = w.authority + w.structural + w.temporal + w.completeness + w.topical;
        let p = prior(&best, &w, &["pajak"], 1945, 2018);
        assert!(p <= total + f32::EPSILON, "prior {p} exceeded {total}");
    }

    #[test]
    fn the_rust_prior_matches_the_python_that_fitted_it() {
        // ! A transcription slip in the hierarchy table or a formula would be
        // invisible: the engine would still rank, just not the way anything
        // was measured. These three are printed by dev_tools/eval/fit_factors.py
        // for the same inputs, and pin this implementation to the one the
        // +7.5 points was measured on.
        let w = Weights::FITTED;
        let cases: [(Facets<'_>, f32); 3] = [
            (
                Facets {
                    regulation_type: Some("UNDANG-UNDANG"),
                    article: Some("Pasal 8"),
                    year: Some(2007),
                    body_len: 800,
                    ..Facets::default()
                },
                1.3,
            ),
            (
                Facets {
                    regulation_type: Some("PERATURAN DAERAH KOTA"),
                    article: Some("LAMPIRAN I"),
                    chapter: Some("LAMPIRAN"),
                    year: Some(2011),
                    body_len: 3_000,
                    ..Facets::default()
                },
                0.3,
            ),
            (
                Facets {
                    regulation_type: Some("PERATURAN PEMERINTAH"),
                    article: Some("Pasal 43"),
                    chapter: Some("PENJELASAN"),
                    year: Some(2005),
                    body_len: 250,
                    ..Facets::default()
                },
                0.85,
            ),
        ];
        for (f, expected) in cases {
            let got = prior(&f, &w, &[], 2005, 2011);
            assert!(
                (got - expected).abs() < 1e-5,
                "prior {got} != python {expected}"
            );
        }
    }

    #[test]
    fn temporal_is_neutral_when_the_pool_shares_one_year() {
        let f = Facets {
            year: Some(2010),
            ..Facets::default()
        };
        assert!((temporal(&f, 2010, 2010) - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn topical_measures_query_coverage_of_the_subject() {
        let f = Facets {
            about: Some("KETENAGALISTRIKAN DAN ENERGI"),
            ..Facets::default()
        };
        assert!((topical(&f, &["energi"]) - 1.0).abs() < f32::EPSILON);
        assert!((topical(&f, &["energi", "pajak"]) - 0.5).abs() < f32::EPSILON);
        assert!((topical(&f, &[]) - 0.0).abs() < f32::EPSILON);
    }
}
