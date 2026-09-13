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
//! | text arm, no factors | 40.0% |
//! | best in-sample weights | 57.5% |
//! | **leave-one-out** | **47.5%** |
//!
//! ! The honest number is the leave-one-out one: **+7.5 points**, ✗ +17.5.
//! Picking the maximum of 625 weight combinations on 40 cases overfits, and
//! reporting the in-sample peak would be claiming a number that was never
//! measured out of sample (invariant 15).
//!
//! Two results say the gain is real rather than a lucky peak: **539 of those
//! 625 combinations (86%) beat the baseline**, median 50.0% — the whole weight
//! space is better than no factors — and leave-one-out chose exactly
//! [`Weights::FITTED`] in **36 of 40 folds**.

/// The tier of an Indonesian regulation, 1–10. `None` for a type the corpus
/// does not contain.
///
/// A lookup, ✗ a model: Indonesian regulation is a strict published hierarchy.
/// These ten types are the complete set in the corpus, verified with
/// `SELECT DISTINCT regulation_type FROM chunks`.
#[must_use]
pub fn tier(regulation_type: &str) -> Option<u8> {
    // Matched case-insensitively on the trimmed value; ingestion is consistent
    // today, and a stray space silently scoring 0.0 would be invisible.
    Some(match regulation_type.trim().to_uppercase().as_str() {
        "UNDANG-UNDANG" => 8,
        "PERATURAN PEMERINTAH" => 6,
        "PERATURAN PRESIDEN" | "INSTRUKSI PRESIDEN" => 5,
        "PERATURAN GUBERNUR" => 4,
        "PERATURAN BUPATI" | "PERATURAN WALIKOTA" | "PERATURAN DAERAH PROVINSI" => 3,
        "PERATURAN DAERAH KABUPATEN" | "PERATURAN DAERAH KOTA" => 2,
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

/// Relative influence of each factor. All-zero reproduces the arms' own order
/// exactly, which is what makes this safe to ship dark.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Weights {
    pub authority: f32,
    pub structural: f32,
    pub temporal: f32,
    pub completeness: f32,
    pub topical: f32,
}

impl Weights {
    /// Every factor off · scoring becomes the identity on the fused order.
    pub const OFF: Self = Self {
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
        authority: 0.5,
        structural: 0.25,
        temporal: 0.0,
        completeness: 0.25,
        topical: 0.0,
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

/// Rescore a fused pool in place, best first.
///
/// `pool` is `(relevance, facets)` — relevance being the arms' fused score, the
/// only signal that a candidate is relevant at all.
///
/// ! The **relevance gate of `docs/SCORING.md` §3 is structural here, ✗ a
/// threshold**: multiplying by relevance means a candidate with no retrieval
/// score has nothing for its metadata to multiply. Membership of the pool is
/// the floor, and every member earned it from an arm.
///
/// Ties break on the incoming order, so an all-zero [`Weights`] is exactly the
/// identity — `sort_by` is stable.
pub fn rescore<T: Copy>(pool: &mut [(f32, Facets<'_>, T)], w: &Weights, query_terms: &[&str]) {
    if *w == Weights::OFF {
        return;
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

    #[test]
    fn the_hierarchy_orders_national_above_local() {
        assert!(tier("UNDANG-UNDANG") > tier("PERATURAN PEMERINTAH"));
        assert!(tier("PERATURAN PEMERINTAH") > tier("PERATURAN BUPATI"));
        assert!(tier("PERATURAN BUPATI") > tier("PERATURAN DAERAH KOTA"));
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
        rescore(&mut pool, &Weights::FITTED, &[]);
        assert_eq!(pool[0].2, 2, "the statute should lead");
    }

    #[test]
    fn metadata_cannot_rescue_a_candidate_retrieval_ranked_far_below() {
        // Invariant 9: authority without relevance is not a result. With the
        // fitted weights the prior is bounded by 1.0, so it can at most double
        // a score -- never close a 3x relevance gap.
        let mut pool = vec![
            (
                0.90_f32,
                facets("PERATURAN DAERAH KOTA", "LAMPIRAN I"),
                1_u8,
            ),
            (0.20, facets("UNDANG-UNDANG", "Pasal 8"), 2),
        ];
        rescore(&mut pool, &Weights::FITTED, &[]);
        assert_eq!(pool[0].2, 1, "relevance must still dominate");
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
                0.866_166,
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
                0.349_862,
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
                0.491_185,
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
