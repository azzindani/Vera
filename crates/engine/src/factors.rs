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
//! Measured through the real server by `dev_tools/eval/e2e.py` — one binary,
//! the weights varied per request, so every row is the product, ✗ an arm
//! (`docs/EVAL.md` §1). 44 cases:
//!
//! | | Recall@5 | Recall@10 | MRR |
//! |---|---|---|---|
//! | factors off | 50.0% | 52.3% | 0.360 |
//! | **[`Weights::FITTED`]** | **54.5%** | **59.1%** | **0.454** |
//!
//! Fitted by `fit_factors.py --pool fused` over 3,125 configurations:
//! leave-one-out **55.0%**, in-sample 60.0%. Quoting the in-sample number
//! would be claiming one never measured out of sample (invariant 15).
//!
//! ! The fit cannot pin these five numbers down. **23 configurations tie** at
//! the top in-sample score, and [`Weights::FITTED`] is one of them, chosen
//! among equals for keeping the annex penalty live — `structural` is what
//! demotes LAMPIRAN, and 22.5% of the corpus is annex material, which is a
//! property of the corpus rather than of 40 labelled cases. It also has the
//! best MRR of the tied set in situ. That is a selection criterion, ✗ an
//! out-of-sample result.
//!
//! # Why the weights are small
//!
//! The prior is bounded by `1 + Σwᵢ`, and that bound must stay **inside the
//! pool's own relevance spread**, or metadata alone can lift the last
//! candidate in the pool to first — invariant 9's exact failure mode.
//! Measured over the 40 cases, the fused pool spans `best/worst` of **2.56×
//! median** (min 1.31×, max 2.62×). `Σw = 1.0` here, so the bound is 2.00×,
//! inside the spread in 38 of 40 cases.
//!
//! The unconstrained best fit is `authority=1.0 structural=0.25
//! completeness=1.0 topical=0.5` — bound **3.75×, out-spanning the pool in 40
//! of 40 cases**. In situ it scores the *best* Recall@5 of anything measured,
//! 56.8%, and it is the worst answer on the list: Recall@10 is also 56.8%, so
//! positions six through ten find nothing, and **MRR falls to 0.382** against
//! 0.454 here. It hoists a few metadata-favoured chunks into the top five and
//! scrambles everything else. Recall@5 on n=40 cannot see that, which is why
//! the constraint is in the fit and in a test rather than in a comment.
//!
//! ! `completeness` is why the constraint is load-bearing rather than
//! decorative: it fits to 1.0 unconstrained, and it is a relevance proxy
//! wearing a factor's name — longer chunks contain more query terms. 0.0.
//!
//! # The relevance floor is 0.3, ✗ 0.4
//!
//! 0.4 was fitted against the **text arm alone**, which is what the fitter
//! reranked before it could reach an embedder. The engine does not rank the
//! text arm; it ranks an RRF pool that already contains BM25. On the pool that
//! actually ships:
//!
//! | floor | Recall@5 |
//! |---|---|
//! | 0.0 · 0.2 · **0.3** | 55.0% |
//! | 0.4 | 52.5% |
//! | 0.5 | 50.0% |
//!
//! The floor earns nothing on its own here — BM25 is a better lexical filter
//! than counting content terms, and refitting it against `bm25::evidence`
//! (the better instrument, named as the next experiment when 0.4 shipped)
//! does not change that. What it does is stay free up to 0.3 while every
//! top-scoring bounded configuration uses it, so it is kept where it costs
//! nothing and dropped from where it cost 2.5 points.
//!
/// What a corpus declares about its own labels, so the factor functions above
/// it stay general.
///
/// # Why this is data
///
/// `authority`, `structural` and the term splitter used to compile in three
/// Indonesian tables — ten `UNDANG-UNDANG`/`PERATURAN …` literals, the strings
/// `LAMPIRAN`/`PENJELASAN`/`PASAL`, and 25 Indonesian function words. Vera is
/// not a legal engine; Indonesian regulation is the corpus it was **first**
/// built for, the way `Qwen3-Embedding` is the model it was first built for.
/// Invariant 2 already settles the model: the corpus declares it and the
/// engine matches. Scoring vocabulary is the same claim and had the same
/// answer missing.
///
/// ! On a corpus these tables do not describe, every one of them fails
/// **silently and neutrally**: `authority` returns 0.0 for every row,
/// `structural` returns its default for every row, and the weights an operator
/// set do nothing at all. That is the exact shape of the `topical`-against-NULL
/// bug this project already shipped once — a weight that cannot act is
/// indistinguishable from a weight of zero, and only a check on the *input*
/// catches it. [`Vocabulary::missing_for`] is that check.
///
/// ! Also: `identity.authority` in Ravel's `id_regulation@1.0.yaml` is the
/// **same table**, already declared beside the corpus. Two copies in two repos
/// that must agree are two copies that will eventually disagree, and the copy
/// beside the corpus is the one that can be right for a corpus.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Vocabulary {
    /// Source tier per label, uppercased. Higher binds harder.
    ///
    /// Legal: `UNDANG-UNDANG` → 8. It generalises to any corpus with a source
    /// hierarchy — venue rank for papers, `STD`/`PROPOSED`/`INFORMATIONAL` for
    /// RFCs, normative versus informative for standards.
    pub authority: Vec<(String, u8)>,
    /// What [`Vocabulary::authority`] normalises against · the top of the
    /// declared scale, ✗ the top present in this particular corpus.
    pub max_tier: f32,
    /// Ordered · **first match wins**, so the most specific rule goes first.
    pub structural: Vec<StructuralRule>,
    /// Score for a chunk no [`StructuralRule`] matched. Neither promoted nor
    /// penalised: a corpus that labels nothing ranks on the other factors.
    pub structural_default: f32,
    /// Function words dropped before term overlap is measured. They occur in
    /// nearly every chunk, so counting them makes every candidate look equally
    /// relevant.
    pub stopwords: Vec<String>,
    /// Shortest term kept. Below this a token carries no discriminative power
    /// in any language this has been measured on.
    pub min_term_chars: usize,
}

/// Which stored label a [`StructuralRule`] reads.
///
/// ! Faithful to what shipped, ✗ tidied. The original checked `LAMPIRAN`
/// against article **or** chapter, `PENJELASAN` against chapter only and
/// `PASAL` against article only. Collapsing those to "check both" would move
/// results, and the shipped weights were fitted against this behaviour.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Field {
    Article,
    Chapter,
    Either,
}

/// One rule in the structural ladder: is this the operative body, or an annex?
#[derive(Debug, Clone, PartialEq)]
pub struct StructuralRule {
    /// Uppercased. The caller uppercases the stored label before comparing, so
    /// a corpus whose casing is inconsistent still matches.
    pub label: String,
    pub field: Field,
    /// `true` compares with `starts_with`, `false` with `contains`.
    ///
    /// ! Not cosmetic. `PASAL` is a prefix rule because "Pasal 9 dihapus" —
    /// an amending clause — contains the word without being one, which is the
    /// same distinction Ravel's profile makes with `marker_only`.
    pub prefix: bool,
    pub score: f32,
}

impl Vocabulary {
    /// The vocabulary of the corpus this engine was first built for.
    ///
    /// ! A **fallback, ✗ a default** — [`Vocabulary::is_builtin`] is reported
    /// at startup so an operator running a different corpus finds out before a
    /// weight silently does nothing, rather than after.
    ///
    /// The authority ladder follows UU 12/2011: the Art 7 ladder
    /// (UU → PP → Perpres → Perda Provinsi → Perda Kab/Kota), with Art 8
    /// instruments — a governor's, regent's or mayor's own regulation — placed
    /// **below** the Perda they implement.
    ///
    /// ! An earlier table had `PERATURAN BUPATI` (3) above `PERATURAN DAERAH
    /// KOTA` (2), inverting legislation and the regulation implementing it.
    /// Corrected — and **the eval cannot tell the difference**: fitted against
    /// both, Recall@5 is identical (57.5% in-sample, 47.5% leave-one-out), and
    /// a coarse national-vs-local split does at least as well. At n=40 the fine
    /// ordering is not evidence-backed; it is here because a table stating
    /// something legally false is wrong whether or not this eval set can detect
    /// it (`docs/EVAL.md` §5).
    #[must_use]
    pub fn id_regulation() -> Self {
        let tiers: &[(&str, u8)] = &[
            ("UNDANG-UNDANG", 8),
            ("PERATURAN PEMERINTAH", 7),
            ("PERATURAN PRESIDEN", 6),
            // An instruction binds the officials it addresses, ✗ the public.
            // High issuer, low normativity.
            ("INSTRUKSI PRESIDEN", 5),
            ("PERATURAN DAERAH PROVINSI", 4),
            ("PERATURAN GUBERNUR", 3),
            ("PERATURAN DAERAH KABUPATEN", 3),
            ("PERATURAN DAERAH KOTA", 3),
            ("PERATURAN BUPATI", 2),
            ("PERATURAN WALIKOTA", 2),
        ];
        Self {
            authority: tiers
                .iter()
                .map(|(k, v)| ((*k).to_owned(), *v))
                .collect(),
            // ! The top of the published hierarchy (UUD 1945 = 9, and 10 leaves
            // headroom), ✗ the top present in this corpus. Normalising against
            // what happens to be loaded would make the same chunk score
            // differently in a corpus that merely lacks a constitution.
            max_tier: 10.0,
            structural: vec![
                // ! 80,121 of 355,621 indexable chunks are LAMPIRAN — 22.5% of
                // the corpus is annex material: tables, forms and schedules
                // that are rarely the answer to a question about obligations.
                StructuralRule {
                    label: "LAMPIRAN".to_owned(),
                    field: Field::Either,
                    prefix: false,
                    score: 0.0,
                },
                // Elucidation sits between the two: it explains a clause
                // authoritatively without being one.
                StructuralRule {
                    label: "PENJELASAN".to_owned(),
                    field: Field::Chapter,
                    prefix: false,
                    score: 0.3,
                },
                StructuralRule {
                    label: "PASAL".to_owned(),
                    field: Field::Article,
                    prefix: true,
                    score: 1.0,
                },
            ],
            structural_default: 0.5,
            stopwords: STOP_ID.iter().map(|s| (*s).to_owned()).collect(),
            min_term_chars: 3,
        }
    }

    /// Whether this is the compiled-in fallback rather than something the
    /// corpus declared.
    #[must_use]
    pub fn is_builtin(&self) -> bool {
        *self == Self::id_regulation()
    }

    /// The tier of a label · `None` for one this vocabulary does not describe.
    ///
    /// Matched case-insensitively on the trimmed value; a stray space silently
    /// scoring 0.0 would be invisible.
    ///
    /// ! Longest match wins, so `PERATURAN DAERAH KABUPATEN` beats `PERATURAN
    /// DAERAH` — the same rule Ravel's profile states as "ordered longest-first".
    /// Relying on declaration order would make a correct table depend on how it
    /// was typed.
    #[must_use]
    pub fn tier(&self, label: &str) -> Option<u8> {
        let want = label.trim().to_uppercase();
        self.authority
            .iter()
            .filter(|(k, _)| want == *k || want.starts_with(k.as_str()))
            .max_by_key(|(k, _)| k.len())
            .map(|(_, t)| *t)
    }

    /// Names the factors this vocabulary cannot evaluate, given weights that
    /// ask it to.
    ///
    /// ! The guard that makes a missing table loud. A weight of 0.5 on a factor
    /// whose table is empty is not a small effect — it is **no effect**, and it
    /// looks exactly like a weight that was measured and found not to help.
    #[must_use]
    pub fn missing_for(&self, w: &Weights) -> Vec<&'static str> {
        let mut out = Vec::new();
        if w.authority > 0.0 && self.authority.is_empty() {
            out.push("authority");
        }
        if w.structural > 0.0 && self.structural.is_empty() {
            out.push("structural");
        }
        out
    }
}

/// Indonesian function words · the fallback vocabulary's stop list.
///
/// ! `dapat`, `harus`, `wajib` and `tidak` are here because they occur in
/// nearly every chunk and carry no discriminative power **as terms**. That is
/// correct for term overlap and is exactly why a separate `modality` factor is
/// worth having: the pairing "query asks for an obligation ∧ chunk states an
/// obligation" is discriminative even though neither word is.
const STOP_ID: &[&str] = &[
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
pub fn authority(f: &Facets<'_>, v: &Vocabulary) -> f32 {
    let max = if v.max_tier > 0.0 { v.max_tier } else { 1.0 };
    f.regulation_type
        .and_then(|l| v.tier(l))
        .map_or(0.0, |t| f32::from(t) / max)
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
pub fn structural(f: &Facets<'_>, v: &Vocabulary) -> f32 {
    let article = f.article.unwrap_or_default().to_uppercase();
    let chapter = f.chapter.unwrap_or_default().to_uppercase();
    let hit = |text: &str, r: &StructuralRule| {
        if r.prefix {
            text.starts_with(&r.label)
        } else {
            text.contains(&r.label)
        }
    };
    // First match wins, so the most specific rule is declared first.
    for r in &v.structural {
        let matched = match r.field {
            Field::Article => hit(&article, r),
            Field::Chapter => hit(&chapter, r),
            Field::Either => hit(&article, r) || hit(&chapter, r),
        };
        if matched {
            return r.score;
        }
    }
    v.structural_default
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
pub fn topical(f: &Facets<'_>, query_terms: &[&str], v: &Vocabulary) -> f32 {
    if query_terms.is_empty() {
        return 0.0;
    }
    // ! Whole terms, ✗ substrings. `about.contains("pajak")` is also true of
    // "PERPAJAKAN", and this weight was fitted against a token-set intersection
    // that says false. A looser rule inflates a score the fit never measured.
    let about = content_terms(f.about.unwrap_or_default(), v);
    let matched = query_terms
        .iter()
        .filter(|t| about.iter().any(|a| a == *t))
        .count();
    #[allow(clippy::cast_precision_loss)]
    {
        matched as f32 / query_terms.len() as f32
    }
}


/// The content words of a text · lowercased, longer than three characters,
/// function words removed.
///
/// ! Deliberately **not** `bm25::tokenize`, which keeps every run of two or
/// more alphanumerics. The floor below was fitted against this rule
/// (`dev_tools/eval/fit_factors.py::terms`) and the two disagree on short
/// words, so using the wrong one would apply a threshold nothing measured.
#[must_use]
pub fn content_terms(text: &str, v: &Vocabulary) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut cur = String::new();
    let flush = |cur: &mut String, out: &mut Vec<String>| {
        if cur.chars().count() > v.min_term_chars
            && !v.stopwords.iter().any(|w| w == cur)
            && !out.contains(cur)
        {
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
pub fn coverage(text: &str, query_terms: &[&str], v: &Vocabulary) -> f32 {
    if query_terms.is_empty() {
        return 1.0;
    }
    let have = content_terms(text, v);
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
    /// ! `Σw = 1.0`, so the prior is bounded at 2.00× — inside the fused
    /// pool's 2.56× median spread. That constraint is part of the fit, ✗ a
    /// coincidence of it; see the module docs.
    pub const FITTED: Self = Self {
        relevance_floor: 0.3,
        authority: 0.5,
        structural: 0.25,
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
pub fn prior(
    f: &Facets<'_>,
    w: &Weights,
    query_terms: &[&str],
    oldest: i32,
    newest: i32,
    v: &Vocabulary,
) -> f32 {
    w.authority * authority(f, v)
        + w.structural * structural(f, v)
        + w.completeness * completeness(f)
        + w.temporal * temporal(f, oldest, newest)
        + w.topical * topical(f, query_terms, v)
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
pub fn rescore<T: Copy>(
    pool: &mut Vec<(f32, Facets<'_>, T)>,
    w: &Weights,
    query_terms: &[&str],
    v: &Vocabulary,
) {
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
                coverage(f.body.unwrap_or_default(), query_terms, v) >= w.relevance_floor
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
        let sa = a.0 * (1.0 + prior(&a.1, w, query_terms, oldest, newest, v));
        let sb = b.0 * (1.0 + prior(&b.1, w, query_terms, oldest, newest, v));
        sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The corpus this engine was first built for · the tests below assert on
    /// Indonesian labels because that is the vocabulary they declare, ✗ because
    /// the engine knows them.
    fn v() -> Vocabulary {
        Vocabulary::id_regulation()
    }

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
        assert!(v().tier("UNDANG-UNDANG") > v().tier("PERATURAN PEMERINTAH"));
        assert!(v().tier("PERATURAN PEMERINTAH") > v().tier("PERATURAN PRESIDEN"));
        assert!(v().tier("PERATURAN PRESIDEN") > v().tier("PERATURAN DAERAH PROVINSI"));
    }

    #[test]
    fn legislation_outranks_the_executive_regulation_implementing_it() {
        // ! The previous table had this backwards -- PERATURAN BUPATI (3) above
        // PERATURAN DAERAH KOTA (2) -- and the old test asserted the error. A
        // Perda is legislation; a Perbup is the regent's own regulation under
        // it, and cannot outrank it.
        assert!(v().tier("PERATURAN DAERAH PROVINSI") > v().tier("PERATURAN GUBERNUR"));
        assert!(v().tier("PERATURAN DAERAH KOTA") > v().tier("PERATURAN WALIKOTA"));
        assert!(v().tier("PERATURAN DAERAH KABUPATEN") > v().tier("PERATURAN BUPATI"));
    }

    #[test]
    fn an_unknown_regulation_type_is_neutral_not_penalised() {
        // An ingestion gap must not read as a legal fact.
        let f = Facets {
            regulation_type: Some("SURAT EDARAN"),
            ..Facets::default()
        };
        assert!((authority(&f, &v()) - 0.0).abs() < f32::EPSILON);
        assert_eq!(v().tier("SURAT EDARAN"), None);
    }

    #[test]
    fn the_hierarchy_lookup_tolerates_case_and_padding() {
        assert_eq!(v().tier("  undang-undang  "), Some(8));
    }

    #[test]
    fn an_annex_scores_below_an_article() {
        assert!(
            structural(&facets("UNDANG-UNDANG", "Pasal 8"), &v())
                > structural(&facets("UNDANG-UNDANG", "LAMPIRAN I"), &v())
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
        assert!((structural(&f, &v()) - 0.0).abs() < f32::EPSILON);
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
        rescore(&mut pool, &Weights::OFF, &[], &v());
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
        rescore(&mut pool, &ordering_only(), &[], &v());
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
        rescore(&mut pool, &w, &[], &v());
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
        let t = content_terms("Dalam Peraturan ini yang dimaksud dengan izin usaha", &v());
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
        let t = content_terms("pajak pajak pajak daerah", &v());
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
            (coverage(body, &q, &v()) - 0.75).abs() < 1e-6,
            "{}",
            coverage(body, &q, &v())
        );
        assert!(coverage("teks pendek", &["izin"], &v()).abs() < f32::EPSILON);
    }

    #[test]
    fn an_empty_query_never_filters_anything() {
        // A query of only function words has no content terms. Filtering on
        // zero terms would reject the whole pool.
        assert!((coverage("apa pun", &[], &v()) - 1.0).abs() < f32::EPSILON);
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
        // alone cannot express "this does not belong in the answer".
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
            &v(),
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
            &v(),
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
        rescore(&mut pool, &Weights::OFF, &["izin"], &v());
        assert_eq!(pool.len(), 2);
        assert_eq!(pool[0].2, 1, "OFF must not reorder either");
    }

    #[test]
    fn a_floor_leaves_room_for_a_partial_match() {
        // The shape a floor has to encode, at whatever value: a candidate
        // answering most of a question must survive, because legal answers are
        // rarely phrased in the querier's words. Half the terms is enough.
        let body = "izin usaha pertambangan wajib memenuhi persyaratan";
        let q = ["izin", "usaha", "reklamasi", "jaminan"];
        assert!(
            coverage(body, &q, &v()) >= Weights::FITTED.relevance_floor,
            "coverage {} fell below the floor",
            coverage(body, &q, &v())
        );
    }

    #[test]
    fn the_shipped_weights_stay_inside_the_pool_spread() {
        // ! Invariant 9, as arithmetic. The fused pool's relevance spans
        // 2.57x median (min 1.31x) across the 40 eval cases — measured, see
        // the module docs. A prior bounded above that can reorder the pool on
        // metadata alone, which is authority without relevance.
        //
        // This is the test that fails if someone raises a weight because
        // Recall@5 went up: the unconstrained fit does score the same
        // Recall@5, and its MRR is below factors-off.
        let w = Weights::FITTED;
        let total = w.authority + w.structural + w.temporal + w.completeness + w.topical;
        assert!(
            1.0 + total <= 2.00 + f32::EPSILON,
            "prior bound {:.2}x exceeds the 2.00x the fit was constrained to",
            1.0 + total
        );
    }

    #[test]
    fn the_shipped_floor_stays_where_it_is_free() {
        // ! 0.4 cost 2.5 points on the pool the engine actually ranks; 0.3
        // costs nothing and every top-scoring bounded configuration uses it.
        // The value is fitted against the FUSED pool, ✗ the text arm — using
        // the old 0.4 would apply a threshold measured on something the
        // engine does not rank.
        assert!((Weights::FITTED.relevance_floor - 0.3).abs() < 1e-6);
    }

    #[test]
    fn hierarchy_matches_the_engine() {
        // ! The fitter carries its own copy of this table in Python, and the
        // two silently diverged once already: 9c50c87 corrected the Rust and
        // left the Python inverted, so weights were fitted against one
        // hierarchy and applied against another. Neither number moved.
        let src = include_str!("../../../dev_tools/eval/fit_factors.py");
        let body = src
            .split_once("HIERARCHY = {")
            .expect("fit_factors.py must define HIERARCHY")
            .1
            .split_once('}')
            .expect("unterminated HIERARCHY")
            .0;
        let mut seen = 0;
        for line in body.lines() {
            let Some((k, raw)) = line.trim().trim_end_matches(',').split_once(':') else {
                continue;
            };
            let name = k.trim().trim_matches('"');
            let want: u8 = raw.trim().parse().expect("tier must be an integer");
            assert_eq!(v().tier(name), Some(want), "{name} disagrees with the fitter");
            seen += 1;
        }
        assert_eq!(seen, 10, "the corpus contains ten regulation types");
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
        let p = prior(&best, &w, &["pajak"], 1945, 2018, &v());
        assert!(p <= total + f32::EPSILON, "prior {p} exceeded {total}");
    }

    #[test]
    fn the_rust_prior_matches_the_python_that_fitted_it() {
        // ! A transcription slip in the hierarchy table or a formula would be
        // invisible: the engine would still rank, just not the way anything
        // was measured. These three are printed by dev_tools/eval/fit_factors.py
        // for the same inputs, and pin this implementation to the one the
        // fit was measured on.
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
                0.65,
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
                0.15,
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
                0.425,
            ),
        ];
        for (f, expected) in cases {
            let got = prior(&f, &w, &[], 2005, 2011, &v());
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
        assert!((topical(&f, &["energi"], &v()) - 1.0).abs() < f32::EPSILON);
        assert!((topical(&f, &["energi", "pajak"], &v()) - 0.5).abs() < f32::EPSILON);
        assert!((topical(&f, &[], &v()) - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn topical_matches_whole_terms_not_substrings() {
        // ! `about.contains("pajak")` is true of "PERPAJAKAN", and the weight
        // was fitted against a token-set intersection that says false. A looser
        // rule here inflates a score nothing measured.
        let f = Facets {
            about: Some("KETENTUAN UMUM PERPAJAKAN"),
            ..Facets::default()
        };
        assert!(
            (topical(&f, &["pajak"], &v()) - 0.0).abs() < f32::EPSILON,
            "substring match leaked in: {}",
            topical(&f, &["pajak"], &v())
        );
        assert!((topical(&f, &["perpajakan"], &v()) - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn topical_ignores_a_subject_line_the_corpus_never_recorded() {
        // Weight 0.25 and a NULL column would otherwise be indistinguishable
        // from weight 0.0 — which is exactly the bug that hid here for a while.
        let f = Facets {
            about: None,
            ..Facets::default()
        };
        assert!((topical(&f, &["energi"], &v()) - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn nothing_indonesian_is_compiled_into_the_factor_functions() {
        // ! The regression this guards. `authority` and `structural` held ten
        // `UNDANG-UNDANG`/`PERATURAN ...` literals and the strings LAMPIRAN /
        // PENJELASAN / PASAL. On any corpus those tables do not describe, every
        // factor evaluated neutrally and the operator's weights did nothing --
        // silently, which is the same shape as `topical` shipping at 0.25
        // against a column the engine never selected.
        //
        // A vocabulary that knows only English labels must rank them, and must
        // NOT rank the Indonesian ones.
        let v = Vocabulary {
            authority: vec![
                ("INTERNET STANDARD".to_owned(), 5),
                ("PROPOSED STANDARD".to_owned(), 3),
            ],
            max_tier: 5.0,
            structural: vec![
                StructuralRule {
                    label: "APPENDIX".to_owned(),
                    field: Field::Either,
                    prefix: false,
                    score: 0.0,
                },
                StructuralRule {
                    label: "SECTION".to_owned(),
                    field: Field::Article,
                    prefix: true,
                    score: 1.0,
                },
            ],
            structural_default: 0.5,
            stopwords: vec!["the".to_owned(), "and".to_owned()],
            min_term_chars: 2,
        };

        let std_doc = Facets {
            regulation_type: Some("Internet Standard"),
            article: Some("Section 4.2"),
            ..Facets::default()
        };
        let appendix = Facets {
            regulation_type: Some("Proposed Standard"),
            article: Some("Appendix B"),
            ..Facets::default()
        };
        assert!(authority(&std_doc, &v) > authority(&appendix, &v));
        assert!(structural(&std_doc, &v) > structural(&appendix, &v));

        // And the Indonesian labels are now just unknown strings.
        let uu = Facets {
            regulation_type: Some("UNDANG-UNDANG"),
            article: Some("Pasal 9"),
            ..Facets::default()
        };
        assert!(
            (authority(&uu, &v) - 0.0).abs() < f32::EPSILON,
            "an unknown label is neutral, ✗ penalised"
        );
        assert!((structural(&uu, &v) - v.structural_default).abs() < f32::EPSILON);
    }

    #[test]
    fn the_term_splitter_takes_its_stop_list_from_the_corpus() {
        // Indonesian function words are not universal function words. An
        // English corpus that kept them would waste term slots, and an
        // Indonesian corpus that dropped "the" would not notice.
        let en = Vocabulary {
            stopwords: vec!["the".to_owned(), "and".to_owned()],
            min_term_chars: 2,
            ..Vocabulary::default()
        };
        let terms = content_terms("the tax and the penalty", &en);
        assert_eq!(terms, vec!["tax".to_owned(), "penalty".to_owned()]);

        // The same text under the Indonesian vocabulary keeps "the" (not a
        // stop word there) but drops it for length: min_term_chars is 3.
        let id = Vocabulary::id_regulation();
        assert!(!content_terms("the tax and the penalty", &id).contains(&"the".to_owned()));
    }

    #[test]
    fn a_weight_the_vocabulary_cannot_evaluate_is_reported() {
        // ! The guard that turns a silent no-op into a startup refusal.
        let empty = Vocabulary::default();
        assert_eq!(
            empty.missing_for(&Weights::FITTED),
            vec!["authority", "structural"]
        );
        // Zero weights ask nothing of the vocabulary, so nothing is missing.
        assert!(empty.missing_for(&Weights::OFF).is_empty());
        // The corpus that declares its tables is clean at any weight.
        assert!(
            Vocabulary::id_regulation()
                .missing_for(&Weights::FITTED)
                .is_empty()
        );
    }

    #[test]
    fn the_longest_matching_label_wins_not_the_first_declared() {
        // ! `PERATURAN DAERAH KABUPATEN` must beat `PERATURAN DAERAH`, and a
        // correct table must not depend on the order someone typed it in.
        // Ravel's profile states the same rule as "ordered longest-first".
        let v = Vocabulary {
            authority: vec![
                ("PERATURAN DAERAH".to_owned(), 3),
                ("PERATURAN DAERAH KABUPATEN".to_owned(), 7),
            ],
            max_tier: 10.0,
            ..Vocabulary::default()
        };
        assert_eq!(v.tier("PERATURAN DAERAH KABUPATEN BANDUNG"), Some(7));
        assert_eq!(v.tier("PERATURAN DAERAH"), Some(3));
    }
}
