//! Corpus profile · the lexical shape of a source, measured once at ingest.
//!
//! ! `FACTORS.md` §5b calls these **corpus statistics**: properties of the
//! collection as a whole, ✗ of any row. They are what decides whether a number
//! measured on one source transfers to another — and the benchmark has already
//! been burnt once by not having them. `METRICS.md` §2.3 records BM25 at 72% of
//! query time on the synthetic fixture, with the caveat that the fixture's
//! vocabulary is 50 words, so an 8-term `OR` matches a large fraction of the
//! corpus. That caveat was written by hand, from memory, after the measurement.
//! Profiling at ingest makes it a **stored number attached to the corpus**, so
//! the caveat travels with the data instead of living in a document.
//!
//! ! Document frequency, ✗ term frequency. BM25 selectivity is `ln(N/df)`: what
//! matters is how many *rows* a term can reach, not how often it is repeated
//! inside one. A term appearing 500 times in a single row is perfectly
//! selective; a term appearing once in every row is worthless. Only `df`
//! distinguishes them.
//!
//! ! Zero I/O, like everything in `vera-core`. The builder is fed rows by
//! whoever has them (`vera-index` at build time), streams, and holds only the
//! vocabulary — so profiling a 750K-row corpus costs a hash map, ✗ a second copy
//! of the corpus.

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

/// What a corpus looks like lexically · stored in corpus metadata at build time.
///
/// Every field exists to answer one question: **will a retrieval number measured
/// here mean anything anywhere else?**
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CorpusProfile {
    pub rows: usize,
    /// Distinct `source_url`s · rows per document says how much a `same_document`
    /// traversal will return, and whether chunking was aggressive.
    pub distinct_documents: usize,

    // ── vocabulary ───────────────────────────────────────────────────────────
    /// Distinct lowercase alphanumeric tokens across the corpus.
    pub vocabulary: usize,
    pub tokens_total: usize,
    /// Share of the vocabulary occurring in exactly one row.
    ///
    /// ! Real text is ~40–60% hapax. A corpus near 0 has a closed vocabulary,
    /// which is the single strongest sign that its BM25 numbers are fiction.
    pub hapax_fraction: f32,
    /// Mean `ln(rows / df)` over the **vocabulary** · one entry per distinct
    /// term, however rare.
    ///
    /// ! Reads high on any corpus with a long tail, and is therefore **not** the
    /// number that predicts BM25 cost. Measured on a Zipfian fixture it came
    /// back at 8.7 — "the average term reaches 0.02% of rows" — while an actual
    /// 8-term query drawn from a document reached **more rows than the corpus
    /// has**. Both are true: the tail dominates the type average, and queries
    /// never draw from the tail. Use [`mean_idf_weighted`](Self::mean_idf_weighted).
    pub mean_idf: f32,
    /// Mean `ln(rows / df)` weighted by `df` · the expected selectivity of a
    /// term drawn by **occurrence**, which is how a query draws.
    ///
    /// ! **This is the number `METRICS.md` §2.3 is about**, and the type-level
    /// average is the trap next to it. A query built from a document's words
    /// samples terms roughly in proportion to how often they occur, so it is
    /// dominated by the head — and in a Zipfian corpus the head is in nearly
    /// every row. Low here means an OR over query terms touches most of the
    /// corpus no matter how large the vocabulary is, so BM25 latency scales with
    /// corpus size and the "inverted indexes prune" argument does not hold.
    #[serde(default)]
    pub mean_idf_weighted: f32,
    /// Fraction of rows the single commonest term appears in.
    pub top_term_row_share: f32,
    /// The commonest terms and the share of rows each reaches, most common
    /// first · bounded to the head.
    ///
    /// ! Stored because the *diagnosis* needs specific terms, ✗ an aggregate. A
    /// mean says keyword search is slow; this says which terms are dragging it,
    /// and it is the data any future selectivity-aware term capping would need
    /// (`METRICS.md` §2.3 names term capping as one of the three responses).
    #[serde(default)]
    pub head_terms: Vec<(String, f32)>,
    /// Least-squares slope of `ln(df)` against `ln(rank)` over the head of the
    /// distribution.
    ///
    /// ! Natural language sits near **−1** (Zipf's law). A flat slope (≈0) means
    /// every term is equally common — a uniform vocabulary, where no query term
    /// prunes anything and the inverted index degenerates into a full scan.
    pub zipf_slope: f32,

    // ── document length ──────────────────────────────────────────────────────
    pub mean_doc_tokens: f32,
    pub p50_doc_tokens: usize,
    pub p95_doc_tokens: usize,

    // ── identifiers ──────────────────────────────────────────────────────────
    /// Share of rows carrying a canonical identifier.
    ///
    /// ! The reach of the routing bypass (`LOOPHOLES.md` §1). At 2% of rows, the
    /// global exact-identifier net covers 2% of the corpus and no more —
    /// exact-match recall can only be measured over the rows this counts.
    pub identifier_density: f32,
    pub distinct_identifiers: usize,
}

impl CorpusProfile {
    /// Expected share of the corpus a `term_count`-term `OR` query reaches.
    ///
    /// ! The quantity BM25 latency actually tracks, and the one an aggregate
    /// IDF is easy to mistake for. Vera ORs **every** query token
    /// (`fts_match_expression`), so the cost is the union of the postings lists,
    /// and one head term in the query puts most of the corpus in that union
    /// however selective the other seven are. `1 − (1 − p)^n` under the
    /// occurrence-weighted per-term reach `p`; approximate, and it does not need
    /// to be better than approximate to separate "prunes hard" from "scans
    /// everything".
    #[must_use]
    pub fn expected_query_reach(&self, term_count: usize) -> f32 {
        let per_term = (-self.mean_idf_weighted).exp().clamp(0.0, 1.0);
        1.0 - (1.0 - per_term).powi(i32::try_from(term_count).unwrap_or(8))
    }

    /// Whether keyword measurements taken on this corpus can be believed.
    ///
    /// ! Gated on the **occurrence-weighted** selectivity, ✗ the type average.
    /// An earlier version used the type average and would have passed a corpus
    /// whose queries reach every row — the exact failure it exists to catch —
    /// because a long rare tail lifts the type average while queries draw
    /// entirely from the head.
    ///
    /// A judgement, deliberately, and a loose one: the failure being caught is a
    /// keyword half that degenerates into a full scan, not a corpus 10% off
    /// ideal.
    #[must_use]
    pub fn keyword_metrics_are_representative(&self) -> bool {
        self.vocabulary >= 1_000
            && self.zipf_slope <= -0.5
            && self.expected_query_reach(8) <= 0.25
    }

    /// One-line reason the corpus fails [`Self::keyword_metrics_are_representative`].
    #[must_use]
    pub fn keyword_caveat(&self) -> Option<String> {
        if self.keyword_metrics_are_representative() {
            return None;
        }
        Some(format!(
            "vocabulary {} terms · Zipf slope {:.2} · weighted IDF {:.2} → an 8-term OR \
             reaches ~{:.0}% of rows{} · BM25 latency measured here describes a scan, \
             ✗ an index lookup, and does not transfer",
            self.vocabulary,
            self.zipf_slope,
            self.mean_idf_weighted,
            self.expected_query_reach(8) * 100.0,
            match self.head_terms.first() {
                Some((term, share)) =>
                    format!(" (commonest term '{term}' alone is in {:.0}%)", share * 100.0),
                None => String::new(),
            }
        ))
    }
}

/// Streaming profiler. Feed it every row once, then [`finish`](Self::finish).
#[derive(Debug, Default)]
pub struct ProfileBuilder {
    rows: usize,
    tokens_total: usize,
    /// term → number of *rows* it appears in.
    document_frequency: HashMap<String, u32>,
    doc_tokens: Vec<u32>,
    documents: HashSet<String>,
    identifiers: HashSet<String>,
    rows_with_identifier: usize,
    /// Reused across rows so a corpus-sized profile does not allocate per row.
    seen: HashSet<String>,
}

impl ProfileBuilder {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record one row.
    ///
    /// `document` is whatever groups chunks into a source document — the
    /// `source_url` in this schema.
    pub fn observe(&mut self, body: &str, document: &str, identifier: Option<&str>) {
        self.rows += 1;
        self.documents.insert(document.to_owned());
        if let Some(id) = identifier {
            self.rows_with_identifier += 1;
            self.identifiers.insert(id.to_owned());
        }

        self.seen.clear();
        let mut length = 0u32;
        for token in tokenize(body) {
            length += 1;
            self.tokens_total += 1;
            // ! Counted once per row · this is document frequency, not term
            // frequency. See the module note.
            if !self.seen.contains(&token) {
                *self.document_frequency.entry(token.clone()).or_insert(0) += 1;
                self.seen.insert(token);
            }
        }
        self.doc_tokens.push(length);
    }

    /// Collapse the observations into a profile.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn finish(mut self) -> CorpusProfile {
        let rows = self.rows;
        let mut by_frequency: Vec<(String, u32)> = self.document_frequency.drain().collect();
        // Ties broken by term so a profile is reproducible across runs.
        by_frequency.sort_unstable_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        let frequencies: Vec<u32> = by_frequency.iter().map(|(_, df)| *df).collect();

        let vocabulary = frequencies.len();
        let hapax = frequencies.iter().filter(|f| **f == 1).count();

        let idf = |df: u32| (rows as f32 / df as f32).ln();
        let mean_idf = if vocabulary == 0 || rows == 0 {
            0.0
        } else {
            frequencies.iter().map(|df| idf(*df)).sum::<f32>() / vocabulary as f32
        };
        // ! Weighted by df, so a term is counted as often as it is met. See
        // `mean_idf_weighted`: this is what a query samples, and the unweighted
        // average is not.
        let total_df: f64 = frequencies.iter().map(|df| f64::from(*df)).sum();
        let mean_idf_weighted = if total_df <= 0.0 {
            0.0
        } else {
            #[allow(clippy::cast_possible_truncation)]
            {
                (frequencies
                    .iter()
                    .map(|df| f64::from(*df) * f64::from(idf(*df)))
                    .sum::<f64>()
                    / total_df) as f32
            }
        };
        let head_terms: Vec<(String, f32)> = by_frequency
            .iter()
            .take(32)
            .map(|(term, df)| {
                (
                    term.clone(),
                    if rows == 0 { 0.0 } else { *df as f32 / rows as f32 },
                )
            })
            .collect();

        self.doc_tokens.sort_unstable();
        let percentile = |p: f64| -> usize {
            if self.doc_tokens.is_empty() {
                return 0;
            }
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let i = ((p * self.doc_tokens.len() as f64).ceil() as usize).saturating_sub(1);
            self.doc_tokens[i.min(self.doc_tokens.len() - 1)] as usize
        };

        CorpusProfile {
            rows,
            distinct_documents: self.documents.len(),
            vocabulary,
            tokens_total: self.tokens_total,
            hapax_fraction: if vocabulary == 0 {
                0.0
            } else {
                hapax as f32 / vocabulary as f32
            },
            mean_idf,
            mean_idf_weighted,
            top_term_row_share: head_terms.first().map_or(0.0, |(_, share)| *share),
            head_terms,
            zipf_slope: zipf_slope(&frequencies),
            mean_doc_tokens: if self.doc_tokens.is_empty() {
                0.0
            } else {
                self.tokens_total as f32 / self.doc_tokens.len() as f32
            },
            p50_doc_tokens: percentile(0.50),
            p95_doc_tokens: percentile(0.95),
            identifier_density: if rows == 0 {
                0.0
            } else {
                self.rows_with_identifier as f32 / rows as f32
            },
            distinct_identifiers: self.identifiers.len(),
        }
    }
}

/// Lowercase alphanumeric runs · the same split
/// `vera_store::sqlite::fts_match_expression` applies to a query.
///
/// ! Deliberately identical to the query-side tokenizer. Profiling with a
/// different splitter than the one that builds MATCH expressions would report a
/// selectivity the search engine never sees.
fn tokenize(text: &str) -> impl Iterator<Item = String> + '_ {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(str::to_lowercase)
}

/// Least-squares slope of `ln(df)` on `ln(rank)`, over the head of the
/// distribution.
///
/// ! The head only. Zipf's law describes the head well and the tail badly — past
/// a few thousand ranks everything has `df` 1 or 2 and the flat tail drags the
/// fitted slope toward zero, which would make every corpus look uniform.
#[allow(clippy::cast_precision_loss)]
fn zipf_slope(sorted_desc: &[u32]) -> f32 {
    let n = sorted_desc.len().min(1_000);
    if n < 8 {
        return 0.0;
    }
    let points: Vec<(f64, f64)> = (0..n)
        .map(|i| ((i as f64 + 1.0).ln(), f64::from(sorted_desc[i]).max(1.0).ln()))
        .collect();
    let mean_x = points.iter().map(|(x, _)| x).sum::<f64>() / n as f64;
    let mean_y = points.iter().map(|(_, y)| y).sum::<f64>() / n as f64;
    let covariance: f64 = points
        .iter()
        .map(|(x, y)| (x - mean_x) * (y - mean_y))
        .sum();
    let variance: f64 = points.iter().map(|(x, _)| (x - mean_x).powi(2)).sum();
    if variance.abs() < f64::EPSILON {
        return 0.0;
    }
    #[allow(clippy::cast_possible_truncation)]
    {
        (covariance / variance) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A corpus where every row draws from the same tiny closed vocabulary ·
    /// the synthetic fixture's failure mode, in miniature.
    fn uniform_corpus(rows: usize) -> CorpusProfile {
        let mut b = ProfileBuilder::new();
        let vocab = ["wajib", "pajak", "sanksi", "ketentuan", "umum"];
        for i in 0..rows {
            let body = vocab.join(" ");
            b.observe(&body, &format!("doc-{}", i % 10), None);
        }
        b.finish()
    }

    /// A corpus with a Zipfian vocabulary · what real text looks like.
    fn zipfian_corpus(rows: usize) -> CorpusProfile {
        let mut b = ProfileBuilder::new();
        for i in 0..rows {
            // Term j appears in roughly rows/j documents.
            let mut body = String::new();
            for j in 1..=40usize {
                if i % j == 0 {
                    body.push_str(&format!("t{j} "));
                }
            }
            // A term unique to this row · the hapax tail.
            body.push_str(&format!("uniq{i}"));
            b.observe(&body, &format!("doc-{}", i / 3), Some("UU 28/2007"));
        }
        b.finish()
    }

    #[test]
    fn a_closed_vocabulary_is_flagged_as_unrepresentative() {
        // ! The whole reason this module exists. Every BM25 number measured on
        // such a corpus describes a full scan wearing an index's clothes.
        let p = uniform_corpus(500);
        assert_eq!(p.vocabulary, 5);
        assert!(!p.keyword_metrics_are_representative());
        let caveat = p.keyword_caveat().expect("must explain itself");
        assert!(caveat.contains("does not transfer"), "{caveat}");
    }

    #[test]
    fn every_term_in_every_row_means_zero_selectivity() {
        // mean IDF = ln(N/N) = 0 · the average term prunes nothing at all.
        let p = uniform_corpus(500);
        assert!(p.mean_idf.abs() < 1e-4, "{}", p.mean_idf);
        assert!((p.top_term_row_share - 1.0).abs() < 1e-4);
    }

    #[test]
    fn a_zipfian_corpus_slopes_down_and_carries_real_selectivity() {
        let p = zipfian_corpus(2_000);
        assert!(p.zipf_slope < -0.5, "slope {}", p.zipf_slope);
        assert!(p.mean_idf > 2.0, "mean idf {}", p.mean_idf);
    }

    #[test]
    fn a_long_tail_lifts_the_type_average_while_queries_still_reach_everything() {
        // ! The measurement error this module shipped once and had to fix. The
        // fixture below is Zipfian and has a big vocabulary, so `mean_idf` reads
        // high — and yet term `t1` is in every row, so any query containing it
        // touches the whole corpus. A representativeness check keyed on the type
        // average passes this corpus; keyed on the weighted average it does not,
        // and the weighted one is right.
        let p = zipfian_corpus(2_000);
        assert!(p.mean_idf > 4.0, "type average is high: {}", p.mean_idf);
        assert!(
            p.mean_idf_weighted < p.mean_idf,
            "weighted {} must sit below type {}",
            p.mean_idf_weighted,
            p.mean_idf
        );
        assert!(
            p.expected_query_reach(8) > 0.25,
            "an 8-term OR should still reach a large share of rows: {}",
            p.expected_query_reach(8)
        );
        assert!(
            !p.keyword_metrics_are_representative(),
            "a corpus whose queries reach every row is not representative"
        );
    }

    #[test]
    fn a_zipf_head_that_is_still_selective_is_the_case_that_passes() {
        // ! Both conditions have to hold, and they catch opposite fixtures. A
        // Zipfian shape whose head is in every row fails on reach; a corpus of
        // nothing but unique terms has perfect reach but a flat slope and is
        // just as unlike real text, so its BM25 numbers would be optimistic in
        // the other direction. What passes is a real head that still prunes:
        // term `t_j` here reaches at most 5% of rows.
        let mut b = ProfileBuilder::new();
        for i in 0..20_000usize {
            let mut body = format!("uniq{i}");
            for j in 1..=40usize {
                if i % (20 * j) == 0 {
                    body.push_str(&format!(" t{j}"));
                }
            }
            b.observe(&body, &format!("doc-{}", i / 4), None);
        }
        let p = b.finish();
        assert!(p.zipf_slope <= -0.5, "slope {}", p.zipf_slope);
        assert!(p.top_term_row_share < 0.10, "head share {}", p.top_term_row_share);
        assert!(
            p.expected_query_reach(8) < 0.25,
            "reach {}",
            p.expected_query_reach(8)
        );
        assert!(p.keyword_metrics_are_representative(), "{:?}", p.keyword_caveat());
    }

    #[test]
    fn the_head_terms_name_what_is_dragging_keyword_search() {
        // ! An aggregate says search is slow; the head names the culprit — and
        // is the data a selectivity-aware term cap would need.
        let mut b = ProfileBuilder::new();
        for i in 0..1_000 {
            b.observe(&format!("wajib rare{i}"), "doc", None);
        }
        let p = b.finish();
        let (term, share) = &p.head_terms[0];
        assert_eq!(term, "wajib");
        assert!((share - 1.0).abs() < 1e-4, "{share}");
        assert!(p.keyword_caveat().unwrap().contains("'wajib'"));
    }

    #[test]
    fn hapax_terms_are_counted() {
        // The `uniq{i}` term of every row occurs exactly once.
        let p = zipfian_corpus(1_000);
        assert!(p.hapax_fraction > 0.9, "{}", p.hapax_fraction);
    }

    #[test]
    fn document_frequency_counts_a_row_once_however_often_a_term_repeats() {
        // ! The distinction the module note argues for. A term hammered 100
        // times into one row reaches exactly one row, and BM25 prunes on reach.
        let mut b = ProfileBuilder::new();
        b.observe(&"pajak ".repeat(100), "doc-1", None);
        b.observe("sanksi", "doc-2", None);
        let p = b.finish();
        assert_eq!(p.vocabulary, 2);
        assert_eq!(p.tokens_total, 101, "token count still counts repeats");
        // Each term reaches 1 of 2 rows → idf = ln(2) for both.
        assert!((p.mean_idf - 2.0f32.ln()).abs() < 1e-4, "{}", p.mean_idf);
    }

    #[test]
    fn identifier_density_measures_the_reach_of_the_routing_bypass() {
        let mut b = ProfileBuilder::new();
        for i in 0..100 {
            b.observe("ketentuan umum", "doc", (i % 10 == 0).then_some("UU 28/2007"));
        }
        let p = b.finish();
        assert!((p.identifier_density - 0.10).abs() < 1e-5, "{}", p.identifier_density);
        assert_eq!(p.distinct_identifiers, 1);
    }

    #[test]
    fn document_lengths_are_reported_as_percentiles_not_a_mean_alone() {
        // ! BM25 length-normalizes, so a corpus of mixed-length chunks scores
        // differently than its mean suggests. A mean of 50 over lengths of
        // {1, 99} is the same mean and a different corpus.
        let mut b = ProfileBuilder::new();
        for i in 0..100 {
            let n = if i < 95 { 10 } else { 500 };
            b.observe(&"kata ".repeat(n), "doc", None);
        }
        let p = b.finish();
        assert_eq!(p.p50_doc_tokens, 10);
        assert_eq!(p.p95_doc_tokens, 10);
        assert!(p.mean_doc_tokens > 10.0, "the tail moves the mean: {}", p.mean_doc_tokens);
    }

    #[test]
    fn an_empty_corpus_profiles_to_zeroes_rather_than_dividing_by_zero() {
        let p = ProfileBuilder::new().finish();
        assert_eq!(p.rows, 0);
        assert_eq!(p.vocabulary, 0);
        assert!(p.mean_idf.abs() < f32::EPSILON);
        assert!(p.zipf_slope.abs() < f32::EPSILON);
        assert!(!p.keyword_metrics_are_representative());
    }

    #[test]
    fn tokenization_matches_what_the_keyword_half_will_see() {
        // ! "UU 28/2007" must split the same way here and in the FTS5 MATCH
        // expression, or the profile describes a vocabulary the search engine
        // never queries.
        let got: Vec<String> = tokenize("Wajib Pajak · UU 28/2007").collect();
        assert_eq!(got, ["wajib", "pajak", "uu", "28", "2007"]);
    }

    #[test]
    fn a_profile_survives_a_json_round_trip() {
        // It crosses the process boundary as corpus metadata.
        let p = zipfian_corpus(200);
        let back: CorpusProfile = serde_json::from_str(&serde_json::to_string(&p).unwrap()).unwrap();
        assert_eq!(back, p);
    }
}
