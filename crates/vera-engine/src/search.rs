//! The query orchestrator: route → scan → fuse → hydrate.
//!
//! ! Synchronous on purpose. Everything here is CPU and disk; the only async
//! work on the query path is the embedding call, which happens *before* this and
//! is passed in as a vector. Keeping the orchestrator sync means a benchmark
//! measures retrieval, ✗ executor scheduling, and it lets the concurrency
//! ceiling live in exactly one place — the semaphore at the MCP boundary
//! (`MCP_ENGINE.md` §4).

use std::collections::HashMap;
use std::time::{Duration, Instant};

use vera_core::{
    Chunk, ComponentScores, Confidence, Config, EmbeddingSpace, ExactMatch, SearchResponse,
    SearchResult, SummaryPayload,
};
use vera_core::contract::citation_block;
use vera_embed::cosine;
use vera_store::{ChunkStore, DomainAnchor, StoreError};

use crate::fusion::{Fused, RankedList, merge_by_score, reciprocal_rank_fusion};
use crate::identifier;
use crate::routing::{self, ProbedCluster};
use crate::topk::{Scored, TopK};

#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    #[error(transparent)]
    Store(#[from] StoreError),

    #[error(transparent)]
    Space(#[from] vera_core::SpaceMismatch),

    #[error("query vector has {got} dimensions but the corpus is {expected}")]
    QueryWidth { expected: usize, got: usize },
}

/// How many layer-2 clusters to open.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Probe {
    /// Normal operation: the `n` nearest centroids.
    Nearest(usize),
    /// Every cluster in the domain · the exhaustive baseline.
    ///
    /// ! Not a serving mode. It exists so the eval harness can measure what
    /// routing *costs* in recall: routed results are compared against this.
    /// Without a ground truth to compare against, a recall number is a guess.
    Exhaustive,
}

/// Where the wall-clock went. The unit of the speed report.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StageTimings {
    pub route_domain: Duration,
    pub route_cluster: Duration,
    pub leaf_dense: Duration,
    pub leaf_keyword: Duration,
    pub exact_path: Duration,
    pub fuse: Duration,
    pub hydrate: Duration,
    pub total: Duration,
    /// Rows the dense scan actually touched · the real cost driver, and the
    /// number that shows whether routing pruned anything.
    pub rows_scanned: usize,
    pub clusters_probed: usize,
}

/// Every candidate that reached fusion, by the list it arrived on.
///
/// ! Exists to make **candidate-cap loss observable** (`METRICS.md` §3.1,
/// `LOOPHOLES.md` §7). The leaf scan keeps only `per_cluster_top_k` rows per
/// cluster, so a correct chunk ranking 51st inside its own cluster is discarded
/// before fusion — and no metric the harness had could see it. `recall@k` scores
/// against a baseline that applies the *same* cap, so both sides drop the same
/// row and recall reads 100%; `route/D` asks only whether the row sat in a
/// probed cluster, and a row that was probed and then cut counts as a routing
/// *success*. The loss is real, silent, and dialled by a knob nothing measures.
///
/// Knowing which ids survived to fusion closes that: a truth item in a probed
/// cluster but absent from here was cut by the cap, ✗ missed by routing.
///
/// ! Costs no allocation on the query path. These `String`s are moved out of the
/// candidate lists after fusion has consumed them, ✗ cloned.
#[derive(Debug, Clone, Default)]
pub struct Candidates {
    pub dense: Vec<String>,
    pub keyword: Vec<String>,
    pub exact: Vec<String>,
}

impl Candidates {
    /// Whether an id reached fusion at all, by any route.
    #[must_use]
    pub fn reached_fusion(&self, id: &str) -> bool {
        self.dense.iter().any(|c| c == id)
            || self.keyword.iter().any(|c| c == id)
            || self.exact.iter().any(|c| c == id)
    }

    #[must_use]
    pub fn total(&self) -> usize {
        self.dense.len() + self.keyword.len() + self.exact.len()
    }
}

/// A search, plus everything needed to explain and measure it.
#[derive(Debug, Clone)]
pub struct SearchOutcome {
    pub response: SearchResponse,
    pub timings: StageTimings,
    pub probed: Vec<ProbedCluster>,
    /// What fusion was given · the input side of the ranking, kept so recall
    /// loss can be attributed to routing, the cap, or fusion separately.
    pub candidates: Candidates,
}

/// The stateless query engine.
///
/// Holds the hot routing structures and nothing else. Two instances over the
/// same corpus are interchangeable, which is what makes replicas trivial.
pub struct Engine<S: ChunkStore> {
    store: S,
    config: Config,
    domains: Vec<DomainAnchor>,
    centroids: HashMap<String, Vec<vera_store::Centroid>>,
    /// Layer-1 threshold per domain, calibrated at build time unless overridden.
    thresholds: HashMap<String, f32>,
}

impl<S: ChunkStore> Engine<S> {
    /// Load the hot routing structures and verify the corpus space.
    ///
    /// ! The space check happens **here**, at startup, and fails closed. Serving
    /// one query against a mismatched corpus is already a wrong answer a human
    /// may act on (`CLAUDE.md` §7 rule 3).
    ///
    /// # Errors
    /// [`EngineError::Space`] on a corpus/engine mismatch, or a store failure.
    pub fn load(store: S, config: Config) -> Result<Self, EngineError> {
        let corpus = store.corpus_space()?;
        config.embedding.assert_matches(&corpus)?;

        let domains = store.domains()?;
        let mut centroids = HashMap::new();
        let mut thresholds = HashMap::new();
        for d in &domains {
            centroids.insert(d.id.clone(), store.centroids(&d.id)?);
            // ! Prefer what the corpus measured about itself over the config's
            // guess, and *derive* the threshold from those measurements rather
            // than read a scalar fixed at build time — so the policy can be
            // retuned without re-ingesting the corpus (`AnchorStats`).
            //
            // Precedence: an explicit operator override wins; else the recorded
            // distribution under the configured margin; else a threshold an
            // older corpus baked in; else the permissive fallback. The order
            // matters because a threshold set too tight returns a *successful*
            // empty result for every query, which no caller can tell from an
            // empty corpus.
            let threshold = match config.routing.domain_threshold {
                Some(explicit) => explicit,
                None => match store.anchor_stats(&d.id)? {
                    Some(stats) => stats.threshold_at(config.routing.threshold_margin),
                    None => store
                        .calibrated_domain_threshold(&d.id)?
                        .unwrap_or(vera_core::config::FALLBACK_DOMAIN_THRESHOLD),
                },
            };
            thresholds.insert(d.id.clone(), threshold);
        }
        Ok(Self {
            store,
            config,
            domains,
            centroids,
            thresholds,
        })
    }

    /// The layer-1 threshold actually in force for a domain, after calibration
    /// and any operator override. Surfaced for `search(dry_run)` and the bench.
    #[must_use]
    pub fn threshold_for(&self, domain_id: &str) -> f32 {
        self.thresholds
            .get(domain_id)
            .copied()
            .unwrap_or(vera_core::config::FALLBACK_DOMAIN_THRESHOLD)
    }

    #[must_use]
    pub const fn config(&self) -> &Config {
        &self.config
    }

    #[must_use]
    pub fn space(&self) -> &EmbeddingSpace {
        &self.config.embedding
    }

    #[must_use]
    pub fn store(&self) -> &S {
        &self.store
    }

    /// Total rows the corpus holds across all probed-able clusters.
    #[must_use]
    pub fn corpus_rows(&self) -> i64 {
        self.domains.iter().map(|d| d.row_count).sum()
    }

    /// Run a search for an already-embedded query.
    ///
    /// # Errors
    /// [`EngineError::QueryWidth`] if the vector does not match the corpus, or a
    /// store failure.
    pub fn search(
        &self,
        query_text: &str,
        query_vector: &[f32],
        probe: Probe,
    ) -> Result<SearchOutcome, EngineError> {
        let started = Instant::now();
        let mut timings = StageTimings::default();
        let mut progress = Vec::new();

        if query_vector.len() != self.config.embedding.dim {
            return Err(EngineError::QueryWidth {
                expected: self.config.embedding.dim,
                got: query_vector.len(),
            });
        }

        // ── Global exact-identifier path · runs BEFORE any routing ───────────
        // ! Hoisted above layer 1 deliberately. `CLAUDE.md` §7 rule 4 says an
        // exact identifier must never be gated by routing, and layer-1 domain
        // detection is routing: a query naming "UU 28/2007" whose vector fell
        // below the domain anchor threshold would return empty, and the caller
        // could not distinguish that from "no such regulation exists". Silent
        // zero-recall on a citation is the worst failure this engine can
        // produce, so the lookup happens first and unconditionally.
        let t = Instant::now();
        let identifiers = identifier::extract(query_text);
        let mut exact_matches = Vec::new();
        let mut exact_ids: Vec<Scored> = Vec::new();
        for ident in &identifiers {
            for chunk in self
                .store
                .exact_identifier(&ident.canonical, self.config.search.bm25_limit)?
            {
                exact_matches.push(ExactMatch {
                    id: chunk.id.clone(),
                    matched_on: ident.matched_on.clone(),
                });
                // Rank 1-equivalent: an exact citation hit is the strongest
                // evidence the engine has.
                exact_ids.push(Scored {
                    id: chunk.id,
                    score: 1.0,
                });
            }
        }
        timings.exact_path = t.elapsed();
        if !identifiers.is_empty() {
            progress.push(format!(
                "exact:global {} identifier(s) → {} hit(s)",
                identifiers.len(),
                exact_matches.len()
            ));
        }

        // ── Layer 1 ──────────────────────────────────────────────────────────
        let t = Instant::now();
        let detected = routing::detect_domain_with(query_vector, &self.domains, |id| {
            self.threshold_for(id)
        });
        timings.route_domain = t.elapsed();

        let Some(domain) = detected else {
            progress.push("route:layer1 no anchor above threshold".to_owned());
            timings.total = started.elapsed();

            // ! Even with no domain, exact hits are still returned. Routing
            // failing is not a reason to withhold a regulation the caller named
            // outright and the corpus demonstrably contains.
            if exact_ids.is_empty() {
                // Empty, ✗ a guess. See routing::detect_domain.
                return Ok(SearchOutcome {
                    response: SearchResponse::no_matching_domain(query_text, progress),
                    timings,
                    probed: Vec::new(),
                    candidates: Candidates::default(),
                });
            }
            let response =
                self.exact_only_response(query_text, &exact_ids, exact_matches, progress)?;
            return Ok(SearchOutcome {
                response,
                timings,
                probed: Vec::new(),
                candidates: Candidates {
                    exact: exact_ids.into_iter().map(|s| s.id).collect(),
                    ..Candidates::default()
                },
            });
        };
        progress.push(format!(
            "route:layer1 domain={} sim={:.3}",
            domain.id, domain.similarity
        ));

        // ── Layer 2 ──────────────────────────────────────────────────────────
        let t = Instant::now();
        let empty = Vec::new();
        let centroids = self.centroids.get(&domain.id).unwrap_or(&empty);
        let probed = match probe {
            Probe::Nearest(n) => routing::nearest_clusters(query_vector, centroids, n),
            Probe::Exhaustive => routing::nearest_clusters(query_vector, centroids, centroids.len()),
        };
        timings.route_cluster = t.elapsed();
        timings.clusters_probed = probed.len();
        progress.push(format!(
            "route:layer2 probing {} of {} clusters",
            probed.len(),
            centroids.len()
        ));

        // ── Layer 3 · sequential leaf scan ───────────────────────────────────
        // ! One cluster at a time. `per_cluster` holds at most
        // `per_cluster_top_k` ids and the scan itself streams — so the working
        // set does not grow with the number of clusters probed.
        let mut dense_lists: Vec<Vec<Scored>> = Vec::with_capacity(probed.len());

        for cluster in &probed {
            let t = Instant::now();
            let mut top = TopK::new(self.config.routing.per_cluster_top_k);
            let scanned = self.store.scan_cluster(cluster.cluster_id, &mut |row| {
                top.offer(row.id, cosine(query_vector, row.vector));
            })?;
            timings.rows_scanned += scanned;
            timings.leaf_dense += t.elapsed();
            dense_lists.push(top.into_ranked());
        }
        progress.push(format!("scan:layer3 {} rows", timings.rows_scanned));

        // ── Keyword half · ONE global query, ✗ one per probed cluster ────────
        // ! Routing exists because dense search has no index — a 4096-dim
        // vector cannot be indexed, so the only way to avoid scanning 100M rows
        // is to not look at them. BM25 has the opposite problem: it *is* an
        // index, and an inverted index already prunes to the matching
        // postings. Scoping it per cluster does not make it cheaper — FTS5
        // evaluates the match corpus-wide and then discards rows whose
        // cluster_id is wrong — so probing N clusters cost N full-corpus
        // keyword scans. Measured on the bench corpus this was 87% of total
        // query time, and it grows with `clusters_probed`, which is meant to be
        // the cheap dial.
        //
        // One global query is both faster and strictly higher recall: it can
        // surface a keyword match that routing pruned away, which is the same
        // guarantee the exact-identifier path provides, generalized.
        let t = Instant::now();
        let keyword: Vec<Scored> = self
            .store
            .keyword_search(query_text, None, self.config.search.bm25_limit)?
            .into_iter()
            .map(|h| Scored {
                id: h.id,
                score: h.score,
            })
            .collect();
        timings.leaf_keyword = t.elapsed();
        progress.push(format!("keyword:global {} candidates", keyword.len()));

        // ── Fusion ───────────────────────────────────────────────────────────
        let t = Instant::now();
        let candidate_cap = self.config.search.max_results * 10;
        let dense = merge_by_score(dense_lists, candidate_cap);
        let mut lists = vec![
            RankedList {
                label: "dense",
                items: &dense,
            },
            RankedList {
                label: "bm25",
                items: &keyword,
            },
        ];
        if !exact_ids.is_empty() {
            lists.push(RankedList {
                label: "exact",
                items: &exact_ids,
            });
        }
        let mut fused = reciprocal_rank_fusion(&lists, self.config.search.rrf_k);
        fused.truncate(self.config.search.max_results);
        timings.fuse = t.elapsed();
        progress.push(format!("fuse:rrf {} candidates", fused.len()));

        // ! Moved, ✗ cloned. `lists` borrowed these and fusion has consumed
        // them, so recording what reached fusion costs nothing on the query
        // path — which is what lets the diagnostic be always-on rather than a
        // debug mode nobody remembers to enable. See `Candidates`.
        let candidates = Candidates {
            dense: dense.into_iter().map(|s| s.id).collect(),
            keyword: keyword.into_iter().map(|s| s.id).collect(),
            exact: exact_ids.into_iter().map(|s| s.id).collect(),
        };

        // ── Hydrate ──────────────────────────────────────────────────────────
        let t = Instant::now();
        let results = self.hydrate(&fused)?;
        timings.hydrate = t.elapsed();

        let confidence =
            confidence_for(&results, !exact_matches.is_empty(), &self.config.search.confidence);
        let citation_block = citation_block(&results);
        let summary_payload = summary_payload(&results, probed.len());

        timings.total = started.elapsed();

        let mut response = SearchResponse {
            success: true,
            op: "search",
            query: query_text.to_owned(),
            detected_domain: Some(domain.id),
            domain_confidence: domain.similarity,
            clusters_probed: probed.len(),
            results,
            citation_block,
            summary_payload,
            exact_matches,
            confidence,
            progress,
            token_estimate: 0,
            truncated: false,
            hint: None,
        };
        response.token_estimate = response.estimate_tokens();

        Ok(SearchOutcome {
            response,
            timings,
            probed,
            candidates,
        })
    }

    /// Response for a query whose identifiers hit but whose routing did not.
    ///
    /// Reports `detected_domain: null` honestly — routing genuinely failed —
    /// while still returning the evidence the corpus holds. `confidence` is
    /// `High` because an exact identifier match is the least ambiguous signal
    /// the engine has; it did not need routing to be right.
    fn exact_only_response(
        &self,
        query_text: &str,
        exact_ids: &[Scored],
        exact_matches: Vec<ExactMatch>,
        mut progress: Vec<String>,
    ) -> Result<SearchResponse, EngineError> {
        progress.push("exact:global returning identifier hits without a routed domain".to_owned());
        let fused = reciprocal_rank_fusion(
            &[RankedList {
                label: "exact",
                items: exact_ids,
            }],
            self.config.search.rrf_k,
        );
        let mut results = self.hydrate(&fused)?;
        results.truncate(self.config.search.max_results);

        let citation_block = citation_block(&results);
        let summary_payload = summary_payload(&results, 0);
        let mut response = SearchResponse {
            success: true,
            op: "search",
            query: query_text.to_owned(),
            detected_domain: None,
            domain_confidence: 0.0,
            clusters_probed: 0,
            results,
            citation_block,
            summary_payload,
            exact_matches,
            confidence: Confidence::High,
            progress,
            token_estimate: 0,
            truncated: false,
            hint: Some(
                "no domain anchor matched, so these are exact-identifier hits only · \
                 semantic results were not searched".into(),
            ),
        };
        response.token_estimate = response.estimate_tokens();
        Ok(response)
    }

    /// Turn fused ids into full results with snippets and provenance.
    ///
    /// ! One batched fetch, ✗ one query per id. At `max_results` = 10 the
    /// difference is 1 query against 10; the shape matters more than the
    /// constant, because this runs inside the concurrency ceiling.
    fn hydrate(&self, fused: &[Fused]) -> Result<Vec<SearchResult>, EngineError> {
        if fused.is_empty() {
            return Ok(Vec::new());
        }
        let ids: Vec<String> = fused.iter().map(|f| f.id.clone()).collect();
        let by_id: HashMap<String, Chunk> = self
            .store
            .chunks_by_id(&ids)?
            .into_iter()
            .map(|c| (c.id.clone(), c))
            .collect();

        // Preserve fusion order · the map lookup must not reorder the ranking.
        Ok(fused
            .iter()
            .filter_map(|f| {
                let chunk = by_id.get(&f.id)?;
                Some(SearchResult {
                    id: f.id.clone(),
                    snippet: chunk.snippet(self.config.search.snippet_chars),
                    score: f.score,
                    scores: ComponentScores {
                        dense: f.components.get("dense").copied().unwrap_or(0.0),
                        bm25: f.components.get("bm25").copied().unwrap_or(0.0),
                    },
                    source: chunk.source(),
                })
            })
            .collect())
    }
}

/// The published `confidence` signal · `OUTPUT_CONTRACT.md` §4.
///
/// The contract defines it in terms of **result quality**:
/// > `high` — strong top scores, exact-identifier match present, or tight
/// > agreement between dense and BM25.
/// > `medium` — moderate scores, single-half support.
/// > `low` — top scores clustered and low, no exact match.
///
/// ! Every one of those is computable without a model — an exact-match flag, a
/// cosine, and whether both halves contributed — so there is no tension with
/// "the engine never calls an LLM". An earlier version derived this from the
/// layer-1 domain-anchor similarity instead, which reports how sure the
/// *routing* was, a different quantity: on a coherent corpus the anchor
/// similarity is nearly constant across queries (p1 0.71 to p95 0.75 on the
/// benchmark corpus), so it carried almost no signal and would have read `high`
/// for every query including the bad ones.
fn confidence_for(
    results: &[SearchResult],
    exact_match_present: bool,
    cfg: &vera_core::ConfidenceConfig,
) -> Confidence {
    if results.is_empty() {
        return Confidence::None;
    }
    // An exact identifier hit is the least ambiguous evidence the engine has;
    // it did not need routing or ranking to be right (`LOOPHOLES.md` §1).
    if exact_match_present {
        return Confidence::High;
    }

    let top = &results[0];
    let both_halves = top.scores.dense > 0.0 && top.scores.bm25 > 0.0;

    // "top scores clustered" · nothing separated itself from the pack, so the
    // ranking carries no information even if the absolute numbers look fine.
    let clustered = results.len() > 1
        && (top.score - results[results.len() - 1].score).abs() < cfg.spread_epsilon;

    if clustered || top.scores.dense < cfg.low_dense {
        return Confidence::Low;
    }
    if both_halves && top.scores.dense >= cfg.high_dense {
        return Confidence::High;
    }
    Confidence::Medium
}

/// De-duplicated snippets and citation references for the agent.
///
/// ! `sources` holds `"[1]"`-style references into `citation_block`, ✗ repeated
/// title+url lines (`OUTPUT_CONTRACT.md` §2). `coverage` says how broadly the
/// corpus was consulted, which is what the agent needs to decide whether to
/// widen the search — a bare result count says nothing about that.
fn summary_payload(results: &[SearchResult], clusters_probed: usize) -> SummaryPayload {
    let mut documents: Vec<&str> = Vec::new();
    for r in results {
        if !documents.contains(&r.source.url.as_str()) {
            documents.push(&r.source.url);
        }
    }
    let top_score = results.first().map_or(0.0, |r| r.score);
    SummaryPayload {
        snippets: results.iter().map(|r| r.snippet.clone()).collect(),
        sources: (1..=results.len()).map(|i| format!("[{i}]")).collect(),
        coverage: format!(
            "{clusters_probed} clusters probed, {} distinct documents, top score {top_score:.3}",
            documents.len()
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vera_core::{ConfidenceConfig, Locator, Source};

    fn result(id: &str, fused: f32, dense: f32, bm25: f32) -> SearchResult {
        SearchResult {
            id: id.into(),
            snippet: "Wajib Pajak…".into(),
            score: fused,
            scores: ComponentScores { dense, bm25 },
            source: Source {
                title: "Doc".into(),
                url: format!("https://e/{id}"),
                locator: Locator::default(),
            },
        }
    }

    fn cfg() -> ConfidenceConfig {
        ConfidenceConfig::default()
    }

    #[test]
    fn an_exact_identifier_match_is_always_high() {
        // OUTPUT_CONTRACT.md §4 · the least ambiguous evidence the engine has.
        let weak = [result("a", 0.01, 0.05, 0.0)];
        assert_eq!(confidence_for(&weak, true, &cfg()), Confidence::High);
    }

    #[test]
    fn agreement_between_both_halves_at_a_strong_score_is_high() {
        let r = [result("a", 0.03, 0.82, 5.0), result("b", 0.01, 0.4, 0.0)];
        assert_eq!(confidence_for(&r, false, &cfg()), Confidence::High);
    }

    #[test]
    fn a_strong_dense_score_with_no_keyword_support_is_medium() {
        // "single-half support" · one signal, however strong, is one signal.
        let r = [result("a", 0.03, 0.82, 0.0), result("b", 0.01, 0.2, 0.0)];
        assert_eq!(confidence_for(&r, false, &cfg()), Confidence::Medium);
    }

    #[test]
    fn a_weak_top_score_is_low() {
        let r = [result("a", 0.03, 0.12, 3.0), result("b", 0.01, 0.05, 0.0)];
        assert_eq!(confidence_for(&r, false, &cfg()), Confidence::Low);
    }

    #[test]
    fn clustered_scores_are_low_even_when_the_absolute_numbers_look_fine() {
        // ! "top scores clustered and low" · nothing separated itself, so the
        // ranking carries no information. A threshold on the top score alone
        // would call this high and be confidently useless.
        let r = [
            result("a", 0.0300, 0.90, 5.0),
            result("b", 0.0299, 0.89, 4.9),
            result("c", 0.0298, 0.89, 4.8),
        ];
        assert_eq!(confidence_for(&r, false, &cfg()), Confidence::Low);
    }

    #[test]
    fn no_results_is_never_reported_as_confident() {
        assert_eq!(confidence_for(&[], true, &cfg()), Confidence::None);
        assert_eq!(confidence_for(&[], false, &cfg()), Confidence::None);
    }

    #[test]
    fn the_thresholds_are_config_so_the_eval_set_can_move_them() {
        // EVAL.md §1 lists the confidence thresholds among the dials evidence
        // must set. A stricter high bar must demote the same result set.
        let r = [result("a", 0.03, 0.55, 5.0), result("b", 0.01, 0.2, 0.0)];
        assert_eq!(confidence_for(&r, false, &cfg()), Confidence::High);
        let strict = ConfidenceConfig {
            high_dense: 0.9,
            ..ConfidenceConfig::default()
        };
        assert_eq!(confidence_for(&r, false, &strict), Confidence::Medium);
    }

    #[test]
    fn summary_sources_are_citation_references_in_rank_order() {
        // OUTPUT_CONTRACT.md §2 · they index into citation_block.
        let p = summary_payload(&[result("a", 1.0, 0.5, 1.0), result("b", 0.9, 0.4, 1.0)], 5);
        assert_eq!(p.sources, ["[1]", "[2]"]);
        assert_eq!(p.snippets.len(), 2, "every snippet is kept");
    }

    #[test]
    fn coverage_reports_clusters_distinct_documents_and_top_score() {
        let mut a = result("a", 0.871, 0.5, 1.0);
        let mut b = result("b", 0.5, 0.4, 1.0);
        // Two chunks of one document · one distinct document.
        a.source.url = "https://e/doc1".into();
        b.source.url = "https://e/doc1".into();
        let p = summary_payload(&[a, b], 5);
        assert!(p.coverage.contains("5 clusters probed"), "{}", p.coverage);
        assert!(p.coverage.contains("1 distinct documents"), "{}", p.coverage);
        assert!(p.coverage.contains("top score 0.871"), "{}", p.coverage);
    }

    #[test]
    fn coverage_over_nothing_does_not_panic() {
        let p = summary_payload(&[], 0);
        assert!(p.sources.is_empty());
        assert!(p.coverage.contains("0 distinct documents"), "{}", p.coverage);
    }
}
