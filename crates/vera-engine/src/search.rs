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

/// A search, plus everything needed to explain and measure it.
#[derive(Debug, Clone)]
pub struct SearchOutcome {
    pub response: SearchResponse,
    pub timings: StageTimings,
    pub probed: Vec<ProbedCluster>,
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
        for d in &domains {
            centroids.insert(d.id.clone(), store.centroids(&d.id)?);
        }
        Ok(Self {
            store,
            config,
            domains,
            centroids,
        })
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

        // ── Layer 1 ──────────────────────────────────────────────────────────
        let t = Instant::now();
        let detected = routing::detect_domain(
            query_vector,
            &self.domains,
            self.config.routing.domain_threshold,
        );
        timings.route_domain = t.elapsed();

        let Some(domain) = detected else {
            // ! Empty, ✗ a guess. See routing::detect_domain.
            progress.push("route:layer1 no anchor above threshold".to_owned());
            timings.total = started.elapsed();
            return Ok(SearchOutcome {
                response: SearchResponse::no_matching_domain(query_text, progress),
                timings,
                probed: Vec::new(),
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
        let mut keyword_lists: Vec<Vec<Scored>> = Vec::with_capacity(probed.len());

        for cluster in &probed {
            let t = Instant::now();
            let mut top = TopK::new(self.config.routing.per_cluster_top_k);
            let scanned = self.store.scan_cluster(cluster.cluster_id, &mut |row| {
                top.offer(row.id, cosine(query_vector, row.vector));
            })?;
            timings.rows_scanned += scanned;
            timings.leaf_dense += t.elapsed();
            dense_lists.push(top.into_ranked());

            let t = Instant::now();
            let hits = self.store.keyword_search(
                query_text,
                Some(cluster.cluster_id),
                self.config.search.bm25_limit,
            )?;
            timings.leaf_keyword += t.elapsed();
            keyword_lists.push(
                hits.into_iter()
                    .map(|h| Scored {
                        id: h.id,
                        score: h.score,
                    })
                    .collect(),
            );
        }
        progress.push(format!("scan:layer3 {} rows", timings.rows_scanned));

        // ── Global exact-identifier path · bypasses routing entirely ─────────
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

        // ── Fusion ───────────────────────────────────────────────────────────
        let t = Instant::now();
        let candidate_cap = self.config.search.max_results * 10;
        let dense = merge_by_score(dense_lists, candidate_cap);
        let keyword = merge_by_score(keyword_lists, candidate_cap);
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

        // ── Hydrate ──────────────────────────────────────────────────────────
        let t = Instant::now();
        let results = self.hydrate(&fused)?;
        timings.hydrate = t.elapsed();

        let confidence = confidence_for(domain.similarity, results.len());
        let citation_block = citation_block(&results);
        let summary_payload = summary_payload(&results);

        timings.total = started.elapsed();

        let mut response = SearchResponse {
            success: true,
            op: "search_knowledge",
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
        })
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

/// Map domain similarity and result count onto the published confidence.
///
/// ! Reports how sure the *routing* was, ✗ how good the answers are — the
/// engine has no way to judge the latter without a model, and pretending
/// otherwise is exactly the LLM-in-the-engine this design forbids.
fn confidence_for(domain_similarity: f32, result_count: usize) -> Confidence {
    if result_count == 0 {
        return Confidence::None;
    }
    if domain_similarity >= 0.60 {
        Confidence::High
    } else if domain_similarity >= 0.40 {
        Confidence::Medium
    } else {
        Confidence::Low
    }
}

/// De-duplicated snippets and sources for the agent to write prose from.
fn summary_payload(results: &[SearchResult]) -> SummaryPayload {
    let mut sources: Vec<String> = Vec::new();
    for r in results {
        let line = format!("{} — {}", r.source.title, r.source.url);
        if !sources.contains(&line) {
            sources.push(line);
        }
    }
    SummaryPayload {
        snippets: results.iter().map(|r| r.snippet.clone()).collect(),
        sources,
        coverage: format!("{} result(s)", results.len()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn confidence_tracks_routing_certainty() {
        assert_eq!(confidence_for(0.9, 5), Confidence::High);
        assert_eq!(confidence_for(0.5, 5), Confidence::Medium);
        assert_eq!(confidence_for(0.3, 5), Confidence::Low);
    }

    #[test]
    fn no_results_is_never_reported_as_confident() {
        // ! Even a perfect domain match means nothing if the leaves were empty.
        assert_eq!(confidence_for(0.99, 0), Confidence::None);
    }

    #[test]
    fn the_summary_payload_dedupes_sources_but_keeps_every_snippet() {
        let r = |snippet: &str, title: &str| SearchResult {
            id: "x".into(),
            snippet: snippet.into(),
            score: 1.0,
            scores: ComponentScores {
                dense: 0.0,
                bm25: 0.0,
            },
            source: vera_core::Source {
                title: title.into(),
                url: "https://e/1".into(),
                locator: vera_core::Locator::default(),
            },
        };
        // Two chunks from one document · one source line, two snippets.
        let p = summary_payload(&[r("a", "Doc"), r("b", "Doc")]);
        assert_eq!(p.snippets.len(), 2);
        assert_eq!(p.sources.len(), 1);
    }
}
