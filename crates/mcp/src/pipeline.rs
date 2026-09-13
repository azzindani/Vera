//! ROUTE → SEARCH → FUSE · composing store, engine and embed into one answer.
//!
//! ! No LLM anywhere on this path (`CLAUDE.md` §7.1). The only outbound model
//! call is an *embedding* call. What comes back is evidence — snippets,
//! addresses, citations — and the calling agent writes the prose.

use std::sync::Arc;

use contract::{
    Citation, ComponentScores, Confidence, ExactMatch, Locator, SearchResponse, SearchResult,
    Source, SummaryPayload, citation_block,
};
use embed::EmbeddingProvider;
use engine::{
    Arm, Centroid, DEFAULT_K, Trust, reciprocal_rank_fusion, routing::cosine, select_clusters,
    trust_from,
};
use store::{CorpusMeta, SearchOps, sparse_literal};

use crate::bm25::{self, QueryVectorizer};
use crate::identifier;

/// Tunables. ! Config, ✗ constants (`CLAUDE.md` §7.12) — bigger hardware and
/// bigger corpora move these without touching code.
#[derive(Debug, Clone)]
pub struct Config {
    pub clusters_probed: usize,
    pub per_cluster_k: i64,
    pub per_arm_k: i64,
    pub top_k: usize,
    /// How many fused candidates carry through to scoring.
    ///
    /// ! Metadata is fetched for this many, ✗ for `top_k`. Ranking on
    /// anything beyond text similarity needs the candidate's `regulation_type`,
    /// `year`, `chapter` and `article`, and a candidate whose metadata was
    /// never loaded cannot be reordered — so cutting to `top_k` before the
    /// fetch would mean factors could only ever re-rank the results that
    /// already won (`docs/SCORING.md` §8).
    ///
    /// Bounds the fetch too: the fused union is at most `3 × per_arm_k`, and
    /// this caps it independently of that.
    pub candidate_pool: usize,
    pub snippet_chars: usize,
    pub dense_weight: f32,
    pub sparse_weight: f32,
    pub text_weight: f32,
    /// Minimum cosine for the startup canary round-trip.
    ///
    /// Vectors are stored as `halfvec`, so an exact round-trip through the
    /// same model lands around 0.999 rather than 1.0 — the gap is fp16
    /// rounding, not drift. A genuinely different model scores far below this,
    /// so the threshold separates "same space" from "different space" with
    /// room to spare.
    pub canary_min_cosine: f32,
    /// Below this nearest-centroid similarity, results carry a weak-match hint.
    ///
    /// ! A warning, ✗ the invariant-13 gate. It does not suppress results,
    /// because the measurement behind it does not separate cleanly enough to
    /// justify returning nothing — see `domain_confidence` in `assemble`.
    pub domain_floor: f32,
    /// Minimum IDF-mass of the query that the retrieved evidence must account
    /// for. The lexical half of the domain gate.
    pub domain_lexical_floor: f32,
    /// How many sparse hits the lexical evidence is pooled over.
    pub gate_sample: usize,
    /// Relative influence of each metadata factor (`docs/SCORING.md` §2).
    ///
    /// ! `Weights::OFF` is exactly the identity on the fused order, so this
    /// layer can be disabled in production without a rebuild.
    pub factor_weights: engine::Weights,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            clusters_probed: 5,
            per_cluster_k: 20,
            per_arm_k: 20,
            top_k: 10,
            candidate_pool: 60,
            snippet_chars: 280,
            // ! Measured, ✗ chosen — and re-measured after every re-chunk,
            // because these are properties of the CORPUS, not of the engine.
            //
            // Through the real server on spike-02, 44 retrievable cases:
            //
            //   dense=0 sparse=1 text=0    38.6% Recall@5   MRR 0.344
            //   dense=0 sparse=1 text=1    50.0%            MRR 0.360
            //   dense=1 sparse=1 text=1    50.0%            MRR 0.360
            //
            // ! The previous values (text 0.0) were fitted on spike-01, where
            // the text arm scored 0.0% because the corpus was unchunked and a
            // whole 32,000-character article was one tsvector. Chunking took
            // that arm to 40.9% on its own — the best of the three — and the
            // weights were never refitted, so the engine shipped at 38.6%
            // while the eval reported 50.0%. Stale weights are silent: every
            // arm still runs, the results still look reasonable.
            //
            // Recall@5 is flat for any text weight in 0.3..=1.0, and MRR wanders
            // 0.360..=0.383 with no trend, so that spread is noise on n=44.
            // Equal weight is the RRF paper's default and claims no precision
            // the measurement cannot support.
            //
            // dense stays 0.0: it adds nothing at any weight (row 3 above), and
            // re-embedding with contextual headers is the open candidate.
            dense_weight: 0.0,
            sparse_weight: 1.0,
            text_weight: 1.0,
            canary_min_cosine: 0.98,
            // ! Measured on the 50-case set in dev_tools/eval/, not chosen. With
            // identifier queries exempt (invariant 4), this pair rejects 6 of
            // 6 out-of-domain queries and 0 of 39 real ones:
            //
            //   lexical   in-domain min 0.469 · worst junk caught 0.333
            //   centroid  in-domain min 0.570 · worst junk caught 0.437
            //
            // Both halves are load bearing. Centroid similarity alone cannot
            // reject "cara memperbaiki keran air yang bocor di dapur" (0.717,
            // above the in-domain mean) because it tracks language, not
            // subject. Lexical evidence alone cannot reject "what is the
            // capital of France" (0.789), because English function words do
            // occur in this corpus. Each covers the other's blind spot.
            domain_floor: 0.45,
            domain_lexical_floor: 0.40,
            gate_sample: 5,
            // Fitted, ✗ chosen: dev_tools/eval/fit_factors.py, +7.5 points
            // leave-one-out over the text arm.
            factor_weights: engine::Weights::FITTED,
        }
    }
}

/// Everything the query path needs, assembled once at startup.
pub struct Pipeline {
    ops: SearchOps,
    provider: Arc<dyn EmbeddingProvider>,
    vectorizer: QueryVectorizer,
    centroids: Vec<Centroid>,
    meta: CorpusMeta,
    cfg: Config,
}

fn log_canary(chunk_id: &str, got: f32) {
    // stderr only · stdout is the MCP channel (invariant 10).
    eprintln!("[vera] canary ok · chunk {chunk_id} round-trip cosine {got:.5}");
}

impl Pipeline {
    /// Wire the pipeline and verify the corpus agrees with the engine.
    ///
    /// # Errors
    /// A corpus whose declared vector space differs from the provider's.
    pub async fn new(
        ops: SearchOps,
        provider: Arc<dyn EmbeddingProvider>,
        vectorizer: QueryVectorizer,
        model: &str,
        cfg: Config,
    ) -> Result<Self, store::StoreError> {
        let meta = ops.corpus_meta().await?;

        // ! Startup refuses rather than degrading. This is the check whose
        // absence let an unreproducible corpus serve confident nonsense.
        meta.ensure_compatible(model, usize::try_from(meta.dense_dim).unwrap_or(0))?;
        if meta.dense_model != model {
            return Err(store::StoreError::CorpusMismatch {
                corpus_model: meta.dense_model.clone(),
                corpus_dim: meta.dense_dim,
                engine_model: model.to_owned(),
                engine_dim: usize::try_from(meta.dense_dim).unwrap_or(0),
            });
        }

        // ! The real canary. `ensure_compatible` above compares two strings;
        // this compares two vector spaces. Re-embed a chunk's own text through
        // the configured provider and check it lands where ingestion put it.
        // An endpoint quietly serving different weights under the same model
        // name gets caught here and nowhere else.
        let (chunk_id, body, stored) = ops.canary_sample().await?;
        match provider.embed_query(&body).await {
            Ok(fresh) => {
                let got = cosine(&fresh, &stored);
                if got < cfg.canary_min_cosine {
                    return Err(store::StoreError::CanaryFailed {
                        chunk_id,
                        got,
                        want: cfg.canary_min_cosine,
                    });
                }
                log_canary(&chunk_id, got);
            }
            Err(e) => {
                // Refuse rather than degrade: serving without having verified
                // the space is the failure mode invariant 2 exists to prevent.
                return Err(store::StoreError::Pool(format!(
                    "canary embed failed · cannot verify the vector space: {e}"
                )));
            }
        }

        let centroids = ops
            .centroids()
            .await?
            .into_iter()
            .map(|(id, vector)| Centroid { id, vector })
            .collect();

        Ok(Self {
            ops,
            provider,
            vectorizer,
            centroids,
            meta,
            cfg,
        })
    }

    #[must_use]
    pub fn corpus(&self) -> &CorpusMeta {
        &self.meta
    }

    #[must_use]
    pub fn cluster_count(&self) -> usize {
        self.centroids.len()
    }

    /// Embed a query the way the corpus was embedded.
    ///
    /// ! The instruction comes from `corpus_meta`, ✗ from a constant here.
    /// Qwen3-Embedding is instruction-aware, and an instruction applied to the
    /// query but not the documents (or the reverse) puts the two sides in
    /// different spaces — silently, since the vectors still normalize and the
    /// rankings still look plausible. Only the corpus knows which convention it
    /// was built under, so only the corpus gets to say.
    ///
    /// # Errors
    /// Whatever the provider returns.
    async fn embed_query(&self, query: &str) -> Result<Vec<f32>, PipelineError> {
        let v = match self.meta.dense_instruction.as_deref() {
            Some(prefix) => {
                self.provider
                    .embed_query(&format!("{prefix}{query}"))
                    .await?
            }
            None => self.provider.embed_query(query).await?,
        };
        Ok(v)
    }

    /// The routing decision alone · backs `explain_routing`.
    ///
    /// # Errors
    /// Embedding or database failure.
    pub async fn explain(&self, query: &str) -> Result<RouteExplain, PipelineError> {
        let q = self.embed_query(query).await?;
        let scores = select_clusters(&q, &self.centroids, self.cfg.clusters_probed);
        Ok(RouteExplain {
            domain: self.meta.id.clone(),
            clusters_probed: scores.iter().map(|(id, _)| *id).collect(),
            cluster_scores: scores,
            total_clusters: self.centroids.len(),
            identifiers: identifier::extract(query)
                .into_iter()
                .map(|i| format!("{}/{}", i.number, i.year.unwrap_or(0)))
                .collect(),
            provider: self.provider.describe(),
        })
    }

    /// The workhorse.
    ///
    /// # Errors
    /// Embedding or database failure.
    pub async fn search(&self, query: &str) -> Result<SearchResponse, PipelineError> {
        let mut progress = Vec::new();

        let qvec = self.embed_query(query).await?;
        progress.push(format!("embedded query ({} dims)", qvec.len()));

        let probed = select_clusters(&qvec, &self.centroids, self.cfg.clusters_probed);
        progress.push(format!(
            "probed {} of {} clusters",
            probed.len(),
            self.centroids.len()
        ));

        let dense = self.dense_arm(&qvec, &probed).await?;
        progress.push(format!("dense: {} candidates", dense.len()));

        let sparse = self
            .ops
            .sparse(
                &sparse_literal(&self.vectorizer.query(query), self.vectorizer.dim()),
                self.cfg.per_arm_k,
            )
            .await?;
        progress.push(format!("sparse: {} candidates", sparse.len()));

        let text = self.ops.text(query, self.cfg.per_arm_k).await?;
        progress.push(format!("text: {} candidates", text.len()));

        let exact_matches = self.exact_arm(query, &mut progress).await?;

        let top_cluster = probed.first().map_or(0.0, |(_, s)| *s);
        if let Some(refused) = self
            .domain_gate(query, &sparse, &exact_matches, top_cluster, &mut progress)
            .await?
        {
            return Ok(refused);
        }

        let dense_ids: Vec<String> = dense.iter().map(|s| s.id.clone()).collect();
        let sparse_ids: Vec<String> = sparse.iter().map(|s| s.id.clone()).collect();
        let text_ids: Vec<String> = text.iter().map(|s| s.id.clone()).collect();

        let fused = reciprocal_rank_fusion(
            &[
                Arm {
                    name: "dense",
                    ids: &dense_ids,
                    weight: self.cfg.dense_weight,
                },
                Arm {
                    name: "sparse",
                    ids: &sparse_ids,
                    weight: self.cfg.sparse_weight,
                },
                Arm {
                    name: "text",
                    ids: &text_ids,
                    weight: self.cfg.text_weight,
                },
            ],
            DEFAULT_K,
        );
        // ! The pool, ✗ the answer. Cutting to `top_k` here is what made the
        // metadata fetch below useless for ranking: a candidate at rank 15
        // carrying the governing law could never be promoted, because nothing
        // about it was ever loaded. The cut moves after scoring.
        let pool: Vec<_> = fused.into_iter().take(self.cfg.candidate_pool).collect();
        progress.push(format!("fused to {} candidates", pool.len()));

        if pool.is_empty() {
            let mut empty = SearchResponse::no_matching_domain(query, progress);
            empty.exact_matches = exact_matches;
            return Ok(empty);
        }

        let ids: Vec<String> = pool.iter().map(|f| f.id.clone()).collect();
        let rows = self.ops.chunks_by_id(&ids).await?;

        // Pair each candidate with its metadata, preserving fused order. A
        // candidate whose row is missing is dropped rather than ranked blind.
        let mut ranked: Vec<(&engine::Fused, &store::ChunkRow)> = pool
            .iter()
            .filter_map(|f| rows.iter().find(|r| r.id == f.id).map(|row| (f, row)))
            .collect();

        // Factor scoring (`docs/SCORING.md` §2). Every candidate has its
        // metadata loaded by now, which is the whole point of fetching the pool
        // rather than the answer.
        //
        // ! The relevance gate of §3 is structural here, ✗ a threshold: the
        // prior MULTIPLIES the fused score, so a candidate no arm ranked has
        // nothing for its pedigree to multiply. Pool membership is the floor.
        let before = ranked.first().map(|(f, _)| f.id.clone());
        let terms = bm25::tokenize(query);
        let term_refs: Vec<&str> = terms.iter().map(String::as_str).collect();
        let mut scored: Vec<(f32, engine::Facets<'_>, usize)> = ranked
            .iter()
            .enumerate()
            .map(|(i, (f, row))| (f.score, Self::facets_of(row), i))
            .collect();
        engine::rescore(&mut scored, &self.cfg.factor_weights, &term_refs);
        let reordered: Vec<_> = scored.iter().map(|(_, _, i)| ranked[*i]).collect();
        ranked = reordered;
        if let Some(was) = before {
            let now = &ranked[0].0.id;
            if *now != was {
                progress.push(format!("factors promoted {now} over {was}"));
            }
        }

        ranked.truncate(self.cfg.top_k);
        progress.push(format!(
            "returning {} of {} candidates",
            ranked.len(),
            pool.len()
        ));
        let results = self.to_results(&ranked, &dense, &sparse);

        Ok(self.assemble(
            query,
            probed.len(),
            top_cluster,
            results,
            exact_matches,
            progress,
        ))
    }

    /// Map a stored row onto the scoring facets.
    ///
    /// ! Borrowed, so scoring a pool of 60 clones nothing. Lives here rather
    /// than in `engine` because `engine` holds zero sibling dependencies by
    /// design and must not learn about `store`.
    fn facets_of(row: &store::ChunkRow) -> engine::Facets<'_> {
        engine::Facets {
            regulation_type: row.regulation_type.as_deref(),
            article: row.article.as_deref(),
            chapter: row.chapter.as_deref(),
            year: row.year,
            // ! `about` is in the schema and not selected by `chunks_by_id`,
            // so the topical factor sees nothing. Harmless today — it carries
            // weight 0.0, having been measured to add nothing — and the one
            // line to add when that changes is in `SearchOps::chunks_by_id`.
            about: None,
            body_len: row.body.len(),
        }
    }

    /// The domain gate · does this corpus hold anything that accounts for the
    /// question? `Some(response)` is a refusal.
    ///
    /// ! Gated on the evidence actually retrieved, ✗ on a judgement about the
    /// query. The corpus is asked whether it holds anything that accounts for
    /// the question, which is the only question that matters.
    ///
    /// ! Identifier queries are NEVER gated. Invariant 4 says a named
    /// regulation must not be lost, and it outranks this check: a bare
    /// "PP 26 tahun 2009" carries almost no semantic or lexical signal and
    /// would be rejected here on both halves.
    async fn domain_gate(
        &self,
        query: &str,
        sparse: &[store::Scored],
        exact_matches: &[ExactMatch],
        top_cluster: f32,
        progress: &mut Vec<String>,
    ) -> Result<Option<SearchResponse>, PipelineError> {
        if !exact_matches.is_empty() {
            return Ok(None);
        }
        let sample: Vec<String> = sparse
            .iter()
            .take(self.cfg.gate_sample)
            .map(|s| s.id.clone())
            .collect();
        let bodies: Vec<String> = self
            .ops
            .chunks_by_id(&sample)
            .await?
            .into_iter()
            .map(|r| r.body)
            .collect();
        let lexical = self.vectorizer.evidence(query, &bodies);
        progress.push(format!(
            "domain gate: lexical {lexical:.3} (floor {:.2}), centroid {top_cluster:.3} (floor {:.2})",
            self.cfg.domain_lexical_floor, self.cfg.domain_floor
        ));
        if lexical >= self.cfg.domain_lexical_floor && top_cluster >= self.cfg.domain_floor {
            return Ok(None);
        }
        let mut empty = SearchResponse::no_matching_domain(query, std::mem::take(progress));
        // One literal · a `\` + newline only folds cleanly with LF endings,
        // and this message is read by a person.
        empty.hint = Some(format!(
            "this corpus does not appear to cover the question · the best matching documents account for only {pct:.0}% of its distinctive terms · check list_domains for what this engine covers",
            pct = lexical * 100.0
        ));
        empty.token_estimate = empty.estimate_tokens();
        Ok(Some(empty))
    }

    /// Scan the probed clusters **one at a time**.
    ///
    /// ! This loop is the OOM guarantee (`docs/MCP_ENGINE.md` §5): a request holds
    /// one cluster at a time, so the working set does not grow with
    /// `clusters_probed`.
    async fn dense_arm(
        &self,
        qvec: &[f32],
        probed: &[(i32, f32)],
    ) -> Result<Vec<store::Scored>, PipelineError> {
        let mut dense: Vec<store::Scored> = Vec::new();
        for (cluster_id, _) in probed {
            dense.extend(
                self.ops
                    .dense_in_cluster(*cluster_id, qvec, self.cfg.per_cluster_k)
                    .await?,
            );
        }
        dense.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        dense.truncate(usize::try_from(self.cfg.per_arm_k).unwrap_or(20));
        Ok(dense)
    }

    /// Global identifier lookup · never gated by cluster selection.
    async fn exact_arm(
        &self,
        query: &str,
        progress: &mut Vec<String>,
    ) -> Result<Vec<ExactMatch>, PipelineError> {
        let mut out = Vec::new();
        for id in identifier::extract(query) {
            let hits = self
                .ops
                .exact_identifier(&id.number, id.year, id.reg_type, 5)
                .await?;
            if !hits.is_empty() {
                progress.push(format!(
                    "exact identifier {}/{}: {} hits (routing bypassed)",
                    id.number,
                    id.year.unwrap_or(0),
                    hits.len()
                ));
            }
            out.extend(hits.into_iter().map(|h| ExactMatch {
                id: h.id,
                matched_on: h.matched_on,
            }));
        }
        Ok(out)
    }

    /// Render the surviving candidates as the wire contract.
    ///
    /// ! Takes candidates already paired with their metadata and already cut
    /// to `top_k`. Building a `SearchResult` for the whole pool would compute
    /// a snippet for every candidate and throw most of them away.
    fn to_results(
        &self,
        ranked: &[(&engine::Fused, &store::ChunkRow)],
        dense: &[store::Scored],
        sparse: &[store::Scored],
    ) -> Vec<SearchResult> {
        let score_in = |arm: &[store::Scored], id: &str| {
            arm.iter().find(|s| s.id == id).map_or(0.0, |s| s.score)
        };
        ranked
            .iter()
            .map(|(f, row)| SearchResult {
                id: row.id.clone(),
                snippet: snippet(&row.body, self.cfg.snippet_chars),
                score: f.score,
                scores: ComponentScores {
                    dense: score_in(dense, &row.id),
                    bm25: score_in(sparse, &row.id),
                },
                source: Source {
                    title: row.source_title.clone(),
                    // ! Straight from ingest. Never synthesised (invariant 8).
                    url: row.source_url.clone(),
                    locator: Locator {
                        page: None,
                        section: locator_of(row),
                    },
                },
            })
            .collect()
    }

    fn assemble(
        &self,
        query: &str,
        probed: usize,
        top_cluster: f32,
        results: Vec<SearchResult>,
        exact_matches: Vec<ExactMatch>,
        progress: Vec<String>,
    ) -> SearchResponse {
        let citations: Vec<Citation> = citation_block(&results);
        let incomplete = results.iter().filter(|r| r.source.url.is_none()).count();
        let summary_payload = SummaryPayload {
            snippets: results.iter().map(|r| r.snippet.clone()).collect(),
            sources: citations.iter().map(|c| c.text.clone()).collect(),
            coverage: format!(
                "{} results from {probed} of {} clusters",
                results.len(),
                self.centroids.len()
            ),
        };
        let top_score = results.first().map_or(0.0, |r| r.score);
        let confidence = match trust_from(true, results.len(), top_score) {
            Trust::High => Confidence::High,
            Trust::Medium => Confidence::Medium,
            Trust::Low => Confidence::Low,
            Trust::None => Confidence::None,
        };
        let weak = top_cluster < self.cfg.domain_floor;
        let mut notes: Vec<String> = Vec::new();
        if weak {
            notes.push(format!(
                "weak domain match ({top_cluster:.3}) · the corpus may not cover \
                 this question · check the citations before relying on these results"
            ));
        }
        if incomplete > 0 {
            notes.push(format!(
                "{incomplete} of {} results have no source_url · this corpus was \
                     ingested without one · cite by title and locator, and treat the \
                     link as unavailable",
                results.len()
            ));
        }
        let hint = (!notes.is_empty()).then(|| notes.join(" · "));

        let mut resp = SearchResponse {
            success: true,
            op: "search_knowledge",
            query: query.to_owned(),
            detected_domain: Some(self.meta.id.clone()),
            // ! Measured, not asserted. This used to be a hardcoded 1.0, which
            // meant the engine reported total confidence in its domain for
            // "chocolate chip cookie recipe" as readily as for a real legal
            // question. It is now the similarity to the nearest probed
            // centroid — a number that is actually about this query.
            //
            // ! It is NOT yet the gate invariant 13 asks for, and pretending
            // otherwise would be worse than the honest gap. Measured over the
            // 50-case set in dev_tools/eval/, nearest-centroid similarity does not
            // separate in-domain from out-of-domain: "cara memperbaiki keran
            // air yang bocor di dapur" scores 0.7163, above the in-domain mean
            // of 0.6993, while the exact_ref query "PP 60/2014" sits at 0.4178.
            // The signal separates LANGUAGE, not domain. Until a gate exists
            // that actually discriminates, the engine reports what it knows
            // and flags a weak match rather than silently guessing.
            domain_confidence: top_cluster,
            clusters_probed: probed,
            results,
            citation_block: citations,
            summary_payload,
            exact_matches,
            confidence,
            progress,
            token_estimate: 0,
            truncated: false,
            hint,
        };
        resp.token_estimate = resp.estimate_tokens();
        resp
    }

    /// Bounded read of one chunk · `read_chunk`.
    ///
    /// # Errors
    /// Database failure.
    pub async fn read_chunk(
        &self,
        id: &str,
        max_chars: usize,
    ) -> Result<Option<(store::ChunkRow, String, bool)>, PipelineError> {
        let rows = self.ops.chunks_by_id(&[id.to_owned()]).await?;
        Ok(rows.into_iter().next().map(|row| {
            let full = row.body.chars().count();
            let truncated = full > max_chars;
            let body = if truncated {
                row.body.chars().take(max_chars).collect()
            } else {
                row.body.clone()
            };
            (row, body, truncated)
        }))
    }

    /// Provenance bundle for result ids · `get_provenance`.
    ///
    /// # Errors
    /// Database failure.
    pub async fn provenance(&self, ids: &[String]) -> Result<Vec<store::ChunkRow>, PipelineError> {
        Ok(self.ops.chunks_by_id(ids).await?)
    }
}

/// What routing decided, for `explain_routing`.
#[derive(Debug, Clone)]
pub struct RouteExplain {
    pub domain: String,
    pub clusters_probed: Vec<i32>,
    pub cluster_scores: Vec<(i32, f32)>,
    pub total_clusters: usize,
    pub identifiers: Vec<String>,
    pub provider: String,
}

#[derive(Debug, thiserror::Error)]
pub enum PipelineError {
    #[error(transparent)]
    Store(#[from] store::StoreError),

    #[error(transparent)]
    Embed(#[from] embed::EmbedError),
}

/// The most precise locator ingest recorded · never a placeholder.
fn locator_of(row: &store::ChunkRow) -> Option<String> {
    match (row.chapter.as_deref(), row.article.as_deref()) {
        (Some(c), Some(a)) if c != "N/A" && !c.is_empty() => Some(format!("{c} / {a}")),
        (_, Some(a)) if !a.is_empty() => Some(a.to_owned()),
        _ => None,
    }
}

/// Bounded preview · whitespace collapsed, cut on a word break, multibyte-safe.
fn snippet(body: &str, max_chars: usize) -> String {
    let flat = body.split_whitespace().collect::<Vec<_>>().join(" ");
    if flat.chars().count() <= max_chars {
        return flat;
    }
    let cut: String = flat.chars().take(max_chars).collect();
    let keep = match cut.rfind(char::is_whitespace) {
        Some(i) if i >= max_chars.saturating_mul(3) / 4 => &cut[..i],
        _ => cut.as_str(),
    };
    format!("{}…", keep.trim_end())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snippets_collapse_whitespace_and_stay_within_budget() {
        let s = snippet("Menimbang :\n\n  Mengingat :\n Menetapkan", 200);
        assert_eq!(s, "Menimbang : Mengingat : Menetapkan");
    }

    #[test]
    fn long_bodies_are_cut_on_a_word_break() {
        let body = "Wajib Pajak yang terlambat menyampaikan laporan dikenai sanksi";
        let s = snippet(body, 20);
        assert!(s.ends_with('…'), "{s}");
        assert!(s.starts_with("Wajib Pajak"), "{s}");
    }

    #[test]
    fn snippets_never_split_a_multibyte_character() {
        let body = "pératuran émbedding ünicode ✓ 日本語のテキスト";
        for n in 1..40 {
            let s = snippet(body, n);
            assert!(s.is_char_boundary(s.len()));
        }
    }

    fn row(chapter: Option<&str>, article: Option<&str>) -> store::ChunkRow {
        store::ChunkRow {
            id: "c1".into(),
            body: "x".into(),
            source_title: "t".into(),
            source_url: None,
            chapter: chapter.map(Into::into),
            article: article.map(Into::into),
            regulation_type: None,
            regulation_number: None,
            year: None,
            truncated_at_source: false,
            cluster_id: None,
        }
    }

    #[test]
    fn locators_prefer_chapter_and_article_together() {
        assert_eq!(
            locator_of(&row(Some("BAB I"), Some("Pasal 1"))).as_deref(),
            Some("BAB I / Pasal 1")
        );
    }

    #[test]
    fn the_placeholder_chapter_is_not_treated_as_a_locator() {
        // ! 45,000 rows in this corpus have chapter = 'N/A'. Rendering that as
        // provenance would be inventing an address that does not exist.
        assert_eq!(
            locator_of(&row(Some("N/A"), Some("Pembukaan"))).as_deref(),
            Some("Pembukaan")
        );
    }

    #[test]
    fn a_row_with_nothing_recorded_has_no_locator() {
        assert!(locator_of(&row(None, None)).is_none());
        assert!(locator_of(&row(Some("N/A"), Some(""))).is_none());
    }
}
