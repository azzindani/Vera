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

use crate::bm25::QueryVectorizer;
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
            // dense stays 0.0: it adds nothing at any weight (row 3 above).
            //
            // ! Re-embedding the corpus is NOT queued behind this. An arm at
            // weight 0.0 contributing 0.0% is absent, ✗ broken, and paying
            // hours of GPU to improve it is a bet nothing has measured. See
            // `docs/EMBEDDING.md` §5 for what would have to be measured first.
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

/// Everything `assemble` needs to build the response.
///
/// A struct rather than eight positional arguments: at that width the compiler
/// stops catching a transposed pair, and two `usize` fields next to each other
/// is exactly the shape that silently swaps.
struct Assembly<'a> {
    query: &'a str,
    probed: usize,
    top_cluster: f32,
    results: Vec<SearchResult>,
    exact_matches: Vec<ExactMatch>,
    progress: Vec<String>,
    applied: contract::AppliedOptions,
}

/// `engine::Weights` -> the wire shape. Two types on purpose: `engine` holds
/// zero sibling dependencies, so it cannot name a `contract` type, and the wire
/// format is a compatibility surface that must be free to diverge from the
/// internal one.
fn weights_to_contract(w: &engine::Weights) -> contract::FactorWeights {
    contract::FactorWeights {
        relevance_floor: w.relevance_floor,
        authority: w.authority,
        structural: w.structural,
        temporal: w.temporal,
        completeness: w.completeness,
        topical: w.topical,
    }
}

/// The inverse. Caller-supplied weights arrive here.
fn weights_from_contract(w: &contract::FactorWeights) -> engine::Weights {
    engine::Weights {
        relevance_floor: w.relevance_floor,
        authority: w.authority,
        structural: w.structural,
        temporal: w.temporal,
        completeness: w.completeness,
        topical: w.topical,
    }
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

    /// The server's ceilings · what a caller may narrow toward, never past.
    fn ceilings(&self) -> contract::Ceilings {
        contract::Ceilings {
            top_k: self.cfg.top_k,
            candidate_pool: self.cfg.candidate_pool,
        }
    }

    /// The workhorse · search under caller-supplied options
    /// (`docs/TOOL_SURFACE.md`).
    ///
    /// ! Options **narrow**, never widen. `resolve` clamps every numeric field
    /// to the configured ceiling and records each clamp, so a caller that asks
    /// for 500 results and receives 10 can see why without reading the
    /// server's configuration. `SearchOptions::default()` is exactly the
    /// behaviour `EVAL.md` scored.
    ///
    /// # Errors
    /// Embedding or database failure.
    pub async fn search_with(
        &self,
        query: &str,
        opts: &contract::SearchOptions,
    ) -> Result<SearchResponse, PipelineError> {
        let applied = opts.resolve(
            self.ceilings(),
            weights_to_contract(&self.cfg.factor_weights),
        );
        let mut progress = Vec::new();
        for line in &applied.clamped {
            progress.push(format!("clamped: {line}"));
        }

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

        let (w_dense, w_sparse, w_text) = self.arm_weights(applied.mode);
        let fused = reciprocal_rank_fusion(
            &[
                Arm {
                    name: "dense",
                    ids: &dense_ids,
                    weight: w_dense,
                },
                Arm {
                    name: "sparse",
                    ids: &sparse_ids,
                    weight: w_sparse,
                },
                Arm {
                    name: "text",
                    ids: &text_ids,
                    weight: w_text,
                },
            ],
            DEFAULT_K,
        );
        // ! The pool, ✗ the answer. Cutting to `top_k` here is what made the
        // metadata fetch below useless for ranking: a candidate at rank 15
        // carrying the governing law could never be promoted, because nothing
        // about it was ever loaded. The cut moves after scoring.
        let pool: Vec<_> = fused.into_iter().take(applied.candidate_pool).collect();
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

        // ! The floor is skipped for a named regulation, exactly as the
        // domain gate is. See `apply_factors`.
        Self::apply_factors(
            &mut ranked,
            query,
            &applied,
            exact_matches.is_empty(),
            &mut progress,
        );
        ranked.truncate(applied.top_k);
        progress.push(format!(
            "returning {} of {} candidates",
            ranked.len(),
            pool.len()
        ));
        let results = self.to_results(&ranked, &dense, &sparse);

        Ok(self.assemble(Assembly {
            query,
            probed: probed.len(),
            top_cluster,
            results,
            exact_matches,
            progress,
            applied,
        }))
    }

    /// Arm weights for a requested mode.
    ///
    /// ! Mode selects arms by ZEROING weights, ✗ by skipping retrieval. RRF
    /// already treats a zero-weight arm as absent (`fusion.rs`), so one code
    /// path serves every mode and there is no second fusion to keep in step.
    fn arm_weights(&self, mode: contract::Mode) -> (f32, f32, f32) {
        match mode {
            contract::Mode::Hybrid => (
                self.cfg.dense_weight,
                self.cfg.sparse_weight,
                self.cfg.text_weight,
            ),
            contract::Mode::Keyword => (0.0, self.cfg.sparse_weight, self.cfg.text_weight),
            contract::Mode::Semantic => (self.cfg.dense_weight.max(1.0), 0.0, 0.0),
        }
    }

    /// Reorder a pool on its metadata (`docs/SCORING.md` §2).
    ///
    /// ! `apply_floor` is false when the query named a regulation, and the
    /// floor is then skipped entirely — the same exemption the domain gate
    /// makes, for the same reason. "PP 26 tahun 2009" reduces to the content
    /// terms `{tahun, 2009}`, which the clause bodies do not contain, so the
    /// floor would drop the whole pool. Measured on the fixture: **19 of 21
    /// candidates dropped, `results` collapsing from ten to two.** Invariant 4
    /// survived — `exact_matches` is a separate channel — but a result list
    /// that quietly empties for the queries users are most confident about is
    /// its own defect. An identifier IS the relevance signal.
    ///
    /// ! The relevance gate of §3 is **structural here, ✗ a threshold**: the
    /// prior MULTIPLIES the fused score, so a candidate no arm ranked has
    /// nothing for its pedigree to multiply. Pool membership is the floor, and
    /// every member earned it from an arm.
    ///
    /// A promotion is recorded in `progress` — a reordering the caller cannot
    /// see is one they cannot check.
    fn apply_factors(
        ranked: &mut Vec<(&engine::Fused, &store::ChunkRow)>,
        query: &str,
        applied: &contract::AppliedOptions,
        apply_floor: bool,
        progress: &mut Vec<String>,
    ) {
        let before = ranked.first().map(|(f, _)| f.id.clone());
        // ! `content_terms`, ✗ `bm25::tokenize`. The relevance floor was fitted
        // against the former — words longer than three characters with function
        // words removed — and the two tokenisers disagree on short words, so
        // the wrong one applies a threshold nothing measured.
        let terms = engine::factors::content_terms(query);
        let term_refs: Vec<&str> = terms.iter().map(String::as_str).collect();

        let scored_len_before = ranked.len();
        let mut scored: Vec<(f32, engine::Facets<'_>, usize)> = ranked
            .iter()
            .enumerate()
            .map(|(i, (f, row))| (f.score, Self::facets_of(row), i))
            .collect();
        let mut weights = weights_from_contract(&applied.factor_weights);
        if !apply_floor {
            weights.relevance_floor = 0.0;
            progress.push("relevance floor skipped · the query names a regulation".into());
        }
        engine::rescore(&mut scored, &weights, &term_refs);

        let kept = scored.len();
        *ranked = scored.iter().map(|(_, _, i)| ranked[*i]).collect();

        let dropped = scored_len_before.saturating_sub(kept);
        if dropped > 0 {
            progress.push(format!(
                "relevance floor dropped {dropped} of {scored_len_before} candidates"
            ));
        }
        if let (Some(was), Some((now, _))) = (before, ranked.first())
            && now.id != was
        {
            progress.push(format!("factors promoted {} over {was}", now.id));
        }
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
            about: row.about.as_deref(),
            body: Some(&row.body),
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

    fn assemble(&self, parts: Assembly<'_>) -> SearchResponse {
        let Assembly {
            query,
            probed,
            top_cluster,
            results,
            exact_matches,
            progress,
            applied,
        } = parts;
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
        // ! The caller asked for an arm this corpus measures at 0.0%. Serve it
        // and say so -- silently returning weak results from a mode we know is
        // empty would be the dishonest option (`TOOL_SURFACE.md` §3).
        if applied.mode.is_measured_empty() {
            notes.push(
                "mode=semantic uses the dense arm alone, which scores 0.0% Recall@5 on                  this corpus and ships at weight 0.0 · use hybrid or keyword for results                  that reflect what was measured"
                    .into(),
            );
        }
        if applied.experimental {
            notes.push(
                "experimental: factor_weights were supplied by the caller rather than                  fitted · the effective weights are in `applied` so this ranking can be                  reproduced"
                    .into(),
            );
        }
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
            applied: Some(applied),
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
            about: None,
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

// ---------------------------------------------------------------------------
// Live pipeline tests · the whole engine, no embedding server, no GPU.
//
// ! These need the FIXTURE corpus, not the real one:
//
//     DATABASE_URL="host=... dbname=vera_fx ..." python dev_tools/fixtures/seed.py
//     VERA_FX_DSN="host=... dbname=vera_fx ..." cargo test -p vera-mcp -- --ignored
//
// The fixture's dense vectors are a pure function of (theme, body)
// (`dev_tools/fixtures/seed.py`), so `FixtureProvider` below reproduces them
// exactly and the STARTUP CANARY PASSES FOR REAL rather than being bypassed. A
// test constructor that skipped the canary would be a hole in invariant 3, and
// these tests exist partly to prove the canary works.
//
// ! What these do NOT test is retrieval quality. The vectors are derived from
// hashes and say nothing about meaning; `dev_tools/eval/e2e.py` is the only
// thing that scores that. These test MECHANICS -- that options are honoured,
// that the pool is built and cut in the right order, that factors reorder
// rather than filter, and that the contract comes back whole.
// ---------------------------------------------------------------------------
#[cfg(test)]
mod live {
    use super::*;
    use std::collections::HashMap;

    const DIM: usize = 1024;
    const FIXTURE_MODEL: &str = "qwen/qwen3-embedding-0.6b";

    fn dsn() -> String {
        std::env::var("VERA_FX_DSN").expect("set VERA_FX_DSN to the fixture database")
    }

    /// Reproduces `dev_tools/fixtures/seed.py::dense_for`.
    ///
    /// ! Pinned to the Python by `the_provider_reproduces_the_stored_vectors`,
    /// which is the canary in miniature. Two implementations of one formula
    /// drift silently otherwise.
    struct FixtureProvider {
        /// body text -> theme, read from `seed.dense.json`.
        themes: HashMap<String, String>,
        theme_share: f32,
    }

    impl FixtureProvider {
        fn load() -> Self {
            let path = concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../dev_tools/fixtures/seed.dense.json"
            );
            let raw = std::fs::read_to_string(path)
                .unwrap_or_else(|e| panic!("run dev_tools/fixtures/seed.py first · {path}: {e}"));
            let v: serde_json::Value = serde_json::from_str(&raw).expect("seed.dense.json");
            let themes = v["bodies"]
                .as_object()
                .expect("bodies")
                .iter()
                .map(|(k, t)| (k.clone(), t.as_str().unwrap_or_default().to_owned()))
                .collect();
            #[allow(clippy::cast_possible_truncation)]
            let theme_share = v["theme_share"].as_f64().unwrap_or(0.95) as f32;
            Self {
                themes,
                theme_share,
            }
        }

        fn unit_of(text: &str) -> Vec<f32> {
            Self::normalize(embed::StubProvider::vector_for(text, DIM))
        }

        fn normalize(mut v: Vec<f32>) -> Vec<f32> {
            let n = v
                .iter()
                .map(|x| x * x)
                .sum::<f32>()
                .sqrt()
                .max(f32::EPSILON);
            for x in &mut v {
                *x /= n;
            }
            v
        }

        /// The vector for a known body · exactly what ingestion stored.
        fn dense_for(&self, theme: &str, body: &str) -> Vec<f32> {
            let t = Self::unit_of(theme);
            let b = Self::unit_of(body);
            let s = self.theme_share;
            Self::normalize(
                t.iter()
                    .zip(&b)
                    .map(|(ti, bi)| s * ti + (1.0 - s) * bi)
                    .collect(),
            )
        }

        /// The themes this fixture holds, so a test can aim at a cluster.
        fn theme_names(&self) -> Vec<String> {
            let mut t: Vec<String> = self.themes.values().cloned().collect();
            t.sort();
            t.dedup();
            t
        }
    }

    #[async_trait::async_trait]
    impl embed::EmbeddingProvider for FixtureProvider {
        async fn embed_query(&self, text: &str) -> Result<Vec<f32>, embed::EmbedError> {
            // A stored body: reproduce ingestion exactly. This is the path the
            // startup canary takes.
            if let Some(theme) = self.themes.get(text) {
                return Ok(self.dense_for(theme, text));
            }
            // A query: route by a theme word when one is present, so a test can
            // aim at a cluster deliberately. Otherwise an arbitrary direction,
            // which is honest -- a hash-based fixture has no semantics.
            for theme in self.theme_names() {
                if text.to_lowercase().contains(&theme) {
                    return Ok(Self::unit_of(&theme));
                }
            }
            Ok(Self::unit_of(text))
        }

        fn describe(&self) -> String {
            format!("fixture(dim={DIM}, bodies={})", self.themes.len())
        }
    }

    async fn pipeline(cfg: Config) -> Pipeline {
        let ops = SearchOps::new(store::connect(&dsn(), 4, 15_000).expect("pool"));
        let vocab = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../dev_tools/fixtures/seed.bm25.json"
        );
        let vectorizer = QueryVectorizer::load(std::path::Path::new(vocab)).expect("vocab");
        Pipeline::new(
            ops,
            Arc::new(FixtureProvider::load()),
            vectorizer,
            FIXTURE_MODEL,
            cfg,
        )
        .await
        .expect("pipeline · did the canary fail?")
    }

    /// Floors off, so the domain gate cannot swallow a mechanics test. The gate
    /// has its own tests below, which use the real floors.
    fn open_cfg() -> Config {
        Config {
            domain_floor: -1.0,
            domain_lexical_floor: -1.0,
            ..Config::default()
        }
    }

    fn no_factors() -> contract::FactorWeights {
        contract::FactorWeights {
            relevance_floor: 0.0,
            authority: 0.0,
            structural: 0.0,
            temporal: 0.0,
            completeness: 0.0,
            topical: 0.0,
        }
    }

    #[tokio::test]
    #[ignore = "needs the fixture corpus · VERA_FX_DSN"]
    async fn the_provider_reproduces_the_stored_vectors() {
        // ! The canary in miniature, and the reason every test below can run
        // without a GPU. If seed.py's formula and FixtureProvider's ever
        // diverge, this fails first and names the drift.
        let ops = SearchOps::new(store::connect(&dsn(), 2, 15_000).expect("pool"));
        let (id, body, stored) = ops.canary_sample().await.expect("canary sample");
        let p = FixtureProvider::load();
        let fresh = embed::EmbeddingProvider::embed_query(&p, &body)
            .await
            .expect("embed");
        let got = cosine(&fresh, &stored);
        assert!(got >= 0.999, "chunk {id} round-tripped at {got:.6}");
    }

    #[tokio::test]
    #[ignore = "needs the fixture corpus · VERA_FX_DSN"]
    async fn a_search_returns_a_whole_contract() {
        let p = pipeline(open_cfg()).await;
        let r = p
            .search_with("kelas jalan", &contract::SearchOptions::default())
            .await
            .expect("search");

        assert!(r.success);
        assert_eq!(r.op, "search_knowledge");
        assert!(!r.results.is_empty(), "fixture should match 'kelas jalan'");
        assert!(r.token_estimate > 0, "an unset estimate is a broken budget");
        assert!(
            !r.progress.is_empty(),
            "progress is how routing is auditable"
        );
        for res in &r.results {
            assert!(!res.id.is_empty(), "a result must be addressable");
            assert!(!res.snippet.is_empty(), "a result must be quotable");
        }
    }

    #[tokio::test]
    #[ignore = "needs the fixture corpus · VERA_FX_DSN"]
    async fn the_applied_options_come_back_even_when_none_were_sent() {
        // "The defaults were used" is itself the reproducibility record.
        let p = pipeline(open_cfg()).await;
        let r = p
            .search_with("retribusi", &contract::SearchOptions::default())
            .await
            .expect("search");
        let a = r.applied.expect("applied must always be present");
        assert_eq!(a.mode, contract::Mode::Hybrid);
        assert_eq!(a.top_k, Config::default().top_k);
        assert!(!a.experimental);
        assert!(a.clamped.is_empty());
    }

    #[tokio::test]
    #[ignore = "needs the fixture corpus · VERA_FX_DSN"]
    async fn top_k_narrows_the_answer() {
        let p = pipeline(open_cfg()).await;
        let r = p
            .search_with(
                "jalan",
                &contract::SearchOptions {
                    top_k: Some(2),
                    ..contract::SearchOptions::default()
                },
            )
            .await
            .expect("search");
        assert!(r.results.len() <= 2, "got {}", r.results.len());
        assert_eq!(r.applied.expect("applied").top_k, 2);
    }

    #[tokio::test]
    #[ignore = "needs the fixture corpus · VERA_FX_DSN"]
    async fn a_caller_cannot_widen_past_the_server_ceiling() {
        let p = pipeline(open_cfg()).await;
        let r = p
            .search_with(
                "jalan",
                &contract::SearchOptions {
                    top_k: Some(9_999),
                    ..contract::SearchOptions::default()
                },
            )
            .await
            .expect("search");
        let a = r.applied.clone().expect("applied");
        assert_eq!(a.top_k, Config::default().top_k);
        assert!(!a.clamped.is_empty(), "the clamp must be reported");
        // And visible without reading the server's configuration.
        assert!(
            r.progress.iter().any(|l| l.contains("clamped")),
            "{:?}",
            r.progress
        );
    }

    #[tokio::test]
    #[ignore = "needs the fixture corpus · VERA_FX_DSN"]
    async fn keyword_mode_still_answers_without_the_dense_arm() {
        let p = pipeline(open_cfg()).await;
        let r = p
            .search_with(
                "kelas jalan",
                &contract::SearchOptions {
                    mode: Some(contract::Mode::Keyword),
                    ..contract::SearchOptions::default()
                },
            )
            .await
            .expect("search");
        assert!(!r.results.is_empty(), "the lexical arms carry this corpus");
        assert_eq!(r.applied.expect("applied").mode, contract::Mode::Keyword);
    }

    #[tokio::test]
    #[ignore = "needs the fixture corpus · VERA_FX_DSN"]
    async fn semantic_mode_is_served_and_labelled() {
        // Dense scores 0.0% on the real corpus. Serving it silently would be
        // the dishonest option; the hint is the contract.
        let p = pipeline(open_cfg()).await;
        let r = p
            .search_with(
                "energi",
                &contract::SearchOptions {
                    mode: Some(contract::Mode::Semantic),
                    ..contract::SearchOptions::default()
                },
            )
            .await
            .expect("search");
        let hint = r.hint.unwrap_or_default();
        assert!(hint.contains("semantic"), "hint was {hint:?}");
    }

    #[tokio::test]
    #[ignore = "needs the fixture corpus · VERA_FX_DSN"]
    async fn caller_supplied_weights_are_marked_and_echoed() {
        let p = pipeline(open_cfg()).await;
        let mine = contract::FactorWeights {
            authority: 1.5,
            ..no_factors()
        };
        let r = p
            .search_with(
                "jalan",
                &contract::SearchOptions {
                    factor_weights: Some(mine),
                    ..contract::SearchOptions::default()
                },
            )
            .await
            .expect("search");
        let a = r.applied.expect("applied");
        assert!(a.experimental);
        assert_eq!(a.factor_weights, mine, "the ranking must be reproducible");
    }

    #[tokio::test]
    #[ignore = "needs the fixture corpus · VERA_FX_DSN"]
    async fn weights_without_a_floor_reorder_but_never_filter() {
        // ! With the floor OFF the layer is pure ordering, and that must stay
        // true: a weight that silently dropped a candidate would make a recall
        // change impossible to attribute to either half of the layer.
        //
        // ! `candidate_pool == top_k` is what makes the claim testable, ✗ a
        // convenience. A response only carries `top_k`, so against a wider pool
        // reordering legitimately changes WHICH candidates clear the cut — that
        // is ranking working, not filtering, and an earlier version of this
        // test read the difference as a failure. Pinning the pool to the answer
        // fixes membership at fusion and leaves order as the only free variable.
        let p = pipeline(open_cfg()).await;
        let pinned = |w: contract::FactorWeights| contract::SearchOptions {
            factor_weights: Some(w),
            top_k: Some(10),
            candidate_pool: Some(10),
            ..contract::SearchOptions::default()
        };
        let ordering_only = contract::FactorWeights {
            relevance_floor: 0.0,
            ..weights_to_contract(&engine::Weights::FITTED)
        };
        let base = p
            .search_with("bangunan gedung", &pinned(no_factors()))
            .await
            .expect("search");
        let scored = p
            .search_with("bangunan gedung", &pinned(ordering_only))
            .await
            .expect("search");
        assert_eq!(
            base.results.len(),
            scored.results.len(),
            "weights alone must reorder, not filter"
        );
        let mut a: Vec<&str> = base.results.iter().map(|r| r.id.as_str()).collect();
        let mut b: Vec<&str> = scored.results.iter().map(|r| r.id.as_str()).collect();
        a.sort_unstable();
        b.sort_unstable();
        assert_eq!(a, b, "the same candidates, in a different order");
    }

    #[tokio::test]
    #[ignore = "needs the fixture corpus · VERA_FX_DSN"]
    async fn the_relevance_floor_accounts_for_what_it_drops() {
        // ! The floor is the one part of scoring that REMOVES candidates, and
        // it is where most of the measured gain is (+12.5 of +17.5 points).
        // Whatever it drops it must report: a silent filter is unauditable, and
        // the progress line is the only place a caller can see it happened.
        let p = pipeline(open_cfg()).await;
        let r = p
            .search_with(
                "kelas jalan provinsi ditetapkan gubernur",
                &contract::SearchOptions::default(),
            )
            .await
            .expect("search");
        if r.results.len() < Config::default().top_k {
            assert!(
                r.progress.iter().any(|l| l.contains("relevance floor")),
                "a filter that leaves no trace is unauditable: {:?}",
                r.progress
            );
        }
    }

    #[tokio::test]
    #[ignore = "needs the fixture corpus · VERA_FX_DSN"]
    async fn the_floor_never_turns_a_match_into_a_refusal() {
        // ! "This corpus cannot answer the question" is the DOMAIN GATE's
        // answer (invariant 13). If the floor could empty the pool, two
        // components would be refusing for different reasons and a caller could
        // not tell which one declined.
        let p = pipeline(open_cfg()).await;
        for q in ["jalan", "energi", "bangunan gedung", "retribusi pasar"] {
            let r = p
                .search_with(q, &contract::SearchOptions::default())
                .await
                .expect("search");
            assert!(!r.results.is_empty(), "floor emptied the answer for {q:?}");
        }
    }

    #[tokio::test]
    #[ignore = "needs the fixture corpus · VERA_FX_DSN"]
    async fn zeroed_factor_weights_are_the_identity_on_the_fused_order() {
        // The layer must be switchable off in production without a rebuild,
        // and "off" has to mean exactly the order fusion produced.
        let p = pipeline(open_cfg()).await;
        let r = p
            .search_with(
                "jalan",
                &contract::SearchOptions {
                    factor_weights: Some(no_factors()),
                    ..contract::SearchOptions::default()
                },
            )
            .await
            .expect("search");
        let scores: Vec<f32> = r.results.iter().map(|x| x.score).collect();
        assert!(
            scores.windows(2).all(|w| w[0] >= w[1]),
            "fused order must be monotonic when factors are off: {scores:?}"
        );
    }

    #[tokio::test]
    #[ignore = "needs the fixture corpus · VERA_FX_DSN"]
    async fn the_same_query_returns_the_same_answer() {
        // ! CLAUDE.md §2 calls this a deterministic retrieval function, and it
        // was not one. `ORDER BY <distance> LIMIT k` hands back an arbitrary
        // member of any tie group straddling the limit, so 7 of 44 eval
        // queries moved between runs of ONE process and two identical
        // evaluations of one binary scored 52.3% and 54.5%.
        //
        // ! Asserted on the ORDER, ✗ on the set. Membership was stable in most
        // of the observed cases; what moved was which of two tied candidates
        // came first, and that is enough to change every RRF rank downstream.
        let p = pipeline(open_cfg()).await;
        for q in ["bangunan gedung", "jalan", "izin lingkungan", "retribusi"] {
            let first: Vec<String> = p
                .search_with(q, &contract::SearchOptions::default())
                .await
                .expect("search")
                .results
                .into_iter()
                .map(|r| r.id)
                .collect();
            for round in 1..4 {
                let again: Vec<String> = p
                    .search_with(q, &contract::SearchOptions::default())
                    .await
                    .expect("search")
                    .results
                    .into_iter()
                    .map(|r| r.id)
                    .collect();
                assert_eq!(first, again, "{q:?} moved on round {round}");
            }
        }
    }

    #[tokio::test]
    #[ignore = "needs the fixture corpus · VERA_FX_DSN"]
    async fn an_identifier_query_finds_the_named_regulation() {
        // Invariant 4: the exact path is never gated by routing. Run it with
        // the REAL floors, which is the condition that matters.
        let p = pipeline(Config::default()).await;
        let r = p
            .search_with(
                "PERATURAN PEMERINTAH 26 tahun 2009",
                &contract::SearchOptions::default(),
            )
            .await
            .expect("search");
        assert!(
            !r.exact_matches.is_empty(),
            "identifier path returned nothing · progress: {:?}",
            r.progress
        );
    }

    #[tokio::test]
    #[ignore = "needs the fixture corpus · VERA_FX_DSN"]
    async fn an_identifier_query_still_returns_a_usable_result_list() {
        // ! An identifier query carries almost no CONTENT terms -- "PP 26 tahun
        // 2009" reduces to {tahun, 2009}, and the clause bodies contain
        // neither. The relevance floor would therefore drop the entire pool and
        // fall back to a single candidate, so the caller would get one result
        // where a normal query gets ten.
        //
        // Invariant 4 is not violated -- `exact_matches` is a separate channel
        // and carries the named regulation regardless -- but a `results` list
        // that silently collapses for exactly the queries users are most
        // confident about is its own defect.
        let p = pipeline(Config::default()).await;
        let r = p
            .search_with(
                "PERATURAN PEMERINTAH 26 tahun 2009",
                &contract::SearchOptions::default(),
            )
            .await
            .expect("search");
        assert!(!r.exact_matches.is_empty(), "invariant 4");
        assert!(
            r.progress
                .iter()
                .any(|l| l.contains("relevance floor skipped")),
            "the exemption must be stated, not silent: {:?}",
            r.progress
        );
        // Before the exemption this returned 2 of a possible 10.
        assert!(
            r.results.len() >= 5,
            "the floor collapsed the result list: {} results · {:?}",
            r.results.len(),
            r.progress
        );
    }

    #[tokio::test]
    #[ignore = "needs the fixture corpus · VERA_FX_DSN"]
    async fn a_question_this_corpus_cannot_answer_is_refused_not_guessed() {
        // Invariant 13, through the whole pipeline, with the real floors.
        let p = pipeline(Config::default()).await;
        let r = p
            .search_with(
                "resep kue coklat untuk ulang tahun anak",
                &contract::SearchOptions::default(),
            )
            .await
            .expect("search");
        assert!(r.success, "a refusal is an answer, not an error");
        assert!(r.results.is_empty(), "{:?}", r.results);
        assert!(r.hint.is_some(), "a refusal must say why");
    }

    #[tokio::test]
    #[ignore = "needs the fixture corpus · VERA_FX_DSN"]
    async fn results_never_carry_a_synthesised_source_url() {
        // Invariant 8. This fixture genuinely has no URLs, so every result must
        // report None rather than inventing one.
        let p = pipeline(open_cfg()).await;
        let r = p
            .search_with("jalan", &contract::SearchOptions::default())
            .await
            .expect("search");
        assert!(!r.results.is_empty());
        for res in &r.results {
            assert!(res.source.url.is_none(), "invented {:?}", res.source.url);
        }
    }

    #[tokio::test]
    #[ignore = "needs the fixture corpus · VERA_FX_DSN"]
    async fn an_unindexable_chunk_never_reaches_a_result() {
        // seed.py marks one chunk unindexable on purpose. Every arm filters on
        // `indexable`, and this is the test that would catch one that stopped.
        const HIDDEN: &str = "fx-PE-26-2009-99-lampiran";
        let p = pipeline(open_cfg()).await;

        // Its own words, so if any arm stopped filtering this is what surfaces.
        for q in [
            "sanksi administrasi denda cukai lampiran tabel",
            "jalan",
            "energi",
            "bangunan",
            "rapat",
        ] {
            let r = p
                .search_with(q, &contract::SearchOptions::default())
                .await
                .expect("search");
            assert!(
                r.results.iter().all(|res| res.id != HIDDEN),
                "unindexable chunk surfaced for {q:?}"
            );
        }
    }

    #[tokio::test]
    #[ignore = "needs the fixture corpus · VERA_FX_DSN"]
    async fn the_response_serialises_with_success_first() {
        // ! serde_json runs with preserve_order. Without it keys sort
        // alphabetically and the contract's "success first" rule silently stops
        // holding (`docs/OUTPUT_CONTRACT.md` §3).
        let p = pipeline(open_cfg()).await;
        let r = p
            .search_with("jalan", &contract::SearchOptions::default())
            .await
            .expect("search");
        let s = serde_json::to_string(&r).expect("serialize");
        assert!(s.starts_with("{\"success\":"), "{}", &s[..40.min(s.len())]);
    }

    #[tokio::test]
    #[ignore = "needs the fixture corpus · VERA_FX_DSN"]
    async fn a_narrower_pool_cannot_cap_the_answer_below_top_k() {
        // A pool smaller than the answer is a wrong answer, not a slow one.
        let p = pipeline(open_cfg()).await;
        let r = p
            .search_with(
                "jalan",
                &contract::SearchOptions {
                    top_k: Some(5),
                    candidate_pool: Some(1),
                    ..contract::SearchOptions::default()
                },
            )
            .await
            .expect("search");
        let a = r.applied.expect("applied");
        assert!(a.candidate_pool >= a.top_k, "{a:?}");
    }
}
