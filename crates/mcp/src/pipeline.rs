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
    /// How many probed clusters go into one statement.
    ///
    /// ! A **latency** dial, ✗ a memory one. Measured engine peak RSS is flat
    /// across 1/2/5 (14/12/14 MB) because the store returns ranked ids, never
    /// vectors; `5` is worth −182 ms p50 and `2` is worth nothing.
    /// `ARCHITECTURE.md` §4 records what this was previously claimed to do and
    /// why the measurement refuted it. `1` is the default because every
    /// published number was taken at it.
    pub cluster_batch: usize,
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
    /// How many top-ranked candidates sibling expansion walks from.
    ///
    /// ! Seeds, ✗ the whole pool. Expanding 60 candidates at up to 1,313
    /// chunks each is a different query; expanding the few that ranked is the
    /// bet that the right clause sits beside a chunk that already ranked.
    pub expand_seeds: usize,
    /// Sibling rows read per seed, before admission.
    pub expand_per_seed: i64,
    /// Relative influence of each metadata factor (`docs/SCORING.md` §2).
    ///
    /// ! `Weights::OFF` is exactly the identity on the fused order, so this
    /// layer can be disabled in production without a rebuild.
    pub factor_weights: engine::Weights,
    /// What the corpus's own labels mean · the tables `authority`,
    /// `structural` and the term splitter read (`engine::Vocabulary`).
    ///
    /// ! Data, ✗ constants. Vera is not a legal engine; Indonesian regulation
    /// is the corpus it was first built for, and these tables belong beside the
    /// corpus for the same reason the model width does (invariant 2).
    pub vocabulary: engine::Vocabulary,
    /// Every algorithm a caller may name (`contract::algorithms`).
    ///
    /// ! Held here rather than looked up per request. It is validated once, at
    /// startup, like every other limit (invariant 12) — an algorithm whose
    /// prior out-spans the pool must kill the process, ✗ fail the one request
    /// that happens to name it.
    pub algorithms: contract::Registry,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            clusters_probed: 5,
            cluster_batch: 1,
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
            // ! p50 is 22 chunks per regulation and p95 is 210, so 5 seeds x 40
            // is the common case whole and the tail bounded. Measured cost of
            // the widest possible walk — siblings of 10 seeds drawn from the 10
            // largest regulations — is 11,750 rows in 24 ms (`SCORING.md` §7).
            expand_seeds: 5,
            expand_per_seed: 40,
            // Fitted, ✗ chosen: dev_tools/eval/fit_factors.py, +7.5 points
            // leave-one-out over the text arm.
            factor_weights: engine::Weights::FITTED,
            // ! Derived from `factor_weights` immediately above, ✗ a second
            // copy of the fitted numbers. Two literals that must agree are two
            // literals that will eventually disagree.
            vocabulary: engine::Vocabulary::id_regulation(),
            algorithms: contract::Registry::builtin(weights_to_contract(
                &engine::Weights::FITTED,
            )),
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
/// One candidate on its way to being ranked.
///
/// ! Exists because an expanded sibling has **no `Fused`** — no arm retrieved
/// it, so there is no rank to derive a score from. Pairing `(&Fused, &Row)`
/// made that unrepresentable, which is the right default and the wrong one
/// once expansion is admitted.
struct Candidate<'a> {
    /// ! `Cow`, ✗ a reference. A retrieved candidate borrows the row already
    /// fetched for the pool; an expanded one owns a row that was fetched
    /// afterwards and has nowhere older to live. Two vectors with two
    /// lifetimes is the same thing spelled worse.
    row: std::borrow::Cow<'a, store::ChunkRow>,
    /// The retrieval score the factor prior multiplies. For a retrieved
    /// candidate this is RRF's; for a sibling see `admit_siblings`.
    relevance: f32,
    /// `Some(seed id)` when this chunk was expanded rather than retrieved.
    expanded_from: Option<String>,
}

struct Assembly<'a> {
    query: &'a str,
    probed: usize,
    top_cluster: f32,
    results: Vec<SearchResult>,
    /// Candidates that survived scoring, before the cut to `top_k`.
    scored: usize,
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
    ) -> Result<Self, StartupError> {
        let meta = ops.corpus_meta().await?;

        // ! The CORPUS decides, and the engine matches -- the same rule invariant 2
        // applies to the vector space, applied to the tables that rank. Ravel stamps
        // its profile's `scoring:` block into `corpus_meta.scoring_vocabulary`; when
        // it is there it wins outright, because a vocabulary that describes a
        // different corpus evaluates neutrally for every row and the operator's
        // weights then do nothing AT ALL, silently.
        //
        // `VOCABULARY_PATH` is consulted only when the corpus declares nothing --
        // for a corpus loaded before the column existed. A file that could override
        // a corpus's own declaration would reintroduce exactly the drift this
        // replaces; the two copies of the Indonesian ladder had already reached 27
        // entries in Ravel against 10 in the engine.
        let mut cfg = cfg;
        let mut vocab_source = if cfg.vocabulary.is_builtin() {
            "built-in id_regulation"
        } else {
            "VOCABULARY_PATH"
        };
        if let Some(declared) = meta.scoring_vocabulary.as_deref() {
            cfg.vocabulary = crate::vocabulary::CorpusVocabulary::load(declared)
                .map_err(|e| StartupError::CorpusVocabulary { why: e.to_string() })?;
            vocab_source = "corpus_meta (declared by the corpus)";
        }
        let cfg = cfg;

        // ! The guard, here rather than in Settings::validate, because the corpus is
        // only known now. A weight of 0.5 on a factor whose table is empty is not a
        // small effect -- it is NO effect, and it is indistinguishable from a weight
        // that was measured and found not to help.
        let mut unusable: Vec<&str> = Vec::new();
        let mut blamed: Vec<&str> = Vec::new();
        for (name, algo) in &cfg.algorithms.algorithms {
            let missing = cfg.vocabulary.missing_for(&weights_from_contract(&algo.factors));
            if !missing.is_empty() && !blamed.contains(&name.as_str()) {
                blamed.push(name);
            }
            for f in missing {
                if !unusable.contains(&f) {
                    unusable.push(f);
                }
            }
        }
        if !unusable.is_empty() {
            // ! Names the ALGORITHMS as well as the factors. "authority is
            // unusable" sends an operator to the environment; "`balanced` and
            // `sanction` weight authority" sends them to the file that says so.
            return Err(StartupError::FactorWithoutVocabulary {
                algorithms: blamed
                    .iter()
                    .map(|n| format!("`{n}`"))
                    .collect::<Vec<_>>()
                    .join(", "),
                factors: unusable
                    .iter()
                    .map(|f| format!("`{f}`"))
                    .collect::<Vec<_>>()
                    .join(", "),
                plural: if unusable.len() == 1 { "it" } else { "them" },
                from_where: vocab_source,
            });
        }
        eprintln!(
            "[vera] scoring vocabulary \u{b7} {vocab_source} \u{b7} {} authority labels, \
             {} structural rules, {} stopwords",
            cfg.vocabulary.authority.len(),
            cfg.vocabulary.structural.len(),
            cfg.vocabulary.stopwords.len()
        );

        // ! Startup refuses rather than degrading. This is the check whose
        // absence let an unreproducible corpus serve confident nonsense.
        meta.ensure_compatible(model, usize::try_from(meta.dense_dim).unwrap_or(0))?;
        if meta.dense_model != model {
            return Err(store::StoreError::CorpusMismatch {
                corpus_model: meta.dense_model.clone(),
                corpus_dim: meta.dense_dim,
                engine_model: model.to_owned(),
                engine_dim: usize::try_from(meta.dense_dim).unwrap_or(0),
            }.into());
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
                    }.into());
                }
                log_canary(&chunk_id, got);
            }
            Err(e) => {
                // Refuse rather than degrade: serving without having verified
                // the space is the failure mode invariant 2 exists to prevent.
                return Err(store::StoreError::Pool(format!(
                    "canary embed failed · cannot verify the vector space: {e}"
                )).into());
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
        let top_cluster = scores.first().map_or(0.0, |(_, s)| *s);
        let identifiers: Vec<String> = identifier::extract(query)
            .into_iter()
            .map(|i| format!("{}/{}", i.number, i.year.unwrap_or(0)))
            .collect();

        // ! This tool used to report `detected_domain: <corpus id>`
        // unconditionally, having never run the gate — so asked about a query
        // `search_knowledge` REFUSES, it reported a matched domain. The one
        // tool built to make routing falsifiable could not falsify the gate.
        //
        // ! The gate's lexical half needs the sparse arm, so this now costs a
        // real retrieval. A transparency tool that is cheaper than the thing it
        // explains is explaining something else.
        let exempt = !identifiers.is_empty();
        let lexical = if exempt {
            1.0
        } else {
            let sparse = self
                .ops
                .sparse(
                    query,
                    &sparse_literal(&self.vectorizer.query(query), self.vectorizer.dim()),
                    i64::try_from(self.cfg.gate_sample).unwrap_or(i64::MAX),
                )
                .await?;
            let ids: Vec<String> = sparse.iter().map(|s| s.id.clone()).collect();
            let bodies: Vec<String> = self
                .ops
                .chunks_by_id(&ids)
                .await?
                .into_iter()
                .map(|r| r.body)
                .collect();
            self.vectorizer.evidence(query, &bodies)
        };
        let passes = exempt
            || (lexical >= self.cfg.domain_lexical_floor && top_cluster >= self.cfg.domain_floor);

        Ok(RouteExplain {
            // ! `None` when the gate would refuse, matching what
            // `search_knowledge` puts on the wire for the same query.
            domain: passes.then(|| self.meta.id.clone()),
            clusters_probed: scores.iter().map(|(id, _)| *id).collect(),
            cluster_scores: scores,
            total_clusters: self.centroids.len(),
            lexical_evidence: lexical,
            centroid_similarity: top_cluster,
            lexical_floor: self.cfg.domain_lexical_floor,
            centroid_floor: self.cfg.domain_floor,
            would_refuse: !passes,
            gate_bypassed: exempt,
            identifiers,
            provider: self.provider.describe(),
        })
    }

    /// The algorithms this engine will accept, for the tool schema and for
    /// `list_algorithms`.
    #[must_use]
    pub const fn algorithms(&self) -> &contract::Registry {
        &self.cfg.algorithms
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
        let applied = opts.resolve(self.ceilings(), &self.cfg.algorithms)?;
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

        let (dense, sparse, text) = self.arms(query, &qvec, &probed, &mut progress).await?;

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
            return Ok(self.nothing_matched(
                query,
                top_cluster,
                probed.len(),
                exact_matches,
                progress,
            ));
        }

        let ids: Vec<String> = pool.iter().map(|f| f.id.clone()).collect();
        let rows = self.ops.chunks_by_id(&ids).await?;

        // Pair each candidate with its metadata, preserving fused order. A
        // candidate whose row is missing is dropped rather than ranked blind.
        let mut ranked: Vec<Candidate<'_>> = pool
            .iter()
            .filter_map(|f| {
                rows.iter().find(|r| r.id == f.id).map(|row| Candidate {
                    row: std::borrow::Cow::Borrowed(row),
                    relevance: f.score,
                    expanded_from: None,
                })
            })
            .collect();

        // ! The floor is skipped for a named regulation, exactly as the
        // domain gate is. See `apply_factors`.
        let apply_floor = exact_matches.is_empty();
        let vocab = &self.cfg.vocabulary;
        Self::apply_factors(&mut ranked, query, &applied, apply_floor, &mut progress, vocab);

        // ! Expansion runs AFTER ranking, ✗ before. Its seeds are the
        // candidates that actually ranked; walking the whole pool would be a
        // different and much larger query (`SCORING.md` §7).
        if applied.expand.contains(&contract::Expansion::Siblings) {
            self.expand_pool(query, &mut ranked, &applied, apply_floor, &mut progress)
                .await?;
        }

        // ! Counted BEFORE the cut. `truncated` is the only signal a caller
        // has that raising `top_k` would return something it has not seen, and
        // after `truncate` the number that would have said so is gone.
        let scored = ranked.len();
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
            scored,
            exact_matches,
            progress,
            applied,
        }))
    }

    /// Run the three arms.
    ///
    /// ! Only `dense` is routed. `sparse` and `text` scan globally, which is
    /// what makes a routing miss a latency cost rather than a zero-recall one
    /// (`FAILURE_MODES.md` §1) — so this takes the probed clusters and uses
    /// them for exactly one of the three.
    async fn arms(
        &self,
        query: &str,
        qvec: &[f32],
        probed: &[(i32, f32)],
        progress: &mut Vec<String>,
    ) -> Result<(Vec<store::Scored>, Vec<store::Scored>, Vec<store::Scored>), PipelineError> {
        let dense = self.dense_arm(qvec, probed).await?;
        progress.push(format!("dense: {} candidates", dense.len()));

        let sparse = self
            .ops
            .sparse(
                query,
                &sparse_literal(&self.vectorizer.query(query), self.vectorizer.dim()),
                self.cfg.per_arm_k,
            )
            .await?;
        progress.push(format!("sparse: {} candidates", sparse.len()));

        let text = self.ops.text(query, self.cfg.per_arm_k).await?;
        progress.push(format!("text: {} candidates", text.len()));

        Ok((dense, sparse, text))
    }

    /// Admit siblings into the pool, then rank the whole thing again.
    ///
    /// ! Re-ranked as ONE pool. An expanded chunk earns its place against the
    /// same weights as a retrieved one, or it does not get one — appending it
    /// below the retrieved results would make expansion cosmetic.
    async fn expand_pool(
        &self,
        query: &str,
        ranked: &mut Vec<Candidate<'_>>,
        applied: &contract::AppliedOptions,
        apply_floor: bool,
        progress: &mut Vec<String>,
    ) -> Result<(), PipelineError> {
        // ! The REQUEST's floor, ✗ the server's. Admission and rescoring have
        // to use one threshold: an algorithm declaring 0.5 that admitted its
        // siblings at the server's 0.3 would let in candidates its own rescore
        // then drops, paying for a walk whose results cannot survive it.
        let floor = applied.factor_weights.relevance_floor;
        let admitted = self.admit_siblings(query, ranked, floor, progress).await?;
        if admitted.is_empty() {
            return Ok(());
        }
        ranked.extend(admitted);
        Self::apply_factors(ranked, query, applied, apply_floor, progress, &self.cfg.vocabulary);
        Ok(())
    }

    /// Pull in other chunks of the regulations that ranked, and admit the ones
    /// that are relevant on their own evidence.
    ///
    /// `docs/SCORING.md` §7. In 10% of labelled cases the engine held the right
    /// regulation and returned the wrong clause of it; the right clause was
    /// never a candidate, so no amount of re-weighting reaches it.
    ///
    /// ! Admitted, ✗ merged. A sibling arrives with **no retrieval score** —
    /// nothing matched it — and the score it is given here is derived from two
    /// measured quantities and nothing else:
    ///
    /// ```text
    /// relevance(sibling) = seed.relevance × (evidence(sibling) / evidence(seed))
    /// ```
    ///
    /// capped at the best relevance in the pool. A sibling that accounts for
    /// the query's IDF mass as well as its seed did inherits the seed's
    /// standing; one that accounts for half of it gets half. It **may outrank
    /// its seed** — that is the entire point, since the seed is the wrong
    /// clause — but it may not outrank the best thing retrieval actually
    /// found, because it was not found.
    ///
    /// ! Admission is the §3 relevance gate reused, on `bm25::evidence`: one
    /// definition of "relevant to this query" for the whole engine.
    async fn admit_siblings(
        &self,
        query: &str,
        ranked: &[Candidate<'_>],
        floor: f32,
        progress: &mut Vec<String>,
    ) -> Result<Vec<Candidate<'static>>, PipelineError> {
        let seen: Vec<String> = ranked.iter().map(|c| c.row.id.clone()).collect();
        let ceiling = ranked.iter().map(|c| c.relevance).fold(0.0_f32, f32::max);

        // Seeds, de-duplicated by regulation: two chunks of one law walk to the
        // same siblings, and doing it twice costs twice and admits duplicates.
        let mut walked: Vec<store::SiblingKey<'_>> = Vec::new();
        let mut seeds: Vec<(store::SiblingKey<'_>, String, f32, f32)> = Vec::new();
        for c in ranked.iter().take(self.cfg.expand_seeds) {
            let Some(key) = store::SiblingKey::of(&c.row) else {
                continue;
            };
            if walked.contains(&key) {
                continue;
            }
            walked.push(key);
            let evidence = self
                .vectorizer
                .evidence(query, std::slice::from_ref(&c.row.body));
            // ! A seed accounting for none of the query cannot scale anything —
            // the ratio is undefined — so it does not get to sponsor siblings.
            if evidence > 0.0 {
                seeds.push((key, c.row.id.clone(), evidence, c.relevance));
            }
        }

        let mut admitted = Vec::new();
        let mut read = 0;
        for (key, seed_id, seed_evidence, seed_relevance) in seeds {
            let found = self
                .ops
                .siblings(&key, &seen, self.cfg.expand_per_seed)
                .await?;
            read += found.len();
            for row in found {
                let evidence = self
                    .vectorizer
                    .evidence(query, std::slice::from_ref(&row.body));
                if evidence < floor {
                    continue;
                }
                let relevance = (seed_relevance * (evidence / seed_evidence)).min(ceiling);
                admitted.push(Candidate {
                    row: std::borrow::Cow::Owned(row),
                    relevance,
                    expanded_from: Some(seed_id.clone()),
                });
            }
        }
        progress.push(format!(
            "expansion: read {read} siblings of {} regulations, admitted {} at evidence >= {floor:.2}",
            walked.len(),
            admitted.len()
        ));
        Ok(admitted)
    }

    /// The domain matched and retrieval came back empty.
    ///
    /// ! ✗ `no_matching_domain`. The gate has already passed by the time this
    /// is reachable, so reporting a null domain would tell the agent the
    /// corpus does not cover the subject — the one answer that makes it stop
    /// asking (`FAILURE_MODES.md` §11).
    fn nothing_matched(
        &self,
        query: &str,
        top_cluster: f32,
        probed: usize,
        exact_matches: Vec<ExactMatch>,
        progress: Vec<String>,
    ) -> SearchResponse {
        let mut empty = SearchResponse::no_match_in_domain(
            query,
            self.meta.id.clone(),
            top_cluster,
            probed,
            progress,
        );
        empty.exact_matches = exact_matches;
        empty.token_estimate = empty.estimate_tokens();
        empty
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
        ranked: &mut Vec<Candidate<'_>>,
        query: &str,
        applied: &contract::AppliedOptions,
        apply_floor: bool,
        progress: &mut Vec<String>,
        vocab: &engine::Vocabulary,
    ) {
        let before = ranked.first().map(|c| c.row.id.clone());
        // ! `content_terms`, ✗ `bm25::tokenize`. The relevance floor was fitted
        // against the former — words longer than three characters with function
        // words removed — and the two tokenisers disagree on short words, so
        // the wrong one applies a threshold nothing measured.
        let terms = engine::factors::content_terms(query, vocab);
        let term_refs: Vec<&str> = terms.iter().map(String::as_str).collect();

        let scored_len_before = ranked.len();
        let mut scored: Vec<(f32, engine::Facets<'_>, usize)> = ranked
            .iter()
            .enumerate()
            .map(|(i, c)| (c.relevance, Self::facets_of(&c.row), i))
            .collect();
        let mut weights = weights_from_contract(&applied.factor_weights);
        if !apply_floor {
            weights.relevance_floor = 0.0;
            progress.push("relevance floor skipped · the query names a regulation".into());
        }
        engine::rescore(&mut scored, &weights, &term_refs, vocab);

        let kept = scored.len();
        let order: Vec<usize> = scored.iter().map(|(_, _, i)| *i).collect();
        let mut taken: Vec<Option<Candidate<'_>>> = ranked.drain(..).map(Some).collect();
        *ranked = order.into_iter().filter_map(|i| taken[i].take()).collect();

        let dropped = scored_len_before.saturating_sub(kept);
        if dropped > 0 {
            progress.push(format!(
                "relevance floor dropped {dropped} of {scored_len_before} candidates"
            ));
        }
        if let (Some(was), Some(now)) = (before, ranked.first())
            && now.row.id != was
        {
            progress.push(format!("factors promoted {} over {was}", now.row.id));
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

    /// Scan the probed clusters in batches of `CLUSTER_BATCH`.
    ///
    /// ! This loop is **not** the OOM guarantee, though it was documented as one.
    /// The store returns `(id, score)` rows under a `LIMIT`, so no shape of this
    /// loop puts a cluster in this process — measured peak RSS is flat across
    /// `cluster_batch` (`docs/HARDWARE.md` §6a). What the loop decides is how
    /// many round trips the dense arm costs.
    ///
    /// ! `cluster_batch` is validated at startup, so the chunk size here cannot
    /// be zero and the loop cannot spin.
    async fn dense_arm(
        &self,
        qvec: &[f32],
        probed: &[(i32, f32)],
    ) -> Result<Vec<store::Scored>, PipelineError> {
        let batch = self.cfg.cluster_batch.max(1);
        let ids: Vec<i32> = probed.iter().map(|(id, _)| *id).collect();
        let mut dense: Vec<store::Scored> = Vec::new();
        for window in ids.chunks(batch) {
            // ! per_cluster_k × the window, ✗ per_cluster_k. The store applies
            // ONE limit across the whole window, so the per-cluster figure
            // would make a batch of 5 yield a fifth of the rows five separate
            // calls yield.
            //
            // ! At the shipped defaults that is invisible — per_arm_k equals
            // per_cluster_k, so both spellings truncate to the same true top 20
            // (verified against the corpus: identical ids and scores). It bites
            // the moment PER_ARM_K > PER_CLUSTER_K, where the unscaled version
            // hands back fewer candidates than per_arm_k asked for and a memory
            // knob silently becomes a recall knob. Scaling is also strictly
            // better than the sequential loop there: one window returns the true
            // top k×m, where five separate calls return a per-cluster quota.
            let k = self.cfg.per_cluster_k * i64::try_from(window.len()).unwrap_or(1);
            dense.extend(self.ops.dense_in_clusters(window, qvec, k).await?);
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
        ranked: &[Candidate<'_>],
        dense: &[store::Scored],
        sparse: &[store::Scored],
    ) -> Vec<SearchResult> {
        let score_in = |arm: &[store::Scored], id: &str| {
            arm.iter().find(|s| s.id == id).map_or(0.0, |s| s.score)
        };
        ranked
            .iter()
            .map(|c| {
                let row = &c.row;
                SearchResult {
                    id: row.id.clone(),
                    snippet: snippet(&row.body, self.cfg.snippet_chars),
                    score: c.relevance,
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
                    expanded_from: c.expanded_from.clone(),
                }
            })
            .collect()
    }

    fn assemble(&self, parts: Assembly<'_>) -> SearchResponse {
        let Assembly {
            query,
            probed,
            top_cluster,
            results,
            scored,
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
        let results_len = results.len();
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
            // ! Was hardcoded `false` while `top_k` dropped candidates
            // silently, so a caller could not tell a complete answer from the
            // visible tenth of one.
            truncated: scored > results_len,
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
    /// `None` when the gate would refuse · the same value `search_knowledge`
    /// reports for this query, ✗ the corpus id unconditionally.
    pub domain: Option<String>,
    pub clusters_probed: Vec<i32>,
    pub cluster_scores: Vec<(i32, f32)>,
    pub total_clusters: usize,
    /// Both halves of the gate, with the thresholds they were compared to, so
    /// a refusal can be attributed to one of them rather than guessed at.
    pub lexical_evidence: f32,
    pub centroid_similarity: f32,
    pub lexical_floor: f32,
    pub centroid_floor: f32,
    pub would_refuse: bool,
    /// The query names a regulation, so the gate is skipped (invariant 4).
    pub gate_bypassed: bool,
    pub identifiers: Vec<String>,
    pub provider: String,
}

#[derive(Debug, thiserror::Error)]
pub enum PipelineError {
    #[error(transparent)]
    Store(#[from] store::StoreError),

    #[error(transparent)]
    Embed(#[from] embed::EmbedError),

    /// The caller named an algorithm this engine does not serve.
    ///
    /// ! A request error, ✗ a startup one: the registry was valid, the name was
    /// not. It carries the full list so the `hint` can tell the agent what it
    /// may name instead.
    #[error(transparent)]
    Options(#[from] contract::ResolveError),
}

/// Why the engine would not start.
///
/// ! A separate type from [`store::StoreError`], and it exists because reusing
/// `CorpusMismatch` for a configuration fault produced the message
///
/// ```text
/// corpus was embedded with 'a corpus declaring no table for: authority,
/// structural' at 0 dims but this engine ... at 0 dims
/// ```
///
/// which is not what happened and sends the reader to the wrong file. A refusal
/// is the one message an operator reads under time pressure; `docs/OPERATIONS.md`
/// asks every one of them to name the variable and the value.
#[derive(Debug, thiserror::Error)]
pub enum StartupError {
    #[error(transparent)]
    Store(#[from] store::StoreError),

    /// `corpus_meta.scoring_vocabulary` holds something this engine cannot score
    /// with.
    #[error(
        "the corpus declares a scoring vocabulary this engine cannot use \u{b7} {why} \u{b7} \
         it was written by Ravel from its profile, so the profile is where to fix it"
    )]
    CorpusVocabulary { why: String },

    /// A weight asks for a factor the live vocabulary has no table for.
    ///
    /// ! Fatal, and this is the whole reason a vocabulary is declared. A weight of
    /// 0.5 on a factor whose table is empty is not a small effect -- it is NO
    /// effect, and it is indistinguishable from a weight that was measured and
    /// found not to help. This project shipped that bug once already, with
    /// `topical` fitted at 0.25 against a column the engine never selected.
    #[error(
        "algorithm(s) {algorithms} weight {factors}, but the scoring vocabulary in use \
         ({from_where}) declares no table for {plural} \u{b7} every candidate would score 0.0 \
         and the weight would do nothing at all. Either set those weights to 0, or \
         declare the table -- in the corpus profile if Ravel built this corpus, \
         otherwise in VOCABULARY_PATH"
    )]
    FactorWithoutVocabulary {
        algorithms: String,
        factors: String,
        plural: &'static str,
        // ! Named `from_where`, ✗ `source`. thiserror treats a field called
        // `source` as the error's CAUSE and requires it to implement Error, so
        // the obvious name silently changes what the type means.
        from_where: &'static str,
    },
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
    async fn truncated_says_whether_raising_top_k_would_show_more() {
        // ! It was hardcoded `false`. A caller cannot tell a complete answer
        // from the visible tenth of one, and the only way to find out was to
        // re-ask with a bigger `top_k` and compare — which is the work this
        // field exists to save.
        let p = pipeline(open_cfg()).await;
        let narrow = p
            .search_with(
                "bangunan gedung",
                &contract::SearchOptions {
                    top_k: Some(1),
                    ..contract::SearchOptions::default()
                },
            )
            .await
            .expect("search");
        assert_eq!(narrow.results.len(), 1);
        assert!(
            narrow.truncated,
            "one result out of a 60-candidate pool is a cut, and must say so"
        );

        // And the converse: a pool that fits must NOT claim it was cut.
        let whole = p
            .search_with(
                "bangunan gedung",
                &contract::SearchOptions {
                    top_k: Some(10),
                    candidate_pool: Some(1),
                    ..contract::SearchOptions::default()
                },
            )
            .await
            .expect("search");
        assert!(
            !whole.truncated,
            "nothing was dropped · {} results",
            whole.results.len()
        );
    }

    #[tokio::test]
    #[ignore = "needs the fixture corpus · VERA_FX_DSN"]
    async fn explain_routing_agrees_with_what_search_would_do() {
        // ! `explain_routing` reported the corpus id unconditionally — it
        // never ran the gate. Asked about a query `search_knowledge` refuses,
        // the transparency tool said the domain matched, so the one tool built
        // to make routing falsifiable could not falsify the gate.
        let p = pipeline(Config::default()).await;
        for q in [
            "bangunan gedung",
            "what is the capital of France",
            "cara memperbaiki keran air yang bocor di dapur",
            "PP 26 tahun 2009",
        ] {
            let explained = p.explain(q).await.expect("explain");
            let searched = p
                .search_with(q, &contract::SearchOptions::default())
                .await
                .expect("search");
            assert_eq!(
                explained.domain, searched.detected_domain,
                "the two tools disagree about {q:?}"
            );
            assert_eq!(
                explained.would_refuse,
                searched.detected_domain.is_none(),
                "would_refuse must mean what it says for {q:?}"
            );
        }
    }

    #[tokio::test]
    #[ignore = "needs the fixture corpus · VERA_FX_DSN"]
    async fn the_gate_reports_which_half_refused() {
        // A refusal the caller cannot attribute is a refusal they cannot act
        // on: widening the wording and picking a different engine are
        // different responses to the two halves.
        let p = pipeline(Config::default()).await;
        let r = p
            .explain("what is the capital of France")
            .await
            .expect("explain");
        assert!(r.would_refuse, "junk must not clear the gate");
        assert!(
            r.lexical_evidence < r.lexical_floor || r.centroid_similarity < r.centroid_floor,
            "would_refuse with both halves passing is incoherent: {r:?}"
        );
    }

    #[tokio::test]
    #[ignore = "needs the fixture corpus · VERA_FX_DSN"]
    async fn an_identifier_bypasses_the_gate_in_both_tools() {
        // Invariant 4, asserted on the explanation as well as the answer.
        let p = pipeline(Config::default()).await;
        let r = p.explain("PP 26 tahun 2009").await.expect("explain");
        assert!(r.gate_bypassed, "a named regulation skips the gate");
        assert!(!r.would_refuse);
        assert!(r.domain.is_some());
    }

    /// ! A NARROW pool, deliberately. The fixture holds 32 indexable chunks
    /// and at most 4 per regulation, so the default 60-candidate pool contains
    /// the entire corpus — every sibling is already retrieved, `exclude`
    /// removes all of them, and expansion is unexercisable. Narrowing the pool
    /// is what puts chunks outside it for the walk to find.
    ///
    /// `candidate_pool` is floored at `top_k` by `resolve`, so both move.
    fn siblings() -> contract::SearchOptions {
        contract::SearchOptions {
            expand: Some(vec![contract::Expansion::Siblings]),
            top_k: Some(3),
            candidate_pool: Some(3),
            ..contract::SearchOptions::default()
        }
    }

    fn narrow() -> contract::SearchOptions {
        contract::SearchOptions {
            expand: None,
            ..siblings()
        }
    }

    #[tokio::test]
    #[ignore = "needs the fixture corpus · VERA_FX_DSN"]
    async fn expansion_is_off_unless_asked_for() {
        // ! An expansion admits chunks NO ARM RETRIEVED. Doing that silently
        // would change what "the engine found this" means for every caller
        // that never opted in.
        let p = pipeline(open_cfg()).await;
        let r = p
            .search_with("bangunan gedung", &contract::SearchOptions::default())
            .await
            .expect("search");
        assert!(
            r.results.iter().all(|x| x.expanded_from.is_none()),
            "nothing may be expanded without `expand`"
        );
        assert!(
            !r.progress.iter().any(|l| l.contains("expansion")),
            "and the walk must not run: {:?}",
            r.progress
        );
    }

    #[tokio::test]
    #[ignore = "needs the fixture corpus · VERA_FX_DSN"]
    async fn an_expanded_chunk_says_which_result_it_came_from() {
        // ! The provenance is the point. A sibling was admitted on its own
        // evidence, not retrieved, and a caller weighing it as a hit is
        // drawing a stronger conclusion than the evidence supports.
        let p = pipeline(open_cfg()).await;
        let r = p
            .search_with("bangunan gedung", &siblings())
            .await
            .expect("search");
        assert!(
            r.progress.iter().any(|l| l.contains("expansion")),
            "the walk must report what it read and admitted: {:?}",
            r.progress
        );
        let ids: Vec<&str> = r.results.iter().map(|x| x.id.as_str()).collect();
        for x in &r.results {
            if let Some(seed) = &x.expanded_from {
                assert_ne!(seed, &x.id, "a chunk cannot be its own seed");
                assert!(
                    ids.iter().filter(|i| **i == x.id).nth(1).is_none(),
                    "an expanded chunk must not also appear as a retrieved one"
                );
            }
        }
    }

    #[tokio::test]
    #[ignore = "needs the fixture corpus · VERA_FX_DSN"]
    async fn expansion_actually_admits_something_on_the_fixture() {
        // ! Without this the three tests around it pass vacuously: every
        // assertion about expanded chunks holds trivially when none exist.
        let p = pipeline(open_cfg()).await;
        let mut seen = 0;
        for q in ["bangunan gedung", "jalan", "retribusi", "izin lingkungan"] {
            let r = p.search_with(q, &siblings()).await.expect("search");
            seen += r
                .results
                .iter()
                .filter(|x| x.expanded_from.is_some())
                .count();
        }
        assert!(seen > 0, "the walk admitted nothing across four queries");
    }

    #[tokio::test]
    #[ignore = "needs the fixture corpus · VERA_FX_DSN"]
    async fn expansion_only_ever_adds_candidates() {
        // ! Expansion widens the pool; it must not cost a retrieved result its
        // place by some accounting slip. Ranking may reorder freely, but every
        // id the un-expanded search returned must still be somewhere in the
        // expanded pool's answer or have been outranked by a real candidate.
        let p = pipeline(open_cfg()).await;
        let plain = p
            .search_with("bangunan gedung", &narrow())
            .await
            .expect("search");
        let grown = p
            .search_with("bangunan gedung", &siblings())
            .await
            .expect("search");
        assert!(
            grown.results.len() >= plain.results.len(),
            "expansion returned fewer results: {} -> {}",
            plain.results.len(),
            grown.results.len()
        );
    }

    #[tokio::test]
    #[ignore = "needs the fixture corpus · VERA_FX_DSN"]
    async fn an_expanded_chunk_belongs_to_a_regulation_that_ranked() {
        // ! The walk is keyed on (type, number, year), all three. Number and
        // year alone match 12 regulations on the live corpus, so a two-part key
        // would pull in clauses of a different law and label them siblings.
        let p = pipeline(open_cfg()).await;
        let r = p
            .search_with("bangunan gedung", &siblings())
            .await
            .expect("search");
        let seeds: Vec<&str> = r
            .results
            .iter()
            .filter(|x| x.expanded_from.is_none())
            .map(|x| x.id.as_str())
            .collect();
        for x in &r.results {
            if let Some(seed) = &x.expanded_from {
                assert!(
                    seeds.contains(&seed.as_str()) || r.results.iter().any(|o| &o.id == seed),
                    "{} claims a seed that is not in the answer: {seed}",
                    x.id
                );
            }
        }
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
