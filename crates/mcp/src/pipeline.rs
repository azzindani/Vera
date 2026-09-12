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
    Arm, Centroid, DEFAULT_K, Trust, reciprocal_rank_fusion, select_clusters, trust_from,
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
    pub snippet_chars: usize,
    pub dense_weight: f32,
    pub sparse_weight: f32,
    pub text_weight: f32,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            clusters_probed: 5,
            per_cluster_k: 20,
            per_arm_k: 20,
            top_k: 10,
            snippet_chars: 280,
            // ! Measured, ✗ chosen. On the labeled query set in `eval/`, with
            // this corpus and this model:
            //
            //   sparse      40.0% Recall@5   MRR 0.301
            //   RRF(all)    30.0%            MRR 0.188
            //   dense        0.0%            MRR 0.000
            //   text         0.0%            MRR 0.007
            //
            // Equal weights let two arms that find nothing outvote the one that
            // works, demoting correct answers (q003 rank 4 → 11, q007 11 → 32).
            // So they are off until they earn their place: re-embedding with
            // contextual headers is the open candidate for dense
            // (`EVAL.md` §4). Override with DENSE_WEIGHT / TEXT_WEIGHT.
            dense_weight: 0.0,
            sparse_weight: 1.0,
            text_weight: 0.0,
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

    /// The routing decision alone · backs `explain_routing`.
    ///
    /// # Errors
    /// Embedding or database failure.
    pub async fn explain(&self, query: &str) -> Result<RouteExplain, PipelineError> {
        let q = self.provider.embed_query(query).await?;
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

        let qvec = self.provider.embed_query(query).await?;
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
        let top: Vec<_> = fused.into_iter().take(self.cfg.top_k).collect();
        progress.push(format!("fused to {} results", top.len()));

        if top.is_empty() {
            let mut empty = SearchResponse::no_matching_domain(query, progress);
            empty.exact_matches = exact_matches;
            return Ok(empty);
        }

        let ids: Vec<String> = top.iter().map(|f| f.id.clone()).collect();
        let rows = self.ops.chunks_by_id(&ids).await?;
        let results = self.to_results(&top, &rows, &dense, &sparse);

        Ok(self.assemble(query, probed.len(), results, exact_matches, progress))
    }

    /// Scan the probed clusters **one at a time**.
    ///
    /// ! This loop is the OOM guarantee (`MCP_ENGINE.md` §5): a request holds
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
            let hits = self.ops.exact_identifier(&id.number, id.year, 5).await?;
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

    fn to_results(
        &self,
        top: &[engine::Fused],
        rows: &[store::ChunkRow],
        dense: &[store::Scored],
        sparse: &[store::Scored],
    ) -> Vec<SearchResult> {
        let score_in = |arm: &[store::Scored], id: &str| {
            arm.iter().find(|s| s.id == id).map_or(0.0, |s| s.score)
        };
        top.iter()
            .filter_map(|f| rows.iter().find(|r| r.id == f.id).map(|row| (f, row)))
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
        let hint = (incomplete > 0).then(|| {
            format!(
                "{incomplete} of {} results have no source_url · this corpus was \
                 ingested without one · cite by title and locator, and treat the \
                 link as unavailable",
                results.len()
            )
        });

        let mut resp = SearchResponse {
            success: true,
            op: "search_knowledge",
            query: query.to_owned(),
            detected_domain: Some(self.meta.id.clone()),
            domain_confidence: 1.0,
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
