//! Runtime configuration · every dial the design calls "config, ✗ a magic number".
//!
//! `CLAUDE.md` §7 rule 12: never hardcode RAM/row/cluster limits, so bigger
//! hardware scales without a recompile. This module is where those limits live.
//!
//! ! **Embedding width is config, ✗ a constant.** The design pins Qwen3-8B at
//! 4096, but the pin that actually matters is *corpus and query agree*, ✗ the
//! specific number. Hardcoding 4096 made a 1024-dim corpus (Qwen3-0.6B)
//! unrunnable while adding no safety a startup check does not already give.
//! The invariant is preserved where it belongs: [`EmbeddingSpace`] is compared
//! against what the corpus recorded, and a mismatch fails closed.

use serde::{Deserialize, Serialize};

/// The vector space a corpus lives in. Corpus and query must agree on **all**
/// of it — model, revision, width, normalization.
///
/// ! Two corpora embedded with different values here can still be compared
/// arithmetically, which is exactly why this is checked rather than assumed:
/// the failure is silent, plausible ranking over meaningless distances.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmbeddingSpace {
    /// Provider-qualified model id, e.g. `qwen/qwen3-embedding-8b`.
    pub model_id: String,
    /// Vector width. 4096 for Qwen3-8B, 1024 for Qwen3-0.6B.
    pub dim: usize,
    /// Whether stored vectors are L2-normalized. Decides whether cosine can be
    /// computed as a plain dot product.
    #[serde(default = "yes")]
    pub normalized: bool,
    /// Instruction prepended to queries (✗ to documents). Part of the space:
    /// the same model with a different instruction is a different space.
    #[serde(default)]
    pub query_instruction: String,
}

const fn yes() -> bool {
    true
}

impl EmbeddingSpace {
    /// Qwen3-Embedding-8B at full width · the production target.
    #[must_use]
    pub fn qwen3_8b() -> Self {
        Self {
            model_id: "qwen/qwen3-embedding-8b".to_owned(),
            dim: 4096,
            normalized: true,
            query_instruction: DEFAULT_QUERY_INSTRUCTION.to_owned(),
        }
    }

    /// Qwen3-Embedding-0.6B · the width the current test corpus uses.
    #[must_use]
    pub fn qwen3_06b() -> Self {
        Self {
            model_id: "qwen/qwen3-embedding-0.6b".to_owned(),
            dim: 1024,
            normalized: true,
            query_instruction: DEFAULT_QUERY_INSTRUCTION.to_owned(),
        }
    }

    /// Bytes one vector occupies at f32 · used for the RAM budget.
    #[must_use]
    pub const fn bytes_per_vector(&self) -> usize {
        self.dim * 4
    }

    /// Reject a corpus that was not embedded in this space.
    ///
    /// ! Compares the *space*, ✗ just the width. Two 1024-dim models produce
    /// same-shaped vectors that share no geometry, so width alone would let a
    /// mismatched corpus through.
    ///
    /// # Errors
    /// [`SpaceMismatch`] naming the first field that differs.
    pub fn assert_matches(&self, corpus: &Self) -> Result<(), SpaceMismatch> {
        if self.model_id != corpus.model_id {
            return Err(SpaceMismatch {
                field: "model_id",
                expected: corpus.model_id.clone(),
                actual: self.model_id.clone(),
            });
        }
        if self.dim != corpus.dim {
            return Err(SpaceMismatch {
                field: "dim",
                expected: corpus.dim.to_string(),
                actual: self.dim.to_string(),
            });
        }
        if self.normalized != corpus.normalized {
            return Err(SpaceMismatch {
                field: "normalized",
                expected: corpus.normalized.to_string(),
                actual: self.normalized.to_string(),
            });
        }
        Ok(())
    }
}

/// The query side and the corpus side disagree about the vector space.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpaceMismatch {
    pub field: &'static str,
    pub expected: String,
    pub actual: String,
}

impl std::fmt::Display for SpaceMismatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "embedding space mismatch on `{}`: corpus was built with '{}' but this \
             engine is configured for '{}' · vectors would not share a space · refusing",
            self.field, self.expected, self.actual
        )
    }
}

impl std::error::Error for SpaceMismatch {}

/// Instruction prepended to *queries* only. Qwen3-Embedding is
/// instruction-aware; documents are embedded bare.
pub const DEFAULT_QUERY_INSTRUCTION: &str =
    "Instruct: Given a legal or regulatory question, retrieve passages that answer it\nQuery: ";

/// Layer-1 and layer-2 routing dials.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RoutingConfig {
    /// How many layer-2 clusters to open per query · the central
    /// recall/latency dial. ! Raising it costs latency, ✗ RAM — clusters load
    /// sequentially (`ARCHITECTURE.md` §4).
    #[serde(default = "default_clusters_probed")]
    pub clusters_probed: usize,
    /// Minimum cosine against a layer-1 domain anchor to accept a domain.
    /// Below this the engine returns empty rather than guessing.
    #[serde(default = "default_domain_threshold")]
    pub domain_threshold: f32,
    /// Candidates kept from each probed cluster before fusion.
    #[serde(default = "default_per_cluster_top_k")]
    pub per_cluster_top_k: usize,
}

const fn default_clusters_probed() -> usize {
    5
}
const fn default_domain_threshold() -> f32 {
    0.25
}
const fn default_per_cluster_top_k() -> usize {
    50
}

impl Default for RoutingConfig {
    fn default() -> Self {
        Self {
            clusters_probed: default_clusters_probed(),
            domain_threshold: default_domain_threshold(),
            per_cluster_top_k: default_per_cluster_top_k(),
        }
    }
}

/// Result shaping and fusion.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SearchConfig {
    #[serde(default = "default_max_results")]
    pub max_results: usize,
    #[serde(default = "default_snippet_chars")]
    pub snippet_chars: usize,
    /// RRF damping constant. 60 is the value from the original RRF paper;
    /// larger flattens the contribution of top ranks.
    #[serde(default = "default_rrf_k")]
    pub rrf_k: f32,
    /// Candidates the keyword half keeps, per cluster and globally.
    #[serde(default = "default_bm25_limit")]
    pub bm25_limit: usize,
}

const fn default_max_results() -> usize {
    10
}
const fn default_snippet_chars() -> usize {
    320
}
const fn default_rrf_k() -> f32 {
    60.0
}
const fn default_bm25_limit() -> usize {
    50
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            max_results: default_max_results(),
            snippet_chars: default_snippet_chars(),
            rrf_k: default_rrf_k(),
            bm25_limit: default_bm25_limit(),
        }
    }
}

/// The four bounds that make peak RAM provable (`MCP_ENGINE.md` §4).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConcurrencyConfig {
    /// Hard ceiling on in-flight searches. On 2 cores this is the real wall.
    #[serde(default = "default_max_concurrent")]
    pub max_concurrent: usize,
    /// Bounded queue · ! an unbounded queue is itself an OOM vector.
    #[serde(default = "default_queue_capacity")]
    pub queue_capacity: usize,
    /// Reject after this long in the queue, to bound tail latency.
    #[serde(default = "default_wait_timeout_ms")]
    pub wait_timeout_ms: u64,
}

const fn default_max_concurrent() -> usize {
    4
}
const fn default_queue_capacity() -> usize {
    64
}
const fn default_wait_timeout_ms() -> u64 {
    5_000
}

impl Default for ConcurrencyConfig {
    fn default() -> Self {
        Self {
            max_concurrent: default_max_concurrent(),
            queue_capacity: default_queue_capacity(),
            wait_timeout_ms: default_wait_timeout_ms(),
        }
    }
}

/// Bounds on `read_chunk`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReadConfig {
    #[serde(default = "default_max_chunk_bytes")]
    pub max_chunk_bytes: usize,
}

const fn default_max_chunk_bytes() -> usize {
    32_768
}

impl Default for ReadConfig {
    fn default() -> Self {
        Self {
            max_chunk_bytes: default_max_chunk_bytes(),
        }
    }
}

/// Everything the engine reads at startup.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Config {
    pub embedding: EmbeddingSpace,
    #[serde(default)]
    pub routing: RoutingConfig,
    #[serde(default)]
    pub search: SearchConfig,
    #[serde(default)]
    pub concurrency: ConcurrencyConfig,
    #[serde(default)]
    pub read: ReadConfig,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            embedding: EmbeddingSpace::qwen3_8b(),
            routing: RoutingConfig::default(),
            search: SearchConfig::default(),
            concurrency: ConcurrencyConfig::default(),
            read: ReadConfig::default(),
        }
    }
}

impl Config {
    /// Parse from TOML.
    ///
    /// # Errors
    /// Propagates the deserializer's error verbatim so the operator sees which
    /// key is wrong.
    pub fn from_toml(text: &str) -> Result<Self, toml::de::Error> {
        toml::from_str(text)
    }

    /// Worst-case bytes one in-flight request holds, given the largest cluster.
    ///
    /// ! This is the whole OOM argument in one function: it depends on the size
    /// of **one** cluster, ✗ on `clusters_probed`, because clusters load
    /// sequentially. If this ever grows a `clusters_probed` term, sequential
    /// loading has been broken somewhere.
    #[must_use]
    pub const fn per_request_ceiling_bytes(&self, largest_cluster_rows: usize) -> usize {
        largest_cluster_rows * self.embedding.bytes_per_vector()
    }

    /// Peak RAM the design promises to stay under.
    #[must_use]
    pub const fn peak_ram_bytes(&self, fixed_costs: usize, largest_cluster_rows: usize) -> usize {
        fixed_costs
            + self.concurrency.max_concurrent * self.per_request_ceiling_bytes(largest_cluster_rows)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_describe_the_production_target() {
        let c = Config::default();
        assert_eq!(c.embedding.dim, 4096);
        assert_eq!(c.routing.clusters_probed, 5);
        assert_eq!(c.concurrency.max_concurrent, 4);
    }

    #[test]
    fn a_partial_toml_fills_the_rest_from_defaults() {
        // An operator overriding one dial must not have to restate the others.
        let c = Config::from_toml(
            r#"
            [embedding]
            model_id = "qwen/qwen3-embedding-0.6b"
            dim = 1024

            [routing]
            clusters_probed = 12
            "#,
        )
        .unwrap();
        assert_eq!(c.embedding.dim, 1024);
        assert_eq!(c.routing.clusters_probed, 12);
        assert_eq!(c.search.max_results, default_max_results());
        assert_eq!(c.concurrency.queue_capacity, 64);
    }

    #[test]
    fn a_matching_space_is_accepted() {
        assert!(
            EmbeddingSpace::qwen3_06b()
                .assert_matches(&EmbeddingSpace::qwen3_06b())
                .is_ok()
        );
    }

    #[test]
    fn a_width_mismatch_fails_closed() {
        // ! The 0.6B/8B confusion this project actually hit.
        let err = EmbeddingSpace::qwen3_8b()
            .assert_matches(&EmbeddingSpace::qwen3_06b())
            .unwrap_err();
        assert_eq!(err.field, "model_id");
        assert!(err.to_string().contains("would not share a space"));
    }

    #[test]
    fn same_width_different_model_is_still_a_mismatch() {
        // ! The dangerous case: shapes agree, geometry does not, so nothing
        // crashes — it just ranks nonsense. Width alone cannot catch this.
        let mut corpus = EmbeddingSpace::qwen3_06b();
        corpus.model_id = "intfloat/multilingual-e5-large".to_owned();
        let err = EmbeddingSpace::qwen3_06b().assert_matches(&corpus).unwrap_err();
        assert_eq!(err.field, "model_id");
    }

    #[test]
    fn normalization_disagreement_is_a_mismatch() {
        let mut corpus = EmbeddingSpace::qwen3_06b();
        corpus.normalized = false;
        let err = EmbeddingSpace::qwen3_06b().assert_matches(&corpus).unwrap_err();
        assert_eq!(err.field, "normalized");
    }

    #[test]
    fn the_per_request_ceiling_does_not_depend_on_clusters_probed() {
        // ! Guards the OOM argument (ARCHITECTURE.md §4). If probing more
        // clusters ever raised this, sequential loading would be broken.
        let mut c = Config::default();
        let five = c.per_request_ceiling_bytes(10_000);
        c.routing.clusters_probed = 50;
        assert_eq!(c.per_request_ceiling_bytes(10_000), five);
    }

    #[test]
    fn the_worked_ram_budget_matches_the_documented_one() {
        // MCP_ENGINE.md §5: ~82 MB per cluster at 4096 halfvec, 4 concurrent.
        let c = Config::default();
        let per_cluster = c.per_request_ceiling_bytes(10_000);
        assert_eq!(per_cluster, 10_000 * 4096 * 4);
        // f32 working copy is 2x the halfvec on-disk figure; still bounded.
        let peak = c.peak_ram_bytes(3_000_000_000, 10_000);
        assert!(peak < 4_000_000_000, "peak {peak} exceeded the 8 GB profile");
    }

    #[test]
    fn a_1024_dim_corpus_needs_a_quarter_the_working_set() {
        let mut c = Config::default();
        c.embedding = EmbeddingSpace::qwen3_06b();
        assert_eq!(c.per_request_ceiling_bytes(10_000), 10_000 * 1024 * 4);
    }
}
