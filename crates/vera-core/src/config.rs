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
    /// Provider ids proven to reproduce this space by the offline cosine
    /// round-trip preflight (`EMBEDDING.md` §4.5).
    ///
    /// ! An **allowlist**, ✗ a single pin, and that is the whole point.
    /// `EMBEDDING.md` §5 and `LOOPHOLES.md` §4 are one dilemma: pinning a single
    /// provider for consistency makes its outage fatal, but failing over to an
    /// arbitrary host silently changes the vector space. The resolution is to
    /// validate the fallback *offline, in advance*, and let exactly those hosts
    /// serve. Which one is primary is a runtime choice ([`ProviderConfig`]);
    /// which ones are *permitted* is a property of the corpus, recorded here.
    ///
    /// ! Empty means "this corpus predates provider validation", which is
    /// permitted with a warning rather than refused — see
    /// [`assert_provider_validated`](Self::assert_provider_validated).
    #[serde(default)]
    pub validated_providers: Vec<String>,
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
            validated_providers: Vec::new(),
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
            validated_providers: Vec::new(),
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

    /// Refuse a provider this corpus never validated · `EMBEDDING.md` §5,
    /// `LOOPHOLES.md` §4, `CLAUDE.md` §7 rule 7.
    ///
    /// ! Enforced only when the corpus **declares** an allowlist. A corpus built
    /// before validation existed records none, and refusing it would break every
    /// existing index to guard against a risk the operator has not yet opted
    /// into. Once one provider is validated, the list becomes closed: an unknown
    /// host is then a hard failure, because at that point silence means someone
    /// pointed a validated corpus at an unproven provider.
    ///
    /// This is the check that makes a *fallback* safe. Failover picks the next
    /// id from the same list, so an outage cannot widen the space.
    ///
    /// # Errors
    /// [`UnvalidatedProvider`] naming the ids that were proven.
    pub fn assert_provider_validated(&self, provider_id: &str) -> Result<(), UnvalidatedProvider> {
        if self.validated_providers.is_empty()
            || self.validated_providers.iter().any(|p| p == provider_id)
        {
            return Ok(());
        }
        Err(UnvalidatedProvider {
            attempted: provider_id.to_owned(),
            validated: self.validated_providers.clone(),
        })
    }

    /// Whether this corpus has been pinned to any provider at all.
    ///
    /// ! Surfaced so startup can *warn* about the unpinned case rather than let
    /// it pass in silence — an un-enforced guarantee that nobody knows is
    /// un-enforced is the failure mode this whole document set is shaped around.
    #[must_use]
    pub fn provider_is_pinned(&self) -> bool {
        !self.validated_providers.is_empty()
    }
}

/// A provider that was never validated against the corpus's vector space.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnvalidatedProvider {
    pub attempted: String,
    pub validated: Vec<String>,
}

impl std::fmt::Display for UnvalidatedProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // ! Names the remedy. This fires most often not on a genuine mismatch
        // but on a corpus that recorded only the GPU that embedded it, while the
        // query side legitimately runs somewhere else (`EMBEDDING.md` §2) — a
        // configuration mistake, and an error that only says "refusing" leaves
        // the operator to guess which of the two ids is wrong.
        write!(
            f,
            "embedding provider '{}' was never validated against this corpus · \
             validated providers are [{}] · a wrong-space embedding is worse than \
             a brief outage · refusing.\n\
             If '{}' really does serve this corpus's space, record it: run the \
             EMBEDDING.md §4.5 round-trip preflight, then re-ingest with \
             --providers listing every validated host (the query-side ones, not \
             only the GPU that produced the vectors). Otherwise set \
             [provider] primary to one of the listed ids.",
            self.attempted,
            self.validated.join(", "),
            self.attempted,
        )
    }
}

impl std::error::Error for UnvalidatedProvider {}

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
    ///
    /// ! `None` (the default) means **calibrate from the corpus**, which is
    /// almost always what you want: the right value depends on how narrow a
    /// cone the embedding model's vectors occupy, which differs per model and
    /// per corpus. A hardcoded value that is too high rejects every genuine
    /// query and reports it as a *successful* empty result — a silent failure
    /// indistinguishable from "this corpus has no answer". Set `Some` only to
    /// deliberately override a calibration.
    #[serde(default)]
    pub domain_threshold: Option<f32>,
    /// Candidates kept from each probed cluster before fusion.
    #[serde(default = "default_per_cluster_top_k")]
    pub per_cluster_top_k: usize,
    /// How far below p1 to place the calibrated layer-1 threshold, in units of
    /// `(p50 − p1)`. See [`crate::AnchorStats::threshold_at`].
    ///
    /// ! Raising this rejects fewer queries. 0 means "trust p1 exactly", which
    /// measurably over-rejects, because the calibration is taken over documents
    /// and applied to queries.
    #[serde(default = "default_threshold_margin")]
    pub threshold_margin: f32,
}

const fn default_threshold_margin() -> f32 {
    1.0
}

const fn default_clusters_probed() -> usize {
    5
}
const fn default_per_cluster_top_k() -> usize {
    50
}

impl Default for RoutingConfig {
    fn default() -> Self {
        Self {
            clusters_probed: default_clusters_probed(),
            domain_threshold: None,
            per_cluster_top_k: default_per_cluster_top_k(),
            threshold_margin: default_threshold_margin(),
        }
    }
}

/// Fallback layer-1 threshold for a corpus that records no calibration.
///
/// ! Deliberately permissive. An over-tight fallback returns empty for every
/// query and looks like an empty corpus; an over-loose one merely lets an
/// off-topic query through to a search that will rank it poorly anyway. Given
/// the choice, fail loud rather than silent.
pub const FALLBACK_DOMAIN_THRESHOLD: f32 = 0.0;

/// Thresholds behind the published `confidence` signal.
///
/// ! Config, ✗ constants. `EVAL.md` §1 names "the routing-confidence and
/// `confidence:low` thresholds" among the dials that can only be set with
/// evidence — the eval set decides them, so they must be settable without a
/// recompile.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConfidenceConfig {
    /// Top dense cosine at or above which agreement reads as `high`.
    #[serde(default = "default_high_dense")]
    pub high_dense: f32,
    /// Top dense cosine below which the result set reads as `low`.
    #[serde(default = "default_low_dense")]
    pub low_dense: f32,
    /// If the best and worst fused scores differ by less than this, the results
    /// are "clustered" — nothing stands out — which `OUTPUT_CONTRACT.md` §4
    /// treats as a `low` signal even when the absolute scores look acceptable.
    #[serde(default = "default_spread_epsilon")]
    pub spread_epsilon: f32,
}

const fn default_high_dense() -> f32 {
    0.50
}
const fn default_low_dense() -> f32 {
    0.30
}
const fn default_spread_epsilon() -> f32 {
    0.002
}

impl Default for ConfidenceConfig {
    fn default() -> Self {
        Self {
            high_dense: default_high_dense(),
            low_dense: default_low_dense(),
            spread_epsilon: default_spread_epsilon(),
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
    #[serde(default)]
    pub confidence: ConfidenceConfig,
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
            confidence: ConfidenceConfig::default(),
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

/// Which embedding host serves queries, and what it must prove to keep serving.
///
/// ! Separate from [`EmbeddingSpace`] on purpose. The space is a property of the
/// **corpus** — recorded at ingest, identical on every replica, changed only by
/// re-embedding. Which host answers *today* is a property of the **deployment**,
/// and it changes during an outage. Merging them would make failover look like a
/// corpus change, which is exactly the confusion `EMBEDDING.md` §5 warns about.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProviderConfig {
    /// The pinned primary. Must appear in
    /// [`EmbeddingSpace::validated_providers`] when the corpus declares any.
    #[serde(default = "default_provider_id")]
    pub primary: String,
    /// The single pre-validated fallback (`EMBEDDING.md` §5).
    ///
    /// ! One, ✗ a list of "whatever is up". Every candidate costs an offline
    /// round-trip preflight to validate, and an unvalidated fallback is the
    /// failure this field exists to make impossible.
    #[serde(default)]
    pub fallback: Option<String>,
    /// Minimum cosine the startup canary must reach against the stored
    /// reference vector before the engine will serve (`EMBEDDING.md` §4.6).
    #[serde(default = "default_canary_threshold")]
    pub canary_threshold: f32,
    /// Attempts against the primary before failover · a blip is not an outage,
    /// and switching hosts is the more expensive answer.
    #[serde(default = "default_max_retries")]
    pub max_retries: u32,
}

fn default_provider_id() -> String {
    "openrouter/exacto".to_owned()
}
const fn default_canary_threshold() -> f32 {
    0.999
}
const fn default_max_retries() -> u32 {
    3
}

impl Default for ProviderConfig {
    fn default() -> Self {
        Self {
            primary: default_provider_id(),
            fallback: None,
            canary_threshold: default_canary_threshold(),
            max_retries: default_max_retries(),
        }
    }
}

impl ProviderConfig {
    /// The hosts this deployment may use, primary first.
    #[must_use]
    pub fn candidates(&self) -> Vec<&str> {
        let mut out = vec![self.primary.as_str()];
        out.extend(self.fallback.as_deref());
        out
    }

    /// Check every host this deployment could reach against the corpus's
    /// allowlist, ✗ only the one in use.
    ///
    /// ! Checked at **startup**, for the fallback too. Validating only the
    /// primary would pass boot and fail during the outage the fallback exists
    /// for — the one moment nobody is watching a config error.
    ///
    /// # Errors
    /// [`UnvalidatedProvider`] for the first host the corpus does not permit.
    pub fn assert_all_validated(&self, space: &EmbeddingSpace) -> Result<(), UnvalidatedProvider> {
        for id in self.candidates() {
            space.assert_provider_validated(id)?;
        }
        Ok(())
    }
}

/// Bounds on `fetch(depth="full")`.
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
    #[serde(default)]
    pub provider: ProviderConfig,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            embedding: EmbeddingSpace::qwen3_8b(),
            routing: RoutingConfig::default(),
            search: SearchConfig::default(),
            concurrency: ConcurrencyConfig::default(),
            read: ReadConfig::default(),
            provider: ProviderConfig::default(),
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

    fn validated(ids: &[&str]) -> EmbeddingSpace {
        EmbeddingSpace {
            validated_providers: ids.iter().map(|s| (*s).to_owned()).collect(),
            ..EmbeddingSpace::qwen3_06b()
        }
    }

    #[test]
    fn an_unvalidated_provider_is_refused_once_the_corpus_pins_any() {
        // ! CLAUDE.md §7 rule 7. The failure being prevented is not an outage —
        // it is serving *through* one, from a host whose vectors live somewhere
        // else, with every result still looking plausible.
        let space = validated(&["openrouter/exacto"]);
        assert!(space.assert_provider_validated("openrouter/exacto").is_ok());
        let err = space.assert_provider_validated("together").unwrap_err();
        assert_eq!(err.attempted, "together");
        assert!(err.to_string().contains("worse than"), "{err}");
    }

    #[test]
    fn a_corpus_that_pins_nothing_permits_anything_but_says_so() {
        // ! Refusing here would break every corpus built before validation
        // existed. `provider_is_pinned` is what lets startup warn instead.
        let space = EmbeddingSpace::qwen3_06b();
        assert!(!space.provider_is_pinned());
        assert!(space.assert_provider_validated("anything at all").is_ok());
    }

    #[test]
    fn the_fallback_is_validated_at_startup_not_during_the_outage() {
        // ! EMBEDDING.md §5. Checking only the primary passes boot and fails at
        // the exact moment the fallback is needed.
        let space = validated(&["openrouter/exacto"]);
        let cfg = ProviderConfig {
            primary: "openrouter/exacto".to_owned(),
            fallback: Some("some-other-host".to_owned()),
            ..ProviderConfig::default()
        };
        let err = cfg.assert_all_validated(&space).unwrap_err();
        assert_eq!(err.attempted, "some-other-host");

        let ok = ProviderConfig {
            fallback: None,
            ..cfg
        };
        assert!(ok.assert_all_validated(&space).is_ok());
    }

    #[test]
    fn a_validated_pair_lets_failover_stay_inside_the_space() {
        let space = validated(&["openrouter/exacto", "deepinfra"]);
        let cfg = ProviderConfig {
            primary: "openrouter/exacto".to_owned(),
            fallback: Some("deepinfra".to_owned()),
            ..ProviderConfig::default()
        };
        assert!(cfg.assert_all_validated(&space).is_ok());
        assert_eq!(cfg.candidates(), ["openrouter/exacto", "deepinfra"]);
    }

    #[test]
    fn the_provider_pin_survives_a_config_round_trip() {
        let c = Config::from_toml(
            r#"
            [embedding]
            model_id = "qwen/qwen3-embedding-0.6b"
            dim = 1024
            validated_providers = ["openrouter/exacto", "deepinfra"]

            [provider]
            primary = "openrouter/exacto"
            fallback = "deepinfra"
            "#,
        )
        .unwrap();
        assert!(c.provider.assert_all_validated(&c.embedding).is_ok());
        assert!((c.provider.canary_threshold - 0.999).abs() < 1e-6);
    }

    #[test]
    fn the_provider_list_is_not_part_of_the_space_equality_check() {
        // ! Deliberate. The corpus is embedded on a GPU box and queries come
        // from an API host; requiring those ids to match would reject the
        // architecture EMBEDDING.md §2 actually describes. Space equality is
        // about geometry — model, width, normalization — and the allowlist is a
        // separate, one-directional permission check.
        let corpus = validated(&["gpu/local-qwen3"]);
        let engine = validated(&["openrouter/exacto"]);
        assert!(engine.assert_matches(&corpus).is_ok());
    }

    #[test]
    fn a_1024_dim_corpus_needs_a_quarter_the_working_set() {
        let mut c = Config::default();
        c.embedding = EmbeddingSpace::qwen3_06b();
        assert_eq!(c.per_request_ceiling_bytes(10_000), 10_000 * 1024 * 4);
    }
}
