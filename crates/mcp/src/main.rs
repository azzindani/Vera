//! `vera-mcp` · the MCP server: transport and tool dispatch, no domain logic.
//!
//! ! Every log line goes to **stderr** (`CLAUDE.md` §7.10). stdout carries the
//! JSON-RPC stream and nothing else; one stray `println!` corrupts the channel
//! and the client sees a protocol error rather than a message.
//!
//! All configuration is read from the environment (invariant 12) and resolved
//! once, at startup, by [`Settings::from_env`]. See `docs/operations/CONFIGURATION.md` for
//! the full table and `.env.example` for a working set.

mod bm25;
mod domains;
mod http;
mod identifier;
mod pipeline;
mod protocol;
mod region;
mod tools;
mod vocabulary;

use std::sync::Arc;

use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use pipeline::{Config, Pipeline};

/// JSON-RPC codes and framing live in `protocol`, which is testable without a
/// corpus. `BUSY_CODE` is re-exported because `http` maps it to 503.
pub(crate) use protocol::BUSY_CODE;

/// Log to stderr. ! Never stdout.
macro_rules! log {
    ($($arg:tt)*) => { eprintln!($($arg)*) };
}

/// Connections one permitted search may hold: the three arms run concurrently.
const ARM_FANOUT: usize = 3;

/// A required variable is missing, or one that is set cannot be parsed.
///
/// ! Both are fatal, ✗ defaulted. A typo in `MAX_CONCURRENCY` that silently
/// became 4 would be a memory bound nobody chose, and a `DATABASE_URL` that
/// defaults to localhost is how a container ends up serving an empty corpus and
/// saying nothing about it.
#[derive(Debug, thiserror::Error)]
enum ConfigError {
    #[error("{0} is not set · see .env.example")]
    Missing(&'static str),
    #[error("{key}={value:?} is not a valid {kind}")]
    Invalid {
        key: &'static str,
        value: String,
        kind: &'static str,
    },
    /// `VOCABULARY_PATH` names a file that cannot be read, or one whose
    /// contents the engine will not score with.
    #[error("VOCABULARY_PATH={path} · {why}")]
    Vocabulary { path: String, why: String },
    /// `ALGORITHMS_PATH` names a file that cannot be read, or one whose contents
    /// are not a registry this engine will serve.
    ///
    /// ! Fatal, like every other configuration fault. An algorithm whose prior
    /// out-spans the candidate pool ranks on metadata alone (invariant 9), and
    /// the request that would reveal it is whichever one happens to name it —
    /// possibly weeks later. Refuse at startup, where a rollout can still see.
    #[error("ALGORITHMS_PATH={path} · {why}")]
    Algorithms { path: String, why: String },
}

/// A variable that must be set. There is no sensible default for any of these:
/// each names something outside the process that only the operator knows.
fn required(key: &'static str) -> Result<String, ConfigError> {
    std::env::var(key)
        .ok()
        .filter(|v| !v.trim().is_empty())
        .ok_or(ConfigError::Missing(key))
}

/// A variable with a defensible default, parsed strictly.
fn parsed<T: std::str::FromStr>(
    key: &'static str,
    kind: &'static str,
    fallback: T,
) -> Result<T, ConfigError> {
    match std::env::var(key) {
        Err(_) => Ok(fallback),
        Ok(v) if v.trim().is_empty() => Ok(fallback),
        Ok(v) => v.trim().parse().map_err(|_| ConfigError::Invalid {
            key,
            value: v,
            kind,
        }),
    }
}

/// The corpus's scoring vocabulary · a file when `VOCABULARY_PATH` names one,
/// otherwise the tables of the corpus this engine was first built for.
///
/// ! Unset is supported and **reported**. An operator serving a corpus that is
/// not Indonesian regulation gets a line at startup saying which vocabulary is
/// in use, because the failure it prevents is silent: every factor evaluates
/// neutrally and the weights do nothing.
fn load_vocabulary() -> Result<engine::Vocabulary, ConfigError> {
    let Some(path) = std::env::var("VOCABULARY_PATH")
        .ok()
        .filter(|v| !v.trim().is_empty())
    else {
        return Ok(engine::Vocabulary::id_regulation());
    };
    let path = path.trim();
    let body = std::fs::read_to_string(path).map_err(|e| ConfigError::Vocabulary {
        path: path.to_owned(),
        why: e.to_string(),
    })?;
    vocabulary::VocabularyFile::load(&body).map_err(|e| ConfigError::Vocabulary {
        path: path.to_owned(),
        why: e.to_string(),
    })
}

/// The algorithm registry · a file when `ALGORITHMS_PATH` names one, otherwise
/// exactly today's behaviour under the name every caller already gets.
///
/// ! Unset is a supported deployment, ✗ a degraded one. A corpus with one
/// fitted way to rank needs no file, and requiring one would mean every
/// deployment carries a copy of numbers the binary already has.
fn load_algorithms(env_weights: &engine::Weights) -> Result<contract::Registry, ConfigError> {
    load_algorithms_from(
        std::env::var("ALGORITHMS_PATH").ok().as_deref(),
        env_weights,
    )
}

/// [`load_algorithms`] from a named file, for a domain whose `DOMAINS_FILE` entry names
/// its own: Indonesia ranks by `balanced`, a corpus that declares no prior by `no_prior`.
fn load_algorithms_from(
    path: Option<&str>,
    env_weights: &engine::Weights,
) -> Result<contract::Registry, ConfigError> {
    let fallback = contract::Registry::builtin(contract::FactorWeights {
        relevance_floor: env_weights.relevance_floor,
        authority: env_weights.authority,
        structural: env_weights.structural,
        temporal: env_weights.temporal,
        completeness: env_weights.completeness,
        topical: env_weights.topical,
    });
    let Some(path) = path.filter(|v| !v.trim().is_empty()) else {
        // ! Validated too. `builtin` is assembled from the FACTOR_* variables,
        // so an operator can push the prior bound past MAX_PRIOR_BOUND without
        // ever writing a registry file -- and `.env.example` itself shipped
        // values summing to 2.75x while this was unchecked. A rule enforced on
        // one of two construction paths is not enforced.
        fallback.validate().map_err(|e| ConfigError::Algorithms {
            path: "<FACTOR_* environment>".to_owned(),
            why: e.to_string(),
        })?;
        return Ok(fallback);
    };
    let path = path.trim();
    let body = std::fs::read_to_string(path).map_err(|e| ConfigError::Algorithms {
        path: path.to_owned(),
        why: e.to_string(),
    })?;
    // ! `parse` validates. A Registry that exists has been checked, so nothing
    // downstream re-checks it and nothing downstream can forget to.
    contract::Registry::parse(&body).map_err(|e| ConfigError::Algorithms {
        path: path.to_owned(),
        why: e.to_string(),
    })
}

/// Which transport the process serves on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Transport {
    Stdio,
    Http,
}

impl std::str::FromStr for Transport {
    type Err = ();
    fn from_str(s: &str) -> Result<Self, ()> {
        match s {
            "stdio" => Ok(Self::Stdio),
            "http" => Ok(Self::Http),
            _ => Err(()),
        }
    }
}

/// Everything the process reads from its environment, resolved once.
///
/// ! Resolved up front, ✗ read where needed. A process that discovers a missing
/// variable on its first request has already told its orchestrator it is
/// healthy; this way a misconfigured deployment dies at startup, where a
/// rollout can still see it.
#[derive(Debug)]
struct Settings {
    database_url: String,
    embed_endpoint: String,
    bm25_vocab: std::path::PathBuf,
    /// `DOMAINS_FILE`: the domains to serve, when more than the one `DATABASE_URL` names
    /// (`docs/operations/SCALING.md` §4a). `None` serves exactly that one, as before.
    domains_file: Option<std::path::PathBuf>,
    max_concurrency: usize,
    transport: Transport,
    http_addr: String,
    queue_wait: std::time::Duration,
    read_chunk_chars: usize,
    max_provenance_ids: usize,
    /// `DOMAIN_SELECT_LEXICAL`: with several domains, the lexical evidence below which a
    /// domain answers only if no other serves (`domains::gates`).
    domain_select_lexical: f32,
    statement_timeout: u64,
    pipeline: Config,
}

impl Settings {
    fn from_env() -> Result<Self, ConfigError> {
        let d = Config::default();
        let mut s = Self {
            // ! No defaults. Each names something the process cannot guess, and
            // a wrong guess fails silently rather than loudly: the wrong
            // vocabulary compares unrelated sparse dimensions, and the wrong
            // database serves a corpus the operator did not intend.
            database_url: required("DATABASE_URL")?,
            embed_endpoint: required("EMBED_ENDPOINT")?,
            bm25_vocab: required("BM25_VOCAB")?.into(),
            domains_file: std::env::var("DOMAINS_FILE")
                .ok()
                .filter(|v| !v.trim().is_empty())
                .map(|v| v.trim().into()),

            max_concurrency: parsed("MAX_CONCURRENCY", "positive integer", 4usize)?,
            transport: parsed("TRANSPORT", "transport (stdio|http)", Transport::Stdio)?,
            http_addr: std::env::var("HTTP_ADDR").unwrap_or_else(|_| "0.0.0.0:8081".into()),
            queue_wait: std::time::Duration::from_millis(parsed(
                "QUEUE_WAIT_MS",
                "duration in milliseconds",
                2000u64,
            )?),
            read_chunk_chars: parsed("READ_CHUNK_CHARS", "positive integer", 4000usize)?,
            max_provenance_ids: parsed("MAX_PROVENANCE_IDS", "positive integer", 50usize)?,
            domain_select_lexical: parsed("DOMAIN_SELECT_LEXICAL", "number", 0.2f32)?,
            // ! The third concurrency bound. The semaphore caps what runs and
            // QUEUE_WAIT_MS caps what waits; without this nothing caps how LONG
            // a request runs, and a slow query holds its permit for its whole
            // duration.
            statement_timeout: parsed(
                "STATEMENT_TIMEOUT_MS",
                "duration in milliseconds",
                15_000u64,
            )?,

            pipeline: Config {
                clusters_probed: parsed("CLUSTERS_PROBED", "positive integer", d.clusters_probed)?,
                // ! 0 is valid and means one level: every leaf held and scanned, the
                // routing every number before 2026-11-14 was measured with.
                parents_probed: parsed("PARENTS_PROBED", "integer", d.parents_probed)?,
                two_level_from: parsed("TWO_LEVEL_FROM", "integer", d.two_level_from)?,
                named_chunks: parsed("NAMED_CHUNKS", "positive integer", d.named_chunks)?,
                // ! Config, ✗ a constant, because §7.12 forbids a hardcoded
                // limit — not because it trades memory for speed. Measured
                // engine RSS is flat across it; what it buys is −182 ms
                // (ARCHITECTURE.md §4).
                cluster_batch: parsed("CLUSTER_BATCH", "positive integer", d.cluster_batch)?,
                per_cluster_k: parsed("PER_CLUSTER_K", "positive integer", d.per_cluster_k)?,
                per_arm_k: parsed("PER_ARM_K", "positive integer", d.per_arm_k)?,
                top_k: parsed("TOP_K", "positive integer", d.top_k)?,
                // ! Metadata is fetched for this many candidates, not for
                // TOP_K. Raising TOP_K above it silently returns fewer
                // results than asked for, which validate() rejects.
                candidate_pool: parsed("CANDIDATE_POOL", "positive integer", d.candidate_pool)?,
                snippet_chars: parsed("SNIPPET_CHARS", "positive integer", d.snippet_chars)?,
                // Invariant 12: read from the environment, ✗ hardcoded. Both
                // bound how much a single `expand` walk may read.
                expand_seeds: parsed("EXPAND_SEEDS", "positive integer", d.expand_seeds)?,
                expand_per_seed: parsed("EXPAND_PER_SEED", "positive integer", d.expand_per_seed)?,
                // ! 0 is valid and means off: the order every number before 2026-11-19
                // was measured with.
                dense_tail_from: parsed("DENSE_TAIL_FROM", "integer", d.dense_tail_from)?,
                rounds: parsed("ROUNDS", "positive integer", d.rounds)?,
                round_documents: parsed("ROUND_DOCUMENTS", "positive integer", d.round_documents)?,
                round_per_document: parsed(
                    "ROUND_PER_DOCUMENT",
                    "positive integer",
                    d.round_per_document,
                )?,
                round_scan: parsed("ROUND_SCAN", "positive integer", d.round_scan)?,
                // ! 0 is valid and turns rewordings off.
                max_rewordings: parsed("MAX_REWORDINGS", "integer", d.max_rewordings)?,
                // ! Arm weights are properties of the CORPUS, not of the engine,
                // and a corpus is re-chunked far more often than the engine is
                // rebuilt. Refitting must not require a release.
                dense_weight: parsed("DENSE_WEIGHT", "number", d.dense_weight)?,
                sparse_weight: parsed("SPARSE_WEIGHT", "number", d.sparse_weight)?,
                text_weight: parsed("TEXT_WEIGHT", "number", d.text_weight)?,
                rrf_k: parsed("RRF_K", "number", d.rrf_k)?,
                region_weight: parsed("REGION_WEIGHT", "number", d.region_weight)?,
                text_min_idf: parsed("TEXT_MIN_IDF", "number", d.text_min_idf)?,
                // ! Loosening the canary is a deliberate, visible act. There is
                // no code path that quietly skips it.
                canary_min_cosine: parsed("CANARY_MIN_COSINE", "number", d.canary_min_cosine)?,
                domain_floor: parsed("DOMAIN_FLOOR", "number", d.domain_floor)?,
                domain_lexical_floor: parsed(
                    "DOMAIN_LEXICAL_FLOOR",
                    "number",
                    d.domain_lexical_floor,
                )?,
                gate_sample: parsed("GATE_SAMPLE", "positive integer", d.gate_sample)?,
                // ! Declared by the corpus (`corpus_meta.gate.foreign`), ✗ configured.
                gate_foreign: d.gate_foreign,
                // ! Fitted against the eval set, never chosen by hand
                // (`docs/design/SCORING.md` §4) -- and env-driven for the same reason
                // the arm weights are: they are properties of the CORPUS, and
                // refitting after a re-chunk must not require a release.
                factor_weights: engine::Weights {
                    // ! A floor, not a weight -- and the single most valuable
                    // dial measured (`docs/design/SCORING.md` §3).
                    relevance_floor: parsed(
                        "RELEVANCE_FLOOR",
                        "number",
                        d.factor_weights.relevance_floor,
                    )?,
                    authority: parsed("FACTOR_AUTHORITY", "number", d.factor_weights.authority)?,
                    structural: parsed("FACTOR_STRUCTURAL", "number", d.factor_weights.structural)?,
                    temporal: parsed("FACTOR_TEMPORAL", "number", d.factor_weights.temporal)?,
                    completeness: parsed(
                        "FACTOR_COMPLETENESS",
                        "number",
                        d.factor_weights.completeness,
                    )?,
                    topical: parsed("FACTOR_TOPICAL", "number", d.factor_weights.topical)?,
                },
                vocabulary: load_vocabulary()?,
                // Replaced below, once the env weights above are known.
                algorithms: d.algorithms.clone(),
            },
        };
        // ! After the struct, because the fallback registry is built FROM the
        // weights the environment just resolved. Reading ALGORITHMS_PATH first
        // would make `balanced` mean the compiled-in fit even where the
        // operator had overridden FACTOR_AUTHORITY.
        s.pipeline.algorithms = load_algorithms(&s.pipeline.factor_weights)?;
        s.validate()?;
        Ok(s)
    }

    /// Reject values that parse but cannot work.
    ///
    /// ! `MAX_CONCURRENCY=0` builds a semaphore that admits nobody: the server
    /// starts, reports healthy, and refuses every request. A parser cannot
    /// catch that; only a range check can.
    fn validate(&self) -> Result<(), ConfigError> {
        let positive = [
            ("MAX_CONCURRENCY", self.max_concurrency),
            ("READ_CHUNK_CHARS", self.read_chunk_chars),
            ("MAX_PROVENANCE_IDS", self.max_provenance_ids),
            ("CLUSTERS_PROBED", self.pipeline.clusters_probed),
            // ! 0 would chunk into empty windows and scan nothing, returning an
            // empty dense arm with no error — the same class of silent failure
            // MAX_CONCURRENCY=0 causes.
            ("CLUSTER_BATCH", self.pipeline.cluster_batch),
            ("TOP_K", self.pipeline.top_k),
            ("CANDIDATE_POOL", self.pipeline.candidate_pool),
            ("SNIPPET_CHARS", self.pipeline.snippet_chars),
            ("GATE_SAMPLE", self.pipeline.gate_sample),
            ("ROUNDS", self.pipeline.rounds),
            ("ROUND_DOCUMENTS", self.pipeline.round_documents),
            ("ROUND_PER_DOCUMENT", self.pipeline.round_per_document),
        ];
        for (key, value) in positive {
            if value == 0 {
                return Err(ConfigError::Invalid {
                    key,
                    value: "0".into(),
                    kind: "positive integer",
                });
            }
        }
        // ! 0 rows ranked per regulation makes every later round find nothing and stop,
        // which looks exactly like a round that had nothing to add.
        if self.pipeline.round_scan <= 0 {
            return Err(ConfigError::Invalid {
                key: "ROUND_SCAN",
                value: self.pipeline.round_scan.to_string(),
                kind: "positive integer",
            });
        }
        // ! A later round's chunks are placed by dense distance below DENSE_TAIL_FROM.
        // With it off they would have nowhere principled to go, and appending them
        // would be the merge the simulation measured as a loss (`rounds_sim.py`).
        if self.pipeline.rounds > 1 && self.pipeline.dense_tail_from == 0 {
            return Err(ConfigError::Invalid {
                key: "ROUNDS",
                value: self.pipeline.rounds.to_string(),
                kind: "1 unless DENSE_TAIL_FROM is set",
            });
        }
        // ! A pool smaller than the answer caps the reply below TOP_K without
        // saying so: the engine would return `candidate_pool` results and
        // report nothing unusual.
        if self.pipeline.candidate_pool < self.pipeline.top_k {
            return Err(ConfigError::Invalid {
                key: "CANDIDATE_POOL",
                value: self.pipeline.candidate_pool.to_string(),
                kind: "pool at least as large as TOP_K",
            });
        }
        // ! The "a weight the vocabulary cannot evaluate" check is NOT here. It
        // moved to `Pipeline::new`, which is the first place the corpus's own
        // declaration is known -- and the corpus's declaration is the one that
        // decides. Checking the fallback here would pass a config the corpus then
        // invalidates, which is a green startup check for the wrong thing.
        // ! Negative or NaN would read as "off" in `prune` and hide the typo.
        if self.pipeline.text_min_idf.is_nan() || self.pipeline.text_min_idf < 0.0 {
            return Err(ConfigError::Invalid {
                key: "TEXT_MIN_IDF",
                value: self.pipeline.text_min_idf.to_string(),
                kind: "non-negative number",
            });
        }
        // ! A share of the query: above 1 no domain could hold a question, and every
        // selection would fall back to similarity alone without saying so.
        if !(0.0..=1.0).contains(&self.domain_select_lexical) {
            return Err(ConfigError::Invalid {
                key: "DOMAIN_SELECT_LEXICAL",
                value: self.domain_select_lexical.to_string(),
                kind: "number from 0 to 1",
            });
        }
        // ! `k` ≤ 0 divides by zero at rank 1 or inverts the fusion; NaN ranks nothing.
        if !(self.pipeline.rrf_k.is_finite() && self.pipeline.rrf_k > 0.0) {
            return Err(ConfigError::Invalid {
                key: "RRF_K",
                value: self.pipeline.rrf_k.to_string(),
                kind: "positive number",
            });
        }
        // ! Negative would push the named region's chunks DOWN; NaN fuses nothing.
        if !(self.pipeline.region_weight.is_finite() && self.pipeline.region_weight >= 0.0) {
            return Err(ConfigError::Invalid {
                key: "REGION_WEIGHT",
                value: self.pipeline.region_weight.to_string(),
                kind: "non-negative number",
            });
        }
        // ! All three weights at zero fuses nothing and returns nothing, which
        // looks exactly like a corpus with no matches.
        let weights =
            self.pipeline.dense_weight + self.pipeline.sparse_weight + self.pipeline.text_weight;
        if weights <= 0.0 {
            return Err(ConfigError::Invalid {
                key: "DENSE_WEIGHT/SPARSE_WEIGHT/TEXT_WEIGHT",
                value: format!("{weights}"),
                kind: "set of weights with a positive sum",
            });
        }
        Ok(())
    }
}

#[tokio::main]
async fn main() {
    // ! Display, not Debug. A refusal to serve has to tell the operator what
    // went wrong and why; `CanaryFailed { got: 0.61, want: 0.98 }` makes them
    // go read the source, and the message already explains itself.
    if let Err(e) = run().await {
        log!("refusing to serve \u{b7} {e}");
        std::process::exit(1);
    }
}

async fn run() -> Result<(), Box<dyn std::error::Error>> {
    let s = Settings::from_env()?;

    // ! The connection string is never logged. It carries a password, and an
    // engine log is the one place an operator pastes into a ticket without
    // thinking about it.
    log!(
        "starting \u{b7} embed={} vocab={} transport={:?} \u{b7} statement timeout {}ms",
        s.embed_endpoint,
        s.bm25_vocab.display(),
        s.transport,
        s.statement_timeout
    );

    log!(
        "algorithms \u{b7} {} loaded, {} offered to the agent \u{b7} default `{}`",
        s.pipeline.algorithms.algorithms.len(),
        s.pipeline.algorithms.offered().len(),
        contract::DEFAULT_ALGORITHM
    );

    let (domain_list, registry) = open_domains(&s).await?;
    let domain_list = Arc::new(domain_list);

    // ! Bounded concurrency (`CLAUDE.md` \u{a7}7.6). Peak RAM is fixed costs plus
    // this ceiling times the per-request ceiling, and both terms are bounded.
    let permits = Arc::new(Semaphore::new(s.max_concurrency));
    let limits = Limits {
        read_chunk_chars: s.read_chunk_chars,
        max_provenance_ids: s.max_provenance_ids,
        domain_select_lexical: s.domain_select_lexical,
    };

    match s.transport {
        Transport::Http => {
            // ! A wait ceiling is the other half of invariant 6. A bounded
            // queue alone still lets a caller block indefinitely behind a full
            // one; the bound has to be on TIME as well as on depth, or
            // backpressure never actually reaches the client.
            let server = Arc::new(Server {
                domains: domain_list.clone(),
                registry,
                permits,
                queue_wait: Some(s.queue_wait),
                limits,
            });
            log!(
                "transport http \u{b7} {} \u{b7} queue wait {}ms",
                s.http_addr,
                s.queue_wait.as_millis()
            );
            http::serve(server, &s.http_addr).await?;
            Ok(())
        }
        // ! stdio is serial: one line is read, handled, answered, and only then
        // is the next read. Nothing can queue, so there is nobody to refuse and
        // a wait ceiling could only ever fire against the request already being
        // served.
        Transport::Stdio => {
            let server = Arc::new(Server {
                domains: domain_list.clone(),
                registry,
                permits,
                queue_wait: None,
                limits,
            });
            serve_stdio(&server).await
        }
    }
}

/// Every domain `DOMAINS_FILE` lists, or the one `DATABASE_URL` names when it is unset,
/// opened and checked; with the algorithms they offer between them.
async fn open_domains(
    s: &Settings,
) -> Result<(Vec<domains::Domain>, contract::Registry), Box<dyn std::error::Error>> {
    // ! One entry, ✗ a list, when no DOMAINS_FILE is set: the deployment that existed
    // before domains did, served by the same code path.
    let entries = match &s.domains_file {
        None => vec![domains::Entry {
            name: String::new(),
            schema: None,
            vocab: s.bm25_vocab.clone(),
            algorithms: None,
            dsn_env: "DATABASE_URL".to_owned(),
        }],
        Some(path) => {
            let body = std::fs::read_to_string(path)
                .map_err(|e| format!("DOMAINS_FILE={} \u{b7} {e}", path.display()))?;
            domains::parse(&body)
                .map_err(|e| format!("DOMAINS_FILE={} \u{b7} {e}", path.display()))?
        }
    };
    let mut opened: Vec<domains::Domain> = Vec::with_capacity(entries.len());
    let mut provider: Option<(Arc<embed::HttpProvider>, String, i32)> = None;
    for entry in &entries {
        opened.push(open_domain(s, entry, &mut provider).await?);
    }
    let registry = domains::merged_registry(&opened);
    Ok((opened, registry))
}

/// One domain: its pool, pinned to its schema; its corpus's declaration; its canary.
async fn open_domain(
    s: &Settings,
    entry: &domains::Entry,
    provider: &mut Option<(Arc<embed::HttpProvider>, String, i32)>,
) -> Result<domains::Domain, Box<dyn std::error::Error>> {
    let dsn = if entry.dsn_env == "DATABASE_URL" {
        s.database_url.clone()
    } else {
        std::env::var(&entry.dsn_env)
            .map_err(|_| format!("{} is not set \u{b7} DOMAINS_FILE names it", entry.dsn_env))?
    };
    // ! `ARM_FANOUT` connections per permitted search: the dense, sparse and text
    // arms run concurrently (`pipeline.rs::arms`), so one search can hold three.
    // +2: plus the startup canary and health probes, which must not have to wait
    // behind a full queue.
    let pool = store::connect_in(
        &dsn,
        entry.schema.as_deref(),
        s.max_concurrency * ARM_FANOUT + 2,
        s.statement_timeout,
    )?;
    let ops = store::SearchOps::new(pool);

    // The corpus decides the vector space; the provider is built to match it.
    let meta = ops.corpus_meta().await?;
    log!(
        "corpus {}{} \u{b7} {} @ {}d \u{b7} {} pooling{}",
        meta.id,
        entry
            .schema
            .as_deref()
            .map(|x| format!(" (schema {x})"))
            .unwrap_or_default(),
        meta.dense_model,
        meta.dense_dim,
        meta.dense_pooling,
        if meta.dense_instruction.is_some() {
            " \u{b7} instruction-aware"
        } else {
            ""
        }
    );

    // ! One provider for every domain, because they must share one space: a second
    // space needs its own embedder, which `SCALING.md` §4 designs and this does not
    // build. Refused at startup, ✗ embedded with the wrong model (§7.7).
    let shared = if let Some((p, model, dim)) = provider.as_ref() {
        if *model != meta.dense_model || *dim != meta.dense_dim {
            return Err(format!(
                "domain {} declares {} @ {}d, the first declares {model} @ {dim}d \u{b7} \
                 one engine serves one vector space",
                meta.id, meta.dense_model, meta.dense_dim
            )
            .into());
        }
        p.clone()
    } else {
        let dim = usize::try_from(meta.dense_dim).unwrap_or(0);
        let p = Arc::new(embed::HttpProvider::new(
            &s.embed_endpoint,
            &meta.dense_model,
            dim,
        )?);
        *provider = Some((p.clone(), meta.dense_model.clone(), meta.dense_dim));
        p
    };
    let vectorizer = bm25::QueryVectorizer::load(&entry.vocab)?;
    let mut cfg = s.pipeline.clone();
    if entry.algorithms.is_some() {
        cfg.algorithms = load_algorithms_from(entry.algorithms.as_deref(), &cfg.factor_weights)?;
    }
    let pipe =
        Arc::new(Pipeline::new(ops, shared, vectorizer, &meta.dense_model.clone(), cfg).await?);
    log!(
        "ready \u{b7} {} clusters, probing {} \u{b7} concurrency {}",
        pipe.cluster_count(),
        s.pipeline.clusters_probed,
        s.max_concurrency
    );
    let name = if entry.name.is_empty() {
        meta.id.clone()
    } else {
        entry.name.clone()
    };
    Ok(domains::Domain { name, pipe })
}
async fn serve_stdio(server: &Server) -> Result<(), Box<dyn std::error::Error>> {
    let mut lines = BufReader::new(tokio::io::stdin()).lines();
    let mut stdout = tokio::io::stdout();

    while let Some(line) = lines.next_line().await? {
        if line.trim().is_empty() {
            continue;
        }
        let Ok(req): Result<Value, _> = serde_json::from_str(&line) else {
            log!("dropping unparseable line ({} bytes)", line.len());
            continue;
        };

        if let Some(resp) = server.handle(&req).await {
            let mut bytes = serde_json::to_vec(&resp)?;
            bytes.push(b'\n');
            stdout.write_all(&bytes).await?;
            stdout.flush().await?;
        }
    }
    Ok(())
}

/// Tool dispatch, shared by every transport.
///
/// ! One implementation, ✗ one per transport. stdio and HTTP differ in exactly
/// one respect — what happens when the concurrency ceiling is already full —
/// and that difference is `queue_wait`. Everything else a client can observe
/// about a tool call is identical either way, which is the only reason an eval
/// harness driving stdio says anything about the HTTP deployment.
pub(crate) struct Server {
    /// What is served: one domain without `DOMAINS_FILE`, as many as it lists with it.
    pub(crate) domains: Arc<Vec<domains::Domain>>,
    /// Every domain's algorithms, merged: what `tools/list` advertises.
    pub(crate) registry: contract::Registry,
    pub(crate) permits: Arc<Semaphore>,
    /// How long a call may wait for a permit. `None` waits forever, which is
    /// correct only for a serial transport.
    pub(crate) queue_wait: Option<std::time::Duration>,
    pub(crate) limits: Limits,
}

/// Caps on what one response may contain.
///
/// ! Carried, \u{2717} compiled in. These bound the size of a reply, and the right
/// bound depends on the caller's context window and the corpus's chunk size,
/// neither of which this binary knows.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Limits {
    pub(crate) read_chunk_chars: usize,
    pub(crate) max_provenance_ids: usize,
    /// `DOMAIN_SELECT_LEXICAL` (`domains::gates`).
    pub(crate) domain_select_lexical: f32,
}

impl Server {
    /// Handle one JSON-RPC message. `None` means "notification, say nothing".
    pub(crate) async fn handle(&self, req: &Value) -> Option<Value> {
        let id = req.get("id").cloned();
        match protocol::route(req) {
            protocol::Route::Initialize => {
                Some(protocol::ok(id.as_ref(), &protocol::initialize_result()))
            }
            protocol::Route::ToolsList => Some(protocol::ok(
                id.as_ref(),
                &json!({ "tools": tools::definitions(&self.registry) }),
            )),
            protocol::Route::ToolsCall(params) => {
                // ! The permit is taken BEFORE dispatch and dropped after, so
                // the ceiling bounds work in flight rather than requests
                // accepted.
                let Some(permit) = self.admit().await else {
                    return Some(protocol::busy(id.as_ref(), self.queue_wait));
                };
                let out = dispatch(&self.domains, &self.registry, &params, self.limits).await;
                drop(permit);
                Some(protocol::ok(id.as_ref(), &protocol::tool_content(&out)))
            }
            // Notifications carry no id and must not be answered.
            protocol::Route::Notification => None,
            protocol::Route::Unknown(m) => Some(protocol::method_not_found(id.as_ref(), &m)),
        }
    }

    /// Take a concurrency permit, or refuse.
    async fn admit(&self) -> Option<OwnedSemaphorePermit> {
        admit_within(&self.permits, self.queue_wait).await
    }
}

/// Take a permit, waiting at most `wait`.
///
/// `None` is a refusal, ✗ an error: nothing was attempted, so the caller can
/// retry safely. Free-standing so the ceiling can be tested without standing up
/// a corpus — a guarantee that needs a database to test is a guarantee nobody
/// tests.
async fn admit_within(
    permits: &Arc<Semaphore>,
    wait: Option<std::time::Duration>,
) -> Option<OwnedSemaphorePermit> {
    let sem = permits.clone();
    match wait {
        None => sem.acquire_owned().await.ok(),
        Some(d) => tokio::time::timeout(d, sem.acquire_owned())
            .await
            .ok()?
            .ok(),
    }
}

async fn dispatch(
    domains: &[domains::Domain],
    registry: &contract::Registry,
    params: &Value,
    limits: Limits,
) -> Value {
    let name = params.get("name").and_then(Value::as_str).unwrap_or("");
    let args = params.get("arguments").cloned().unwrap_or(json!({}));

    // ! One domain is the deployment that existed before domains did, and it is served
    // by exactly the code it was: no gate is asked twice, no reply is rewritten.
    if let [only] = domains {
        let pipe = &only.pipe;
        return match name {
            "list_domains" => list_domains(pipe),
            "list_algorithms" => list_algorithms(pipe.algorithms()),
            "search_knowledge" => search_knowledge(pipe, &args).await,
            "read_chunk" => read_chunk(pipe, &args, limits).await,
            "get_provenance" => get_provenance(pipe, &args, limits).await,
            "explain_routing" => explain_routing(pipe, &args).await,
            other => unknown_tool(other),
        };
    }
    match name {
        "list_domains" => list_all_domains(domains),
        "list_algorithms" => list_algorithms(registry),
        "search_knowledge" => {
            let Some(q) = args.get("query").and_then(Value::as_str) else {
                return search_knowledge(&domains[0].pipe, &args).await;
            };
            let verdicts = domains::gates(domains, q, limits.domain_select_lexical).await;
            let mut out = search_knowledge(&verdicts[0].domain.pipe, &args).await;
            note(&mut out, &verdicts);
            out
        }
        "explain_routing" => {
            let Some(q) = args.get("query").and_then(Value::as_str) else {
                return explain_routing(&domains[0].pipe, &args).await;
            };
            let verdicts = domains::gates(domains, q, limits.domain_select_lexical).await;
            let mut out = explain_routing(&verdicts[0].domain.pipe, &args).await;
            note(&mut out, &verdicts);
            if let Some(obj) = out.as_object_mut() {
                obj.insert(
                    "domains".into(),
                    verdicts
                        .iter()
                        .map(|v| {
                            json!({
                                "domain": v.domain.name,
                                "centroid_similarity": v.similarity,
                                "lexical_evidence": v.lexical,
                                "holds_query": v.holds,
                                "would_serve": v.serves,
                                "gate_failed": v.failed,
                            })
                        })
                        .collect(),
                );
            }
            out
        }
        "read_chunk" => {
            let Some(id) = args.get("id").and_then(Value::as_str) else {
                return read_chunk(&domains[0].pipe, &args, limits).await;
            };
            match holder(domains, id).await {
                Ok(Some(d)) => read_chunk(&d.pipe, &args, limits).await,
                Ok(None) => read_chunk(&domains[0].pipe, &args, limits).await,
                Err(e) => e,
            }
        }
        "get_provenance" => get_provenance_all(domains, &args, limits).await,
        other => unknown_tool(other),
    }
}

fn unknown_tool(other: &str) -> Value {
    tools::error(
        "tools/call",
        &format!("unknown tool: {other}"),
        "call tools/list to see the available tools",
    )
}

/// Put which domain answered, and how the others judged the query, first in `progress`.
fn note(out: &mut Value, verdicts: &[domains::Verdict<'_>]) {
    let line = Value::String(domains::describe(verdicts));
    if let Some(progress) = out.get_mut("progress").and_then(Value::as_array_mut) {
        progress.insert(0, line);
    } else if let Some(obj) = out.as_object_mut() {
        obj.insert("progress".into(), json!([line]));
    }
}

/// The one domain holding chunk `id`, if any.
///
/// ! Two is an error, ✗ the first: an id the engine cannot attribute to one source is a
/// provenance it would be guessing (`CLAUDE.md` §7.8).
async fn holder<'a>(
    domains: &'a [domains::Domain],
    id: &str,
) -> Result<Option<&'a domains::Domain>, Value> {
    let mut found = None;
    for d in domains {
        match d.pipe.provenance(&[id.to_owned()]).await {
            Ok(rows) if !rows.is_empty() => {
                if let Some(first) = found.replace(d) {
                    return Err(tools::error(
                        "read_chunk",
                        &format!("chunk {id} is in domains {} and {}", first.name, d.name),
                        "an id must name one source · report this corpus pair",
                    ));
                }
            }
            Ok(_) => {}
            Err(e) => {
                return Err(tools::error(
                    "read_chunk",
                    &e.to_string(),
                    "check the database",
                ));
            }
        }
    }
    Ok(found)
}

fn list_all_domains(domains: &[domains::Domain]) -> Value {
    let rows: Vec<Value> = domains
        .iter()
        .map(|d| {
            let m = d.pipe.corpus();
            json!({
                "id": d.name,
                "description": format!(
                    "{} embedded with {} at {} dims · {} clusters",
                    m.id, m.dense_model, m.dense_dim, d.pipe.cluster_count()
                ),
            })
        })
        .collect();
    let n = rows.len();
    tools::sized(json!({
        "success": true,
        "op": "list_domains",
        "domains": rows,
        "progress": [format!("listed {n} domains")],
        "token_estimate": 0,
    }))
}

async fn get_provenance_all(domains: &[domains::Domain], args: &Value, limits: Limits) -> Value {
    let ids = provenance_ids(args, limits);
    if ids.is_empty() {
        return get_provenance(&domains[0].pipe, args, limits).await;
    }
    let mut rows = Vec::new();
    let mut seen: std::collections::BTreeMap<String, &str> = std::collections::BTreeMap::new();
    for d in domains {
        match d.pipe.provenance(&ids).await {
            Ok(found) => {
                for r in found {
                    if let Some(other) = seen.insert(r.id.clone(), &d.name) {
                        return tools::error(
                            "get_provenance",
                            &format!("chunk {} is in domains {other} and {}", r.id, d.name),
                            "an id must name one source · report this corpus pair",
                        );
                    }
                    rows.push(r);
                }
            }
            Err(e) => return tools::error("get_provenance", &e.to_string(), "check the database"),
        }
    }
    provenance_reply(&rows)
}

fn list_domains(pipe: &Pipeline) -> Value {
    let m = pipe.corpus();
    tools::sized(json!({
        "success": true,
        "op": "list_domains",
        "domains": [{
            "id": m.id,
            "description": format!(
                "{} embedded with {} at {} dims · {} clusters",
                m.id, m.dense_model, m.dense_dim, pipe.cluster_count()
            ),
        }],
        "progress": ["listed 1 domain"],
        "token_estimate": 0,
    }))
}

/// What each algorithm is for, and what it scored.
///
/// ! Lists the **unfitted** ones too, marked. `tools/list` offers only fitted
/// names because an offer is a recommendation; this is introspection, and
/// hiding an algorithm the engine will accept would leave a caller unable to
/// find out why a name it was told about works.
fn list_algorithms(reg: &contract::Registry) -> Value {
    let rows: Vec<Value> = reg
        .algorithms
        .iter()
        .map(|(name, a)| {
            json!({
                "name": name,
                "description": a.description,
                // The bound is the one number a caller can act on: it says how
                // far metadata may move a result away from the arms' own order.
                "prior_bound": a.prior_bound(),
                "relevance_floor": a.factors.relevance_floor,
                "expand": a.expand,
                "fitted": a.fitted,
                // ! Stated per row, ✗ inferred from `fitted` being null. An
                // agent should not have to know that rule to read this.
                "experimental": !a.is_fitted(),
            })
        })
        .collect();
    let n = rows.len();
    tools::sized(json!({
        "success": true,
        "op": "list_algorithms",
        "default": contract::DEFAULT_ALGORITHM,
        "algorithms": rows,
        "progress": [format!("listed {n} algorithms")],
        "token_estimate": 0,
    }))
}

/// Pull the option fields out of the tool's argument object.
///
/// ! `query` is removed first. It belongs to the tool, ✗ to the options, and
/// `SearchOptions` denies unknown fields so leaving it in would reject every
/// well-formed call.
fn parse_options(args: &Value) -> Result<contract::SearchOptions, serde_json::Error> {
    let mut obj = args.as_object().cloned().unwrap_or_default();
    obj.remove("query");
    obj.remove("rewordings");
    serde_json::from_value(Value::Object(obj))
}

/// The caller's rewordings of its question: an array of strings, or absent.
///
/// ! Anything else is an error, ✗ ignored: an agent that sends one string where a list
/// belongs would otherwise be served a single search and believe it was fused.
fn parse_rewordings(args: &Value) -> Result<Vec<String>, &'static str> {
    match args.get("rewordings") {
        None | Some(Value::Null) => Ok(Vec::new()),
        Some(Value::Array(items)) => items
            .iter()
            .map(|v| {
                v.as_str()
                    .map(str::to_owned)
                    .ok_or("every rewording must be a string")
            })
            .collect(),
        Some(_) => Err("`rewordings` must be an array of strings"),
    }
}

async fn search_knowledge(pipe: &Pipeline, args: &Value) -> Value {
    let Some(q) = args.get("query").and_then(Value::as_str) else {
        return tools::error(
            "search_knowledge",
            "missing `query`",
            "pass {\"query\": \"...\"} · this tool takes a query and nothing else",
        );
    };
    if q.trim().is_empty() {
        return tools::error(
            "search_knowledge",
            "empty `query`",
            "describe what you are looking for, or name a regulation",
        );
    }
    // ! Options are parsed from the SAME object as `query`, and an unknown key
    // is an error rather than a silently ignored default (`deny_unknown_fields`
    // on SearchOptions). An agent that misspells a knob and is served the
    // default has been given a wrong answer wearing a correct one's clothes.
    let opts = match parse_options(args) {
        Ok(o) => o,
        Err(e) => {
            return tools::error(
                "search_knowledge",
                &format!("invalid options · {e}"),
                "see the inputSchema · every option is optional and narrows the server's own limit",
            );
        }
    };
    let rewordings = match parse_rewordings(args) {
        Ok(r) => r,
        Err(e) => {
            return tools::error(
                "search_knowledge",
                &format!("invalid rewordings · {e}"),
                "pass {\"rewordings\": [\"...\", \"...\"]} · other wordings of the same question",
            );
        }
    };
    match pipe.search_reworded(q, &rewordings, &opts).await {
        Ok(r) => serde_json::to_value(r).unwrap_or_else(|e| {
            tools::error("search_knowledge", &e.to_string(), "retry the query")
        }),
        Err(e) => tools::error(
            "search_knowledge",
            &e.to_string(),
            "check the embedding service and database are reachable",
        ),
    }
}

async fn read_chunk(pipe: &Pipeline, args: &Value, limits: Limits) -> Value {
    let Some(id) = args.get("id").and_then(Value::as_str) else {
        return tools::error(
            "read_chunk",
            "missing `id`",
            "use an id from search_knowledge results",
        );
    };
    let cap = args
        .get("max_chars")
        .and_then(Value::as_u64)
        .and_then(|v| usize::try_from(v).ok())
        .map_or(limits.read_chunk_chars, |n| n.min(limits.read_chunk_chars));
    match pipe.read_chunk(id, cap).await {
        Ok(Some((row, body, truncated))) => tools::sized(json!({
            "success": true,
            "op": "read_chunk",
            "id": row.id,
            "body": body,
            "truncated": truncated,
            "truncated_at_source": row.truncated_at_source,
            "source": { "title": row.source_title, "url": row.source_url },
            "progress": ["read 1 chunk"],
            "token_estimate": 0,
        })),
        Ok(None) => tools::error(
            "read_chunk",
            &format!("no chunk {id}"),
            "ids come from search_knowledge · they are not guessable",
        ),
        Err(e) => tools::error("read_chunk", &e.to_string(), "check the database"),
    }
}

/// The ids a `get_provenance` call asks for, capped.
///
/// ! Capped. An agent that pastes a whole result set back is the normal
/// case, and an uncapped IN-list is an unbounded query and an unbounded
/// reply — the one place a read-only server can still be made to allocate
/// without limit.
fn provenance_ids(args: &Value, limits: Limits) -> Vec<String> {
    args.get("ids")
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_str().map(str::to_owned))
                .take(limits.max_provenance_ids)
                .collect()
        })
        .unwrap_or_default()
}

fn provenance_reply(rows: &[store::ChunkRow]) -> Value {
    let incomplete = rows.iter().filter(|r| !r.provenance_complete()).count();
    tools::sized(json!({
        "success": true,
        "op": "get_provenance",
        "sources": rows.iter().map(|r| json!({
            "id": r.id,
            "title": r.source_title,
            "url": r.source_url,
            "chapter": r.chapter,
            "article": r.article,
            "truncated_at_source": r.truncated_at_source,
            "provenance_complete": r.provenance_complete(),
        })).collect::<Vec<_>>(),
        "progress": [format!("resolved {} ids", rows.len())],
        "token_estimate": 0,
        "hint": (incomplete > 0).then(|| format!(
            "{incomplete} of {} have no source_url · cite by title and locator",
            rows.len()
        )),
    }))
}

async fn get_provenance(pipe: &Pipeline, args: &Value, limits: Limits) -> Value {
    let ids = provenance_ids(args, limits);
    if ids.is_empty() {
        return tools::error(
            "get_provenance",
            "missing `ids`",
            "pass {\"ids\": [\"...\"]} from search_knowledge results",
        );
    }
    match pipe.provenance(&ids).await {
        Ok(rows) => provenance_reply(&rows),
        Err(e) => tools::error("get_provenance", &e.to_string(), "check the database"),
    }
}

async fn explain_routing(pipe: &Pipeline, args: &Value) -> Value {
    let Some(q) = args.get("query").and_then(Value::as_str) else {
        return tools::error("explain_routing", "missing `query`", "pass a query");
    };
    match pipe.explain(q).await {
        Ok(r) => tools::sized(json!({
            "success": true,
            "op": "explain_routing",
            // ! `null` when the gate would refuse — the same value
            // `search_knowledge` reports for this query.
            "detected_domain": r.domain,
            "would_refuse": r.would_refuse,
            // Both halves, with their thresholds, so a refusal can be
            // attributed rather than guessed at.
            "domain_gate": {
                "lexical_evidence": r.lexical_evidence,
                "lexical_floor": r.lexical_floor,
                "centroid_similarity": r.centroid_similarity,
                "centroid_floor": r.centroid_floor,
                "bypassed": r.gate_bypassed,
            },
            "clusters_probed": r.clusters_probed,
            "cluster_scores": r.cluster_scores.iter()
                .map(|(id, s)| json!({ "cluster": id, "similarity": s }))
                .collect::<Vec<_>>(),
            "total_clusters": r.total_clusters,
            "identifiers_detected": r.identifiers,
            "routing_bypass": !r.identifiers.is_empty(),
            "provider": r.provider,
            "progress": ["routed", if r.would_refuse {
                "domain gate: would REFUSE this query"
            } else {
                "domain gate: would serve this query"
            }],
            "token_estimate": 0,
        })),
        Err(e) => tools::error(
            "explain_routing",
            &e.to_string(),
            "check the embedding service",
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A settings value that is valid, so a test can change one field and
    /// assert that field alone is what rejects it.
    fn settings() -> Settings {
        Settings {
            database_url: "host=db dbname=vera".into(),
            embed_endpoint: "http://embed:80".into(),
            bm25_vocab: "/vocab/corpus.bm25.json".into(),
            domains_file: None,
            max_concurrency: 4,
            transport: Transport::Http,
            http_addr: "0.0.0.0:8081".into(),
            queue_wait: std::time::Duration::from_secs(2),
            read_chunk_chars: 4000,
            max_provenance_ids: 50,
            domain_select_lexical: 0.2,
            statement_timeout: 15_000,
            pipeline: Config::default(),
        }
    }

    #[test]
    fn the_shipped_defaults_are_a_valid_configuration() {
        // ! If this ever fails, a default was changed to something the server
        // would refuse to start on \u2014 which no operator would ever see, because
        // they would have set the variable.
        assert!(settings().validate().is_ok());
    }

    #[test]
    fn a_ceiling_of_zero_is_refused_rather_than_served() {
        // ! It parses. A semaphore of 0 admits nobody, so the server would
        // start, report healthy, and refuse every request forever.
        let mut s = settings();
        s.max_concurrency = 0;
        let e = s.validate().unwrap_err();
        assert!(e.to_string().contains("MAX_CONCURRENCY"), "{e}");
    }

    #[test]
    fn a_pool_smaller_than_the_answer_is_refused() {
        // ! It parses, and it would serve: the engine would return
        // CANDIDATE_POOL results while TOP_K said 10, and report nothing
        // unusual about the short reply.
        let mut s = settings();
        s.pipeline.top_k = 10;
        s.pipeline.candidate_pool = 5;
        let e = s.validate().unwrap_err();
        assert!(e.to_string().contains("CANDIDATE_POOL"), "{e}");
    }

    #[test]
    fn a_pool_equal_to_the_answer_is_allowed() {
        // Degenerate but coherent: no room for scoring to promote anything,
        // which is a choice an operator may make.
        let mut s = settings();
        s.pipeline.top_k = 10;
        s.pipeline.candidate_pool = 10;
        assert!(s.validate().is_ok());
    }

    #[test]
    fn the_default_pool_leaves_room_to_reorder() {
        // ! The point of the pool. If these were equal by default, factor
        // scoring could only ever re-rank the results that already won.
        let d = Config::default();
        assert!(
            d.candidate_pool > d.top_k,
            "pool {} must exceed top_k {}",
            d.candidate_pool,
            d.top_k
        );
    }

    #[test]
    fn probing_zero_clusters_is_refused() {
        let mut s = settings();
        s.pipeline.clusters_probed = 0;
        assert!(s.validate().is_err());
    }

    #[test]
    fn a_batch_of_zero_clusters_is_refused() {
        // ! `chunks(0)` panics, and guarding it with `.max(1)` in the hot loop
        // would silently serve a configuration the operator did not ask for.
        // Refusing at startup is the only honest reading of `CLUSTER_BATCH=0`.
        let mut s = settings();
        s.pipeline.cluster_batch = 0;
        assert!(s.validate().is_err());
    }

    #[test]
    fn the_default_batch_is_the_sequential_scan() {
        // ! Not a memory floor — engine RSS is flat across this knob. The
        // default is 1 because every published latency number was taken at it,
        // so a different default would silently invalidate the docs.
        assert_eq!(Config::default().cluster_batch, 1);
    }

    #[test]
    fn a_batch_wider_than_the_probe_is_harmless() {
        // ! `chunks` yields a short final window rather than padding, so a batch
        // larger than `clusters_probed` degenerates to one query over every
        // probed cluster — the fast end of the trade, not a misconfiguration.
        let mut s = settings();
        s.pipeline.cluster_batch = 64;
        assert!(s.validate().is_ok());
    }

    #[test]
    fn an_rrf_k_that_cannot_fuse_is_refused() {
        // ! 0 divides by zero at rank 1; a negative k inverts the arms' order.
        for k in [0.0, -1.0, f32::NAN] {
            let mut s = settings();
            s.pipeline.rrf_k = k;
            let e = s.validate().unwrap_err();
            assert!(e.to_string().contains("RRF_K"), "{e}");
        }
    }

    #[test]
    fn a_region_weight_that_cannot_fuse_is_refused() {
        for w in [-0.5, f32::NAN, f32::INFINITY] {
            let mut s = settings();
            s.pipeline.region_weight = w;
            let e = s.validate().unwrap_err();
            assert!(e.to_string().contains("REGION_WEIGHT"), "{e}");
        }
        let mut s = settings();
        s.pipeline.region_weight = 0.0;
        assert!(s.validate().is_ok());
    }

    #[test]
    fn rewordings_are_a_list_of_strings_or_absent() {
        assert!(parse_rewordings(&json!({"query": "q"})).unwrap().is_empty());
        assert!(
            parse_rewordings(&json!({"query": "q", "rewordings": null}))
                .unwrap()
                .is_empty()
        );
        assert_eq!(
            parse_rewordings(&json!({"rewordings": ["a", "b"]})).unwrap(),
            ["a", "b"]
        );
        assert!(parse_rewordings(&json!({"rewordings": "a"})).is_err());
        assert!(parse_rewordings(&json!({"rewordings": ["a", 1]})).is_err());
        // Not an option: the options parser never sees it, so it is not "unknown".
        assert!(parse_options(&json!({"query": "q", "rewordings": ["a"]})).is_ok());
    }

    #[test]
    fn later_rounds_need_the_dense_tail_and_a_scan() {
        let mut s = settings();
        s.pipeline.rounds = 2;
        let e = s.validate().unwrap_err();
        assert!(e.to_string().contains("ROUNDS"), "{e}");
        s.pipeline.dense_tail_from = 10;
        assert!(s.validate().is_ok());
        s.pipeline.round_scan = 0;
        let e = s.validate().unwrap_err();
        assert!(e.to_string().contains("ROUND_SCAN"), "{e}");
        let mut s = settings();
        s.pipeline.rounds = 0;
        assert!(s.validate().unwrap_err().to_string().contains("ROUNDS"));
    }

    #[test]
    fn weights_that_sum_to_nothing_are_refused() {
        // ! Every arm at zero fuses nothing and returns nothing, which is
        // indistinguishable from a corpus that simply has no match.
        let mut s = settings();
        s.pipeline.dense_weight = 0.0;
        s.pipeline.sparse_weight = 0.0;
        s.pipeline.text_weight = 0.0;
        let e = s.validate().unwrap_err();
        assert!(e.to_string().contains("WEIGHT"), "{e}");
    }

    #[test]
    fn one_live_arm_is_enough() {
        // dense currently carries 0.0 weight and the server is expected to run.
        let mut s = settings();
        s.pipeline.dense_weight = 0.0;
        s.pipeline.text_weight = 0.0;
        assert!(s.validate().is_ok());
    }

    #[test]
    fn a_transport_it_cannot_serve_is_a_parse_failure() {
        assert_eq!("stdio".parse::<Transport>(), Ok(Transport::Stdio));
        assert_eq!("http".parse::<Transport>(), Ok(Transport::Http));
        assert!("grpc".parse::<Transport>().is_err());
        // ! Not case-folded. Accepting "HTTP" here would mean the documented
        // spelling and the accepted spellings drift apart over time.
        assert!("HTTP".parse::<Transport>().is_err());
    }

    #[test]
    fn a_missing_variable_names_itself_and_where_to_look() {
        let e = ConfigError::Missing("DATABASE_URL").to_string();
        assert!(e.contains("DATABASE_URL"), "{e}");
        assert!(e.contains(".env.example"), "{e}");
    }

    #[test]
    fn an_unparseable_value_is_quoted_back_at_the_operator() {
        // ! The bad value is echoed. "not a valid positive integer" without it
        // sends someone reading their own deployment manifest line by line.
        let e = ConfigError::Invalid {
            key: "MAX_CONCURRENCY",
            value: "four".into(),
            kind: "positive integer",
        }
        .to_string();
        assert!(e.contains("MAX_CONCURRENCY"), "{e}");
        assert!(e.contains("four"), "{e}");
    }

    fn ms(n: u64) -> std::time::Duration {
        std::time::Duration::from_millis(n)
    }

    #[tokio::test]
    async fn a_free_slot_is_granted_immediately() {
        let p = Arc::new(Semaphore::new(2));
        let a = admit_within(&p, Some(ms(50))).await;
        assert!(a.is_some(), "a server below its ceiling must serve");
        assert_eq!(p.available_permits(), 1);
    }

    #[tokio::test]
    async fn the_ceiling_is_a_ceiling() {
        let p = Arc::new(Semaphore::new(2));
        let _a = admit_within(&p, Some(ms(50))).await.expect("first");
        let _b = admit_within(&p, Some(ms(50))).await.expect("second");

        // ! This is the OOM guarantee (`CLAUDE.md` §5.4). Peak RAM is fixed
        // costs plus concurrency times the per-request ceiling; if a third
        // request could slip through, the second term is unbounded and so is
        // the total.
        assert!(
            admit_within(&p, Some(ms(50))).await.is_none(),
            "a third request must be refused, ✗ queued"
        );
    }

    #[tokio::test]
    async fn a_permit_returns_when_its_request_finishes() {
        let p = Arc::new(Semaphore::new(1));
        {
            let _held = admit_within(&p, Some(ms(50))).await.expect("first");
            assert!(admit_within(&p, Some(ms(20))).await.is_none());
        }
        assert!(
            admit_within(&p, Some(ms(50))).await.is_some(),
            "capacity must come back, or the server refuses forever after one burst"
        );
    }

    #[tokio::test]
    async fn waiting_is_bounded_by_the_wait_ceiling() {
        let p = Arc::new(Semaphore::new(1));
        let _held = admit_within(&p, Some(ms(50))).await.expect("first");

        // ! A bounded QUEUE alone still lets a caller block indefinitely behind
        // a full one. Invariant 6 needs the bound on time as well as on depth.
        let t0 = std::time::Instant::now();
        assert!(admit_within(&p, Some(ms(120))).await.is_none());
        let waited = t0.elapsed();
        assert!(waited >= ms(120), "returned too early: {waited:?}");
        assert!(waited < ms(2000), "did not honour the ceiling: {waited:?}");
    }

    #[test]
    fn a_refusal_says_it_is_retryable() {
        let v = protocol::busy(Some(&json!(7)), Some(ms(1500)));
        assert_eq!(v["error"]["code"], BUSY_CODE);
        assert_eq!(v["id"], 7, "a refusal must answer the id it refused");
        // ! The client has to be able to tell "never attempted" from "failed
        // halfway". Only the first is safe to retry.
        assert_eq!(v["error"]["data"]["retry"], true);
        assert!(
            v["error"]["message"].as_str().unwrap().contains("1500ms"),
            "say how long it waited, so an operator can tune the ceiling"
        );
    }
}
