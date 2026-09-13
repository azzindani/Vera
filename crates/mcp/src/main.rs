//! `vera-mcp` · the MCP server: transport and tool dispatch, no domain logic.
//!
//! ! Every log line goes to **stderr** (`CLAUDE.md` §7.10). stdout carries the
//! JSON-RPC stream and nothing else; one stray `println!` corrupts the channel
//! and the client sees a protocol error rather than a message.
//!
//! All configuration is read from the environment (invariant 12) and resolved
//! once, at startup, by [`Settings::from_env`]. See `docs/CONFIGURATION.md` for
//! the full table and `.env.example` for a working set.

mod bm25;
mod http;
mod identifier;
mod pipeline;
mod tools;

use std::sync::Arc;

use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use pipeline::{Config, Pipeline};

const PROTOCOL_VERSION: &str = "2024-11-05";

/// JSON-RPC code for "at capacity, try again". The spec reserves
/// -32768..=-32000 and defines only part of it; -32000 is where
/// implementations are told to put their own.
pub(crate) const BUSY_CODE: i64 = -32000;

/// Log to stderr. ! Never stdout.
macro_rules! log {
    ($($arg:tt)*) => { eprintln!($($arg)*) };
}

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
    max_concurrency: usize,
    transport: Transport,
    http_addr: String,
    queue_wait: std::time::Duration,
    read_chunk_chars: usize,
    max_provenance_ids: usize,
    pipeline: Config,
}

impl Settings {
    fn from_env() -> Result<Self, ConfigError> {
        let d = Config::default();
        let s = Self {
            // ! No defaults. Each names something the process cannot guess, and
            // a wrong guess fails silently rather than loudly: the wrong
            // vocabulary compares unrelated sparse dimensions, and the wrong
            // database serves a corpus the operator did not intend.
            database_url: required("DATABASE_URL")?,
            embed_endpoint: required("EMBED_ENDPOINT")?,
            bm25_vocab: required("BM25_VOCAB")?.into(),

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

            pipeline: Config {
                clusters_probed: parsed("CLUSTERS_PROBED", "positive integer", d.clusters_probed)?,
                per_cluster_k: parsed("PER_CLUSTER_K", "positive integer", d.per_cluster_k)?,
                per_arm_k: parsed("PER_ARM_K", "positive integer", d.per_arm_k)?,
                top_k: parsed("TOP_K", "positive integer", d.top_k)?,
                snippet_chars: parsed("SNIPPET_CHARS", "positive integer", d.snippet_chars)?,
                // ! Arm weights are properties of the CORPUS, not of the engine,
                // and a corpus is re-chunked far more often than the engine is
                // rebuilt. Refitting must not require a release.
                dense_weight: parsed("DENSE_WEIGHT", "number", d.dense_weight)?,
                sparse_weight: parsed("SPARSE_WEIGHT", "number", d.sparse_weight)?,
                text_weight: parsed("TEXT_WEIGHT", "number", d.text_weight)?,
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
            },
        };
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
            ("TOP_K", self.pipeline.top_k),
            ("SNIPPET_CHARS", self.pipeline.snippet_chars),
            ("GATE_SAMPLE", self.pipeline.gate_sample),
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
        "starting \u{b7} embed={} vocab={} transport={:?}",
        s.embed_endpoint,
        s.bm25_vocab.display(),
        s.transport
    );

    // +2: the pool serves `max_concurrency` searches plus the startup canary
    // and health probes, which must not have to wait behind a full queue.
    let pool = store::connect(&s.database_url, s.max_concurrency + 2)?;
    let ops = store::SearchOps::new(pool);

    // The corpus decides the vector space; the provider is built to match it.
    let meta = ops.corpus_meta().await?;
    log!(
        "corpus {} \u{b7} {} @ {}d \u{b7} {} pooling{}",
        meta.id,
        meta.dense_model,
        meta.dense_dim,
        meta.dense_pooling,
        if meta.dense_instruction.is_some() {
            " \u{b7} instruction-aware"
        } else {
            ""
        }
    );

    let dim = usize::try_from(meta.dense_dim).unwrap_or(0);
    let provider = Arc::new(embed::HttpProvider::new(
        &s.embed_endpoint,
        &meta.dense_model,
        dim,
    )?);
    let vectorizer = bm25::QueryVectorizer::load(&s.bm25_vocab)?;

    let pipe = Arc::new(
        Pipeline::new(
            ops,
            provider,
            vectorizer,
            &meta.dense_model.clone(),
            s.pipeline.clone(),
        )
        .await?,
    );
    log!(
        "ready \u{b7} {} clusters, probing {} \u{b7} concurrency {}",
        pipe.cluster_count(),
        s.pipeline.clusters_probed,
        s.max_concurrency
    );

    // ! Bounded concurrency (`CLAUDE.md` \u{a7}7.6). Peak RAM is fixed costs plus
    // this ceiling times the per-request ceiling, and both terms are bounded.
    let permits = Arc::new(Semaphore::new(s.max_concurrency));
    let limits = Limits {
        read_chunk_chars: s.read_chunk_chars,
        max_provenance_ids: s.max_provenance_ids,
    };

    match s.transport {
        Transport::Http => {
            // ! A wait ceiling is the other half of invariant 6. A bounded
            // queue alone still lets a caller block indefinitely behind a full
            // one; the bound has to be on TIME as well as on depth, or
            // backpressure never actually reaches the client.
            let server = Arc::new(Server {
                pipe,
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
                pipe,
                permits,
                queue_wait: None,
                limits,
            });
            serve_stdio(&server).await
        }
    }
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
    pub(crate) pipe: Arc<Pipeline>,
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
}

impl Server {
    /// Handle one JSON-RPC message. `None` means "notification, say nothing".
    pub(crate) async fn handle(&self, req: &Value) -> Option<Value> {
        let id = req.get("id").cloned();
        let method = req.get("method").and_then(Value::as_str).unwrap_or("");
        let params = req.get("params").cloned().unwrap_or(json!({}));

        match method {
            "initialize" => Some(ok(
                id.as_ref(),
                &json!({
                    "protocolVersion": PROTOCOL_VERSION,
                    "serverInfo": { "name": "vera", "version": env!("CARGO_PKG_VERSION") },
                    "capabilities": { "tools": {} }
                }),
            )),
            "tools/list" => Some(ok(id.as_ref(), &json!({ "tools": tools::definitions() }))),
            "tools/call" => {
                let Some(permit) = self.admit().await else {
                    return Some(busy(id.as_ref(), self.queue_wait));
                };
                let out = dispatch(&self.pipe, &params, self.limits).await;
                drop(permit);
                Some(ok(
                    id.as_ref(),
                    &json!({
                        "content": [{ "type": "text", "text": out.to_string() }]
                    }),
                ))
            }
            // Notifications carry no id and must not be answered.
            m if m.starts_with("notifications/") => None,
            other => Some(json!({
                "jsonrpc": "2.0",
                "id": id,
                "error": { "code": -32601, "message": format!("method not found: {other}") }
            })),
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

fn ok(id: Option<&Value>, result: &Value) -> Value {
    json!({ "jsonrpc": "2.0", "id": id, "result": result })
}

/// ! Backpressure is an answer, ✗ a failure. It states that the request was
/// never attempted, which is what makes a retry safe — a generic 500 does not
/// carry that, and a client that cannot tell the difference must assume the
/// worst and stop.
fn busy(id: Option<&Value>, waited: Option<std::time::Duration>) -> Value {
    let ms = waited.map_or(0, |d| d.as_millis());
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "error": {
            "code": BUSY_CODE,
            "message": format!("at capacity · waited {ms}ms for a slot · retry"),
            "data": { "retry": true }
        }
    })
}

async fn dispatch(pipe: &Pipeline, params: &Value, limits: Limits) -> Value {
    let name = params.get("name").and_then(Value::as_str).unwrap_or("");
    let args = params.get("arguments").cloned().unwrap_or(json!({}));

    match name {
        "list_domains" => list_domains(pipe),
        "search_knowledge" => search_knowledge(pipe, &args).await,
        "read_chunk" => read_chunk(pipe, &args, limits).await,
        "get_provenance" => get_provenance(pipe, &args, limits).await,
        "explain_routing" => explain_routing(pipe, &args).await,
        other => tools::error(
            "tools/call",
            &format!("unknown tool: {other}"),
            "call tools/list to see the available tools",
        ),
    }
}

fn list_domains(pipe: &Pipeline) -> Value {
    let m = pipe.corpus();
    json!({
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
        "token_estimate": 60,
    })
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
    match pipe.search(q).await {
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
        Ok(Some((row, body, truncated))) => json!({
            "success": true,
            "op": "read_chunk",
            "id": row.id,
            "body": body,
            "truncated": truncated,
            "truncated_at_source": row.truncated_at_source,
            "source": { "title": row.source_title, "url": row.source_url },
            "progress": ["read 1 chunk"],
            "token_estimate": body.len() / 4,
        }),
        Ok(None) => tools::error(
            "read_chunk",
            &format!("no chunk {id}"),
            "ids come from search_knowledge · they are not guessable",
        ),
        Err(e) => tools::error("read_chunk", &e.to_string(), "check the database"),
    }
}

async fn get_provenance(pipe: &Pipeline, args: &Value, limits: Limits) -> Value {
    // ! Capped. An agent that pastes a whole result set back is the normal
    // case, and an uncapped IN-list is an unbounded query and an unbounded
    // reply — the one place a read-only server can still be made to allocate
    // without limit.
    let ids: Vec<String> = args
        .get("ids")
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_str().map(str::to_owned))
                .take(limits.max_provenance_ids)
                .collect()
        })
        .unwrap_or_default();
    if ids.is_empty() {
        return tools::error(
            "get_provenance",
            "missing `ids`",
            "pass {\"ids\": [\"...\"]} from search_knowledge results",
        );
    }
    match pipe.provenance(&ids).await {
        Ok(rows) => {
            let incomplete = rows.iter().filter(|r| !r.provenance_complete()).count();
            json!({
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
                "token_estimate": rows.len() * 40,
                "hint": (incomplete > 0).then(|| format!(
                    "{incomplete} of {} have no source_url · cite by title and locator",
                    rows.len()
                )),
            })
        }
        Err(e) => tools::error("get_provenance", &e.to_string(), "check the database"),
    }
}

async fn explain_routing(pipe: &Pipeline, args: &Value) -> Value {
    let Some(q) = args.get("query").and_then(Value::as_str) else {
        return tools::error("explain_routing", "missing `query`", "pass a query");
    };
    match pipe.explain(q).await {
        Ok(r) => json!({
            "success": true,
            "op": "explain_routing",
            "detected_domain": r.domain,
            "clusters_probed": r.clusters_probed,
            "cluster_scores": r.cluster_scores.iter()
                .map(|(id, s)| json!({ "cluster": id, "similarity": s }))
                .collect::<Vec<_>>(),
            "total_clusters": r.total_clusters,
            "identifiers_detected": r.identifiers,
            "routing_bypass": !r.identifiers.is_empty(),
            "provider": r.provider,
            "progress": ["routed"],
            "token_estimate": 120,
        }),
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
            max_concurrency: 4,
            transport: Transport::Http,
            http_addr: "0.0.0.0:8081".into(),
            queue_wait: std::time::Duration::from_secs(2),
            read_chunk_chars: 4000,
            max_provenance_ids: 50,
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
    fn probing_zero_clusters_is_refused() {
        let mut s = settings();
        s.pipeline.clusters_probed = 0;
        assert!(s.validate().is_err());
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
        let v = busy(Some(&json!(7)), Some(ms(1500)));
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
