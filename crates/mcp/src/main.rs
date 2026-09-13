//! `mcp` · the MCP server: transport and tool dispatch, no domain logic.
//!
//! ! Every log line goes to **stderr** (`CLAUDE.md` §7.10). stdout carries the
//! JSON-RPC stream and nothing else; one stray `println!` corrupts the channel
//! and the client sees a protocol error rather than a message.
//!
//! Configuration, all from the environment so nothing is hardcoded:
//!
//! ```text
//! DATABASE_URL     postgres connection string
//! EMBED_ENDPOINT   embedding server base URL      (default http://localhost:8080)
//! BM25_VOCAB       path to the corpus's vocabulary artifact
//! CLUSTERS_PROBED  layer-2 probe width            (default 5)
//! DENSE_WEIGHT     RRF weight for the dense arm   (default 0.0, measured)
//! SPARSE_WEIGHT    RRF weight for BM25            (default 1.0)
//! TEXT_WEIGHT      RRF weight for tsvector        (default 0.0, measured)
//! MAX_CONCURRENCY  in-flight request ceiling      (default 4)
//! TRANSPORT        stdio | http                    (default stdio)
//! HTTP_ADDR        bind address for http           (default 0.0.0.0:8081)
//! QUEUE_WAIT_MS    how long a request may queue    (default 2000)
//! CANARY_MIN_COSINE startup round-trip threshold   (default 0.98)
//! DOMAIN_FLOOR     domain gate · centroid similarity (default 0.45)
//! DOMAIN_LEXICAL_FLOOR domain gate · lexical evidence (default 0.40)
//! ```

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
const DEFAULT_READ_CHUNK_CHARS: usize = 4000;

/// Log to stderr. ! Never stdout.
macro_rules! log {
    ($($arg:tt)*) => { eprintln!($($arg)*) };
}

/// Tunables from the environment · invariant 12, so bigger hardware and a
/// different corpus move these without a rebuild.
fn config_from_env(clusters_probed: usize) -> Config {
    fn f32_from(key: &str, fallback: f32) -> f32 {
        std::env::var(key)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(fallback)
    }
    let d = Config::default();
    Config {
        clusters_probed,
        // ! Loosening the canary is a deliberate, visible act — there is no
        // code path that quietly skips it.
        canary_min_cosine: f32_from("CANARY_MIN_COSINE", d.canary_min_cosine),
        domain_floor: f32_from("DOMAIN_FLOOR", d.domain_floor),
        domain_lexical_floor: f32_from("DOMAIN_LEXICAL_FLOOR", d.domain_lexical_floor),
        // ! These were documented as overridable for three releases while
        // `..d` quietly discarded them, which is invariant 12 violated in the
        // one place it is most expensive: arm weights are corpus-specific, and
        // a corpus is re-chunked far more often than the engine is rebuilt.
        dense_weight: f32_from("DENSE_WEIGHT", d.dense_weight),
        sparse_weight: f32_from("SPARSE_WEIGHT", d.sparse_weight),
        text_weight: f32_from("TEXT_WEIGHT", d.text_weight),
        ..d
    }
}

#[tokio::main]
async fn main() {
    // ! Display, not Debug. A refusal to serve has to tell the operator what
    // went wrong and why; `CanaryFailed { got: 0.61, want: 0.98 }` makes them
    // go read the source, and the message already explains itself.
    if let Err(e) = run().await {
        log!("refusing to serve · {e}");
        std::process::exit(1);
    }
}

async fn run() -> Result<(), Box<dyn std::error::Error>> {
    let db = std::env::var("DATABASE_URL")
        .unwrap_or_else(|_| "host=localhost port=5432 dbname=vera user=vera password=vera".into());
    let endpoint =
        std::env::var("EMBED_ENDPOINT").unwrap_or_else(|_| "http://localhost:8080".into());
    let vocab =
        std::env::var("BM25_VOCAB").unwrap_or_else(|_| ".test/runs/spike-01.bm25.json".into());
    let probed: usize = std::env::var("CLUSTERS_PROBED")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(5);
    let max_conc: usize = std::env::var("MAX_CONCURRENCY")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4);

    log!("starting · db={db} embed={endpoint} vocab={vocab}");

    let pool = store::connect(&db, max_conc + 2)?;
    let ops = store::SearchOps::new(pool);

    // The corpus decides the vector space; the provider is built to match it.
    let meta = ops.corpus_meta().await?;
    log!(
        "corpus {} · {} @ {}d · {} pooling",
        meta.id,
        meta.dense_model,
        meta.dense_dim,
        meta.dense_pooling
    );

    let dim = usize::try_from(meta.dense_dim).unwrap_or(0);
    let provider = Arc::new(embed::HttpProvider::new(&endpoint, &meta.dense_model, dim)?);
    let vectorizer = bm25::QueryVectorizer::load(std::path::Path::new(&vocab))?;

    let cfg = config_from_env(probed);
    let pipe =
        Arc::new(Pipeline::new(ops, provider, vectorizer, &meta.dense_model.clone(), cfg).await?);
    log!(
        "ready · {} clusters, probing {} · concurrency {}",
        pipe.cluster_count(),
        probed,
        max_conc
    );

    // ! Bounded concurrency (`CLAUDE.md` §7.6). Peak RAM is fixed costs plus
    // this ceiling times the per-request ceiling, and both terms are bounded.
    let permits = Arc::new(Semaphore::new(max_conc));

    match std::env::var("TRANSPORT")
        .unwrap_or_else(|_| "stdio".into())
        .as_str()
    {
        "http" => {
            let wait: u64 = std::env::var("QUEUE_WAIT_MS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(2000);
            // ! A wait ceiling is the other half of invariant 6. A bounded
            // queue alone still lets a caller block indefinitely behind a full
            // one; the bound has to be on TIME as well as on depth, or
            // backpressure never actually reaches the client.
            let server = Arc::new(Server {
                pipe,
                permits,
                queue_wait: Some(std::time::Duration::from_millis(wait)),
            });
            let addr = std::env::var("HTTP_ADDR").unwrap_or_else(|_| "0.0.0.0:8081".into());
            log!("transport http · {addr} · queue wait {wait}ms");
            http::serve(server, &addr).await?;
            return Ok(());
        }
        "stdio" => {}
        other => {
            return Err(format!("unknown TRANSPORT '{other}' · expected stdio or http").into());
        }
    }

    // ! stdio is serial: one line is read, handled, answered, and only then is
    // the next read. Nothing can queue, so there is nobody to refuse and a wait
    // ceiling could only ever fire against the request already being served.
    let server = Arc::new(Server {
        pipe,
        permits,
        queue_wait: None,
    });

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
                let out = dispatch(&self.pipe, &params).await;
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
        Some(d) => tokio::time::timeout(d, sem.acquire_owned()).await.ok()?.ok(),
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

async fn dispatch(pipe: &Pipeline, params: &Value) -> Value {
    let name = params.get("name").and_then(Value::as_str).unwrap_or("");
    let args = params.get("arguments").cloned().unwrap_or(json!({}));

    match name {
        "list_domains" => list_domains(pipe),
        "search_knowledge" => search_knowledge(pipe, &args).await,
        "read_chunk" => read_chunk(pipe, &args).await,
        "get_provenance" => get_provenance(pipe, &args).await,
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

async fn read_chunk(pipe: &Pipeline, args: &Value) -> Value {
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
        .unwrap_or(DEFAULT_READ_CHUNK_CHARS);
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

async fn get_provenance(pipe: &Pipeline, args: &Value) -> Value {
    let ids: Vec<String> = args
        .get("ids")
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_str().map(str::to_owned))
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
