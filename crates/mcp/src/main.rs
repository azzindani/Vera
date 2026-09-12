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
//! CANARY_MIN_COSINE startup round-trip threshold   (default 0.98)
//! ```

mod bm25;
mod identifier;
mod pipeline;
mod tools;

use std::sync::Arc;

use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::sync::Semaphore;

use pipeline::{Config, Pipeline};

const PROTOCOL_VERSION: &str = "2024-11-05";
const DEFAULT_READ_CHUNK_CHARS: usize = 4000;

/// Log to stderr. ! Never stdout.
macro_rules! log {
    ($($arg:tt)*) => { eprintln!($($arg)*) };
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

    // ! Config, not a constant (invariant 12). Loosening the canary is a
    // deliberate, visible act — there is no code path that quietly skips it.
    let canary_min: f32 = std::env::var("CANARY_MIN_COSINE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(Config::default().canary_min_cosine);

    let cfg = Config {
        clusters_probed: probed,
        canary_min_cosine: canary_min,
        ..Config::default()
    };
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

        let id = req.get("id").cloned();
        let method = req.get("method").and_then(Value::as_str).unwrap_or("");
        let params = req.get("params").cloned().unwrap_or(json!({}));

        let response = match method {
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
                let permit = permits.clone().acquire_owned().await;
                let out = dispatch(&pipe, &params).await;
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
        };

        if let Some(resp) = response {
            let mut bytes = serde_json::to_vec(&resp)?;
            bytes.push(b'\n');
            stdout.write_all(&bytes).await?;
            stdout.flush().await?;
        }
    }
    Ok(())
}

fn ok(id: Option<&Value>, result: &Value) -> Value {
    json!({ "jsonrpc": "2.0", "id": id, "result": result })
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
