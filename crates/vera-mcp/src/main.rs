//! `vera-mcp` · the MCP server.
//!
//! stdio transport today; streamable-http is the second transport
//! (`MCP_ENGINE.md` §7) and wraps the same core.
//!
//! ! **Nothing but JSON-RPC frames go to stdout.** stdio *is* the protocol
//! channel, so a stray `println!` corrupts the stream and the client's failure
//! is a parse error that points nowhere near the print. Every log, warning and
//! diagnostic goes to stderr (`CLAUDE.md` §7 rule 10).

mod concurrency;
mod tools;

use std::sync::Arc;

use concurrency::Gate;
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use vera_core::{Config, EmbeddingSpace};
use vera_embed::{EmbeddingProvider, StubProvider};
use vera_engine::Engine;
use vera_store::{ChunkStore, sqlite::SqliteStore};

const PROTOCOL_VERSION: &str = "2024-11-05";

struct Server {
    engine: Engine<SqliteStore>,
    provider: Box<dyn EmbeddingProvider>,
    gate: Gate,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let corpus = std::env::var("VERA_CORPUS")
        .map_err(|_| "set VERA_CORPUS to the corpus path")?;

    let store = SqliteStore::open(&corpus)?;
    let space: EmbeddingSpace = store.corpus_space()?;

    // ! The engine is configured **from the corpus**, not from a constant. A
    // config that declared a different width or model would be rejected by
    // Engine::load rather than silently ranking across incompatible spaces.
    let config = Config {
        embedding: space.clone(),
        ..Config::default()
    };
    let gate = Gate::from_config(&config.concurrency);
    let engine = Engine::load(store, config)?;

    // ! The stub is a placeholder for the pinned OpenRouter client, and it is
    // wired here rather than buried so the substitution is obvious. It produces
    // deterministic hashes with no semantics: it must never serve real traffic.
    // The real client, with its pin/retry/canary, is the next piece
    // (`MCP_ENGINE.md` §6).
    let provider: Box<dyn EmbeddingProvider> = Box::new(StubProvider::new(space.clone()));
    eprintln!(
        "vera-mcp · corpus={corpus} model={} dim={} provider={}",
        space.model_id,
        space.dim,
        provider.describe()
    );
    eprintln!("WARNING: stub embedding provider · results carry no semantics");

    let server = Arc::new(Server {
        engine,
        provider,
        gate,
    });

    let mut lines = BufReader::new(tokio::io::stdin()).lines();
    let mut stdout = tokio::io::stdout();

    while let Some(line) = lines.next_line().await? {
        if line.trim().is_empty() {
            continue;
        }
        let Ok(req) = serde_json::from_str::<Value>(&line) else {
            eprintln!("skipping unparseable frame");
            continue;
        };
        let id = req.get("id").cloned().unwrap_or(Value::Null);
        let method = req.get("method").and_then(Value::as_str).unwrap_or("");

        let response = match method {
            "initialize" => json!({
                "jsonrpc": "2.0", "id": id,
                "result": {
                    "protocolVersion": PROTOCOL_VERSION,
                    "serverInfo": { "name": "vera", "version": env!("CARGO_PKG_VERSION") },
                    "capabilities": { "tools": {} }
                }
            }),
            "notifications/initialized" => continue,
            "tools/list" => json!({
                "jsonrpc": "2.0", "id": id,
                "result": { "tools": tools::schemas() }
            }),
            "tools/call" => {
                let params = req.get("params").cloned().unwrap_or(Value::Null);
                let result = dispatch(&server, &params).await;
                json!({
                    "jsonrpc": "2.0", "id": id,
                    "result": {
                        // MCP carries tool output as text content; the payload
                        // is the JSON document the return contract defines.
                        "content": [{
                            "type": "text",
                            "text": serde_json::to_string(&result)?
                        }],
                        "isError": result.get("success") == Some(&json!(false))
                    }
                })
            }
            other => json!({
                "jsonrpc": "2.0", "id": id,
                "error": { "code": -32601, "message": format!("method not found: {other}") }
            }),
        };

        let mut bytes = serde_json::to_vec(&response)?;
        bytes.push(b'\n');
        stdout.write_all(&bytes).await?;
        stdout.flush().await?;
    }
    Ok(())
}

/// Route one `tools/call` to one of the four primitives.
async fn dispatch(server: &Arc<Server>, params: &Value) -> Value {
    let name = params.get("name").and_then(Value::as_str).unwrap_or("");
    let args = params.get("arguments").cloned().unwrap_or(json!({}));

    // ! Admission is taken **before** any work, including embedding. A request
    // that will be rejected must not first spend a network round-trip.
    let _guard = match server.gate.admit().await {
        Ok(g) => g,
        Err(rejected) => return tools::failure(name, &rejected.message(), rejected.hint()),
    };

    match name {
        "describe" => tools::describe(&server.engine),

        "fetch" => {
            let Some(ids) = tools::args::list(&args, "ids") else {
                return tools::failure("fetch", "missing 'ids'", "pass one id or an array of ids");
            };
            let Some(depth) = tools::args::depth(&args) else {
                return tools::failure(
                    "fetch",
                    "unknown depth",
                    "depth must be provenance, snippet or full",
                );
            };
            tools::fetch(&server.engine, &ids, depth)
        }

        "traverse" => {
            let Some(ids) = tools::args::list(&args, "ids") else {
                return tools::failure("traverse", "missing 'ids'", "pass one id or an array of ids");
            };
            let Some(edge) = args.get("edge").and_then(Value::as_str) else {
                return tools::failure(
                    "traverse",
                    "missing 'edge'",
                    "call describe for the available edges",
                );
            };
            let limit = tools::args::usize_field(&args, "limit").unwrap_or(20);
            tools::traverse(&server.engine, &ids, edge, limit)
        }

        "search" => {
            let Some(queries) = tools::args::list(&args, "query") else {
                return tools::failure("search", "missing 'query'", "pass a query or an array");
            };
            if queries.is_empty() {
                return tools::failure("search", "empty 'query'", "pass at least one query");
            }
            // ! Constraints are validated before embedding. Rejecting after the
            // provider round-trip would spend a network call to say no, and —
            // worse — invites a future edit that quietly searches unfiltered.
            if let Err(e) = tools::validate_constraints(&server.engine, &args) {
                return e;
            }

            let mut embedded = Vec::with_capacity(queries.len());
            for text in queries {
                match server.provider.embed_query(&text).await {
                    Ok(vector) => embedded.push(tools::Embedded { text, vector }),
                    Err(e) => {
                        return tools::failure(
                            "search",
                            &e.to_string(),
                            "the embedding provider is unavailable or has drifted · \
                             the engine refuses to serve rather than rank in the wrong space",
                        );
                    }
                }
            }

            tools::search(
                &server.engine,
                &embedded,
                tools::args::usize_field(&args, "k"),
                tools::args::usize_field(&args, "clusters_probed"),
                tools::args::flag(&args, "dry_run"),
            )
        }

        other => tools::failure(
            other,
            &format!("unknown tool '{other}'"),
            "call tools/list for the available tools",
        ),
    }
}
