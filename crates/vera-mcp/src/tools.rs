//! The five agent-facing tools · `MCP_ENGINE.md` §2.
//!
//! Thin wrappers, by design: every one of these translates JSON in, calls
//! [`vera_engine`], and translates the result out. No retrieval logic lives
//! here, so the tool surface can change without touching the engine and the
//! engine can be benchmarked without an MCP client.
//!
//! ! Read-only, all five. There is no write tool and there must never be one:
//! the corpus is mutated by the offline pipelines under an atomic version swap
//! (`CLAUDE.md` §5 rule 6). A write path here would also break statelessness,
//! which is what makes replicas free.

use serde_json::{Value, json};
use vera_core::Confidence;
use vera_engine::{Engine, Probe, identifier, routing};
use vera_store::ChunkStore;

/// `len(str(response)) / 4` · the agent budgets its own context with this.
fn token_estimate(v: &Value) -> usize {
    serde_json::to_string(v).map_or(0, |s| s.len() / 4)
}

/// Attach `token_estimate` last, once the body is final.
fn finish(mut v: Value) -> Value {
    let est = token_estimate(&v);
    if let Some(obj) = v.as_object_mut() {
        obj.insert("token_estimate".into(), json!(est));
    }
    v
}

/// A failure, in the shape the return contract requires.
///
/// ! `error` *and* `hint`. An error the agent cannot act on costs a retry loop;
/// the hint is what turns a failure into a next step (`MCP_ENGINE.md` §3).
#[must_use]
pub fn failure(op: &str, error: &str, hint: &str) -> Value {
    finish(json!({
        "success": false,
        "op": op,
        "error": error,
        "hint": hint,
        "progress": [],
    }))
}

/// `list_domains` — what this engine covers. Zero content.
///
/// ! Introspection only. The agent does **not** feed the result back as a
/// domain argument — `search_knowledge` detects the domain itself. Letting the
/// agent choose would let it name a domain that does not exist, or pick the
/// wrong one, and neither failure is visible in the results.
pub fn list_domains<S: ChunkStore>(engine: &Engine<S>) -> Value {
    match engine.store().domains() {
        Err(e) => failure("list_domains", &e.to_string(), "check the corpus is readable"),
        Ok(domains) => finish(json!({
            "success": true,
            "op": "list_domains",
            "domains": domains.iter().map(|d| json!({
                "id": d.id,
                "description": d.description,
                "row_count": d.row_count,
            })).collect::<Vec<_>>(),
            "progress": [format!("listed {} domain(s)", domains.len())],
        })),
    }
}

/// `search_knowledge` — the workhorse. Takes **only** a query.
pub fn search_knowledge<S: ChunkStore>(
    engine: &Engine<S>,
    query: &str,
    query_vector: &[f32],
    max_results: Option<usize>,
    clusters_probed: Option<usize>,
) -> Value {
    let probe = Probe::Nearest(
        clusters_probed.unwrap_or(engine.config().routing.clusters_probed),
    );
    match engine.search(query, query_vector, probe) {
        Err(e) => failure(
            "search_knowledge",
            &e.to_string(),
            "verify the engine and corpus share an embedding space",
        ),
        Ok(outcome) => {
            let mut response = outcome.response;
            if let Some(cap) = max_results {
                if response.results.len() > cap {
                    response.results.truncate(cap);
                    response.citation_block =
                        vera_core::contract::citation_block(&response.results);
                    response.truncated = true;
                }
            }
            response.token_estimate = response.estimate_tokens();
            serde_json::to_value(&response).unwrap_or_else(|e| {
                failure("search_knowledge", &e.to_string(), "internal serialization")
            })
        }
    }
}

/// `read_chunk` — one chunk's full text, size-capped.
///
/// ! Capped and flagged. `search_knowledge` deliberately returns snippets; this
/// is the escape hatch for the few chunks an agent actually needs, and an
/// uncapped one would let a single call blow the agent's whole context
/// (`CLAUDE.md` §7 rule 11).
pub fn read_chunk<S: ChunkStore>(engine: &Engine<S>, chunk_id: &str) -> Value {
    let cap = engine.config().read.max_chunk_bytes;
    match engine.store().chunks_by_id(&[chunk_id.to_owned()]) {
        Err(e) => failure("read_chunk", &e.to_string(), "check the corpus is readable"),
        Ok(found) => match found.into_iter().next() {
            None => failure(
                "read_chunk",
                &format!("no chunk with id '{chunk_id}'"),
                "ids come from search_knowledge results · check for a typo or a stale id",
            ),
            Some(chunk) => {
                // Cut on a char boundary · a byte-exact truncation would split
                // a multibyte character and produce invalid UTF-8.
                let truncated = chunk.body.len() > cap;
                let body = if truncated {
                    let mut end = cap;
                    while end > 0 && !chunk.body.is_char_boundary(end) {
                        end -= 1;
                    }
                    chunk.body[..end].to_owned()
                } else {
                    chunk.body.clone()
                };
                finish(json!({
                    "success": true,
                    "op": "read_chunk",
                    "id": chunk.id,
                    "text": body,
                    "truncated": truncated,
                    "source": chunk.source(),
                    "progress": ["read 1 chunk"],
                }))
            }
        },
    }
}

/// `get_provenance` — source links and locators, for human verification.
///
/// ! Never synthesized. Every field comes from what ingest recorded; a chunk
/// with no locator reports an empty one rather than a plausible guess
/// (`CLAUDE.md` §7 rule 8).
pub fn get_provenance<S: ChunkStore>(engine: &Engine<S>, ids: &[String]) -> Value {
    match engine.store().provenance(ids) {
        Err(e) => failure("get_provenance", &e.to_string(), "check the corpus is readable"),
        Ok(found) => {
            let missing: Vec<&String> = ids
                .iter()
                .filter(|id| !found.iter().any(|(f, _)| f == *id))
                .collect();
            finish(json!({
                "success": true,
                "op": "get_provenance",
                "provenance": found.iter().map(|(id, src)| json!({
                    "id": id,
                    "source_url": src.url,
                    "title": src.title,
                    "locator": src.locator,
                    "citation": format!("{} {} — {}", src.title, src.locator.render(), src.url),
                })).collect::<Vec<_>>(),
                // ! Reported, ✗ silently dropped. An id that resolves to
                // nothing means a stale result set, which the agent must know.
                "missing": missing,
                "progress": [format!("resolved {}/{} id(s)", found.len(), ids.len())],
            }))
        }
    }
}

/// `explain_routing` — why a query went where it went, without searching.
///
/// For tuning and for trust: it exposes the layer-1 decision and the layer-2
/// distances, including the threshold actually in force, so a surprising empty
/// result can be diagnosed rather than guessed at.
pub fn explain_routing<S: ChunkStore>(
    engine: &Engine<S>,
    query: &str,
    query_vector: &[f32],
) -> Value {
    let domains = match engine.store().domains() {
        Ok(d) => d,
        Err(e) => return failure("explain_routing", &e.to_string(), "check the corpus"),
    };

    let scores: Vec<Value> = domains
        .iter()
        .map(|d| {
            let sim = vera_embed::cosine(query_vector, &d.anchor);
            let threshold = engine.threshold_for(&d.id);
            json!({
                "domain": d.id,
                "similarity": sim,
                "threshold": threshold,
                "accepted": sim >= threshold,
            })
        })
        .collect();

    let detected = routing::detect_domain_with(query_vector, &domains, |id| {
        engine.threshold_for(id)
    });

    let clusters = detected.as_ref().map_or_else(Vec::new, |d| {
        match engine.store().centroids(&d.id) {
            Err(_) => Vec::new(),
            Ok(centroids) => routing::nearest_clusters(
                query_vector,
                &centroids,
                engine.config().routing.clusters_probed,
            )
            .into_iter()
            .map(|p| {
                json!({
                    "cluster_id": p.cluster_id,
                    "similarity": p.similarity,
                    "row_count": p.row_count,
                })
            })
            .collect(),
        }
    });

    let identifiers: Vec<Value> = identifier::extract(query)
        .into_iter()
        .map(|i| json!({ "canonical": i.canonical, "matched_on": i.matched_on }))
        .collect();

    finish(json!({
        "success": true,
        "op": "explain_routing",
        "query": query,
        "detected_domain": detected.as_ref().map(|d| d.id.clone()),
        "domain_scores": scores,
        "clusters_probed": clusters,
        // ! Surfaced because this path *bypasses* routing entirely. Without it,
        // an operator reading this output would conclude a query that found a
        // regulation by identifier had been routed to it.
        "exact_identifiers": identifiers,
        "confidence": detected.as_ref().map_or(Confidence::None, |d| {
            if d.similarity >= 0.60 { Confidence::High }
            else if d.similarity >= 0.40 { Confidence::Medium }
            else { Confidence::Low }
        }),
        "progress": ["routed without searching"],
    }))
}

/// The tool schemas advertised by `tools/list`.
#[must_use]
pub fn schemas() -> Value {
    json!([
        {
            "name": "list_domains",
            "description": "List available knowledge bases. Ids and descriptions.",
            "inputSchema": { "type": "object", "properties": {}, "required": [] },
            "annotations": { "readOnlyHint": true, "idempotentHint": true, "openWorldHint": false }
        },
        {
            "name": "search_knowledge",
            "description": "Routed hybrid search. Returns ranked cited results.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": { "type": "string" },
                    "max_results": { "type": "integer", "default": 10 },
                    "clusters_probed": { "type": "integer", "default": 5 }
                },
                "required": ["query"]
            },
            "annotations": { "readOnlyHint": true, "idempotentHint": true, "openWorldHint": true }
        },
        {
            "name": "read_chunk",
            "description": "Read full text of one chunk by id. Size-capped.",
            "inputSchema": {
                "type": "object",
                "properties": { "chunk_id": { "type": "string" } },
                "required": ["chunk_id"]
            },
            "annotations": { "readOnlyHint": true, "idempotentHint": true, "openWorldHint": false }
        },
        {
            "name": "get_provenance",
            "description": "Source links + locators for chunk ids, to verify.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "chunk_ids": { "type": "array", "items": { "type": "string" } }
                },
                "required": ["chunk_ids"]
            },
            "annotations": { "readOnlyHint": true, "idempotentHint": true, "openWorldHint": false }
        },
        {
            "name": "explain_routing",
            "description": "Show which domain/clusters a query routes to.",
            "inputSchema": {
                "type": "object",
                "properties": { "query": { "type": "string" } },
                "required": ["query"]
            },
            "annotations": { "readOnlyHint": true, "idempotentHint": true, "openWorldHint": true }
        }
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_surface_stays_within_the_eight_tool_budget() {
        // CLAUDE.md §6 · a tool surface an agent cannot hold in mind is a
        // surface it uses badly.
        let n = schemas().as_array().map_or(0, Vec::len);
        assert!(n <= 8, "{n} tools");
        assert_eq!(n, 5);
    }

    #[test]
    fn every_tool_declares_itself_read_only() {
        // ! Structural guard on "the query path never writes".
        for tool in schemas().as_array().unwrap() {
            assert_eq!(
                tool["annotations"]["readOnlyHint"], json!(true),
                "{} is not marked read-only", tool["name"]
            );
        }
    }

    #[test]
    fn search_knowledge_does_not_accept_a_domain_argument() {
        // ! CLAUDE.md §7 rule 13. The agent must not be able to name a domain;
        // if this schema ever grows one, detection has been bypassed.
        let tools = schemas();
        let search = tools
            .as_array()
            .unwrap()
            .iter()
            .find(|t| t["name"] == "search_knowledge")
            .unwrap();
        let props = search["inputSchema"]["properties"].as_object().unwrap();
        assert!(!props.contains_key("domain"), "{props:?}");
        assert_eq!(search["inputSchema"]["required"], json!(["query"]));
    }

    #[test]
    fn a_failure_carries_both_an_error_and_a_hint() {
        let f = failure("read_chunk", "no such chunk", "check the id");
        assert_eq!(f["success"], json!(false));
        assert!(f["error"].is_string() && f["hint"].is_string());
        assert!(f["token_estimate"].as_u64().unwrap() > 0);
    }

    #[test]
    fn every_response_carries_a_token_estimate() {
        let v = finish(json!({ "success": true, "op": "x" }));
        assert!(v["token_estimate"].as_u64().unwrap() > 0);
    }
}
