//! The four agent-facing primitives · `MULTI_DOMAIN.md` §10b.
//!
//! ```text
//! describe   what exists — sources, filterable fields, edges, limits
//! search     query (+ constraints) → ranked, cited results
//! fetch      by id → provenance | snippet | full text
//! traverse   from id → along a declared edge
//! ```
//!
//! ! **Capability grows through declared parameters, ✗ new tools.** Adding a
//! source, an edge type, a filterable field or a validity model must add zero
//! tools. A new verb here means the model was wrong — the surface is the place
//! that failure shows up first, so it is guarded by tests.
//!
//! ! **The agent passes intent; the engine owns strategy.** Query text,
//! constraints, `k`, edge name and fetch depth are intent and safe to accept.
//! Which source is searched, which rankers run, their fusion weights and which
//! clusters are probed are strategy — never agent-supplied. An agent that could
//! set fusion weights could silently disable the keyword half, which is the same
//! hole as letting it name a domain (`CLAUDE.md` §7 rule 13), one level down.
//!
//! Thin wrappers, all four: JSON in, [`vera_engine`] call, JSON out. No
//! retrieval logic lives here.

use serde_json::{Value, json};
use vera_core::Confidence;
use vera_engine::{Engine, Probe, identifier, routing};
use vera_store::{ChunkStore, Edge};

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

/// Accept `"x"` or `["x", "y"]` · bulk is a parameter shape, ✗ a tool.
///
/// ! This is what keeps `search_batch` from existing. Batching matters because
/// the embedding provider is a network round-trip (`MCP_ENGINE.md` §6.6) and one
/// admission slot then covers the whole set — but none of that justifies a
/// second verb.
fn as_list(v: Option<&Value>) -> Option<Vec<String>> {
    match v? {
        Value::String(s) => Some(vec![s.clone()]),
        Value::Array(a) => Some(
            a.iter()
                .filter_map(|x| x.as_str().map(ToOwned::to_owned))
                .collect(),
        ),
        _ => None,
    }
}

/// Fields an agent may filter on, for the detected corpus.
///
/// ! Empty today, and deliberately so. Structured constraints
/// (`MULTI_DOMAIN.md` §5) need per-source metadata and the selectivity
/// statistics the cardinality rule (§7) depends on; neither exists in the
/// schema. Publishing an empty list is honest — publishing field names the
/// engine would then ignore is the failure this whole surface is shaped to
/// avoid.
fn filterable_fields<S: ChunkStore>(_engine: &Engine<S>) -> Vec<&'static str> {
    Vec::new()
}

/// Reject a constraint the engine cannot honour · **loudly**.
///
/// ! Silently dropping an unrecognised filter returns a confident answer to a
/// question that was not asked: the agent believes the results are restricted to
/// `court = "MA"` and they are not. Same failure family as applying a constraint
/// after routing (`MULTI_DOMAIN.md` §5) — a well-formed answer that is wrong in
/// a way nothing in the response reveals.
pub fn validate_constraints<S: ChunkStore>(engine: &Engine<S>, args: &Value) -> Result<(), Value> {
    let Some(obj) = args.get("constraints").and_then(Value::as_object) else {
        return Ok(());
    };
    if obj.is_empty() {
        return Ok(());
    }
    let allowed = filterable_fields(engine);
    let unknown: Vec<&String> = obj.keys().filter(|k| !allowed.contains(&k.as_str())).collect();
    if unknown.is_empty() {
        return Ok(());
    }
    Err(failure(
        "search",
        &format!(
            "unsupported constraint field(s): {}",
            unknown
                .iter()
                .map(|s| s.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        ),
        if allowed.is_empty() {
            "this corpus exposes no filterable fields · call describe to see what it \
             supports, and drop the constraint rather than assuming it was applied"
        } else {
            "call describe for the fields this corpus can filter on"
        },
    ))
}

// ── describe ─────────────────────────────────────────────────────────────────

/// What this engine holds, and what an agent may ask of it.
///
/// ! Introspection only, zero content. The agent does **not** feed a source back
/// as an argument — `search` detects it. `describe` exists so an agent can see
/// which constraints and edges are real *before* using one, rather than
/// discovering it through a failure.
pub fn describe<S: ChunkStore>(engine: &Engine<S>) -> Value {
    let domains = match engine.store().domains() {
        Ok(d) => d,
        Err(e) => return failure("describe", &e.to_string(), "check the corpus is readable"),
    };
    let space = engine.space();
    let cfg = engine.config();

    let sources: Vec<Value> = domains
        .iter()
        .map(|d| {
            json!({
                "id": d.id,
                // Today one level: a source and its domain share an id. The
                // field exists now so the shape does not change when the source
                // level lands (`MULTI_DOMAIN.md` §2).
                "domain": d.id,
                "description": d.description,
                "row_count": d.row_count,
                "embedding": { "model": space.model_id, "dim": space.dim },
                "filterable_fields": filterable_fields(engine),
                "edges": Edge::all().iter().map(|e| e.name()).collect::<Vec<_>>(),
                // Absent until the schema carries validity columns · reported as
                // null rather than omitted, so its absence is visible.
                "validity": Value::Null,
                "identifier_grammar": "indonesian-regulation",
            })
        })
        .collect();

    finish(json!({
        "success": true,
        "op": "describe",
        "sources": sources,
        "edges": Edge::all().iter().map(|e| e.name()).collect::<Vec<_>>(),
        "limits": {
            "max_results": cfg.search.max_results,
            "max_chunk_bytes": cfg.read.max_chunk_bytes,
            "clusters_probed_default": cfg.routing.clusters_probed,
        },
        "progress": [format!("described {} source(s)", sources.len())],
    }))
}

// ── search ───────────────────────────────────────────────────────────────────

/// One query's worth of work, already embedded.
pub struct Embedded {
    pub text: String,
    pub vector: Vec<f32>,
}

/// Routed hybrid search over one or many queries.
///
/// `dry_run` returns the routing decision without searching — the same planning
/// work with execution suppressed. ! Kept as a flag rather than a separate
/// `explain_routing` tool: a second tool would have to duplicate the entire
/// constraint surface, and the two would drift.
pub fn search<S: ChunkStore>(
    engine: &Engine<S>,
    queries: &[Embedded],
    max_results: Option<usize>,
    clusters_probed: Option<usize>,
    dry_run: bool,
) -> Value {
    let probe = Probe::Nearest(clusters_probed.unwrap_or(engine.config().routing.clusters_probed));

    let one = |q: &Embedded| -> Value {
        if dry_run {
            return routing_plan(engine, &q.text, &q.vector);
        }
        match engine.search(&q.text, &q.vector, probe) {
            Err(e) => failure(
                "search",
                &e.to_string(),
                "verify the engine and corpus share an embedding space",
            ),
            Ok(outcome) => {
                let mut response = outcome.response;
                if let Some(cap) = max_results
                    && response.results.len() > cap
                {
                    response.results.truncate(cap);
                    response.citation_block =
                        vera_core::contract::citation_block(&response.results);
                    response.truncated = true;
                }
                response.token_estimate = response.estimate_tokens();
                serde_json::to_value(&response).unwrap_or_else(|e| {
                    failure("search", &e.to_string(), "internal serialization")
                })
            }
        }
    };

    match queries {
        // ! A single query returns the bare response `OUTPUT_CONTRACT.md` §2
        // specifies — batching must not change the shape of the common case.
        [only] => one(only),
        many => finish(json!({
            "success": true,
            "op": "search",
            "searches": many.iter().map(one).collect::<Vec<_>>(),
            "progress": [format!("ran {} queries in one batch", many.len())],
        })),
    }
}

/// The routing decision for a query, without executing the search.
fn routing_plan<S: ChunkStore>(engine: &Engine<S>, query: &str, vector: &[f32]) -> Value {
    let domains = match engine.store().domains() {
        Ok(d) => d,
        Err(e) => return failure("search", &e.to_string(), "check the corpus"),
    };

    let scores: Vec<Value> = domains
        .iter()
        .map(|d| {
            let sim = vera_embed::cosine(vector, &d.anchor);
            let threshold = engine.threshold_for(&d.id);
            json!({
                "source": d.id,
                "similarity": sim,
                "threshold": threshold,
                "accepted": sim >= threshold,
            })
        })
        .collect();

    let detected = routing::detect_domain_with(vector, &domains, |id| engine.threshold_for(id));

    let clusters = detected.as_ref().map_or_else(Vec::new, |d| {
        engine.store().centroids(&d.id).map_or_else(
            |_| Vec::new(),
            |centroids| {
                routing::nearest_clusters(vector, &centroids, engine.config().routing.clusters_probed)
                    .into_iter()
                    .map(|p| {
                        json!({
                            "cluster_id": p.cluster_id,
                            "similarity": p.similarity,
                            "row_count": p.row_count,
                        })
                    })
                    .collect()
            },
        )
    });

    finish(json!({
        "success": true,
        "op": "search",
        "dry_run": true,
        "query": query,
        "detected_source": detected.as_ref().map(|d| d.id.clone()),
        "source_scores": scores,
        "clusters_probed": clusters,
        // ! Surfaced because this path *bypasses* routing entirely. Without it an
        // operator reading this plan would conclude a query that found a
        // regulation by identifier had been routed to it.
        "exact_identifiers": identifier::extract(query)
            .into_iter()
            .map(|i| json!({ "canonical": i.canonical, "matched_on": i.matched_on }))
            .collect::<Vec<_>>(),
        "confidence": detected.as_ref().map_or(Confidence::None, |d| {
            if d.similarity >= 0.60 { Confidence::High }
            else if d.similarity >= 0.40 { Confidence::Medium }
            else { Confidence::Low }
        }),
        "progress": ["planned without searching"],
    }))
}

// ── fetch ────────────────────────────────────────────────────────────────────

/// How much of a chunk to return.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Depth {
    /// Source link + locator only · the verification bundle.
    Provenance,
    /// Bounded preview. The default: always safe to ask for.
    Snippet,
    /// Full text, size-capped, `truncated` flagged.
    Full,
}

impl Depth {
    fn parse(s: Option<&str>) -> Option<Self> {
        match s {
            None | Some("snippet") => Some(Self::Snippet),
            Some("provenance") => Some(Self::Provenance),
            Some("full") => Some(Self::Full),
            _ => None,
        }
    }
}

/// Read chunks by id at a chosen depth.
///
/// ! `read_chunk` and `get_provenance` were the same operation at two depths;
/// collapsing them removes a tool without removing a capability. `full` stays
/// size-capped — an uncapped read lets one call blow the agent's whole context
/// (`CLAUDE.md` §7 rule 11).
pub fn fetch<S: ChunkStore>(engine: &Engine<S>, ids: &[String], depth: Depth) -> Value {
    if ids.is_empty() {
        return failure("fetch", "no ids given", "pass one id or an array of ids");
    }
    let cap = engine.config().read.max_chunk_bytes;
    let snippet_chars = engine.config().search.snippet_chars;

    let found = match engine.store().chunks_by_id(ids) {
        Ok(f) => f,
        Err(e) => return failure("fetch", &e.to_string(), "check the corpus is readable"),
    };

    let items: Vec<Value> = found
        .iter()
        .map(|chunk| {
            let source = chunk.source();
            let mut item = json!({
                "id": chunk.id,
                "source": source,
                "citation": format!(
                    "{} {} — {}",
                    source.title,
                    source.locator.render(),
                    source.url
                ),
            });
            let obj = item.as_object_mut().expect("object");
            // ! Every depth carries it, because it qualifies the citation rather
            // than adding to it (`LOOPHOLES.md` §8). `null` reads as *unknown*,
            // ✗ as unchanged: a corpus ingested without hashes must not look
            // like one whose sources are all verified intact.
            obj.insert("source_hash".into(), json!(chunk.source_hash));
            match depth {
                Depth::Provenance => {}
                Depth::Snippet => {
                    obj.insert("snippet".into(), json!(chunk.snippet(snippet_chars)));
                }
                Depth::Full => {
                    let truncated = chunk.body.len() > cap;
                    // Cut on a char boundary · a byte-exact truncation would
                    // split a multibyte character and produce invalid UTF-8.
                    let body = if truncated {
                        let mut end = cap;
                        while end > 0 && !chunk.body.is_char_boundary(end) {
                            end -= 1;
                        }
                        chunk.body[..end].to_owned()
                    } else {
                        chunk.body.clone()
                    };
                    obj.insert("text".into(), json!(body));
                    obj.insert("truncated".into(), json!(truncated));
                }
            }
            item
        })
        .collect();

    // ! Reported, ✗ silently dropped. An id that resolves to nothing means a
    // stale result set, which the agent must know.
    let missing: Vec<&String> = ids
        .iter()
        .filter(|id| !found.iter().any(|c| &c.id == *id))
        .collect();

    finish(json!({
        "success": true,
        "op": "fetch",
        "depth": match depth {
            Depth::Provenance => "provenance",
            Depth::Snippet => "snippet",
            Depth::Full => "full",
        },
        "items": items,
        "missing": missing,
        "truncated": items.iter().any(|i| i["truncated"] == json!(true)),
        "progress": [format!("resolved {}/{} id(s)", found.len(), ids.len())],
    }))
}

// ── traverse ─────────────────────────────────────────────────────────────────

/// Follow a declared edge from known chunks.
///
/// ! Returns neighbour **addresses**, ✗ bodies. Traversal is navigation; the
/// agent composes it with `fetch` when it wants content. Returning full text
/// here would make every hop a context-sized payload and blur the line between
/// navigating and reading.
pub fn traverse<S: ChunkStore>(
    engine: &Engine<S>,
    ids: &[String],
    edge_name: &str,
    limit: usize,
) -> Value {
    if ids.is_empty() {
        return failure("traverse", "no ids given", "pass one id or an array of ids");
    }
    let available: Vec<&str> = Edge::all().iter().map(|e| e.name()).collect();
    let Some(edge) = Edge::parse(edge_name) else {
        // ! Named alternatives, ✗ a bare rejection. `MULTI_DOMAIN.md` §5 lists
        // edges this schema cannot yet answer (parent, cites, versions); an
        // agent asking for one must be told it does not exist rather than
        // receive an empty result it would read as "no neighbours".
        return failure(
            "traverse",
            &format!("unknown edge '{edge_name}'"),
            &format!("available edges: {}", available.join(", ")),
        );
    };

    let mut relations = Vec::with_capacity(ids.len());
    for id in ids {
        match engine.store().neighbors(id, edge, limit) {
            Err(e) => return failure("traverse", &e.to_string(), "check the corpus is readable"),
            Ok(found) => relations.push(json!({
                "from": id,
                "to": found.iter().map(|c| {
                    let source = c.source();
                    json!({
                        "id": c.id,
                        "title": source.title,
                        "locator": source.locator.render(),
                    })
                }).collect::<Vec<_>>(),
            })),
        }
    }

    let total: usize = relations
        .iter()
        .map(|r| r["to"].as_array().map_or(0, Vec::len))
        .sum();

    finish(json!({
        "success": true,
        "op": "traverse",
        "edge": edge.name(),
        "relations": relations,
        "progress": [format!("followed {edge_name} from {} id(s) → {total}", ids.len())],
    }))
}

// ── schemas ──────────────────────────────────────────────────────────────────

/// The four tool schemas advertised by `tools/list`.
#[must_use]
pub fn schemas() -> Value {
    let edges: Vec<&str> = Edge::all().iter().map(|e| e.name()).collect();
    json!([
        {
            "name": "describe",
            "description": "List sources, filterable fields, edges and limits.",
            "inputSchema": { "type": "object", "properties": {}, "required": [] },
            "annotations": { "readOnlyHint": true, "idempotentHint": true, "openWorldHint": false }
        },
        {
            "name": "search",
            "description": "Routed hybrid search. Returns ranked cited results.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    // One query or many · batching is a parameter shape.
                    "query": {
                        "anyOf": [
                            { "type": "string" },
                            { "type": "array", "items": { "type": "string" } }
                        ]
                    },
                    "k": { "type": "integer", "default": 10 },
                    "constraints": {
                        "type": "object",
                        "description": "Filters. See describe for supported fields."
                    },
                    "clusters_probed": { "type": "integer" },
                    "dry_run": {
                        "type": "boolean",
                        "default": false,
                        "description": "Return the routing plan without searching."
                    }
                },
                "required": ["query"]
            },
            "annotations": { "readOnlyHint": true, "idempotentHint": true, "openWorldHint": true }
        },
        {
            "name": "fetch",
            "description": "Read chunks by id: provenance, snippet or full text.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "ids": {
                        "anyOf": [
                            { "type": "string" },
                            { "type": "array", "items": { "type": "string" } }
                        ]
                    },
                    "depth": {
                        "type": "string",
                        "enum": ["provenance", "snippet", "full"],
                        "default": "snippet"
                    }
                },
                "required": ["ids"]
            },
            "annotations": { "readOnlyHint": true, "idempotentHint": true, "openWorldHint": false }
        },
        {
            "name": "traverse",
            "description": "Follow an edge from known chunk ids to related ones.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "ids": {
                        "anyOf": [
                            { "type": "string" },
                            { "type": "array", "items": { "type": "string" } }
                        ]
                    },
                    "edge": { "type": "string", "enum": edges },
                    "limit": { "type": "integer", "default": 20 }
                },
                "required": ["ids", "edge"]
            },
            "annotations": { "readOnlyHint": true, "idempotentHint": true, "openWorldHint": false }
        }
    ])
}

/// Parse `ids` / `query` / `depth` / `edge` from a `tools/call` argument object.
pub mod args {
    use super::{Depth, Value, as_list};

    #[must_use]
    pub fn list(args: &Value, key: &str) -> Option<Vec<String>> {
        as_list(args.get(key))
    }

    #[must_use]
    pub fn depth(args: &Value) -> Option<Depth> {
        Depth::parse(args.get("depth").and_then(Value::as_str))
    }

    #[must_use]
    pub fn usize_field(args: &Value, key: &str) -> Option<usize> {
        args.get(key)
            .and_then(Value::as_u64)
            .and_then(|n| usize::try_from(n).ok())
    }

    #[must_use]
    pub fn flag(args: &Value, key: &str) -> bool {
        args.get(key).and_then(Value::as_bool).unwrap_or(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names() -> Vec<String> {
        schemas()
            .as_array()
            .unwrap()
            .iter()
            .map(|t| t["name"].as_str().unwrap().to_owned())
            .collect()
    }

    #[test]
    fn the_surface_is_four_primitives() {
        // ! The guard on the whole design. Capability must arrive as a declared
        // parameter; a fifth tool means the model was wrong, not that the
        // feature was large (`MULTI_DOMAIN.md` §10b).
        assert_eq!(names(), ["describe", "search", "fetch", "traverse"]);
    }

    #[test]
    fn every_tool_declares_itself_read_only() {
        for tool in schemas().as_array().unwrap() {
            assert_eq!(
                tool["annotations"]["readOnlyHint"],
                json!(true),
                "{} is not marked read-only",
                tool["name"]
            );
        }
    }

    #[test]
    fn search_accepts_intent_but_never_strategy() {
        // ! The line that must not move. An agent that can set fusion weights
        // can silently disable the keyword half; an agent that can name a
        // source bypasses detection (`CLAUDE.md` §7 rule 13).
        let tools = schemas();
        let search = tools
            .as_array()
            .unwrap()
            .iter()
            .find(|t| t["name"] == "search")
            .unwrap();
        let props = search["inputSchema"]["properties"].as_object().unwrap();
        for forbidden in [
            "source", "domain", "rankers", "weights", "fusion", "dense_weight", "bm25_weight",
        ] {
            assert!(!props.contains_key(forbidden), "search must not accept `{forbidden}`");
        }
        for allowed in ["query", "k", "constraints", "dry_run"] {
            assert!(props.contains_key(allowed), "search should accept `{allowed}`");
        }
    }

    #[test]
    fn bulk_is_a_parameter_shape_not_a_tool() {
        assert!(!names().iter().any(|n| n.contains("batch")));
        let tools = schemas();
        for (tool, field) in [("search", "query"), ("fetch", "ids"), ("traverse", "ids")] {
            let t = tools
                .as_array()
                .unwrap()
                .iter()
                .find(|t| t["name"] == tool)
                .unwrap();
            assert!(
                t["inputSchema"]["properties"][field]["anyOf"].is_array(),
                "{tool}.{field} must accept one value or many"
            );
        }
    }

    #[test]
    fn explain_routing_is_a_flag_rather_than_a_fifth_tool() {
        assert!(!names().iter().any(|n| n.contains("explain")));
        let tools = schemas();
        let search = tools
            .as_array()
            .unwrap()
            .iter()
            .find(|t| t["name"] == "search")
            .unwrap();
        assert!(search["inputSchema"]["properties"]["dry_run"].is_object());
    }

    #[test]
    fn traverse_publishes_a_closed_edge_vocabulary() {
        // ! An agent must not have to guess an edge name, and an edge the schema
        // cannot answer must not appear here.
        let tools = schemas();
        let t = tools
            .as_array()
            .unwrap()
            .iter()
            .find(|t| t["name"] == "traverse")
            .unwrap();
        let listed = t["inputSchema"]["properties"]["edge"]["enum"]
            .as_array()
            .unwrap();
        assert_eq!(listed.len(), Edge::all().len());
        for absent in ["parent", "cites", "versions"] {
            assert!(
                !listed.iter().any(|e| e == absent),
                "{absent} has no schema support and must not be advertised"
            );
        }
    }

    #[test]
    fn fetch_absorbs_both_read_chunk_and_get_provenance() {
        assert!(!names().iter().any(|n| n == "read_chunk" || n == "get_provenance"));
        let tools = schemas();
        let f = tools
            .as_array()
            .unwrap()
            .iter()
            .find(|t| t["name"] == "fetch")
            .unwrap();
        let depths = f["inputSchema"]["properties"]["depth"]["enum"]
            .as_array()
            .unwrap();
        assert_eq!(depths.len(), 3, "provenance | snippet | full");
    }

    #[test]
    fn depth_defaults_to_snippet_and_rejects_nonsense() {
        assert_eq!(Depth::parse(None), Some(Depth::Snippet));
        assert_eq!(Depth::parse(Some("full")), Some(Depth::Full));
        assert_eq!(Depth::parse(Some("everything")), None);
    }

    #[test]
    fn one_value_or_many_both_parse() {
        assert_eq!(as_list(Some(&json!("a"))), Some(vec!["a".to_owned()]));
        assert_eq!(
            as_list(Some(&json!(["a", "b"]))),
            Some(vec!["a".to_owned(), "b".to_owned()])
        );
        assert_eq!(as_list(Some(&json!(42))), None);
        assert_eq!(as_list(None), None);
    }

    #[test]
    fn a_failure_carries_both_an_error_and_a_hint() {
        let f = failure("fetch", "no such chunk", "check the id");
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
