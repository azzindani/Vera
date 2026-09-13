//! The agent-facing tool surface · five read-only tools.
//!
//! ! `search_knowledge` takes a **query and nothing else**. There is no
//! `domain` parameter and the schema sets `additionalProperties: false`
//! (`CLAUDE.md` §7.13). Domain is detected inside the engine; letting an agent
//! assert one invites a confidently wrong knowledge base, which is worse than
//! an empty result because nothing in the output reveals it happened.
//!
//! ! Every tool is annotated `readOnlyHint` and `openWorldHint: false`. Nothing
//! here mutates the corpus, and nothing reaches outside the deployment — the
//! embedding endpoint is part of it, ✗ a third-party API.

use serde_json::{Value, json};

/// Every tool's JSON schema, as returned by `tools/list`.
///
/// One long literal by design: the whole agent-facing surface is visible in one
/// screen, and splitting it per tool would trade that for five helpers that are
/// each read once.
#[must_use]
#[allow(clippy::too_many_lines)]
pub fn definitions() -> Vec<Value> {
    vec![
        json!({
            "name": "list_domains",
            "description": "List knowledge bases this engine serves. No content.",
            "inputSchema": {
                "type": "object",
                "properties": {},
                "additionalProperties": false
            },
            "annotations": {
                "title": "List domains",
                "readOnlyHint": true,
                "destructiveHint": false,
                "idempotentHint": true,
                "openWorldHint": false
            }
        }),
        json!({
            "name": "search_knowledge",
            "description": "Routed hybrid search. Returns ranked, cited evidence.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural-language question or regulation reference."
                    },
                    // ! Every option below NARROWS. A value above the server's
                    // ceiling is clamped and the clamp is reported in
                    // `applied.clamped` (`docs/TOOL_SURFACE.md` §2).
                    "mode": {
                        "type": "string",
                        "enum": ["hybrid", "keyword", "semantic"],
                        "description": "Which arms run. hybrid (default) fuses all three; keyword is sparse+text; semantic is the dense arm alone and scores 0.0% on this corpus."
                    },
                    "top_k": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Results returned. The server's ceiling still applies."
                    },
                    "candidate_pool": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Candidates ranked before the cut. Wider finds more and costs more; the server's ceiling still applies."
                    },
                    "profile": {
                        "type": "string",
                        // ! Only fitted profiles are listed. An unfitted profile
                        // is a raw weight vector with a friendly name.
                        "enum": ["balanced"],
                        "description": "Fitted ranking intent. Pick intent, not numbers."
                    },
                    "factor_weights": {
                        "type": "object",
                        "description": "EXPERIMENTAL. Raw factor weights, unfitted. Prefer `profile`. The effective weights are echoed in `applied`.",
                        "properties": {
                            "relevance_floor": { "type": "number" },
                            "authority": { "type": "number" },
                            "structural": { "type": "number" },
                            "temporal": { "type": "number" },
                            "completeness": { "type": "number" },
                            "topical": { "type": "number" }
                        },
                        "required": [
                            "relevance_floor", "authority", "structural",
                            "temporal", "completeness", "topical"
                        ],
                        "additionalProperties": false
                    }
                },
                "required": ["query"],
                // ! Still no `domain`. See the module note.
                "additionalProperties": false
            },
            "annotations": {
                "title": "Search knowledge",
                "readOnlyHint": true,
                "destructiveHint": false,
                "idempotentHint": true,
                "openWorldHint": false
            }
        }),
        json!({
            "name": "read_chunk",
            "description": "Read one chunk in full, size-capped.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "id": { "type": "string", "description": "Chunk id from search_knowledge." },
                    "max_chars": {
                        "type": "integer",
                        "description": "Cap on characters. The server's own cap still applies."
                    }
                },
                "required": ["id"],
                "additionalProperties": false
            },
            "annotations": {
                "title": "Read chunk",
                "readOnlyHint": true,
                "destructiveHint": false,
                "idempotentHint": true,
                "openWorldHint": false
            }
        }),
        json!({
            "name": "get_provenance",
            "description": "Verification bundle: source and locator for result ids.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "ids": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Chunk ids to resolve."
                    }
                },
                "required": ["ids"],
                "additionalProperties": false
            },
            "annotations": {
                "title": "Get provenance",
                "readOnlyHint": true,
                "destructiveHint": false,
                "idempotentHint": true,
                "openWorldHint": false
            }
        }),
        json!({
            "name": "explain_routing",
            "description": "Which clusters a query probes, and why.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": { "type": "string" }
                },
                "required": ["query"],
                "additionalProperties": false
            },
            "annotations": {
                "title": "Explain routing",
                "readOnlyHint": true,
                "destructiveHint": false,
                "idempotentHint": true,
                "openWorldHint": false
            }
        }),
    ]
}

/// The uniform failure envelope.
///
/// Fill in `token_estimate` by measuring the response, ✗ by guessing it.
///
/// ! Every tool but `search_knowledge` used to carry a constant here — 60 for
/// `list_domains`, 120 for `explain_routing`, `rows * 40` for
/// `get_provenance`, and `body.len() / 4` for `read_chunk`, which counted the
/// body and none of the envelope around it. An agent budgets its own context
/// with this number; a constant is wrong by whatever the response actually is,
/// and wrong in the direction that overruns.
///
/// ! Same rule as `SearchResponse::estimate_tokens` — `len(json) / 4` — so the
/// five tools are comparable to each other. Measuring means serialising twice:
/// once to size it, once to send it. The responses are small and the
/// alternative is a number nobody can trust.
#[must_use]
pub fn sized(mut out: Value) -> Value {
    let n = serde_json::to_string(&out).map_or(0, |s| s.len() / 4);
    if let Some(obj) = out.as_object_mut() {
        obj.insert("token_estimate".into(), json!(n));
    }
    out
}

/// ! `success` first, and always an actionable `hint` (`CLAUDE.md` §6). An
/// error the agent cannot act on just becomes a retry loop.
#[must_use]
pub fn error(op: &str, message: &str, hint: &str) -> Value {
    sized(json!({
        "success": false,
        "op": op,
        "error": message,
        "hint": hint,
        "token_estimate": 0,
        "progress": [],
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sized_measures_the_response_rather_than_guessing_it() {
        let small = sized(json!({ "op": "a", "token_estimate": 0 }));
        let big = sized(json!({
            "op": "a",
            "token_estimate": 0,
            "body": "x".repeat(4_000),
        }));
        let n = |v: &Value| v["token_estimate"].as_u64().unwrap();
        assert!(n(&small) > 0, "an unset estimate is a broken budget");
        assert!(
            n(&big) > n(&small) + 900,
            "4,000 more characters must move the estimate: {} vs {}",
            n(&big),
            n(&small)
        );
    }

    #[test]
    fn sized_counts_the_envelope_and_not_just_the_payload() {
        // ! `read_chunk` used to report `body.len() / 4`, which is the payload
        // with none of the JSON around it. The agent budgets context with this.
        let body = "x".repeat(400);
        let v = sized(json!({
            "op": "read_chunk",
            "token_estimate": 0,
            "source": { "title": "a fairly long source title here", "url": null },
            "body": body.clone(),
        }));
        assert!(
            usize::try_from(v["token_estimate"].as_u64().unwrap()).unwrap() > body.len() / 4,
            "the envelope is not free"
        );
    }

    #[test]
    fn an_error_carries_an_estimate_too() {
        let e = error("search_knowledge", "boom", "try again");
        assert!(e["token_estimate"].as_u64().unwrap() > 0);
        assert_eq!(e["success"], json!(false));
    }

    fn tool(name: &str) -> Value {
        definitions()
            .into_iter()
            .find(|t| t["name"] == name)
            .unwrap_or_else(|| panic!("{name} missing"))
    }

    #[test]
    fn the_surface_stays_within_eight_tools() {
        // CLAUDE.md §6 caps the surface. More tools means more ways for an
        // agent to pick the wrong one.
        assert!(definitions().len() <= 8, "{}", definitions().len());
    }

    #[test]
    fn search_knowledge_never_accepts_a_domain() {
        // ! Invariant 13, as a schema assertion. The tool now takes several
        // options (`docs/TOOL_SURFACE.md`), and this is the one that must never
        // join them: an agent that can assert a knowledge base makes the
        // engine's own detection non-authoritative, and nothing in the output
        // would reveal it happened.
        let s = &tool("search_knowledge")["inputSchema"];
        assert_eq!(s["additionalProperties"], json!(false));
        assert!(
            s["properties"].get("domain").is_none(),
            "domain must not be accepted"
        );
        assert_eq!(s["required"], json!(["query"]));
        assert!(s["properties"].get("query").is_some());
    }

    #[test]
    fn every_search_option_is_optional() {
        // A caller passing only `query` must get the measured defaults, so
        // nothing but `query` may ever become required.
        let s = &tool("search_knowledge")["inputSchema"];
        assert_eq!(s["required"], json!(["query"]));
        for opt in [
            "mode",
            "top_k",
            "candidate_pool",
            "profile",
            "factor_weights",
        ] {
            assert!(s["properties"].get(opt).is_some(), "{opt} missing");
        }
    }

    #[test]
    fn only_fitted_profiles_are_offered() {
        // ! An unfitted profile is a raw weight vector with a friendly name.
        // Listing one invites an agent to select an intent nothing measured.
        let s = &tool("search_knowledge")["inputSchema"];
        assert_eq!(s["properties"]["profile"]["enum"], json!(["balanced"]));
    }

    #[test]
    fn the_non_contributing_mode_is_labelled_in_its_own_description() {
        // Dense scores 0.0% on this corpus. An agent reading only the schema
        // must still learn that before choosing it.
        let s = &tool("search_knowledge")["inputSchema"];
        let d = s["properties"]["mode"]["description"]
            .as_str()
            .unwrap_or("");
        assert!(d.contains("0.0%"), "semantic must declare itself: {d}");
    }

    #[test]
    fn raw_weights_are_marked_experimental_in_the_schema() {
        let s = &tool("search_knowledge")["inputSchema"];
        let d = s["properties"]["factor_weights"]["description"]
            .as_str()
            .unwrap_or("");
        assert!(d.contains("EXPERIMENTAL"), "{d}");
    }

    #[test]
    fn every_tool_forbids_unknown_properties() {
        for t in definitions() {
            assert_eq!(
                t["inputSchema"]["additionalProperties"],
                json!(false),
                "{} accepts unknown properties",
                t["name"]
            );
        }
    }

    #[test]
    fn every_tool_has_a_short_description() {
        for t in definitions() {
            let d = t["description"].as_str().expect("description");
            assert!(!d.is_empty());
            assert!(
                d.len() <= 80,
                "{} description is {} chars",
                t["name"],
                d.len()
            );
        }
    }

    #[test]
    fn the_error_envelope_leads_with_success_and_carries_a_hint() {
        let e = error("search_knowledge", "boom", "try a shorter query");
        assert_eq!(e["success"], json!(false));
        assert_eq!(e["op"], json!("search_knowledge"));
        assert!(e["hint"].as_str().unwrap().contains("shorter"));
        // The key order matters for readability; serde_json preserves it.
        let s = serde_json::to_string(&e).unwrap();
        assert!(s.starts_with(r#"{"success":false"#), "{s}");
    }
}
