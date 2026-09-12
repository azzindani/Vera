//! The agent-facing tool surface · five read-only tools.
//!
//! ! `search_knowledge` takes a **query and nothing else**. There is no
//! `domain` parameter and the schema sets `additionalProperties: false`
//! (`CLAUDE.md` §7.13). Domain is detected inside the engine; letting an agent
//! assert one invites a confidently wrong knowledge base, which is worse than
//! an empty result because nothing in the output reveals it happened.

use serde_json::{Value, json};

/// Every tool's JSON schema, as returned by `tools/list`.
#[must_use]
pub fn definitions() -> Vec<Value> {
    vec![
        json!({
            "name": "list_domains",
            "description": "List knowledge bases this engine serves. No content.",
            "inputSchema": {
                "type": "object",
                "properties": {},
                "additionalProperties": false
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
                    }
                },
                "required": ["query"],
                // ! No `domain`. See the module note.
                "additionalProperties": false
            }
        }),
        json!({
            "name": "read_chunk",
            "description": "Read one chunk in full, size-capped.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "id": { "type": "string", "description": "Chunk id from search_knowledge." },
                    "max_chars": { "type": "integer", "description": "Cap, default 4000." }
                },
                "required": ["id"],
                "additionalProperties": false
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
            }
        }),
    ]
}

/// The uniform failure envelope.
///
/// ! `success` first, and always an actionable `hint` (`CLAUDE.md` §6). An
/// error the agent cannot act on just becomes a retry loop.
#[must_use]
pub fn error(op: &str, message: &str, hint: &str) -> Value {
    json!({
        "success": false,
        "op": op,
        "error": message,
        "hint": hint,
        "token_estimate": (message.len() + hint.len()) / 4,
        "progress": [],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn search_knowledge_accepts_a_query_and_nothing_else() {
        // ! Invariant 13, as a schema assertion. If `domain` ever appears here,
        // the agent can assert a knowledge base and the engine's own detection
        // stops being authoritative.
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
