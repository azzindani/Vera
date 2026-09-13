//! JSON-RPC framing for MCP · the half of the server that needs no corpus.
//!
//! ! Extracted so it can be tested without a database or an embedder. Protocol
//! correctness is what every client sees first — a wrong `id`, an answered
//! notification or a missing envelope field breaks the session before a single
//! query runs — and a guarantee that needs a GPU to test is a guarantee nobody
//! tests.
//!
//! Everything here is a pure function of the request. The one thing that is
//! *not* here is tool execution, which needs the pipeline.

use serde_json::{Value, json};

/// MCP protocol revision this server speaks.
pub(crate) const PROTOCOL_VERSION: &str = "2024-11-05";

/// JSON-RPC: the method does not exist.
pub(crate) const METHOD_NOT_FOUND: i64 = -32601;

/// Vera-specific: the concurrency ceiling is full and nothing was attempted.
/// The HTTP transport maps this to 503 + `Retry-After`.
pub(crate) const BUSY_CODE: i64 = -32000;

/// What a request asks for.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Route {
    Initialize,
    ToolsList,
    /// Tool execution · carries `params`, which the caller passes to dispatch.
    ToolsCall(Value),
    /// A notification: carries no id and **must not be answered**.
    Notification,
    Unknown(String),
}

/// Classify one request.
///
/// ! A missing or non-string `method` routes to `Unknown("")` rather than
/// panicking. Anything may arrive on this socket.
pub(crate) fn route(req: &Value) -> Route {
    let method = req.get("method").and_then(Value::as_str).unwrap_or("");
    match method {
        "initialize" => Route::Initialize,
        "tools/list" => Route::ToolsList,
        "tools/call" => Route::ToolsCall(req.get("params").cloned().unwrap_or_else(|| json!({}))),
        m if m.starts_with("notifications/") => Route::Notification,
        other => Route::Unknown(other.to_owned()),
    }
}

/// The `initialize` result.
pub(crate) fn initialize_result() -> Value {
    json!({
        "protocolVersion": PROTOCOL_VERSION,
        "serverInfo": { "name": "vera", "version": env!("CARGO_PKG_VERSION") },
        "capabilities": { "tools": {} }
    })
}

/// A successful envelope.
///
/// ! `id` is echoed **exactly as it arrived**, including `null` and including a
/// string. A client correlating responses by id cannot match a reply that
/// changed its type, and JSON-RPC permits both forms.
pub(crate) fn ok(id: Option<&Value>, result: &Value) -> Value {
    json!({ "jsonrpc": "2.0", "id": id, "result": result })
}

/// An error envelope.
pub(crate) fn error(id: Option<&Value>, code: i64, message: &str, data: Option<Value>) -> Value {
    let mut err = json!({ "code": code, "message": message });
    if let Some(d) = data {
        err["data"] = d;
    }
    json!({ "jsonrpc": "2.0", "id": id, "error": err })
}

/// Method not found.
pub(crate) fn method_not_found(id: Option<&Value>, method: &str) -> Value {
    error(
        id,
        METHOD_NOT_FOUND,
        &format!("method not found: {method}"),
        None,
    )
}

/// At capacity.
///
/// ! Backpressure is an answer, ✗ a failure. It states that the request was
/// never attempted, which is what makes a retry safe — a generic 500 does not
/// carry that, and a client that cannot tell the difference must assume the
/// worst and stop.
pub(crate) fn busy(id: Option<&Value>, waited: Option<std::time::Duration>) -> Value {
    let ms = waited.map_or(0, |d| d.as_millis());
    error(
        id,
        BUSY_CODE,
        &format!("at capacity · waited {ms}ms for a slot · retry"),
        Some(json!({ "retry": true })),
    )
}

/// Wrap a tool's JSON result in the MCP content envelope.
pub(crate) fn tool_content(out: &Value) -> Value {
    json!({ "content": [{ "type": "text", "text": out.to_string() }] })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req(method: &str, id: &Value) -> Value {
        json!({ "jsonrpc": "2.0", "id": id, "method": method })
    }

    #[test]
    fn the_four_methods_route_to_themselves() {
        assert_eq!(route(&req("initialize", &json!(1))), Route::Initialize);
        assert_eq!(route(&req("tools/list", &json!(1))), Route::ToolsList);
        assert!(matches!(
            route(&req("tools/call", &json!(1))),
            Route::ToolsCall(_)
        ));
    }

    #[test]
    fn every_notification_is_a_notification_not_just_the_known_ones() {
        // ! Prefix match, ✗ an allowlist. A client may send a notification this
        // server has never heard of, and answering it corrupts the stream just
        // as badly as answering a known one.
        for m in [
            "notifications/initialized",
            "notifications/cancelled",
            "notifications/something/nobody/has/written/yet",
        ] {
            assert_eq!(route(&req(m, &Value::Null)), Route::Notification, "{m}");
        }
    }

    #[test]
    fn an_unknown_method_is_named_in_the_route() {
        assert_eq!(
            route(&req("resources/list", &json!(1))),
            Route::Unknown("resources/list".into())
        );
    }

    #[test]
    fn a_request_without_a_method_does_not_panic() {
        assert_eq!(route(&json!({ "id": 1 })), Route::Unknown(String::new()));
        assert_eq!(route(&json!({})), Route::Unknown(String::new()));
        assert_eq!(
            route(&json!({ "method": 42 })),
            Route::Unknown(String::new())
        );
    }

    #[test]
    fn tools_call_without_params_yields_an_empty_object_not_a_panic() {
        let Route::ToolsCall(p) = route(&req("tools/call", &json!(1))) else {
            panic!("expected ToolsCall");
        };
        assert_eq!(p, json!({}));
    }

    #[test]
    fn an_integer_id_comes_back_as_an_integer() {
        let r = ok(Some(&json!(7)), &json!({}));
        assert_eq!(r["id"], json!(7));
    }

    #[test]
    fn a_string_id_comes_back_as_a_string() {
        // JSON-RPC permits either. A client correlating by id cannot match a
        // reply whose id changed type.
        let r = ok(Some(&json!("abc-123")), &json!({}));
        assert_eq!(r["id"], json!("abc-123"));
    }

    #[test]
    fn a_null_id_is_preserved_rather_than_dropped() {
        let r = ok(Some(&Value::Null), &json!({}));
        assert!(r.get("id").is_some(), "id key must be present");
        assert_eq!(r["id"], Value::Null);
    }

    #[test]
    fn every_envelope_declares_jsonrpc_two() {
        let envelopes = [
            ok(Some(&json!(1)), &json!({})),
            method_not_found(Some(&json!(1)), "nope"),
            busy(Some(&json!(1)), Some(std::time::Duration::from_secs(2))),
        ];
        for e in envelopes {
            assert_eq!(e["jsonrpc"], json!("2.0"), "{e}");
        }
    }

    #[test]
    fn a_response_is_either_a_result_or_an_error_never_both() {
        // JSON-RPC forbids both keys on one response, and a client that sees
        // both has no defined behaviour.
        let good = ok(Some(&json!(1)), &json!({"x": 1}));
        assert!(good.get("result").is_some() && good.get("error").is_none());

        let bad = method_not_found(Some(&json!(1)), "nope");
        assert!(bad.get("error").is_some() && bad.get("result").is_none());
    }

    #[test]
    fn method_not_found_names_the_method_the_caller_used() {
        let e = method_not_found(Some(&json!(1)), "resources/list");
        assert_eq!(e["error"]["code"], json!(METHOD_NOT_FOUND));
        assert!(
            e["error"]["message"]
                .as_str()
                .unwrap_or_default()
                .contains("resources/list")
        );
    }

    #[test]
    fn busy_says_it_was_never_attempted_and_is_retryable() {
        let b = busy(Some(&json!(1)), Some(std::time::Duration::from_secs(2)));
        assert_eq!(b["error"]["code"], json!(BUSY_CODE));
        assert_eq!(b["error"]["data"]["retry"], json!(true));
        // The wait is stated so an operator can tell a full queue from a slow
        // one without reading the server's configuration.
        assert!(
            b["error"]["message"]
                .as_str()
                .unwrap_or_default()
                .contains("2000ms")
        );
    }

    #[test]
    fn busy_without_a_wait_reports_zero_rather_than_omitting_it() {
        let b = busy(Some(&json!(1)), None);
        assert!(
            b["error"]["message"]
                .as_str()
                .unwrap_or_default()
                .contains("0ms")
        );
    }

    #[test]
    fn initialize_announces_the_pinned_protocol_version() {
        let r = initialize_result();
        assert_eq!(r["protocolVersion"], json!(PROTOCOL_VERSION));
        assert_eq!(r["serverInfo"]["name"], json!("vera"));
        // Tools are the only capability. Declaring resources or prompts we do
        // not serve makes a client call something that will fail.
        assert!(r["capabilities"].get("tools").is_some());
        assert!(r["capabilities"].get("resources").is_none());
        assert!(r["capabilities"].get("prompts").is_none());
    }

    #[test]
    fn initialize_reports_the_crate_version_not_a_literal() {
        assert_eq!(
            initialize_result()["serverInfo"]["version"],
            json!(env!("CARGO_PKG_VERSION"))
        );
    }

    #[test]
    fn a_tool_result_is_wrapped_as_mcp_text_content() {
        let c = tool_content(&json!({ "success": true }));
        assert_eq!(c["content"][0]["type"], json!("text"));
        // The payload is a JSON *string*, which is what MCP text content is.
        let text = c["content"][0]["text"].as_str().expect("text");
        let parsed: Value = serde_json::from_str(text).expect("round-trips");
        assert_eq!(parsed["success"], json!(true));
    }

    #[test]
    fn every_envelope_serialises_to_one_line() {
        // ! The stdio transport is newline-framed. An envelope containing a raw
        // newline would split into two frames and desynchronise the stream for
        // the rest of the session.
        let envelopes = [
            ok(Some(&json!(1)), &initialize_result()),
            tool_content(&json!({ "hint": "line one\nline two" })),
            method_not_found(Some(&json!(1)), "a\nb"),
        ];
        for e in envelopes {
            let s = serde_json::to_string(&e).expect("serialize");
            assert!(!s.contains('\n'), "envelope must be one line: {s}");
        }
    }
}
