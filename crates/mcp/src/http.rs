//! HTTP transport · the same tool surface, reachable over a socket.
//!
//! ! This exists for invariant 6 as much as for deployment. Backpressure,
//! bounded queues and wait-timeouts cannot be *tested* on stdio: it reads one
//! line, answers it, and only then reads the next, so the semaphore that
//! guarantees bounded RAM never has two callers to arbitrate between. Every
//! concurrency claim in `CLAUDE.md` §5.4 was unfalsifiable until this module.
//!
//! Two endpoints, deliberately:
//!
//!   POST /mcp      one JSON-RPC message in, one out
//!   GET  /health   liveness plus the numbers an operator needs at 3am
//!
//! ! No CORS, no auth, no TLS. Vera is infrastructure an agent calls over a
//! private network, ✗ a public API. Adding a browser-facing surface here would
//! invite exactly the deployment this design does not defend.

use std::sync::Arc;

use axum::{
    Json, Router,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
};
use serde_json::{Value, json};

use crate::{BUSY_CODE, Server};

/// Serve until the process is killed.
///
/// # Errors
/// The address is already bound, or the listener dies.
pub(crate) async fn serve(
    server: Arc<Server>,
    addr: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let app = Router::new()
        .route("/mcp", post(rpc))
        .route("/health", get(health))
        .with_state(server);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

/// Liveness, and enough state to tell "busy" from "wedged".
///
/// ! Reports free permits, ✗ just `{"ok": true}`. A server at its ceiling and a
/// server deadlocked on a dead database both fail to answer searches; only this
/// distinguishes them, and it is the first thing anyone asks.
async fn health(State(s): State<Arc<Server>>) -> Response {
    Json(json!({
        "success": true,
        "clusters": s.pipe.cluster_count(),
        "permits_available": s.permits.available_permits(),
        "version": env!("CARGO_PKG_VERSION"),
    }))
    .into_response()
}

/// One JSON-RPC message in, one out.
async fn rpc(State(s): State<Arc<Server>>, Json(req): Json<Value>) -> Response {
    let Some(reply) = s.handle(&req).await else {
        // A notification. There is no body to send and no id to answer to.
        return StatusCode::ACCEPTED.into_response();
    };

    let busy = reply
        .get("error")
        .and_then(|e| e.get("code"))
        .and_then(Value::as_i64)
        == Some(BUSY_CODE);

    if busy {
        // ! 503 + Retry-After, ✗ 500. The distinction is the whole point of
        // backpressure: this request was never attempted, so a client may retry
        // it without risking a double effect. A 500 forbids that.
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            [("retry-after", "1")],
            Json(reply),
        )
            .into_response();
    }

    // ! 200 even for a JSON-RPC error. The transport succeeded; the error is in
    // the envelope, where a JSON-RPC client looks for it. Mapping method-not-
    // found onto 404 would hide it from every compliant client.
    (StatusCode::OK, Json(reply)).into_response()
}
