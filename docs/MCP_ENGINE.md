# MCP_ENGINE.md

The Rust engine: what it exposes, how a request travels, and what bounds it.
Response schema in `OUTPUT_CONTRACT.md`; every dial in `CONFIGURATION.md`.

---

## 1. Responsibilities

Five things, and nothing else:

1. Embed the query, in the corpus's own vector space.
2. Gate on domain, then route to layer-2 clusters.
3. Search: sequential per-cluster dense scan, global sparse, global text, and the
   global exact-identifier path.
4. Fuse with RRF over ranks.
5. Return structured, cited evidence.

It holds no mutable state except the centroids loaded at startup. It never calls an
LLM. It never writes to the corpus. This is enforced structurally, not by convention:
`crates/embed/tests/ac_embed_provider.rs` scans every query-path crate for
completion-shaped API surfaces and fails the build if one appears.

---

## 2. Tool surface

Five read-only tools. `ROUTE → SEARCH → READ → VERIFY`.

### `list_domains`
```
list_domains() -> dict
```
Domain ids and descriptions, zero content. **Introspection only** — the agent does not
pass the result back as an argument. It exists so an agent or operator can see what
this engine covers.

### `search_knowledge`
```
search_knowledge(
    query: str,
    mode: "hybrid" | "keyword" | "semantic" = "hybrid",
    top_k: int = <TOP_K>,
    candidate_pool: int = <CANDIDATE_POOL>,
    profile: "balanced" = "balanced",
    factor_weights: dict = <fitted>,   # experimental
) -> dict
```
Arguments and the reasoning behind each are in `TOOL_SURFACE.md`. Three rules:
every option is optional and defaults to the measured value; every option
**narrows** a server limit and never widens one; and whatever was used comes back
in `applied`, so a ranking can be reproduced.

**There is no `domain` parameter, and there never will be.** The schema sets
`additionalProperties: false`: an agent that could assert a domain could assert one
that does not exist, or the wrong one, and nothing in the output would reveal it
happened. Domain is detected inside the engine.

Returns snippets, component scores, provenance and a citation block — never full
documents. Below the domain gate it returns `success: true` with an empty result set
and `detected_domain: null`, which is an answer, not an error.

### `read_chunk`
```
read_chunk(id: str, max_chars: int = <server cap>) -> dict
```
`search_knowledge` returns snippets and ids; the agent reads full text only for what
it actually needs. `max_chars` may lower the server's cap, never raise it. Sets
`truncated` when it bites.

### `get_provenance`
```
get_provenance(ids: list[str]) -> dict
```
`source_url` and locator per id — the verification bundle a human clicks through.
Capped at `MAX_PROVENANCE_IDS`. The locator is section-only on this corpus; see
`OUTPUT_CONTRACT.md` §3.

### `explain_routing`
```
explain_routing(query: str) -> dict
```
The routing decision without the search: which centroids were nearest, their scores,
any identifiers extracted, and **what the domain gate decided**. This is what makes
routing falsifiable rather than a claim.

```jsonc
{
  "detected_domain": null,        // null when the gate would refuse — the same
                                  // value search_knowledge reports for this query
  "would_refuse": true,
  "domain_gate": {
    "lexical_evidence": 0.333, "lexical_floor": 0.40,
    "centroid_similarity": 0.717, "centroid_floor": 0.45,
    "bypassed": false             // true when the query names a regulation
  },
  "cluster_scores": [ { "cluster": 41, "similarity": 0.717 } ]
}
```

! Both halves are reported **with their thresholds**, because "refused" is not
actionable on its own: a lexical failure means rephrase, a centroid failure means this
may be the wrong engine, and the caller cannot tell which from a bare `null`.

! This costs a real sparse retrieval — the gate's lexical half needs retrieved evidence
(`CONFIGURATION.md` §6). Measured on `spike-02`: **~210 ms**, against **61 ms** for a
query naming a regulation (invariant 4 bypasses the gate, so there is nothing to
retrieve) and a **691 ms** p50 for `search_knowledge`. `explain` used to skip retrieval
and report `domain: self.meta.id` unconditionally, which made the transparency tool
disagree with the tool it explains — and a transparency tool cheaper than the thing it
explains is explaining something else.

Both halves are load-bearing, and this tool now shows it per query:

| query | lexical (floor 0.40) | centroid (floor 0.45) | refused by |
|---|---|---|---|
| "what is the capital of France" | 0.565 | **0.425** | the centroid half |
| "cara memperbaiki keran air yang bocor di dapur" | **0.346** | 0.715 | the lexical half |

Neither half catches both. That was measured once when the floors were chosen; it is
now inspectable at runtime for any query.

---

## 3. Return contract

Every tool returns an object with `success` first. `serde_json` runs with
`preserve_order`; without it keys sort alphabetically and that ordering silently stops
holding.

| Field | When | Purpose |
|---|---|---|
| `success` | always | the agent checks it first |
| `op` | always | which tool ran |
| `error` | on failure | human-readable reason |
| `hint` | on failure | what to do about it |
| `progress` | always | step log: embed / route / scan / fuse |
| `token_estimate` | always | so the agent can budget its own context |
| `truncated` | bounded reads | explicit on `read_chunk` and on `search_knowledge` |

The upstream write-tool fields — snapshot, dry_run, restore — do not apply. The query
path never writes.

! `token_estimate` is **measured on all five tools** — `tools::sized` serialises the
response and applies the same `len / 4` rule as `SearchResponse::estimate_tokens`, so
the numbers are comparable across tools. It was previously a constant on
`list_domains` (60) and `explain_routing` (120), `rows × 40` on `get_provenance`, and
`body.len() / 4` on `read_chunk` — the payload with none of the envelope. A constant
defeats the field's only purpose, and errs in the direction that overruns the caller's
budget.

! `truncated` on `search_knowledge` is `scored > returned`, counted **before** the cut
to `top_k`: after the cut the number that would have said so is gone. It is the only
signal a caller has that raising `top_k` would show something new, and it was hardcoded
`false` while `TOP_K` dropped candidates silently.

---

## 4. Concurrency

```
incoming request
   │
   ▼
[ semaphore: MAX_CONCURRENCY ] ── waited > QUEUE_WAIT_MS ──▶ refuse
   │                                                          503 + Retry-After
   ▼                                                          JSON-RPC -32000
embed (network I/O)
   │
   ▼
route + sequential cluster scan   ← this is what contends on 2 cores
   │
   ▼
fuse → return
```

Two bounds, and both are necessary. The semaphore caps what executes; the wait ceiling
caps how long anything may queue. A bounded queue alone still lets a caller block
indefinitely behind a full one, so the bound has to be on **time** as well as depth.

A **third** bound covers duration: `STATEMENT_TIMEOUT_MS` (default 15,000) is applied
as a connection option, so a query that overruns is cancelled rather than holding its
permit. Without it four slow queries wedge the server at `permits_available: 0`
forever — reachable, ✗ theoretical, since a pathological text-arm query measured 48
seconds over 98.7% of the corpus (`FAILURE_MODES.md` §12).

Measured on the target profile, `MAX_CONCURRENCY=4`:

```
 4 concurrent  →  4 served,  0 refused · p50 2,089 ms
12 concurrent  →  8 served,  4 refused · p50 2,850 ms
```

Latency degrades; the server does not. At 3× its ceiling it still answers two thirds of
the burst and refuses the rest with a retry.

! A refusal is an answer. It states the request was never attempted, which is what
makes retrying safe — so it is `503` + `Retry-After`, never `500`. A client that cannot
tell the difference has to assume the worst and stop.

! None of this is observable on stdio, which reads one line, answers it, and only then
reads the next: the semaphore never has two callers to arbitrate. Every concurrency
claim here was unfalsifiable before the HTTP transport existed. The ceiling itself is
tested without a database at all — `crates/mcp/src/main.rs`, `the_ceiling_is_a_ceiling`
and `waiting_is_bounded_by_the_wait_ceiling` — because a guarantee that needs a corpus
to test is a guarantee nobody tests.

---

## 5. The OOM guarantee

```
Peak RAM = fixed cost + (MAX_CONCURRENCY × per-request ceiling)
```

Sequential cluster loading fixes the per-request ceiling at one cluster, independent of
`CLUSTERS_PROBED`. Measured on the current corpus:

```
cluster size    median 2,030 rows    max 11,448 rows
× 1024 dims × 2 bytes (halfvec)
                median   4.2 MB      worst  23.4 MB
```

At the default ceiling of 4: **8.5 MB + 4 × 23.4 MB ≈ 102 MB**. The container is given
512 MB. The limit exists for the concurrency term, not the resident one.

Full measured budget for the whole stack in `HARDWARE.md` §2.

---

## 6. Startup

The engine refuses to serve rather than degrade. In order:

1. **Resolve configuration.** A missing or unparseable variable stops the process here
   — not on the first request, by which point the orchestrator has been told it is
   healthy.
2. **Read `corpus_meta`.** The corpus declares its model, width, pooling and
   instruction. The engine builds its provider to match, rather than asserting a
   compiled-in model the corpus may not share.
3. **Run the canary.** Re-embed a stored chunk's own text through the configured
   endpoint and compare against the vector ingestion stored for it. Below
   `CANARY_MIN_COSINE`, refuse. Comparing model *names* only proves two strings match;
   this compares two vector spaces, and it is the only thing that catches an endpoint
   quietly serving different weights under the same name.
4. **Load centroids** into memory.
5. **Bind the transport.**

! The health endpoint answers only after all of it. The engine binds its port before
loading centroids, so a socket that accepts proves the process started, not that it
will answer — which is why the container healthcheck is `GET /health` and not a TCP
probe.

---

## 7. Transports

- **stdio** — for harnesses and desktop agents that launch Vera as a subprocess. Serial
  by construction. All logging goes to stderr; one stray `println!` corrupts the
  channel and the client sees a protocol error rather than a message.
- **http** — `POST /mcp` (one JSON-RPC message in, one out) and `GET /health`.

`/health` reports `permits_available`, not just `{"ok": true}`. A server at its ceiling
and a server deadlocked on a dead database both fail to answer searches; only this
distinguishes them, and it is the first thing anyone asks at 3am.

A JSON-RPC error returns HTTP **200**: the transport succeeded, and the error is in the
envelope where a compliant client looks for it. Only backpressure changes the status
code.

! No CORS, no auth, no TLS on the engine itself. Vera is infrastructure an agent calls
over a private network, not a public API — `docker-compose.yml` binds it to loopback.
Exposing it publicly means putting a reverse proxy in front that terminates TLS and
authenticates; see `OPERATIONS.md` §4.
