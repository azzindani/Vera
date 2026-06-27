# MCP_ENGINE.md — Vera query engine

The stateless Rust MCP server. Agent-facing, read-only. For the result/provenance
schema see `OUTPUT_CONTRACT.md`.

---

## 1. Responsibilities

The engine does exactly five things and nothing else:

1. Embed the query via OpenRouter (pinned provider).
2. Route: layer-1 domain anchor, layer-2 cluster centroids (both hot in memory).
3. Search: sequential per-cluster flat halfvec scan + BM25, plus the global
   exact-identifier path.
4. Fuse: RRF over all candidates.
5. Return structured, cited results.

It holds **no mutable state** except the routing centroids loaded at startup. It
**never calls an LLM**. It never writes to the corpus.

---

## 2. Tool surface (≤ 8 tools, read-only)

Read-only analog of the upstream four-tool loop: **ROUTE → SEARCH → READ → VERIFY**.

### `list_domains`
```
list_domains() -> dict
"""List available knowledge bases. Returns domain ids and descriptions."""
annotations: readOnlyHint=True, idempotentHint=True, openWorldHint=False
```
Returns ids + human descriptions + row counts. Zero content. This is **informational /
introspection only** — the agent does *not* pass the result back as a domain argument.
Domain selection happens inside `search_knowledge` via anchor detection. `list_domains`
exists so an agent (or operator) can see what Vera covers.

### `search_knowledge`  (the workhorse)
```
search_knowledge(
    query: str,
    max_results: int = 10,       # final top-k returned
    clusters_probed: int = 5,    # layer-2 dial; recall vs latency
) -> dict
"""Routed hybrid search. Returns ranked cited results with provenance."""
annotations: readOnlyHint=True, idempotentHint=True, openWorldHint=True
```
**Takes only `query`** — no `domain` argument. The domain is detected *inside the
engine* by matching the query vector against pre-embedded domain anchors (layer 1), not
chosen by the agent. This closes a real hole: an agent could pass a domain that does not
exist in the engine, or pick the wrong one. If no anchor matches above threshold, the
engine returns empty results with `detected_domain: null` and `confidence: "none"`
rather than guessing (see `OUTPUT_CONTRACT.md` §4).

`openWorldHint=True` because it calls OpenRouter. Returns **snippets + addresses +
scores + provenance**, never full documents. Bounded by `max_results`
(constrained-mode cap applies). See `OUTPUT_CONTRACT.md`.

### `read_chunk`  (surgical bounded read)
```
read_chunk(chunk_id: str) -> dict
"""Read full text of one chunk by id. Size-capped."""
annotations: readOnlyHint=True, idempotentHint=True, openWorldHint=False
```
`search_knowledge` returns snippets + ids; the agent reads full text for the few
chunks it actually needs. Hard size cap; sets `truncated` if exceeded.

### `get_provenance`  (verification)
```
get_provenance(chunk_ids: list[str]) -> dict
"""Source links + locators for chunk ids, for human verification."""
annotations: readOnlyHint=True, idempotentHint=True, openWorldHint=False
```
Returns `source_url` + `locator` (document `page` / `section` / clause) per id. This
powers the double-check output.

### `explain_routing`  (transparency / debug)
```
explain_routing(query: str) -> dict
"""Show which domain/clusters a query routes to, with scores."""
annotations: readOnlyHint=True, idempotentHint=True, openWorldHint=True
```
For tuning and trust: reveals the detected domain (anchor match) and the layer-2
centroid distances without running the full search. Takes only `query`.

---

## 3. Return value contract (read-only adaptation)

Every tool returns a `dict`, `success` first. Fields that apply to a read-only server:

| Field | When | Purpose |
|---|---|---|
| `success` | always | agent checks first |
| `op` | on success | which tool ran |
| `error` | on failure | human-readable reason |
| `hint` | on failure | actionable recovery ("widen clusters_probed", etc.) |
| `progress` | always | step log (embed / route / scan / fuse) |
| `token_estimate` | always | `len(str(response)) / 4`; agent budgets context |
| `truncated` | bounded reads | explicit on `read_chunk` / capped result sets |

Snapshot/backup/dry_run/restore fields from the upstream write-tool contract **do not
apply** — Vera's query path never writes. They apply only to the offline pipelines.
See `STANDARDS_COMPLIANCE.md` §16-17.

---

## 4. Concurrency model

The engine is the place concurrency is governed, and the goal is: **degrade
gracefully, never crash.**

```
incoming request
   │
   ▼
[ bounded queue (cap ~64) ] ──full──► reject 503 "busy, retry"
   │
   │  wait > T seconds ───────────────► reject 503 (bound tail latency)
   ▼
[ semaphore: max ~4 concurrent ]   ← the real ceiling on 2 cores
   │
   ▼
embed (OpenRouter, overlaps freely — network I/O)
   │
   ▼
route + sequential cluster scan (CPU+disk — this is what contends on 2 cores)
   │
   ▼
fuse → return
```

- **Semaphore (~4):** the hard concurrency cap. On 2 cores, ~4 concurrent is where
  latency stays ~1–1.3 s; beyond that the cores, not RAM, are the wall. The system is
  **CPU-bound long before it is memory-bound** — the safe way to be bounded.
- **Bounded queue (~64) + 503:** an *unbounded* queue is itself an OOM vector (waiters
  accumulate). Cap it; return honest backpressure. The queue holds tiny request
  descriptors, not working sets.
- **Wait-timeout:** reject requests that have waited too long, so a full queue does not
  produce 20-second tail latencies.
- **Embedding throttle:** the semaphore should also respect the OpenRouter rate limit;
  batch decomposed sub-queries into one embedding request.

All four limits are config values, not magic numbers. Bigger hardware raises them.

---

## 5. The OOM guarantee (by construction)

```
Peak RAM = fixed_costs + (concurrency_limit × per_request_ceiling)
```

Both terms are bounded, so peak RAM is bounded. The load-bearing fact is **sequential
cluster loading**: a request loads one cluster, scans it, drops it, loads the next —
so `per_request_ceiling ≈ one cluster (~82 MB at 4096 halfvec) + ~30 MB buffers`,
**independent of `clusters_probed`**.

Worked budget on 2 vCPU / 8 GB (full 4096, halfvec):

| Component | RAM |
|---|---|
| OS + container runtime | ~0.8 GB |
| Rust engine (stateless) | ~0.2 GB |
| Layer-1/2 centroids (hot, mlock'd) | ~0.08 GB |
| Postgres (shared_buffers + capped conns) | ~2.0 GB |
| In-flight: 4 concurrent × ~120 MB | ~0.5 GB |
| **Peak** | **~3.6 GB** |
| **Free margin** | **~4 GB** |

OOM cannot happen because the worst case is fully accounted for. Full hardware detail
in `HARDWARE.md`.

---

## 6. OpenRouter provider client (critical subsystem)

Query embedding is Vera's only external dependency, so the provider client is a
first-class component, not a helper. It must:

1. **Pin one provider** (OpenRouter Exacto mode) — never route across providers, which
   would vary the embedding space. See `EMBEDDING.md`.
2. **Pin the model version** and store it in DB metadata; refuse to serve on mismatch.
3. **Canary check at startup:** embed a known string, compare to the stored reference
   vector; if cosine < threshold (e.g. 0.999), refuse to serve and alert. This catches
   silent provider/model drift before it corrupts results.
4. **Retry with backoff** on 429/529; **queue** through transient outages.
5. **Fail over only to a pre-validated secondary provider** (cosine-matched offline);
   never to an unvalidated host.
6. **Batch** decomposed sub-queries into a single request.
7. **Truncate identically** to the corpus dimension if any truncation is used (for
   full 4096, none — but the truncation policy lives here and must match the corpus).

---

## 7. Transports

Two modes (per upstream §30), but reframed for agentic use:

- **stdio** — for harnesses / desktop agents that launch Vera as a subprocess.
- **streamable-http** — for Claude.ai connectors and remote agents. Auth token
  generated at deploy, never hardcoded.

The OpenAI-compatible HTTP shape may also be exposed by a thin adapter for non-MCP
callers (e.g. local tooling), wrapping the *same* retrieval core with a `mode`
(chunks vs. ready-to-prompt context). The core is written once; adapters are thin.
