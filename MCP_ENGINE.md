# MCP_ENGINE.md — Vera query engine

The stateless Rust MCP server. Agent-facing, read-only. For the result/provenance
schema see `OUTPUT_CONTRACT.md`; for why the surface is shaped this way see
`MULTI_DOMAIN.md` §10b.

---

## 1. Responsibilities

The engine does exactly five things and nothing else:

1. Embed the query via OpenRouter (pinned provider).
2. **Resolve exact identifiers globally**, before any routing runs.
3. Route: layer-1 source anchor, layer-2 cluster centroids (both hot in memory).
4. Search: sequential per-cluster flat vector scan, plus **one global** BM25 query.
5. Fuse with RRF and return structured, cited results.

It holds **no mutable state** except the routing centroids loaded at startup. It
**never calls an LLM**. It never writes to the corpus. It never plans a sequence of
searches — that is the agent's job (§2).

Two ordering facts are load-bearing and easy to get wrong:

- **Step 2 precedes step 3.** An exact identifier must never be gated by routing, and
  layer-1 domain detection *is* routing. A query naming `UU 28/2007` whose vector falls
  below the anchor threshold must still return that regulation. (`LOOPHOLES.md` §1,
  `CLAUDE.md` §7 rule 4.)
- **BM25 runs once, globally — not once per probed cluster.** Routing exists because
  dense vectors cannot be indexed; an inverted index has no such problem and needs no
  routing. Per-cluster keyword search cost N full-corpus scans and *lost* recall.
  Measured, it was 87% of query time. See `ARCHITECTURE.md` §5, which still describes
  the superseded scheme.

---

## 2. Tool surface — **four primitives**

Read-only analog of the upstream four-tool loop:
**DESCRIBE → SEARCH → TRAVERSE → FETCH.**

### `describe`
```
describe() -> dict
"""List sources, filterable fields, edges and limits."""
annotations: readOnlyHint=True, idempotentHint=True, openWorldHint=False
```
Returns, per source: id, domain, description, row count, embedding space, the fields it
can be **filtered** on, the edges it can be **traversed** along, and its validity model.
Plus engine limits (`max_results`, `max_chunk_bytes`, default `clusters_probed`).

Zero content. This is **introspection**, and it is the tool that makes the rest safe:
constraints and edges are closed vocabularies, and `describe` is how an agent learns
them instead of guessing. The agent does **not** feed a source back as an argument —
`search` detects it.

Where a capability does not exist, `describe` says so honestly. `filterable_fields` is
`[]` today because structured constraints need per-source metadata and the selectivity
statistics the cardinality rule depends on. Publishing field names the engine would
then ignore is the exact failure this surface exists to prevent.

### `search` (the workhorse)
```
search(
    query: str | list[str],       # one query, or many in one call
    k: int = 10,                  # final top-k per query
    constraints: dict = {},       # filters; see describe for supported fields
    clusters_probed: int = 5,     # layer-2 dial; recall vs latency
    dry_run: bool = False,        # return the routing plan without searching
) -> dict
"""Routed hybrid search. Returns ranked cited results."""
annotations: readOnlyHint=True, idempotentHint=True, openWorldHint=True
```

**Takes no `domain` or `source`.** Detection happens inside the engine by anchor match
against pre-embedded source anchors. This closes a real hole: an agent could name a
source that does not exist, or the wrong one. If no anchor clears its threshold the
engine returns empty results with `detected_domain: null` and `confidence: "none"`
rather than guessing — *except* for exact-identifier hits, which bypass routing and are
returned regardless (§1).

The threshold is **calibrated per source at build time**, not configured. How close a
query lands to an anchor depends on the embedding model's anisotropy, which is not
knowable in advance; a fixed value that is too tight returns `success: true` with zero
results for every query, which is indistinguishable from an empty corpus. See
`EMBEDDING.md` §4.

**Bulk is a parameter shape, not a separate tool.** `query` accepts a string or a list.
Batching matters because the embedding provider is a network round-trip (§6.6) and one
admission slot then covers the whole set — but that does not justify a `search_batch`
verb. A single query returns the bare `OUTPUT_CONTRACT.md` §2 response; a list returns
`{ "searches": [...] }`. Batching must not change the shape of the common case.

**`dry_run` replaces a separate `explain_routing` tool.** It is the same planning work
with execution suppressed, returning the detected source, per-source anchor scores and
thresholds, the clusters that would be probed with their distances, and any exact
identifiers extracted from the query text. A separate tool would have to duplicate the
entire constraint surface, and the two would drift.

**An unrecognised constraint is an error, never a no-op.** Silently dropping a filter
returns a confident answer to a question that was not asked — the agent believes the
results are restricted and they are not. The error names the valid fields. Constraints
are validated *before* embedding, so a rejected request costs no network call.

Returns **snippets + addresses + scores + provenance**, never full documents.

### `fetch`
```
fetch(
    ids: str | list[str],
    depth: "provenance" | "snippet" | "full" = "snippet",
) -> dict
"""Read chunks by id: provenance, snippet or full text."""
annotations: readOnlyHint=True, idempotentHint=True, openWorldHint=False
```

| `depth` | Returns | Use |
|---|---|---|
| `provenance` | source url + locator + rendered citation | verification; cheapest |
| `snippet` | the above + a bounded preview | the default, always safe |
| `full` | the above + full text, size-capped | the few chunks that actually matter |

One operation at three depths. Reading a chunk and verifying a chunk were never
different jobs — they differ only in how much comes back, so they are one tool.

`full` is hard-capped by `max_chunk_bytes` and sets `truncated`. An uncapped read lets
one call blow the agent's whole context (`CLAUDE.md` §7 rule 11). Ids that resolve to
nothing are reported in `missing`, never silently dropped: an unresolvable id means a
stale result set, which the agent must know.

### `traverse`
```
traverse(
    ids: str | list[str],
    edge: str,                    # see describe for the available edges
    limit: int = 20,
) -> dict
"""Follow an edge from known chunk ids to related ones."""
annotations: readOnlyHint=True, idempotentHint=True, openWorldHint=False
```

Navigation from a known point, along a **declared** edge. This is the single tool that
covers hierarchy, citation graphs and version history as those land — each is an edge
name, not a new verb.

Returns neighbour **addresses** (id, title, locator), not bodies. Traversal is
navigation; the agent composes it with `fetch` when it wants content. Returning full
text here would make every hop a context-sized payload and blur the line between
navigating and reading. It also does not rank: ranking is `search`'s job, and a
traversal that quietly reordered evidence the agent addressed directly would be
misleading.

The edge vocabulary is **closed and published by `describe`**. Currently
`same_document` and `same_identifier`. The `parent`/`children`, `cites`/`cited_by` and
`versions`/`supersedes` edges that `MULTI_DOMAIN.md` §5 calls for have no schema support
yet and are **deliberately absent rather than stubbed** — an edge that exists and
returns nothing is indistinguishable from a document with no neighbours. Asking for an
unknown edge is an error naming the available ones.

### Why exactly four

**Capability grows through declared parameters, not new tools.** Adding a source, an
edge type, a filterable field or a validity model must add **zero** tools. A fifth verb
is evidence that something is missing from the *model*, not from the tool list. This is
enforced by a test on the schema, because the tool surface is where that failure shows
up first.

**The agent passes intent; the engine owns strategy.**

| Agent may pass | Engine owns, always |
|---|---|
| query text | which source — detected, never named by the agent |
| constraints (filter / as-of / subtree) | which rankers run and their fusion weights |
| `k`, `depth`, `edge`, `limit` | which clusters are probed |
| `dry_run` | the fusion method |

Constraints are safe to accept because they are *intent* ("only Supreme Court, in force
in 2019"). Ranker selection and weighting are *strategy*: an agent that could set fusion
weights could silently disable the keyword half, which is the domain-naming hole one
level down.

**Multi-step search is these four composed by the caller.** `describe` → `search` →
`traverse` → `fetch`. The engine needs no planner because the primitives compose, and
gains no planner because composition happens on the agent's side — which is the whole
reason the engine can stay stateless and model-free.

---

## 3. Return value contract (read-only adaptation)

Every tool returns a `dict`, `success` first. Fields that apply to a read-only server:

| Field | When | Purpose |
|---|---|---|
| `success` | always | agent checks first |
| `op` | on success | which tool ran |
| `error` | on failure | human-readable reason |
| `hint` | on failure | actionable recovery ("call describe for valid fields") |
| `progress` | always | step log (exact / route / scan / fuse) |
| `token_estimate` | always | `len(str(response)) / 4`; agent budgets context |
| `truncated` | bounded reads | explicit on `fetch(depth="full")` and capped result sets |
| `missing` | `fetch`, `traverse` | ids that resolved to nothing — never dropped silently |

Snapshot/backup/dry_run/restore fields from the upstream **write**-tool contract do not
apply — Vera's query path never writes. (`search`'s `dry_run` is unrelated: it previews
a *read*, and mutates nothing either way.) They apply only to the offline pipelines. See
`STANDARDS_COMPLIANCE.md` §16-17.

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
  descriptors, not working sets. Depth is reserved **before** waiting, so waiters
  cannot pile up behind the check.
- **Wait-timeout:** reject requests that have waited too long, so a full queue does not
  produce 20-second tail latencies. A full queue sheds load *immediately* rather than
  after the timeout.
- **One admission slot per call, batch included.** A batched `search` takes one slot for
  the whole set, which is part of why batching is a parameter rather than a loop the
  agent runs.

All four limits are config values, not magic numbers. Bigger hardware raises them.

---

## 5. The OOM guarantee (by construction)

```
Peak RAM = fixed_costs + (concurrency_limit × per_request_ceiling)
```

Both terms are bounded, so peak RAM is bounded. The load-bearing fact is **sequential
cluster loading**: a request loads one cluster, scans it, drops it, loads the next — so
`per_request_ceiling` is independent of `clusters_probed`.

The store enforces this structurally rather than by convention: `scan_cluster` takes a
visitor and hands it a **borrowed, reused buffer** instead of returning a `Vec`. A
caller that wants to keep a cluster has to write a visible copy, so materialising one
cannot happen by accident.

**Measured**, on a 903 MB / 200K-row corpus at 1024 dims:

| clusters probed | peak RSS |
|---|---|
| 1 | 7 MB |
| 20 | 7 MB |

Identical — RAM does not move with the dial. Because the scan streams row by row, the
real ceiling is **one row**, not one cluster; the ~82 MB per-cluster figure below is a
correct upper bound that overstates actual use by orders of magnitude. It is kept as
the budget because it is the number that stays true if the streaming API is ever
replaced.

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

Full hardware detail in `HARDWARE.md`.

---

## 6. Embedding provider client (critical subsystem)

Query embedding is Vera's only external dependency, so the provider client is a
first-class component, not a helper. It must:

1. **Pin one provider** (OpenRouter Exacto mode) — never route across providers, which
   would vary the embedding space. See `EMBEDDING.md`.
2. **Pin the model version** and store it in corpus metadata; refuse to serve on
   mismatch. The engine compares its configured space against what the corpus recorded
   at startup and fails closed on model, width **or** normalization.
3. **Canary check at startup:** embed a known string, compare to the stored reference
   vector; if cosine < threshold (e.g. 0.999), refuse to serve and alert. This catches
   silent provider/model drift before it corrupts results.
4. **Retry with backoff** on 429/529; **queue** through transient outages.
5. **Fail over only to a pre-validated secondary provider** (cosine-matched offline);
   never to an unvalidated host.
6. **Batch** the queries of one `search` call into a single request. This is the
   efficiency argument behind bulk being a parameter shape (§2).
7. **Truncate identically** to the corpus dimension if any truncation is used.

The embedding width is **configuration, not a constant**. The production target is
Qwen3-8B at 4096, but a corpus embedded with Qwen3-0.6B at 1024 is equally valid and
must not require a recompile to serve. What is enforced is that corpus and query agree.

---

## 7. Transports

Two modes (per upstream §30), reframed for agentic use:

- **stdio** — for harnesses / desktop agents that launch Vera as a subprocess.
  All logs go to stderr; stdout carries JSON-RPC frames only.
- **streamable-http** — for Claude.ai connectors and remote agents. Auth token
  generated at deploy, never hardcoded.

The OpenAI-compatible HTTP shape may also be exposed by a thin adapter for non-MCP
callers, wrapping the *same* retrieval core with a `mode` (chunks vs. ready-to-prompt
context). The core is written once; adapters are thin.
