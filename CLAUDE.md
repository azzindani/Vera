# CLAUDE.md — Vera

> Context file for AI coding agents working in this repository.
> Read this first. It defines what Vera is, how it is built, the rules you must
> follow, and the deliberate ways it diverges from the upstream
> `azzindani/Standards/local_mcp/STANDARDS.md`.

---

## 1. What Vera is

**Vera** is an agentic retrieval MCP server: a routed, hybrid search engine over
very large document corpora (initial domain: Indonesian regulations and contracts)
that returns **ranked, source-cited results** for an AI agent to reason over, and
for a human to independently verify.

The name means *truth from sources*. Vera never invents knowledge; it surfaces what
already exists and hands back the receipt. Every result is traceable to its origin
(document + page/section, or media + timestamp).

**One-line purpose:** turn a query against 100M+ rows into a handful of
verifiable, cited results — fast enough to feel interactive, light enough to run on
a 2 vCPU / 8 GB VPS, and structured so the calling agent can summarize and the human
can double-check.

Vera is **composable infrastructure**: it is a tool an agent calls, not an
application a person uses directly. It is part of the same family as Folio,
Pipeline, and Sift.

---

## 2. The core mental model

Vera is a **deterministic retrieval function**, not an assistant.

```
Agent's job:   understand intent, choose tools, compose the final summary + citations
Vera's job:    embed → route → search → fuse → return structured, cited results
```

Vera **never calls an LLM**. It does not summarize, rerank with a model, or
"interpret." It returns structured evidence; the agent does the thinking. This
single rule is what keeps the engine stateless, light, and replicable. See
`docs/OUTPUT_CONTRACT.md`.

---

## 3. Architecture in one screen

Three routing layers, hybrid search, two containers. Full detail in
`docs/ARCHITECTURE.md`.

```
LAYER 1  domain          → pick the knowledge base (1 at launch; future-proofed)
LAYER 2  ~10K clusters   → pick ~5 nearest cluster centroids (k-means)
LAYER 3  ~10K rows/cluster → flat halfvec scan + BM25, sequential load, top-50 each
                            → RRF fuse → top-k results + provenance
```

- **Embedding model:** Qwen3-Embedding-8B, **4096 dims**, open weights.
  Corpus is pre-embedded offline on rented GPU; queries are embedded at runtime via
  **OpenRouter** (pinned provider). Same model both ends — non-negotiable. See
  `docs/EMBEDDING.md`.
- **Store:** PostgreSQL + pgvector. Vectors stored as `halfvec`. **No global ANN
  index** — pgvector cannot index 4096 dims, and routing + per-cluster flat scan
  makes one unnecessary. Partition by cluster; rely on routing for pruning.
- **Hybrid:** dense (Qwen3) + Postgres full-text BM25, fused with Reciprocal Rank
  Fusion. No model reranker.
- **Containers:** (1) stateless **Rust** MCP engine; (2) Postgres/pgvector.
- **Two offline pipelines:** pre-embedding (ingest → chunk → embed → load) and
  layer-2 cluster maintenance. These run on the GPU box / operator side, never on
  the query path. See `docs/PRE_EMBEDDING.md` and `docs/CLUSTER_MAINTENANCE.md`.

---

## 4. Repository structure

```
vera/
├── CLAUDE.md                      ← you are here
├── README.md
├── docs/
│   ├── ARCHITECTURE.md            ← full system design, data flow, 2-container split
│   ├── MCP_ENGINE.md              ← Rust query engine: tools, query flow, concurrency, OOM
│   ├── OUTPUT_CONTRACT.md         ← result + provenance schema; agent-summarizes principle
│   ├── EMBEDDING.md               ← Qwen3-8B, 4096 dims, OpenRouter pinning, consistency
│   ├── PRE_EMBEDDING.md           ← offline GPU ingestion pipeline (resumable, idempotent)
│   ├── CLUSTER_MAINTENANCE.md     ← layer-2 drift, 3-tier maintenance, atomic swap
│   ├── HARDWARE.md                ← RAM/disk/latency budgets; 2CPU/8GB profile; scale path
│   ├── LOOPHOLES.md               ← known failure modes + their solutions
│   ├── STANDARDS_COMPLIANCE.md    ← mapping to local_mcp STANDARDS; documented divergences
│   ├── MULTI_DOMAIN.md            ← scaling to many domains/sources; the foundation principles
│   └── EVAL.md                    ← the eval harness that gates retrieval quality
│
├── engine/                        ← Rust MCP engine (stateless)
│   ├── src/
│   │   ├── main.rs                ← MCP server setup (stdio + http transports)
│   │   ├── tools.rs               ← thin tool wrappers (one fn per tool)
│   │   ├── retrieval/             ← embed, route, search, fuse — core logic
│   │   ├── provider/              ← OpenRouter client: pin, retry, canary check
│   │   ├── routing/               ← layer-1/2 centroids, hot in memory
│   │   └── concurrency.rs         ← semaphore + bounded queue + backpressure
│   └── Cargo.toml
│
├── pipelines/                     ← Python (GPU + ingestion libraries)
│   ├── pre_embed/                 ← ingest, chunk, embed on GPU, bulk-load
│   └── cluster_maint/             ← k-means, split, re-cluster, atomic swap
│
└── eval/                          ← labeled query→answer sets + scoring scripts
```

Language split follows the upstream rule "libraries dictate language": the engine is
Rust (tiny stateless footprint, tokio concurrency); the pipelines are Python
(transformers/GPU, PDF extraction, k-means). See `docs/STANDARDS_COMPLIANCE.md` §5.

---

## 5. Architecture principles (do not violate)

1. **The engine never calls an LLM.** No summarization, no model rerank inside Vera.
   Return structured evidence; the agent composes prose. (`docs/OUTPUT_CONTRACT.md`)
2. **Same embedding model, same version, both ends.** Corpus and query must share the
   exact Qwen3-8B weights/version/instruction/pooling. Enforced by a startup canary
   check. (`docs/EMBEDDING.md`)
3. **Routing accelerates; the global keyword net guarantees.** Semantic routing may
   miss; exact-identifier search must bypass routing and scan globally so a known
   regulation number is never silently lost. (`docs/LOOPHOLES.md` §1)
4. **OOM is impossible by construction.** Peak RAM = fixed costs + (concurrency limit
   × per-request ceiling). Sequential cluster load keeps per-request working set to
   one cluster (~82 MB). Both terms bounded → total bounded. (`docs/MCP_ENGINE.md`)
5. **Provenance is captured at ingestion and immutable.** The double-check link/locator
   (source document + page/section) is the product. Never synthesize a source link at
   query time.
6. **Query path is read-only.** No writes, no snapshots, no mutation of the corpus
   while serving. Writes happen only in the offline pipelines, which use atomic
   version swaps. (`docs/CLUSTER_MAINTENANCE.md`)
7. **Design for the 2 vCPU / 8 GB VPS; let bigger hardware benefit automatically.**
   Never hardcode limits — read them from config. (`docs/HARDWARE.md`)

---

## 6. Tool surface (agent-facing, read-only) — **four primitives**

Full schemas in `docs/MCP_ENGINE.md`; the design rationale is `MULTI_DOMAIN.md` §10b.
Vera's read-only analog of the upstream LOCATE→INSPECT→PATCH→VERIFY loop is
**DESCRIBE → SEARCH → TRAVERSE → FETCH**.

| Tool | Role | Returns |
|---|---|---|
| `describe` | what exists: sources, filterable fields, edges, limits | ids + schemas + capabilities, zero content |
| `search` | the workhorse: routed hybrid search, one query or many | ranked results: id, snippet, score, provenance |
| `fetch` | read by id at a chosen depth | `provenance` \| `snippet` \| `full` (size-capped) |
| `traverse` | follow a declared edge from known ids | neighbour addresses, ✗ bodies |

! **Capability grows through declared parameters, ✗ new tools.** Adding a source, an
edge type, a filterable field or a validity model must add **zero** tools. A fifth verb
means the model was wrong, not that the feature was large. Guarded by a test.

! **The agent passes intent; the engine owns strategy.** Query text, constraints, `k`,
edge name and fetch depth are intent. Which source is searched, which rankers run,
their fusion weights and which clusters are probed are strategy and are never
agent-supplied — an agent that could set fusion weights could silently disable the
keyword half.

Consequences of the collapse:

- `list_domains` → `describe`, which now also publishes what can be *asked*, not only
  what exists.
- `read_chunk` + `get_provenance` → `fetch` at two depths. They were one operation.
- `explain_routing` → `search(dry_run: true)`. A separate tool would have to duplicate
  the whole constraint surface and the two would drift.
- **Bulk is a parameter shape, ✗ a tool.** `search` takes one query or many; `fetch` and
  `traverse` take one id or many. This amortises the embedding round-trip and lets one
  admission slot cover a batch, without a `search_batch` verb.
- **An unrecognised constraint fails loudly**, naming the valid fields. Silently
  dropping a filter returns a confident answer to a question that was not asked.

The agent never passes a `domain` or `source` — it is detected inside the engine by
anchor match. If nothing matches above threshold the engine returns empty results
rather than guessing, except for exact-identifier hits, which bypass routing entirely.

Every tool returns a dict with `success` first, plus `token_estimate`, `progress`, and
(on failure) `error` + `hint`. Docstrings ≤ 80 chars.

## 7. What you must NEVER do

1. **Never call an LLM from inside the engine.** Not to summarize, not to rerank.
2. **Never embed the query with a different model/version/provider than the corpus.**
3. **Never serve if the startup canary check fails** (embedding space mismatch).
4. **Never let exact-identifier search be gated by cluster routing.** Reg numbers
   search globally.
13. **Never accept a domain from the agent.** Domain is detected internally by anchor
    match. If no anchor matches above threshold, return empty results — never guess.
5. **Never load all probed clusters into memory at once.** Sequential load only —
   this is the OOM guarantee.
6. **Never use an unbounded request queue.** Bounded + backpressure + wait-timeout.
7. **Never fall back to an unvalidated embedding provider.** Only a pre-validated one.
8. **Never synthesize or guess a provenance link.** Provenance comes from ingestion.
9. **Never mutate a live cluster in place.** Copy-on-write + atomic version swap.
10. **Never block the MCP stdio channel with stdout.** All logs to stderr.
11. **Never return raw vectors or full documents from search.** Snippets + addresses;
    full text only via the bounded `read_chunk`.
12. **Never hardcode RAM/row/cluster limits.** Read from config so bigger hardware
    scales without code changes.

---

## 8. Progress tracker

- [x] Engine skeleton: MCP server (stdio), tool stubs, return contract
- [ ] OpenRouter provider client: pin, batch, retry/backoff, canary check
      *(trait, validation and `canary_check` exist; the HTTP client does not —
      a deterministic stub stands in and logs a warning at startup)*
- [x] Routing: layer-1 anchors + layer-2 centroids loaded hot at startup
- [x] Retrieval: sequential cluster scan + cosine + BM25 + RRF fusion
- [x] Global exact-identifier keyword path (routing bypass)
- [x] Concurrency: semaphore + bounded queue + backpressure + wait-timeout
- [x] Output contract: results + provenance + citation block
- [ ] Pre-embedding pipeline: ingest → chunk → embed (GPU) → bulk-load (resumable)
      *(`vera-ingest` imports an already-embedded corpus; document embedding is
      not built)*
- [x] Consistency: pinned model/version metadata + space check at startup
- [ ] Cluster maintenance: incremental assign, split-on-size, periodic re-cluster
      *(k-means and atomic rebuild exist in `vera-index`; incremental
      maintenance does not. **Split-on-size is needed sooner than expected** —
      see finding 4 below)*
- [x] Eval harness: `vera-bench` sweeps clusters_probed reporting latency and
      recall against an exhaustive baseline
- [ ] Hardware validation on 2 vCPU / 8 GB VPS under concurrency

### Measured findings

Run `vera-bench run --corpus <db>` to reproduce. Numbers below are a synthetic
200K × 1024 corpus on 4 vCPU / 16 GB.

1. **Layer-1 thresholds must be calibrated, never fixed.** A hardcoded 0.25
   rejected 97% of genuine in-corpus queries on the first real corpus tried.
   How close a query lands to a domain anchor depends on the embedding model's
   anisotropy. Worse, the failure is silent: too tight a threshold returns
   `success: true` with zero results, indistinguishable from an empty corpus.
   The build now measures the row-to-anchor distribution and records p1.

2. **BM25 must run globally, once — never per cluster.** Routing exists because
   dense vectors cannot be indexed. An inverted index does not have that
   problem, and FTS5 evaluates a match corpus-wide before filtering by
   `cluster_id`, so per-cluster keyword search cost N full scans and grew with
   the dial that is meant to be cheap. It was 87% of query time. Going global
   cut total time 4.7× *and* raised recall@10 at probe=1 from 61.9% to 99.5%.
   `ARCHITECTURE.md` §5 still describes the per-cluster scheme and is now wrong.

3. **The leaf scan is I/O-bound, not compute-bound — this is the open problem.**
   At 1024 dims the dense scan costs **~5.3 µs/row**, of which the dot product
   is ~0.3 µs. The rest is SQLite row-stepping. One 10K-row cluster therefore
   costs ~53 ms, so probe=5 spends ~265 ms in scan overhead alone and the
   ~150 ms warm target in `ARCHITECTURE.md` §8 is unreachable in this shape.
   The fix is the one the architecture already describes but the storage layer
   does not implement: hold each cluster's vectors as **one contiguous blob**,
   so a probe is one fetch plus an in-memory scan rather than 10K row fetches.
   Expected ~15× on the dominant stage.

4. **k-means produces badly unbalanced clusters.** Targeting 10K rows/cluster
   gave 1,005 min / 25,011 max. The largest cluster sets *both* the per-request
   RAM ceiling and worst-case probe latency, so the tail governs the budget.
   Split-on-size is not a later refinement.

5. **The OOM guarantee holds, measured.** Peak RSS over a 903 MB corpus
   (820 MB of vectors) is **7 MB at probe=1 and 7 MB at probe=20** — identical.
   RAM is independent of `clusters_probed`, as `ARCHITECTURE.md` §4 claims.
   The streaming scan API is what makes this structural rather than a
   convention: `scan_cluster` hands the visitor a borrowed, reused buffer, so
   retaining a cluster would have to be written as a visible copy.

6. **Routing is doing far less work than end-to-end recall suggests.** Adding
   `EVAL.md` §3's *routing recall* — was the true answer in a probed cluster at
   all — splits a number that looked settled:

   | probe | recall@10 | routing recall |
   |---|---|---|
   | 1 | 99.3% | **68.2%** |
   | 5 | 99.5% | **77.5%** |
   | 20 | 100% | 100% |

   Dense routing reaches only ~68% of true answers at probe=1; the **global
   BM25 half recovers the rest**. An earlier read of this project's own numbers
   credited that 99% to routing. It is not routing — and on a corpus where
   query and document vocabulary diverge, the keyword half will not rescue it.
   This is precisely the split `EVAL.md` predicted the metric would expose.

7. **Recall is still not measured on real data.** The synthetic corpus is too
   easy. No recall claim should be trusted until the real corpus runs.

### Known gaps against the docs

Audited against every doc in the repo. These are specified and **not built**:

| Doc | Requirement | Status |
|---|---|---|
| `EMBEDDING.md` §3, `HARDWARE.md` §3 | vectors stored as **halfvec (16-bit)** | store uses f32 — 2× the bytes, and the leaf scan is I/O-bound, so this is also a performance gap |
| `EMBEDDING.md` §4.1 | pin the **provider id** as part of the config | `EmbeddingSpace` has no provider field |
| `EMBEDDING.md` §5 | validated **secondary** provider for fail-over | not built |
| `LOOPHOLES.md` §8 | store a **source hash** to detect moved/changed sources | no such column |
| `LOOPHOLES.md` §10 | **free-space preflight** before bulk-load / split | not built |
| `HARDWARE.md` §2 | centroids **mlock'd** | not built |
| `MCP_ENGINE.md` §7 | **streamable-http** transport | stdio only |
| `CLUSTER_MAINTENANCE.md` §2 | Tier-1 incremental assign, Tier-2 **split-on-size** | only full rebuild exists |
| `EVAL.md` §2 | `eval/` labeled query set | directory does not exist |
| `EVAL.md` §3 | **exact-match recall**, nDCG | not measured (recall@k, routing recall, MRR, p50/p95/p99 are) |
| `CLAUDE.md` §4 | layout: `docs/`, `engine/`, `pipelines/` | actual: docs at root, `crates/` workspace, no Python pipelines |

Two docs are now **wrong** rather than merely unimplemented, and should be
edited rather than built toward:

- `ARCHITECTURE.md` §5 describes the keyword half as scoped to routed clusters.
  Measured, that costs N full-corpus scans and loses recall; it must be global.
- `MCP_ENGINE.md` §5 and `HARDWARE.md` §2 budget `per_request_ceiling` as one
  cluster (~82 MB). The streaming scan makes it one *row* — measured peak RSS
  is 7 MB regardless of `clusters_probed`. The budget is correct as an upper
  bound but overstates actual use by ~4 orders of magnitude.

---

## 9. Scaling beyond one source

Vera's reason to exist is that the predecessor system was locked to **one domain with
one data source**. The target is many domains (legal, medical, …) and many sources
within each (regulations, court decisions, contracts; guidelines, drug labels, ICD-10,
literature) — each with its own metadata, provenance shape, identifier grammar,
validity semantics and mix of retrieval methods.

**`MULTI_DOMAIN.md` is the design record for that foundation.** Read it before changing
the schema, the routing layers, or the tool surface. Its short form:

1. The **source** is the unit of heterogeneity; domain is only a label.
2. The engine never names a domain or source — any `match` on a source id is a failure.
3. A new source is a **manifest**, not a pull request.
4. **Rankers are fused; constraints gate routing.** A constraint applied after routing
   is silent zero-recall.
5. Clusters live inside a source; never cluster across sources.
6. Fan out across sources; never argmax.
7. **Validity is a correctness property**, not a filter option — current-only is default.
8. The engine exposes primitives and never plans; the agent plans and never ranks.
9. **Scale and generality are separate experiments** — one big corpus proves the first,
   several tiny ones prove the second.
10. Schema and storage layout are **one decision, made once** — both are paid for in
    re-ingest.

---

*Upstream standard: `https://github.com/azzindani/Standards/blob/main/local_mcp/STANDARDS.md`.*
*Where this CLAUDE.md or `docs/STANDARDS_COMPLIANCE.md` conflicts with the upstream
standard, this project takes precedence (per the standard's own precedence rule).*
