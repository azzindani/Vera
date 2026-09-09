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
LAYER 3  ~10K rows/cluster → flat halfvec scan, sequential load, top-50 each
         + ONE global BM25 → RRF fuse → top-k results + provenance
```

! **The two "10K"s are the same number**: √100M = 10,000. Cluster count is **k = √N**,
not a fixed rows-per-cluster target — applying "10K rows" at any smaller scale
under-clusters badly (`ARCHITECTURE.md` §2). And the design is **n-layer**, not
two-layer: a routing level exists wherever a level is too big to scan linearly, so 2 is
simply what 100M needs. n ∈ {2, 3} in practice; past that, shard rather than deepen
(`HARDWARE.md` §5).

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
│   ├── METRICS.md                 ← the targets · what "working" means, as falsifiable numbers
│   ├── FACTORS.md                 ← the ~36 variables behind those numbers; which differ per source
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

! **`METRICS.md` holds the targets.** The findings below are what has been *measured*;
`METRICS.md` is what must be *achieved*, with the gap and the blocker named for each.

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

3. **Cluster count must be √N; a fixed rows-per-cluster target under-clusters.**
   The benchmark was built with `--per-cluster 10000` — the 100M design point —
   which gives **20 clusters** over 200K rows, so probing 5 scanned a quarter of
   the corpus. Rebuilt at k=√N=447, at probe=5:

   | | 20 clusters | 447 clusters |
   |---|---|---|
   | rows scanned | 76,831 | **1,856** (41× fewer) |
   | largest cluster | 25,011 | **1,085** |
   | per-request RAM ceiling | 102.4 MB | **4.4 MB** |
   | cluster tightness | 0.751 | **0.881** |

   Every latency and speedup figure measured before this was describing a
   mis-clustered index.

4. **The bottleneck is now the global BM25 query, not the leaf scan.** Scanning
   41× fewer rows made queries only ~2× faster, because the cost that remained
   is fixed:

   | stage | share at probe=50 |
   |---|---|
   | layer-3 BM25 (one global query) | **72.3%** (~291 ms) |
   | layer-3 dense scan | 27.2% (~110 ms) |
   | routing (layers 1+2) | 0.2% |

   BM25 does not vary with `clusters_probed`, so it is a **latency floor**:
   speedup peaks at 4.4× and *falls* beyond probe=10 as the scan re-grows.
   Per-row scan cost is unchanged at ~5.6 µs, so the contiguous-blob storage fix
   now buys ~27% of query time rather than ~76% — **it is no longer the first
   thing to fix.**

   ! Caveat: the fixture's vocabulary is 50 words, so an 8-term OR matches a
   large fraction of the corpus. Real text is Zipfian and most query terms are
   selective, so this figure is probably inflated. It needs the real corpus
   before any BM25 optimisation is justified.

5. **Recall now behaves like a real curve** — it did not before, because
   routing was barely pruning. Post-fix, on the √N-clustered 200K corpus:

   | probe | rows/query | % of corpus | recall@10 | top-1 | speedup |
   |---|---|---|---|---|---|
   | 1 | 414 | 0.21% | 76.0% | 97.5% | 4.7× |
   | 2 | 821 | 0.41% | 90.2% | 100% | 4.7× |
   | **5** | **2,088** | **1.0%** | **99.8%** | 100% | **4.7×** |
   | 10 | 4,353 | 2.2% | 100% | 100% | 4.4× |

   **probe=5 is the knee** — 99.8% recall while touching 1% of the corpus. The
   documented default of 5 is right here. Speedup is flat to probe=5 and falls
   after, because the BM25 floor (finding 4) dominates until the scan re-grows.

6. **Routing works; layer-1 was the problem.** Two measurement errors hid this.
   Routing recall was scored against the **fused** exhaustive baseline, part of
   which is by construction not dense-reachable — a document BM25 found on an
   exact term match may sit nowhere near the query vector, so no probe count
   reaches it. And it conflated a routing miss with a **layer-1 rejection**,
   which produces an empty probe set and scores zero for a completely different
   reason. Scoring against a true **dense-only** baseline and reporting
   rejection separately:

   | probe | route/D | L1-rej | recall@10 |
   |---|---|---|---|
   | 1 | 53.0% | 0% | 76.0% |
   | 2 | 79.2% | 0% | 90.2% |
   | 5 | **99.8%** | 0% | **99.8%** |
   | 10 | **100%** | 0% | 100% |

   At probe=5 routing finds essentially **every** dense top-10 result. The
   earlier claim that "dense routing reaches only ~68% and BM25 carries recall"
   was an artifact of both errors and is withdrawn.

7. **The layer-1 threshold was calibrated on documents and applied to queries.**
   It was p1 of the row-to-anchor cosine distribution — which implies 1%
   rejection but measured **5% on the 20K corpus and 10% on the 200K**. A query
   is not a row: it is *near* a row, and that displacement moves it further from
   the anchor than any document sits, so the row distribution systematically
   understates how far a real query can fall. The threshold is now
   `p1 − margin·(p50 − p1)`, extrapolating one step further down the tail.
   Measured effect: **L1-rej 10% → 0%**, and route/D at probe=5 rose 89.8% →
   99.8% as the masked queries came back.

   ! Erring low is the safe direction. A false reject loses the query and
   answers `success: true` with nothing, which no caller can distinguish from an
   empty corpus. A false accept returns weak results carrying a low `confidence`
   the agent can act on. The margin is config, and the full distribution is
   stored in corpus metadata — so retuning is a config change, ✗ a re-ingest.

8. **k-means imbalance is real but not pathological at √N.** Sizes run p10 148 /
   p50 395 / p90 981 / max 1085 against a mean of 447 — a 2.4× spread, versus
   25× at 20 clusters. Two singleton clusters come from empty-cluster reseeding,
   since k=447 exceeds the fixture's 200 latent topics. The largest cluster
   still sets both the RAM ceiling and worst-case probe latency, so split-on-size
   remains worth having.

9. **The OOM guarantee holds, measured.** Peak RSS over a 903 MB corpus
   (820 MB of vectors) is **7 MB at probe=1 and 7 MB at probe=20** — identical.
   RAM is independent of `clusters_probed`, as `ARCHITECTURE.md` §4 claims.
   The streaming scan API is what makes this structural rather than a
   convention: `scan_cluster` hands the visitor a borrowed, reused buffer, so
   retaining a cluster would have to be written as a visible copy.

10. **Recall is still not measured on real data.** Every number above comes from
   a synthetic corpus whose vocabulary is 50 words and whose topic structure was
   generated to be findable. It is enough to show the *mechanism* works and to
   catch the mis-clustering; it cannot say whether routing pays on Indonesian
   regulation text. No recall claim should be trusted until the real corpus runs.

### Known gaps against the docs

Audited against every doc in the repo. These are specified and **not built**:

| Doc | Requirement | Status |
|---|---|---|
| `EMBEDDING.md` §3, `HARDWARE.md` §3 | vectors stored as **halfvec (16-bit)** | store uses f32 — 2× the bytes. Now a *secondary* perf gap: since the √N fix the leaf scan is only ~27% of query time (finding 4) |
| `EMBEDDING.md` §4.1 | pin the **provider id** as part of the config | `EmbeddingSpace` has no provider field |
| `EMBEDDING.md` §5 | validated **secondary** provider for fail-over | not built |
| `LOOPHOLES.md` §8 | store a **source hash** to detect moved/changed sources | no such column |
| `LOOPHOLES.md` §10 | **free-space preflight** before bulk-load / split | not built |
| `HARDWARE.md` §2 | centroids **mlock'd** | not built |
| `MCP_ENGINE.md` §7 | **streamable-http** transport | stdio only |
| `CLUSTER_MAINTENANCE.md` §2 | Tier-1 incremental assign, Tier-2 **split-on-size** | only full rebuild exists |
| `EVAL.md` §2 | `eval/` labeled query set | directory does not exist |
| `EVAL.md` §3 | **exact-match recall**, nDCG | not measured (recall@k, routing recall, MRR, p50/p95/p99 are) |
| `EVAL.md` §4, `LOOPHOLES.md` §7 | **candidate-cap loss** — is the true answer cut by `per_cluster_top_k` before fusion? | **not measurable** · both recall@k and route/D are blind to it by construction (`METRICS.md` §3.1). Fix before tuning any dial on real data |
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
