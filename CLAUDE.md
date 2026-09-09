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

! **The tree below is the intended layout, ✗ the current one.** Docs live at the repo
root, not under `docs/`; the code is a `crates/` Cargo workspace, not `engine/` +
`pipelines/`; and there is no Python yet. Cross-references in the docs are written as
`docs/X.md` and resolve to `X.md`. Actual:

```
vera/
├── *.md                           ← every doc, at the root
├── Cargo.toml                     ← workspace
├── crates/
│   ├── vera-core/                 ← types, config, output contract, calibration, profile
│   ├── vera-embed/                ← provider trait, canary, stub (no HTTP client yet)
│   ├── vera-store/                ← ChunkStore trait + SQLite backend
│   ├── vera-engine/               ← routing, leaf scan, fusion, orchestration
│   ├── vera-index/                ← k-means, streaming corpus build, split, preflight
│   ├── vera-ingest/               ← import an already-embedded corpus
│   ├── vera-mcp/                  ← the four primitives over stdio
│   └── vera-bench/                ← fixture generation + the eval sweep
└── migrations/                    ← Postgres schema (the production target)
```

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
      *(trait, validation, `canary_check` and the **provider allowlist** exist —
      `EmbeddingSpace.validated_providers` + `ProviderConfig`, both checked at
      startup. The HTTP client does not — a deterministic stub stands in and
      logs a warning)*
- [x] Routing: layer-1 anchors + layer-2 centroids loaded hot at startup
- [x] Retrieval: sequential cluster scan + cosine + BM25 + RRF fusion
- [x] Global exact-identifier keyword path (routing bypass)
- [x] Concurrency: semaphore + bounded queue + backpressure + wait-timeout
- [x] Output contract: results + provenance + citation block
- [~] Pre-embedding pipeline: ingest → chunk → embed (GPU) → bulk-load (resumable)
      *(`vera-ingest` imports an already-embedded corpus and the load now
      **streams** — peak RAM is `train_sample × dim × 4`, ✗ the corpus
      (`PRE_EMBEDDING.md` §2b). Document embedding and chunking are not built,
      and the loader is interruptible but not resumable)*
- [x] Consistency: pinned model/version metadata + space check at startup
- [~] Cluster maintenance: incremental assign, split-on-size, periodic re-cluster
      *(k-means, atomic rebuild and **split-on-size** exist in `vera-index`.
      Split runs at build time behind `--max-cluster-rows`; the online Tier-2
      trigger needs a per-generation assignment table, which is a schema change
      and belongs with `MULTI_DOMAIN.md` §12. Tier-1 incremental assign does
      not exist)*
- [x] Eval harness: `vera-bench` sweeps clusters_probed reporting latency,
      recall, nDCG, exact-match recall and a **recall-loss decomposition**
      (routing / cap / fusion / by-design) against an exhaustive baseline
- [x] Corpus profile at ingest: vocabulary, occurrence-weighted IDF, Zipf slope,
      doc lengths, identifier density — stored with the corpus so the caveat
      that qualifies a keyword number travels with the data
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

   | stage | 50-word fixture (probe=50) | **Zipfian fixture (probe=20)** |
   |---|---|---|
   | layer-3 BM25 (one global query) | 72.3% (~291 ms) | **78.9% (~232 ms)** |
   | layer-3 dense scan | 27.2% (~110 ms) | 20.5% (~60 ms) |
   | routing (layers 1+2) | 0.2% | 0.2% |

   A realistic vocabulary cut BM25 by 20% (291 → 232 ms) and left it **more**
   dominant, because it cut the dense scan further. See finding 11.

   BM25 does not vary with `clusters_probed`, so it is a **latency floor**:
   speedup peaks at 4.4× and *falls* beyond probe=10 as the scan re-grows.
   Per-row scan cost is unchanged at ~5.6 µs, so the contiguous-blob storage fix
   now buys ~27% of query time rather than ~76% — **it is no longer the first
   thing to fix.**

   ! That caveat blamed the fixture, and **it was the wrong diagnosis** — see
   finding 11.

5. **Recall now behaves like a real curve** — it did not before, because
   routing was barely pruning. On the √N-clustered 200K corpus with the
   **Zipfian** vocabulary (200 topics, 447 clusters, 100 queries):

   | probe | rows/query | % of corpus | recall@10 | route/D | nDCG@10 | speedup |
   |---|---|---|---|---|---|---|
   | 1 | 436 | 0.22% | 70.4% | 54.5% | 0.775 | 6.1× |
   | 2 | 820 | 0.41% | 86.8% | 81.2% | 0.898 | 5.9× |
   | **5** | **2,093** | **1.0%** | **99.6%** | **99.4%** | **0.997** | **5.9×** |
   | 10 | 4,454 | 2.2% | 100% | 100% | 1.000 | 5.5× |
   | 20 | 9,048 | 4.5% | 100% | 100% | 1.000 | 5.0× |

   **probe=5 is the knee** — 99.6% recall while touching 1% of the corpus. The
   documented default of 5 is right here. L1-rej is 0% and top-1 is 100%
   throughout. Speedup is flat to probe=5 and falls after, because the BM25
   floor (finding 4) dominates until the scan re-grows.

   ! nDCG is the column that earns its place at probe=1: recall is 70.4% and
   nDCG 0.775, so the misses are concentrated in the *lower* ranks — the top of
   the list survives aggressive pruning better than a recall number suggests.

   Recall-loss decomposition at the same settings (vs the uncapped dense top-10):

   | probe | found | routing | cap | fusion | by design |
   |---|---|---|---|---|---|
   | 1 | 40.8% | 44.0% | 0.0% | 1.1% | 14.1% |
   | 5 | 45.3% | **0.6%** | **0.0%** | 0.0% | 54.1% |
   | 20 | 45.3% | 0.0% | 0.0% | 0.0% | 54.7% |

   At probe=5 routing has stopped losing anything and the cap is not binding at
   all — clusters average 447 rows against a `per_cluster_top_k` of 50, so the
   cap only starts to matter as clusters grow toward the 10K design point. The
   54% "by design" is RRF preferring keyword hits, unchanged by probing, and is
   **not** loss (finding 14).

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
   a synthetic corpus whose topic structure was generated to be findable. It is
   enough to show the *mechanism* works and to catch the mis-clustering; it
   cannot say whether routing pays on Indonesian regulation text. No recall
   claim should be trusted until the real corpus runs.

11. **BM25 cost is set by query-term reach, ✗ by vocabulary size — and Vera's
   query construction makes it corpus-proportional.** Finding 4 blamed the
   fixture's 50-word vocabulary and predicted real Zipfian text would be
   selective. The fixture is now Zipfian (20K terms, slope −0.95 at 200K) and
   **BM25 is still ~82% of query time.** On that corpus:

   | | |
   |---|---|
   | mean IDF per term **type** | 8.68 → "average term reaches 0.02% of rows" |
   | rows an 8-term query **actually** reaches | **20,928 of 20,000** |

   ! A mock Indonesian corpus imported through `vera-ingest` shows the same
   shape for the same reason, so this is not a fixture artefact: vocabulary
   8,010, Zipf slope −0.91, **and `yang` in 100% of rows**. Indonesian function
   words (`yang`, `dan`, `dalam`) are in essentially every legal document, which
   is exactly the head this finding is about. Expect the real corpus to look
   like this.

   Both are correct. The type average is dominated by the rare tail, and a query
   never draws from the tail — it draws from a document's words, which are mostly
   head terms. The commonest term is in 16,499 of 20,000 rows, and
   `fts_match_expression` ORs **every** query token, so the union of postings is
   bounded below by that term however selective the other seven are.

   ! This breaks the second half of `ARCHITECTURE.md` §5's argument. "BM25 *is*
   an index, so it needs no routing" is true **per term** and false for an
   unfiltered OR over all of them. The fix — selectivity-aware term capping —
   is named in `METRICS.md` §2.3 and is **deliberately not built**: it changes
   which documents come back, `EVAL.md` §5 rejects any change that lowers
   recall@k, and there is no labelled set to measure that against. It is the
   first thing to build once there is one, and must not land before it.

12. **A representativeness check keyed on the wrong average would have passed the
   corpus it exists to reject.** `CorpusProfile` originally gated on type-level
   mean IDF, which reads 8.68 on the fixture above — comfortably "representative"
   while its queries touch every row. It now gates on **occurrence-weighted** IDF
   via `expected_query_reach(8)`. Two corpora can agree on vocabulary size, Zipf
   slope and type-level IDF and disagree completely on what BM25 costs.

13. **The recall-loss decomposition reproduced its own blind spot on the first
   attempt.** Scored against the exhaustive *fused* result — the obvious
   reference — the candidate-cap column is **structurally pinned at 0.0%**,
   because that baseline applies the same `per_cluster_top_k` to the same
   clusters, so every row it keeps the routed run keeps too. Turning the cap from
   50 to 1 moved the *routing* column instead. That is precisely the flaw
   `METRICS.md` §3.1 says recall@k has, rebuilt inside the metric written to fix
   it, and the output looked healthy throughout. Cap loss is only visible against
   an **uncapped dense** top-k. Verified: cap now reads 0.0% → 9.2% → 62.8% as
   the dial goes 50 → 5 → 1, while `route/D` sits flat at 99.2% across a recall
   collapse from 98.2% to 85.5%.

14. **A fourth attribution column was needed, and it is not a defect.** Against
   dense-only truth, two thirds of the set reads as "fusion loss" while
   end-to-end recall is 98% — those rows are outranked by keyword hits at *every*
   probe count including exhaustive. Reporting that as loss would argue for
   retuning `rrf_k` to fix nothing, so `fusion` is split by whether the
   exhaustive fused run returned the row: **`fusion`** if it did, **`by design`**
   if it did not.

15. **The loader held the corpus, and three of the four things it held were
   invisible.** `vera-ingest import` collected every `IngestRow` and every vector
   before building — ~100 GB of bodies and **1.6 TB of vectors** at the design
   point. The obvious one is the vector matrix; the other three only show up
   when the arithmetic is done at 100M:

   | held | at 100M × 4096 | replaced by |
   |---|---|---|
   | every `IngestRow` | ~100 GB | one row, written as it arrives |
   | every vector | **1.6 TB** | a bounded training sample |
   | one anchor cosine per row, sorted for percentiles | 400 MB | a fixed-width histogram |
   | one SQL transaction over every insert | a ~1.6 TB WAL | batched commits + a completeness marker |

   Measured peak RSS on a 20K × 1024 import: **89 MB** training on everything,
   **29 MB** at 5,000 rows, **13 MB** at 1,000.

   ! **Sampled training is the standard IVF construction, and assignment stays
   exact.** FAISS trains a coarse quantizer on 30–256 vectors per centroid and
   then adds the full corpus; every row here is still scored against every
   centroid in pass 2. Measured cost on a corpus with real structure: at 36
   points per centroid, cluster tightness fell 0.8814 → 0.8791 (**−0.3%**) and
   recall@10, route/D and nDCG@10 were **identical** (100% / 100% / 1.000).

   ! **Batching commits gave up atomicity, so it needed a marker.** An
   interrupted load leaves a corpus with a valid schema, a working FTS index and
   *some* of the rows — it opens, it answers, it is silently short. The build
   writes `build_state = in_progress` before the first insert and `complete`
   only after the FTS rebuild; `SqliteStore::open` refuses anything else.

16. **`per_cluster_top_k` and `rrf_k` are coupled.** RRF scores by rank *within
   each list*, so a longer dense candidate list gives documents present in both
   halves a second contribution and pushes dense-only documents down. Dense
   faithfulness is therefore **non-monotonic** in the cap (33.0% → 53.0% → 32.8%
   at caps 50 → 5 → 1) even though end-to-end recall is monotonic. The
   one-dial-per-cause table in `METRICS.md` §3.1 is an approximation.

### Known gaps against the docs

Audited against every doc in the repo. These are specified and **not built**:

| Doc | Requirement | Status |
|---|---|---|
| `EMBEDDING.md` §5 | **retry/backoff** through transient provider blips | not built (the allowlist it protects now exists) |
| `HARDWARE.md` §2 | centroids **mlock'd** | not built — and `mlock` is FFI, which the workspace's `unsafe_code = "forbid"` rules out without a wrapper crate. Decide before building |
| `MCP_ENGINE.md` §7 | **streamable-http** transport | stdio only |
| `CLUSTER_MAINTENANCE.md` §2 | Tier-1 incremental assign; the Tier-2 **online trigger** | the split *algorithm* is built and runs at build time; triggering it on a live corpus needs a per-generation assignment table (schema change · `MULTI_DOMAIN.md` §12) |
| `EVAL.md` §2 | `eval/` labeled query set | **directory does not exist — now the top blocker** (`METRICS.md` §8). It gates the knee, honest exact-match recall, and every fix in `METRICS.md` §2.3 |
| `EMBEDDING.md` §3, `HARDWARE.md` §3 | vectors stored as **halfvec (16-bit)** | store uses f32. Secondary since the √N fix: the leaf scan is ~16–27% of query time, BM25 is the rest |
| `CLAUDE.md` §4 | layout: `docs/`, `engine/`, `pipelines/` | actual: docs at root, `crates/` workspace, no Python pipelines. **§4 now states this** rather than only listing the intent |

Closed since the last audit:

| Doc | Requirement | Closed by |
|---|---|---|
| `EMBEDDING.md` §4.1 | pin the **provider id** | `EmbeddingSpace.validated_providers` + `ProviderConfig` |
| `EMBEDDING.md` §5 | validated **secondary** for fail-over | same, both checked at startup |
| `LOOPHOLES.md` §8 | **source hash** | `chunks.source_hash`, `--hash-col`, returned by `fetch` |
| `LOOPHOLES.md` §10 | **free-space preflight** | `vera_index::preflight`, before the k-means |
| `CLUSTER_MAINTENANCE.md` §2 | Tier-2 **split-on-size** algorithm | `vera_index::split`, at build time via `--max-cluster-rows` |
| `EVAL.md` §3 | **exact-match recall**, nDCG | both in the sweep; exact-match runs adversarially |
| `EVAL.md` §4, `LOOPHOLES.md` §7 | **candidate-cap loss** | the decomposition · findings 13–15 |
| `FACTORS.md` §7 | Zipfian fixture, corpus profile | `--vocab`/`--zipf`, `CorpusProfile` |

One doc statement is still **wrong** rather than unimplemented, and should be
edited rather than built toward:

- `ARCHITECTURE.md` §5 argues the keyword half needs no routing because "BM25 is
  an index". True per term; false for an OR over every query token, which is what
  the engine sends. Finding 11 and `METRICS.md` §2.3 have the correction — the
  *conclusion* (global, never per-cluster) still holds and is measured; the
  *reason given for it being cheap* does not.

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
