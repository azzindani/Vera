# ARCHITECTURE.md — Vera

Full system design. For the agent-facing tool contract see `MCP_ENGINE.md`; for the
output schema see `OUTPUT_CONTRACT.md`.

---

## 1. The problem Vera solves

Retrieval over a very large corpus (100M+ chunks) is normally either accurate-but-heavy
(needs a big in-RAM ANN index) or light-but-shallow. Vera makes large-corpus retrieval
viable on a **2 vCPU / 8 GB VPS** by replacing the global index with **routing**: the
intelligence is in knowing *where not to look*.

Vera is exposed as an MCP server so that an agent treats retrieval as a callable
capability (like web search), not as a fixed pipeline baked into an app.

---

## 2. The three routing layers

```
                         query (text)
                            │
                   embed via OpenRouter (Qwen3-8B, 4096)
                            │
          ┌─────────────────┴──────────────────┐
LAYER 1   │  DOMAIN detection                   │  query vs pre-embedded anchors
          │  (engine-owned; agent passes none)  │  → selects KB, or "no match"
          └─────────────────┬──────────────────┘
                            │
          ┌─────────────────┴──────────────────┐
LAYER 2   │  CLUSTER routing                    │  query vs ~10K k-means centroids
          │  (centroids held HOT in engine)     │  → pick ~5 nearest clusters
          └─────────────────┬──────────────────┘
                            │
          ┌─────────────────┴──────────────────┐
LAYER 3   │  LEAF search (~10K rows / cluster)  │  load clusters SEQUENTIALLY:
          │  halfvec distance + BM25            │  scan, keep top-50, drop, next
          └─────────────────┬──────────────────┘
                            │
                   RRF fusion of (dense ⊕ keyword), all probed clusters
                            │
                   + global exact-identifier hits (routing bypass)
                            │
                   top-k results + provenance  →  agent summarizes
```

- **Layer 1 (domain)** detects the knowledge base by matching the query vector against
  pre-embedded **domain anchors** — the engine owns this, the agent never passes a
  domain. At launch there is one domain, so this confirms "the query belongs here";
  with more domains it selects among them. If no anchor matches above a confidence
  threshold, the engine returns empty results rather than forcing the query into a
  domain it does not fit (`LOOPHOLES.md` §9). The routing scaffold means adding domains
  later is configuration (embed a new anchor), not a redesign.
- **Layer 2 (cluster)** is the coarse quantizer. Conceptually this is IVF: clusters =
  inverted lists, centroids = the coarse index. We implement it ourselves rather than
  using pgvector's IVF because we keep full 4096 dims (which pgvector cannot index).
- **Layer 3 (leaf)** is a flat scan within each probed cluster. ~10K rows × 4096 is
  small enough that flat scan is fast; routing is what keeps the scanned set small.

`clusters_probed` (how many layer-2 clusters to open, default ~5) is the central
recall/latency/RAM dial. More clusters = safer recall, more latency, but **not** more
RAM, because of sequential loading (§4).

---

## 3. Why no global ANN index

Two independent reasons converge:

1. **pgvector cannot index 4096 dims.** Its HNSW/IVFFlat indexes cap at 2000 dims
   (`vector`) / 4000 dims (`halfvec`). 4096 exceeds both.
2. **Routing already does the pruning an index would do.** Each query scans only the
   ~5 routed clusters (~50K rows), not 100M. A flat scan over 50K is fast.

So Vera stores `halfvec` columns, **partitions by cluster**, and does per-partition
flat scans using pgvector's distance operators with no ANN index. The HNSW graph
(which would be ~25 GB for 100M rows and cannot fit 8 GB anyway) never exists. See
`HARDWARE.md` for the memory math.

---

## 4. Sequential loading = the OOM guarantee

Probed clusters are loaded **one at a time**: load cluster → scan → keep top-50 → drop
→ load next. A request's working set therefore never exceeds **one cluster (~82 MB at
4096 halfvec)**, regardless of whether it probes 5 clusters or 50. This trades latency
(linear in clusters probed) for bounded RAM. It is the single most important
implementation rule. Detail in `MCP_ENGINE.md` §5.

---

## 5. Hybrid search: dense + keyword

| Half | Engine | Strength | Scope |
|---|---|---|---|
| Dense | Qwen3-8B vectors, halfvec, per-cluster flat scan | paraphrase, concept | routed clusters |
| Keyword | Postgres FTS / BM25 (tsvector + GIN, or pg_search) | exact terms, IDs, citations | routed clusters **and** global (for exact identifiers) |

Fusion is **Reciprocal Rank Fusion (RRF)** — model-free, robust, no reranker. The
keyword half has a **global** mode for exact regulation identifiers that bypasses
routing entirely (see `LOOPHOLES.md` §1) — this is the guarantee against silent
zero-recall.

Qwen3-Embedding is dense-only (no native sparse output), so the keyword side comes
entirely from Postgres, not the model.

---

## 6. Two containers

```
┌────────────────────────────┐        ┌────────────────────────────┐
│  Container A: Vera engine   │        │  Container B: Postgres      │
│  (Rust, stateless)          │  SQL   │  + pgvector                 │
│                             │ ─────► │                             │
│  • MCP server (stdio/http)  │        │  • halfvec vectors          │
│  • OpenRouter client        │        │  • cluster partitions       │
│  • layer-1/2 centroids HOT  │        │  • tsvector / BM25 index    │
│  • sequential leaf scan     │        │  • provenance + metadata    │
│  • RRF fusion               │        │                             │
│  • concurrency control      │        │                             │
└────────────────────────────┘        └────────────────────────────┘
        replicate for QPS                    scale cores / disk for size
```

- **Engine is stateless** (holds only the hot centroids, which are identical on every
  replica). To add throughput, run more engine replicas behind a load balancer.
- **DB holds all state.** To grow corpus, add disk; to speed search under load, add
  cores. Partitioning means cold clusters cost nothing.
- This split is the seed of the production scale-out: single-box and distributed are
  the *same* architecture at two sizes. See `HARDWARE.md` §scale.

---

## 7. The three pipelines / two environments / three cadences

| Pipeline | Environment | Cadence | Doc |
|---|---|---|---|
| Query serving | VPS (always-on) | per request | `MCP_ENGINE.md` |
| Pre-embedding (ingest→embed→load) | rented GPU (transient) | one-time + big batches | `PRE_EMBEDDING.md` |
| Cluster maintenance (assign/split/re-cluster) | VPS (light) + GPU (heavy re-cluster) | continuous + periodic | `CLUSTER_MAINTENANCE.md` |

The two offline pipelines never touch the live query path. Re-clustering publishes via
an **atomic version swap** so the live engine never sees a half-updated index.

---

## 8. End-to-end request trace

```
1. agent → search_knowledge(query="...")          [query only; no domain]
2. engine → OpenRouter.embed(query)                [pinned provider, ~120ms]
3. engine → layer-1: match query vs domain anchors [<1ms, hot; "no match" → empty result]
4. engine → layer-2: 5 nearest centroids           [~2ms, hot]
5. engine → for each of 5 clusters (sequential):
              SELECT ... halfvec distance + BM25 ... LIMIT 50
6. engine → global exact-identifier BM25 (if query has reg numbers/citations)
7. engine → RRF fuse all candidates → top-k
8. engine → attach provenance (source_url + page/section) to each
9. engine → return { detected_domain, results[], citation_block, summary_payload }
10. agent → writes prose summary with inline links + locators for the human
```

Latency: ~150 ms warm, ~250 ms cold single request; ~1–1.8 s worst case at ~4
concurrent cold. Numbers and the concurrency model in `HARDWARE.md` and
`MCP_ENGINE.md`.
