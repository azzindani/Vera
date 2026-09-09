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
          │  halfvec distance only              │  scan, keep top-50, drop, next
          └─────────────────┬──────────────────┘
                            │
                   RRF fusion of three inputs:
                     • dense      ← the routed clusters above
                     • keyword    ← ONE global BM25 query   (§5, never per-cluster)
                     • exact-id   ← global lookup, run BEFORE layer 1 (§8)
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
- **Layer 3 (leaf)** is a flat **vector** scan within each probed cluster. ~10K rows ×
  4096 is small enough that flat scan is fast; routing is what keeps the scanned set
  small. ! The keyword half does **not** live here — it is one global query, for the
  reasons in §5.

### Sizing the layers · **k = √N**

The "~10K clusters of ~10K rows" above is the **100M design point**, ✗ a universal
target. A routed query pays two costs — comparing the query against every centroid, and
scanning the clusters it probes:

```
cost(k) = k  +  nprobe · N/k        minimised at  k = √(nprobe · N)
```

With `nprobe` a small constant that is the familiar **√N** rule, and √100M = 10,000 —
which is why both numbers read "10K" at the design point. **They coincide only there.**

| Corpus | k = √N | rows/cluster | rows scanned at nprobe=5 |
|---|---|---|---|
| 200K | 447 | ~447 | ~2,200  (1.1%) |
| 750K | 866 | ~866 | ~4,300  (0.6%) |
| 100M | 10,000 | 10,000 | 50,000  (0.05%) |

! Reading "10K rows per cluster" as a target below 100M under-clusters badly: at 200K it
yields 20 clusters, so probing 5 scans a **quarter** of the corpus and routing prunes
about 4×. The benchmark was built that way, and its speedup numbers described the
mis-clustering rather than the design.

Cluster size is bounded from the other side too — one cluster is the per-request RAM
ceiling — so a corpus large enough that √N rows exceeds that ceiling takes the smaller
of the two. Split-on-size (`CLUSTER_MAINTENANCE.md` §2) is what enforces it.

### n layers, not two

Two routing levels is what **100M** happens to need. The design is **n-layer**, and the
number is *derived from the corpus*, never chosen.

The rule is recursive and fits in one sentence:

> **If a level is too big to scan linearly, route it.**

Layer 2 exists because 100M rows cannot be scanned. A layer 2½ would exist for exactly
the same reason one level up: a million centroids cannot be scanned either. Same rule
applied to itself.

With fanout `f` per level and `R` routing levels above the leaves:

```
N = f^(R+1)        →        R = log_f(N) − 1
```

`f` is bounded by the per-request RAM ceiling (one cluster must fit). `R` then falls
out. Because each level multiplies capacity by `f`, **R grows extremely slowly**:

| routing levels | max corpus at f = 10K |
|---|---|
| 1 | **100M** ← the design point |
| 2 | **1 trillion** |
| 3 | 10¹⁶ |

One extra level takes 100M to 1T. So n-layer in principle, **n ∈ {2, 3} in practice**,
and past that the answer is to shard rather than to deepen (`HARDWARE.md` §5).

Two constraints on going deeper:

- ! **Recall compounds multiplicatively.** Each level is a probabilistic prune, so
  per-level recall `r` gives `r^R` end to end. At a healthy 90% per level, three levels
  is ~73%; at the ~60% measured on the synthetic corpus it is ~22%. The compensation —
  probing wider at every level — eats the savings that motivated the extra layer. The
  per-level figure is exactly what the eval set exists to establish (`EVAL.md` §3).
- **Layer 1 is not part of this recursion.** Domain/source detection selects *which
  corpus*, not which partition of one: a different decision with a different failure
  mode (`MULTI_DOMAIN.md` §2). The recursion lives entirely in layers 2 and below.

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
| Dense | Qwen3 vectors, halfvec, per-cluster flat scan | paraphrase, concept | **routed clusters** |
| Keyword | Postgres FTS / BM25 (tsvector + GIN, or pg_search) | exact terms, IDs, citations | **always global** |

! **The keyword half is global, always — never scoped to the probed clusters.** The two
halves are asymmetric on purpose, and the reason is the same one that motivates routing
in the first place:

- Dense retrieval **has no index** (pgvector cannot index 4096 dims, §3), so the only
  way to avoid touching 100M rows is to not look at them. Routing is what makes it
  affordable.
- BM25 **is an index.** An inverted index already prunes to the matching postings; it
  does not need routing and gains nothing from it.

Scoping BM25 per cluster is therefore worse on both axes. It is **slower**, because the
engine evaluates the match corpus-wide and then discards rows whose `cluster_id` is
wrong — so probing N clusters costs N full-corpus keyword scans, and the cost grows with
`clusters_probed`, the dial that is meant to be cheap. Measured on the benchmark corpus
it was **87% of total query time**. And it is **less accurate**, because a document the
keyword half would have found is thrown away for sitting in an unprobed cluster: going
global raised recall@10 at probe=1 from 61.9% to 99.5%.

One global query per search, fused with the routed dense candidates.

Fusion is **Reciprocal Rank Fusion (RRF)** — model-free, robust, no reranker. It
consumes **ranks, not scores**, which is what lets a cosine in [-1,1] and a BM25 in
[0,∞) be combined without a per-corpus tuning constant. The same property is why RRF
fuses across sources (`MULTI_DOMAIN.md` §8) and across shards (`HARDWARE.md` §5).

On top of both halves sits the **exact-identifier path**: regulation numbers extracted
from the query text are looked up globally *before any routing runs*, so a named
regulation can never be lost to a routing miss (`LOOPHOLES.md` §1). It is a third input
to the fusion, not a mode of the keyword half.

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
1.  agent  → search(query="...")                  [query only; no domain/source]
2.  engine → OpenRouter.embed(query)              [pinned provider, ~120ms]

    ── nothing below this line may gate step 3 ──────────────────────────────
3.  engine → extract identifiers from query text  [<1ms]
4.  engine → GLOBAL exact-identifier lookup       [routing bypass; LOOPHOLES §1]
    ─────────────────────────────────────────────────────────────────────────

5.  engine → layer-1: query vs source anchors     [<1ms, hot]
              no match → return steps 3-4 only, honestly empty otherwise
6.  engine → layer-2: k nearest centroids         [~2ms, hot]
7.  engine → for each probed cluster, SEQUENTIALLY:
              stream rows → cosine → keep top-50 → drop cluster → next
8.  engine → ONE global BM25 query                [§5; not per cluster]
9.  engine → RRF fuse: dense ⊕ keyword ⊕ exact    → top-k
10. engine → attach provenance (source_url + page/section) from stored fields
11. engine → return { detected_domain, results[], citation_block, summary_payload }
12. agent  → writes prose summary with inline links + locators for the human
```

! Steps 3–4 run **before** layer-1 detection, not after. Layer-1 is routing, and an
exact identifier must never be gated by routing: a query naming `UU 28/2007` whose
vector falls below the anchor threshold must still return that regulation. Ordering
these the other way produces `success: true` with an empty result set, which the caller
cannot distinguish from "no such regulation exists".

Latency: ~150 ms warm, ~250 ms cold single request; ~1–1.8 s worst case at ~4
concurrent cold. Numbers and the concurrency model in `HARDWARE.md` and
`MCP_ENGINE.md`.
