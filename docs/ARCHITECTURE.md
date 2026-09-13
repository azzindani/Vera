# ARCHITECTURE.md

How Vera is built. The tool contract is in `MCP_ENGINE.md`; the response schema in
`OUTPUT_CONTRACT.md`; measured numbers in `HARDWARE.md`.

---

## 1. The problem

Retrieval over a large corpus is normally either accurate and heavy — a global ANN
index that has to be resident — or light and shallow. Vera replaces the global index
with **routing**: the intelligence is in knowing where *not* to look.

And retrieval alone does not rank legal text correctly. Text similarity can say a
chunk addresses the query; it cannot say whether the instrument binds, whether it
is still in force, or whether the passage is an operative clause or an annex. That
is what the factor model in `SCORING.md` is for. **Routing decides what to look at;
factors decide what matters.**

The result is a retrieval server that holds 8.5 MB resident while searching 355,621
chunks, and whose peak memory is a function of its concurrency ceiling rather than of
corpus size.

Vera is exposed as an MCP server so an agent treats retrieval as a callable
capability, not a pipeline baked into an application.

---

## 2. The three routing layers

```
                         query (text)
                            │
                   embed  (the corpus's own model and instruction)
                            │
          ┌─────────────────┴──────────────────┐
LAYER 1   │  DOMAIN gate                        │  centroid similarity + lexical
          │  (engine-owned; agent passes none)  │  evidence → serve, or refuse
          └─────────────────┬──────────────────┘
                            │
          ┌─────────────────┴──────────────────┐
LAYER 2   │  CLUSTER routing                    │  query vs k-means centroids,
          │  (centroids held HOT in the engine) │  held in memory → pick 5
          └─────────────────┬──────────────────┘
                            │
          ┌─────────────────┴──────────────────┐
LAYER 3   │  LEAF search                        │  load ONE cluster, scan it,
          │  halfvec flat scan, per cluster     │  keep top-k, drop it, next
          └─────────────────┬──────────────────┘
                            │
        ⊕ sparse arm (global sparsevec)   ⊕ text arm (global tsvector/RUM)
                            │
                   RRF fusion over RANKS
                            │
                   + global exact-identifier hits (routing bypassed)
                            │
                   [ factor scoring · viewpoints · consensus ]   ← designed, ✗ built
                            │
                   top-k results + provenance  →  agent summarizes
```

**Layer 1** is a gate, not a selector: there is one domain today, so its job is to
decide whether the query belongs to this corpus at all. Two independent signals, both
load-bearing — centroid similarity and lexical IDF-mass. Below either floor the engine
returns `detected_domain: null` and an empty result set rather than a confidently
wrong answer. The scaffold means adding a second domain is configuration, not a
redesign.

**Layer 2** is a coarse quantizer: conceptually IVF, with clusters as inverted lists
and centroids as the index. It is implemented here rather than delegated to pgvector's
IVFFlat because the centroids belong in the engine, where routing can be explained
(`explain_routing`) and where probe width is a request-time dial.

**Layer 3** is a flat scan inside each probed cluster. Clusters are held near 2,000
rows, which is small enough that a flat scan is fast; routing is what keeps the scanned
set small.

`CLUSTERS_PROBED` is the central dial. It trades recall against latency — and
notably **not** against RAM, because of §4.

---

## 3. Why there is no global ANN index

Routing already does the pruning an index would do. A query scans five clusters, not
the corpus; measured, that is 81 ms against 637 ms for the same dense arm run over
everything — an 8× saving that an index would have to beat while also being resident.

So Vera stores `halfvec` columns, partitions by cluster, and scans per partition with
pgvector's distance operators and no ANN index. The graph that would otherwise have to
live in memory never exists.

---

## 4. Sequential loading is the OOM guarantee

Probed clusters are loaded **one at a time**: load, scan, keep top-k, drop, load next.
A request's working set therefore never exceeds one cluster — median 4.2 MB, worst
23.4 MB measured — regardless of whether it probes 5 clusters or 50.

```
Peak RAM = fixed cost + (MAX_CONCURRENCY × one cluster)
```

Both terms are bounded, so the total is. This is the single most important
implementation rule in the system.

! It is not free. One `WHERE cluster_id = ANY($1)` is 3× faster — 81 ms against 268
ms. That 187 ms is the price of the bound, paid deliberately.

---

## 5. Three arms, fused over ranks

| Arm | Mechanism | Scope | Finds |
|---|---|---|---|
| dense | `halfvec` cosine, flat scan | 5 routed clusters | paraphrase, concept |
| sparse | `sparsevec` BM25 | global | term overlap, weighted by IDF |
| text | `tsvector`, ordered by a RUM index | global | exact wording, phrases |

Fusion is **Reciprocal Rank Fusion over ranks, never scores** (`crates/engine/src/fusion.rs`).
Scores from three arms are not commensurable — a cosine of 0.83 and a BM25 of 14.2 have
no shared unit — and normalizing them invents one. Ranks are the only thing the three
arms genuinely agree on.

No model reranker. Adding one would put a model back on the query path, which is the
dependency this design exists to avoid.

! Fusion produces a **candidate pool**, ✗ a final ranking — see `SCORING.md`. The
three arms answer one question between them ("does this text address the query?"),
and ordering legal results needs four more: how binding the instrument is, whether
it is current, whether the passage is operative, and whether the provision is
whole. Those come from metadata the corpus already carries.

**The exact-identifier path bypasses routing entirely.** A query naming a regulation
number is searched against the whole corpus, because semantic routing may miss and a
known identifier must never be silently lost. These hits are reported separately as
`exact_matches` so the agent can tell a high-trust global hit from a routed one.

---

## 6. Three containers

```
┌────────────────────┐     ┌────────────────────┐     ┌────────────────────┐
│  vera-mcp          │     │  vera-db           │     │  vera-embed        │
│  Rust, stateless   │────▶│  Postgres 16       │     │  one model, /embed │
│                    │ SQL │  + pgvector + RUM  │     │  no completions    │
│  • MCP: stdio/http │     │                    │◀────│                    │
│  • centroids HOT   │     │  • halfvec dense   │     │                    │
│  • sequential scan │     │  • sparsevec BM25  │     │                    │
│  • RRF fusion      │     │  • tsvector + RUM  │     │                    │
│  • concurrency     │     │  • provenance      │     │                    │
│  8.5 MB resident   │     │  read-only at qry  │     │  1,088 MB          │
│  read-only rootfs  │     │                    │     │                    │
└────────────────────┘     └────────────────────┘     └────────────────────┘
   replicate for QPS          cores + disk for size       shared by replicas
```

The engine is stateless — it holds only the centroids, identical on every replica — so
throughput scales by running more of it. The database holds all state. Single-box and
distributed are the same architecture at two sizes.

---

## 7. The query path is read-only

No writes, no snapshots, no mutation while serving. The engine container runs with a
read-only root filesystem and all capabilities dropped, because there is nothing it
legitimately needs to write.

All writing happens in `dev_tools/`, offline, on the operator's own hardware:
embedding a corpus and maintaining clusters. Those publish through version swaps; the
live engine never sees a half-updated index.

---

## 8. End-to-end trace

```
1. agent  → search_knowledge(query)            [query only; no domain argument]
2. engine → embed(query)                        [corpus's own model + instruction]
3. engine → layer-1 gate: centroid + lexical    [refuse here, or continue]
4. engine → layer-2: 5 nearest centroids        [hot, in memory]
5. engine → for each cluster, sequentially:     [load, scan, keep, drop]
6. engine → sparse arm, global                  [sparsevec BM25]
7. engine → text arm, global                    [tsvector, RUM-ordered]
8. engine → exact-identifier arm, global        [only if the query names one]
9. engine → RRF over ranks → top-k
10. engine → attach provenance captured at ingest
11. agent  → writes prose, appends the citation block
```

Measured p50 on the target profile: 1,059 ms. Where it goes is in `HARDWARE.md` §3.
