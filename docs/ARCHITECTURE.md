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

The result is a retrieval server that holds ~14 MB resident while searching 355,621
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
                   factor scoring (relevance floor → metadata prior)
                            │
                   sibling expansion, opt-in     ← `expand: ["siblings"]`
                            │
                   [ viewpoints · consensus ]    ← DROPPED, ✗ pending (SCORING.md §5)
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

## 4. Why OOM is impossible · corrected against measurement 2026-09-19

**The engine's PER-REQUEST cost does not move.** Measured at `CLUSTER_BATCH`
1, 2 and 5: **14 / 12 / 14 MB** peak RSS from the container's own `memory.peak`
(`python dev_tools/eval/cluster_batch_sweep.py`). It does not vary with the
batch, with `CLUSTERS_PROBED`, or with load — at 5.1M rows the engine sits at
39 MB idle and 42 MB under a 12-way concurrent burst.

! It **does** vary with corpus size, which an earlier version of this section
denied. The centroids are held hot and `k ∝ n`, so 177 clusters at 355K become
2,478 at 5.1M and the engine's floor goes 9 MB → 39 MB *at startup*, before a
query. `HARDWARE.md` §2. The bound below is about the per-request term, which is
the one the loop in `search.rs` governs.

! This section used to state `peak RAM = fixed + (MAX_CONCURRENCY × one cluster)`,
putting a cluster-sized working set inside the engine and making 102 MB the budget.
That was **arithmetic, never measured**, and the measurement contradicts it: batch=5
would have had to hold five clusters, ~21 MB more than batch=1, and shows none of it.
The real figure is ~7× smaller than the one that was published.

### Where the memory actually is

The engine never receives a vector. Every arm returns ranked **addresses**:

```sql
SELECT id, 1.0 - (dense <=> $1::text::halfvec) AS sim
  FROM chunks WHERE indexable AND cluster_id = ANY($2)
 ORDER BY dense <=> $1::text::halfvec LIMIT $3      -- k × OVERFETCH rows
```

Postgres performs the scan and hands back at most a few hundred `(id, f64)` pairs.
The cluster-sized working set — median 4.2 MB, worst 23.4 MB — is **Postgres's page
working set**, bounded by `shared_buffers` and the database container's own limit.
It was never the engine's, and no engine setting governs it.

What the engine does hold is small and enumerable: the 177 centroids held hot
(177 × 1024 × f32 ≈ 725 KB), the candidate pool's metadata, and the snippets it is
about to return. None of those is a function of `n`.

```
engine   ≈ 14 MB at 355K, 42 MB at 5.1M · the DELTA is centroids (O(k), k ∝ n);
         the per-request term is flat because it never materialises the corpus
postgres  bounded by shared_buffers + work_mem + its container limit
```

! The database side is **unmeasured per setting**. `memory.peak` is cumulative from
container start and this one had served a full day's work, so an honest figure needs
a restart per row. Until that exists, treat Postgres's bound as configured rather
than as demonstrated.

### The corpus-independence claim survives, and it is the important one

A cluster is not a fixed fraction of the corpus, it is a target number of rows —
`k = max(2, n // rows_per_cluster)`, default 2,000
(`dev_tools/cluster_maint/kmeans.py:117`). `k` is proportional to `n`, so a corpus ten
times larger is served by ten times as many clusters of about the same size, not by
clusters ten times bigger. Ten times the corpus is ten times the disk, the same
per-query scan, and the same engine.

That is still the whole reason §3 refuses a global ANN index — the argument simply
relocates. An HNSW graph is one structure over all `n` rows and must be resident to
beat a scan; resident **in Postgres**, which is the constrained process here (1 GB on
the target profile), competing directly with the page cache every arm depends on.
Vera trades a scan it can bound for an index it cannot.

! `k ∝ n` sets the *mean*, ✗ the maximum, and k-means does not divide evenly. The
largest cluster measured is 11,448 rows against a 2,000 target — 5.6×. The bound
survives a growing corpus only while that skew ratio does not also grow, which is
unmeasured beyond the 355,621 chunks in hand. `routing_recall.py` is what would show
it drifting.

### What `CLUSTER_BATCH` is for

Not memory. It is a **latency knob**, and it should be read as nothing else.

One `WHERE cluster_id = ANY($1)` over all five probed clusters is 3× faster than five
separate statements at the arm level — 81 ms against 268 ms. End to end that is worth
**−182 ms p50** at `CLUSTER_BATCH=5` (§`HARDWARE.md` 6a), which matches the 187 ms the
arm-level figure predicts and is 3.1% of a 5,908 ms request on the sub-1 GB profile.

It stays configurable for the reason it was introduced: hardcoding the window at 1
hardcodes a limit, which `CLAUDE.md` §7.12 forbids. It defaults to 1 because that is
the shape the rest of this document describes and the one every published number was
taken at.

! It is safe to raise. The earlier draft of this section told operators that raising
`CLUSTER_BATCH` and raising the container's memory limit were "one decision, not two".
Measured, that is false — the engine's peak is flat across the range tested. What is
true is that a wider window asks Postgres for more rows in one statement, and that
side has not been measured.


---

## 5. Three arms, fused over ranks

| Arm | Mechanism | Scope | Finds |
|---|---|---|---|
| dense | `halfvec` cosine, flat scan | 5 routed clusters | paraphrase, concept |
| sparse | `sparsevec` BM25 | global | term overlap, weighted by IDF |
| text | `tsvector`, ordered by a RUM index | global | exact wording, phrases |

! The **sparse arm has two implementations**, chosen by what the database has.
With `pg_search` and migration 0004 it is Tantivy's BM25 served from an index;
without them it is the original `sparsevec` scan, which has no index and walks
every row. `SearchOps::has_bm25` probes once and caches, the same contract RUM
has in the text arm.

Measured on this corpus: **86 ms against 170 ms at 355K, and 1,858 ms against
6,282 ms at 5M** — the fallback is the worst-scaling thing in the system
(`HARDWARE.md` §6). Fused recall is identical at @20 and @50 and 2.5 points
better at @5, so the index is faster **and** not worse
(`dev_tools/eval/fused_bm25.py`).

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
│  • windowed scan   │     │  • sparsevec BM25  │     │                    │
│  • RRF fusion      │     │  • tsvector + RUM  │     │                    │
│  • concurrency     │     │  • provenance      │     │                    │
│  14 MB @ 355K      │     │  read-only at qry  │     │  378 MB (bf16)     │
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
5. engine → for each window of CLUSTER_BATCH:   [load, scan, keep, drop]
6. engine → sparse arm, global                  [sparsevec BM25]
7. engine → text arm, global                    [tsvector, RUM-ordered]
8. engine → exact-identifier arm, global        [only if the query names one]
9. engine → RRF over ranks → top-k
10. engine → attach provenance captured at ingest
11. agent  → writes prose, appends the citation block
```

Measured p50 with the whole stack pinned to 2 cores and `DTYPE=float32`:
**2,998 ms at 355K**. The dtype is the largest single term — `bfloat16` makes the
same queries 10,251 ms, because PyTorch has no optimised narrow-dtype kernels on
CPU. `HARDWARE.md` §3 carries the profile with every figure; a latency number
without one means nothing here.
