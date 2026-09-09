# HARDWARE.md — Vera

Resource budgets for the target box, the latency profile, and the path to scale. Design
for 2 vCPU / 8 GB; let bigger hardware benefit automatically.

---

## 1. Target profile

| | Target VPS | Notes |
|---|---|---|
| CPU | 2 vCPU | the real ceiling under concurrency |
| RAM | 8 GB | OOM-proof by construction (§2) |
| Disk | fast NVMe, ~1.3 TB | full 4096 halfvec for 100M rows |
| Corpus | ~100M chunks | across ~10K clusters of ~10K rows · **k = √N**, see `ARCHITECTURE.md` §2 |
| Embedding | 4096-dim, halfvec | full precision, no truncation |

The same image runs on local hardware (laptop/mini-PC) for personal use; more RAM/cores
simply raise the configured limits.

---

## 2. RAM budget (the OOM guarantee)

```
Peak RAM = fixed_costs + (concurrency_limit × per_request_ceiling)
```

Sequential cluster loading fixes `per_request_ceiling ≈ one cluster (~82 MB) + buffers`,
independent of how many clusters a query probes.

! **That ceiling is a bound, not a measurement, and the gap is four orders of
magnitude.** The streaming scan API hands the visitor one *row* against a reused buffer,
so nothing ever holds a cluster: measured peak RSS over a 903 MB corpus is **7 MB at
probe=1 and 7 MB at probe=20**. The 82 MB figure stays in the table because it is the
number the guarantee is *proved* against — a caller that chose to retain rows could reach
it, and the budget must survive that — but the table below overstates real use by ~4
orders of magnitude and should be read as a worst case, not a forecast.

| Component | RAM |
|---|---|
| OS + container runtime | ~0.8 GB |
| Rust engine (stateless) | ~0.2 GB |
| Layer-1/2 centroids (hot, mlock'd; 10K × 4096 halfvec) | ~0.08 GB |
| Postgres (shared_buffers ~1.5 GB + capped conns + work_mem) | ~2.0 GB |
| In-flight: 4 concurrent × ~120 MB | ~0.5 GB |
| **Peak** | **~3.6 GB** |
| **Free margin** | **~4 GB** |

Five bounds make it hold (all config, not magic numbers): concurrency semaphore,
bounded queue, capped Postgres memory, streaming sequential scan, fixed candidate
budget. Detail in `MCP_ENGINE.md` §4–5.

---

## 3. Disk budget (where the cost actually lives)

At full 4096, the cost moved from RAM to disk — accepted, since disk is cheap.

| Storage | 100M rows | Notes |
|---|---|---|
| halfvec 4096 (chosen) | ~0.82 TB vectors; ~1.3 TB total (+ table, BM25, overhead) | needs ~1.3 TB NVMe |
| fp32 4096 | ~1.64 TB vectors; ~2.5 TB total | not chosen |

Fast NVMe is **mandatory**, not optional: cold per-cluster reads (~82 MB each) are the
dominant tail-latency factor (§4).

---

## 4. Latency profile

Per-request, full 4096, ~5 clusters sequential, hybrid scoring, no model reranker:

| Stage | Warm | Cold |
|---|---|---|
| Embed query (OpenRouter) | ~120 ms | ~120 ms |
| Route (layer-1 + layer-2, hot) | ~3 ms | ~3 ms |
| 5 clusters sequential (scan + BM25) | ~50 ms | ~325 ms |
| Fuse + provenance | ~7 ms | ~7 ms |
| **Single-request total** | **~180 ms** | **~455 ms** |

! **This table is superseded by `METRICS.md` §2.1** and is kept for the shape only. It
bundles BM25 into the per-cluster scan line, which assumed the keyword half ran per
cluster; it does not (`ARCHITECTURE.md` §5), and as one global query it needs its own
budget. Measured, it is the **dominant** stage rather than a rounding error inside
another one, and `METRICS.md` §2.3 explains why that is structural rather than a
tuning problem.

Concurrency (2 cores; embedding overlaps, scans contend):

| Load | Total per request (worst-case cold) |
|---|---|
| 1 request | ~0.18–0.46 s |
| 5 concurrent | ~1.0–1.3 s |
| 10 concurrent | ~2–3 s (cores saturate) |

The embedding call (~120 ms) dominates a warm query, and it does **not** scale with
corpus size — so 100M is about as fast as 1M when warm. A flat scan of all 100M would
be ~20 s; routing buys the ~100× back.

**Levers:** prefetch the next cluster's read while scanning the current one (softens
cold cost); keep hot clusters cached (real-world latency sits between warm/cold columns
by query-distribution skew). For RAG feeding an agent, ~1–2 s for an accurate, cited,
100M-corpus retrieval is far faster than a human and entirely acceptable.

---

## 5. Scale path — where to grow, and in which direction

The VPS and a production cluster are the **same architecture at two sizes** — no
redesign. What changes with scale is only *which resource runs out first*.

### The four competing constraints

| # | Constraint | Type |
|---|---|---|
| 1 | one cluster must fit the per-request RAM ceiling | **hard** — this is the OOM guarantee (§2) |
| 2 | the hot centroid set must fit RAM | **hard** — the centroids *are* the index |
| 3 | query cost = centroids scanned + rows scanned | soft — a latency budget |
| 4 | the corpus must fit local disk | **hard** — and the one people forget |

At 100M all four are satisfied at 10K clusters × 10K rows. **That agreement is a coincidence of
scale, not a property of the design** (`ARCHITECTURE.md` §2). Away from it they
diverge, and the binding one dictates the move:

| Binding constraint | Response |
|---|---|
| 3 — query cost | re-cluster at k = √N |
| 1 — cluster too big for RAM | split-on-size; cap `f` below √N |
| 2 — too many centroids to hold or scan | **add a routing level** |
| 4 — corpus exceeds local disk | **shard across nodes** |

### Worked example: 10B rows from one source

√N says 100,000 clusters × 100,000 rows. Check it:

| | Value | Verdict |
|---|---|---|
| one cluster | 100K × 4096 halfvec = **819 MB** | ✗ 10× over constraint 1 |
| hot centroids | 100K × 8 KB = **819 MB** | ✗ over constraint 2 |

Forcing cluster size back to 10K rows gives 1,000,000 clusters — 8.2 GB of centroids
held hot and ~4×10⁹ flops just to scan them. Worse. **Two-level routing is finished at
10B.** But routing is not what fails first:

```
10B × 4096 dims × 2 bytes (halfvec)  =  82 TB of vectors alone
                    + text, BM25, overhead  ≈  130 TB
```

Even at 1024 dims that is **20 TB**, against the ~1.3 TB this profile budgets for 100M.
**Constraint 4 binds long before constraint 2 does.** At 10B you are sharding whether or
not you deepen, so the deepening question never arises.

### Deepen or shard

- **Deepen** (add a routing level) when constraint 2 binds and constraint 4 does not —
  roughly the 1B–3B range, where the corpus still fits local disk but the centroid set
  no longer fits RAM. Cost: recall compounds multiplicatively across levels
  (`ARCHITECTURE.md` §2).
- **Shard** when constraint 4 binds. Each shard runs the **unmodified two-level design**
  at ~100M — exactly the profile everything here is budgeted against.

Three properties make sharding nearly free, and they are the same three that make
multi-source work (`MULTI_DOMAIN.md`):

- **Clusters live inside a partition**, so shards never share a centroid space and need
  no coordination.
- **RRF is rank-based**, so fusing N independently-scored shards works exactly as well
  as fusing two lists inside one.
- **The engine is stateless**, so a shard is a replica pointed at different data.

A single source shards fine: by hash, or better by a metadata field (year, issuing body)
so that filters align with shard boundaries and a constrained query touches few shards.

### The other levers

- **More QPS → replicate the engine.** Stateless, only the small hot centroid set is
  resident and identical per replica, so N replicas behind a load balancer scale
  throughput linearly. This is the concurrency answer beyond ~4.
- **A cluster that outgrows flat scan → swappable leaf backend.** It graduates to its
  own on-disk ANN index while small clusters stay flat. The routing layers are
  unchanged; only what happens *inside* a leaf changes.
- **Cheapest lever on this whole table: the rescore pattern** (`EMBEDDING.md` §3, on
  record and not yet chosen). Index truncated 1024-dim vectors for candidate retrieval,
  keep full 4096 only to rescore finalists. At 10B that is 20 TB instead of 82 TB and
  shrinks every cluster 4× — it moves the sharding threshold by a factor of four and
  costs one extra pass over a few hundred candidates.

All scale limits are configuration. The 2 vCPU / 8 GB profile is the floor that proves
the design; nothing about it blocks the ceiling.
