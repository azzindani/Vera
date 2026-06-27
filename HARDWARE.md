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
| Corpus | ~100M chunks | across ~10K clusters of ~10K rows |
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

## 5. Scale path (production, not the VPS)

The VPS and a production cluster are the **same architecture at two sizes** — no
redesign.

- **More QPS → replicate the engine.** It is stateless (only the small hot centroid set
  is resident, identical per replica), so N replicas behind a load balancer scale
  throughput linearly. This is the concurrency answer beyond ~4.
- **Bigger corpus → shard the DB by cluster-range across storage nodes.** The engine's
  router knows which node owns a routed cluster and scatter-gathers. Adding capacity =
  adding nodes and reassigning ranges.
- **Bigger clusters (100K/1M) → swappable per-cluster backend.** A cluster that
  outgrows flat scan graduates to its own on-disk ANN index (hnswlib/qdrant) while small
  clusters stay flat. The routing layer is unchanged; only what happens *inside* a leaf
  changes.

All scale limits are configuration. The 2 vCPU / 8 GB profile is the floor that proves
the design; nothing about it blocks the ceiling.
