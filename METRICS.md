# METRICS.md — Vera

What "working" means, as numbers. Every target here is **falsifiable**: it has a value,
a way to measure it, and a source that justifies it.

This document exists so that "is it fast enough?" and "is it accurate enough?" stop
being judgement calls. If a change moves a number the wrong way, the change is wrong —
not the number.

---

## 0. How to read this

Three status marks:

| Mark | Meaning |
|---|---|
| ✅ | measured, meets target |
| ⚠️ | measured, does **not** meet target — or meets it only under conditions that will not hold |
| ❌ | not measurable yet · the blocker is named |

! **Almost every number measured so far comes from a synthetic corpus on 4 vCPU / 16 GB.**
The target profile is 2 vCPU / 8 GB (`HARDWARE.md` §1) and the target corpus is real
Indonesian regulation text. A green mark against a synthetic corpus on double the
hardware is a *sanity check*, ✗ a pass. Every such row is marked.

Reproduce with:

```bash
vera-bench synth --out corpus.db --rows N --dim D     # build a fixture
vera-ingest inspect --source real.db                  # characterise real data
vera-ingest import  --source real.db --out corpus.db  # build from real data
vera-bench info --corpus corpus.db                    # RAM budget
vera-bench run  --corpus corpus.db --probe 1,2,5,10   # the sweep
```

---

## 1. The three tiers

Targets differ by what is being proven.

| Tier | Corpus | Hardware | Proves |
|---|---|---|---|
| **T1 · Prove it works** | 200K–750K real rows, 1024-dim | anything | retrieval is correct and routing pays on *real* text |
| **T2 · Prove the design** | 100M rows, 4096-dim halfvec | **2 vCPU / 8 GB** | the architecture's actual claim |
| **T3 · Prove it scales** | 1B+ | sharded | the scale path (`HARDWARE.md` §5) |

**T1 is the current goal.** T2 is the number the whole design is justified by. T3 is not
work yet.

---

## 2. Latency

### 2.1 Per-query budget (T2: 100M, probe=5, 2 vCPU / 8 GB)

! The budget below **adds a line the existing docs do not have**. `HARDWARE.md` §4
bundles BM25 into "5 clusters sequential (scan + BM25) — 50 ms", which assumed the
keyword half ran per cluster. It does not (`ARCHITECTURE.md` §5): it is one global
query, and it needs its own budget.

| Stage | Target (warm) | Target (cold) | Basis |
|---|---|---|---|
| Embed query (OpenRouter) | 120 ms | 120 ms | network; not ours to optimise |
| Exact-identifier lookup | ≤ 2 ms | ≤ 5 ms | indexed equality on one column |
| Route L1 + L2 | ≤ 5 ms | ≤ 5 ms | 10K centroids × 4096 = 41 MFLOP, hot |
| Dense scan, 5 × 10K rows | ≤ 50 ms | ≤ 325 ms | 410 MB at ~10 GB/s = 41 ms; bandwidth-bound |
| **Global BM25** | **≤ 40 ms** | **≤ 100 ms** | **new line · unvalidated, see §2.3** |
| RRF fuse + hydrate | ≤ 10 ms | ≤ 15 ms | ~250 candidates |
| **Engine total** | **≤ 110 ms** | **≤ 450 ms** | |
| **End to end** | **≤ 230 ms** | **≤ 570 ms** | includes embed |

The old headline of ~180 ms warm assumed BM25 was free. **Either the target moves to
~230 ms, or BM25 must fit inside 10 ms** — a decision to take once §2.3 is measured, not
before.

### 2.2 The per-row scan cost — the one hard engineering target

| | Value |
|---|---|
| Target | **≤ 0.5 µs/row** (bandwidth-bound) |
| Measured | **5.6 µs/row** ⚠️ |
| Gap | **11×** |

At 100M with probe=5 the scan touches 50K rows. At 5.6 µs that is **280 ms** — over
budget on its own, before BM25 or embedding. At 0.5 µs it is 25 ms.

The 5.6 µs is ~99% per-row SQLite step overhead; the dot product itself is ~0.3 µs. The
fix is the one `ARCHITECTURE.md` §4 already describes but the storage layer does not
implement: **one contiguous blob per cluster**, so a probe is one fetch plus an
in-memory scan rather than 10K row fetches. Combined with halfvec (§5) this is the
single largest lever on T2 latency.

### 2.3 ⚠️ The biggest unvalidated assumption

**Global BM25 latency at 100M is unknown, and it is now the dominant stage.**

Measured 291 ms on a *200K-row* corpus — 72% of query time. But that fixture has a
**50-word vocabulary**, so an 8-term OR matches a large fraction of the corpus. Real
text is Zipfian and most query terms are selective, so the true figure could be far
lower — or, at 500× the rows, far higher.

Nothing else in this document is worth optimising until this is measured on real text.
If BM25 at 100M cannot be made to fit ~100 ms, the hybrid design needs rethinking (term
capping, selectivity-aware query planning, or a different keyword backend), and that is
a bigger change than anything else listed here.

### 2.4 Under concurrency (T2)

| Load | Target p50 | Source |
|---|---|---|
| 1 request | ≤ 0.23 s warm / ≤ 0.57 s cold | §2.1 |
| 4 concurrent | ≤ 1.3 s | `HARDWARE.md` §4 |
| 10 concurrent | degrade **gracefully**, ✗ crash | queue + 503, `MCP_ENGINE.md` §4 |

Throughput at the ceiling: 4 concurrent ÷ 0.23 s ≈ **17 QPS per replica**. Beyond that,
replicate (stateless).

---

## 3. Retrieval quality

`EVAL.md` §3 defines these. Ordered by how much a regression matters.

| Metric | Target | Measured | Status |
|---|---|---|---|
| **Exact-match recall** (identifier queries) | **100%** — no exceptions | not measured | ❌ needs labelled set |
| **Recall@10** vs exhaustive | ≥ 95% at probe=5 | 99.8% | ✅ *synthetic* |
| **Routing recall** (route/D) at probe=5 | ≥ 95% | 99.8% | ✅ *synthetic* |
| **Layer-1 false rejection** | **≤ 1%** | 0% | ✅ *synthetic* |
| **top-1 agreement** | ≥ 95% | 100% | ✅ *synthetic* |
| **MRR** | ≥ 0.90 | 1.000 | ✅ *synthetic* |
| **nDCG@10** | ≥ 0.85 | not implemented | ❌ |

! **Exact-match recall is the only one with no tolerance.** A named regulation that
exists and is not returned is the failure this whole architecture is shaped around
(`LOOPHOLES.md` §1). 99% is not a pass.

! **Layer-1 false rejection is a silent failure.** A rejected query returns
`success: true` with zero results, indistinguishable from an empty corpus. It is
therefore capped tighter than recall, and measured separately (`L1-rej` in the sweep)
rather than folded into a recall figure where it would hide.

**The knee.** `clusters_probed` should be the smallest value holding recall@10 ≥ 95%.
Currently **5**, which matches the documented default. Re-derive this on real data — it
is a per-corpus property, not a constant.

---

## 4. Memory

| Metric | Target | Measured | Status |
|---|---|---|---|
| Peak RSS, engine | ≤ 1.0 GB | 7 MB | ✅ *synthetic, 200K* |
| Peak RSS **independent of `clusters_probed`** | required | 7 MB at probe=1 **and** probe=20 | ✅ |
| Per-request ceiling (T2) | ≤ 120 MB | 4.4 MB at 200K | ✅ *but see below* |
| Total peak incl. Postgres (T2) | ≤ 3.6 GB of 8 GB | not measured | ❌ needs T2 |
| OOM under sustained concurrency | **never** | not measured | ❌ needs T2 |

! The per-request ceiling is set by the **largest** cluster, not the average — so
cluster-size skew is a memory risk, not just a latency one. Measured spread at √N is
2.4× (p50 395, max 1085). **Target: max ≤ 2× mean**, enforced by split-on-size
(`CLUSTER_MAINTENANCE.md` §2, not yet built).

---

## 5. Disk and index build

| Metric | Target | Basis |
|---|---|---|
| Vector storage, T2 | ~0.82 TB (100M × 4096 **halfvec**) | `HARDWARE.md` §3 |
| Total incl. text + BM25, T2 | ~1.3 TB | `HARDWARE.md` §3 |
| Storage format | **halfvec (16-bit)** | ⚠️ currently f32 — **2× the bytes and 2× the scan bandwidth** |
| Index build (k-means), 200K × 1024 | reference: 318 s at k=447 | measured |
| Re-cluster, T2 | ≤ 24 h on GPU | `CLUSTER_MAINTENANCE.md` tier 3 |
| Ingest throughput | not yet targeted | needs real pipeline |

Halfvec is listed under §2.2 as a latency lever as much as a storage one: the leaf scan
is bandwidth-bound, so halving the bytes halves the dominant cost.

---

## 6. Invariant gates — pass/fail, never traded off

These are **not** metrics. They do not have targets to approach; they hold or the build
is broken. Each is guarded by a test.

| Invariant | Guard |
|---|---|
| Engine never calls an LLM | source scan over query-path crates |
| Query and corpus share an embedding space, or startup fails | `assert_matches`, fails closed |
| Exact identifiers are never gated by routing | `routing_never_gates_identifiers.rs` |
| Provenance is never synthesised | derived from stored fields only |
| Query path never writes | no write method exists on `ChunkStore` |
| Peak RAM independent of `clusters_probed` | streaming visitor API + measured RSS |
| An unrecognised constraint errors, never silently drops | tool-surface test |
| The tool surface stays at 4 primitives | schema test |
| `search` accepts intent, never strategy | schema test rejects `rankers`/`weights`/`source` |
| stdout carries only JSON-RPC frames | stdio discipline |

---

## 7. Status summary

| Area | T1 (real 750K) | T2 (100M on 2/8) |
|---|---|---|
| Latency | ❌ no real corpus | ⚠️ scan 11× over budget; BM25 unbudgeted |
| Recall | ❌ no labelled set | ❌ |
| Memory | ✅ synthetic | ❌ not measured |
| Disk | — | ⚠️ f32 not halfvec |
| Invariants | ✅ all guarded | ✅ |

---

## 8. What blocks each unmeasured number

Ordered by what unblocks the most.

1. **A real corpus** → unblocks every latency and recall figure, and settles §2.3, the
   single biggest risk in the document. *Blocked on: the dataset.*
2. **A labelled eval set** (50–100 queries with expert-confirmed answers, `EVAL.md` §2)
   → unblocks exact-match recall, nDCG, and the real knee. Without it, recall is
   measured against an exhaustive scan of the same flawed retrieval, which cannot detect
   a systematic error. *Blocked on: domain expertise, not engineering.*
3. **Contiguous per-cluster storage + halfvec** → the only path to the §2.2 target.
   *Blocked on: nothing. This is the next engineering task, and it is a schema change,
   so it belongs with the source/metadata work (`MULTI_DOMAIN.md` §12).*
4. **A 2 vCPU / 8 GB box** → unblocks every T2 resource number. *Blocked on:
   provisioning. Constrained-cgroup runs are a partial substitute.*

! Note the ordering: **(3) is the only one not blocked on something external**, and it
is also the largest single lever. But it is a schema change, and schema changes are paid
for in re-ingest — so it should land *with* the source-model work, not before it.
