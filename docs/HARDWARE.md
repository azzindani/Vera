# HARDWARE.md — budgets, and what they are measured against

> Every number here was measured, on the corpus named below, with the commands
> given. Where something is budgeted rather than measured it says so. A budget
> nobody has tested is a guess with a table around it.

**Corpus under test:** `spike-02` · 355,621 indexable chunks · 1024 dims
(Qwen3-Embedding-0.6B, last-token pooling) · 177 clusters · 2,417 MB database.

---

## 1. The target

**2 vCPU / 4 GB.**

Tightened from 2 vCPU / 8 GB. The 8 GB figure predated any measurement; the
numbers in §2 say the working set does not need it.

! **The query path cannot run CPU-only today, and the reason is not the
hardware.** The default embedding image pins CUDA
(`text-embeddings-inference:86-1.7.2`, compute 8.6). Three CPU options have now
been measured:

| CPU option | Runs? | Reproduces the corpus space? |
|---|---|---|
| TEI `cpu-1.7.2` | **no** — segfaults (exit 139) at 1.25/4/6 GB, pinned and unpinned. ORT declines last-token pooling, candle takes over, MKL's SGEMM faults. | — |
| TEI `cpu-1.8.2` / `cpu-1.9.3` | **yes** — the fault above is fixed upstream. Starts and serves on 2 pinned cores. | **no** |
| transformers on CPU (`docker/embed_cpu`) | **yes** — 2.49 GiB resident, p50 202 ms at 2 threads | **no** |

! **The blocker is the vector space, ✗ memory.** Both working CPU options
produce vectors that are near-ORTHOGONAL to the stored corpus — and produce
**byte-identical vectors to each other**, agreeing to five decimals across five
chunks:

| chunk | TEI cpu-1.9.3 | `docker/embed_cpu` | required |
|---|---|---|---|
| `0` | 0.19040 | 0.19040 | ≥ 0.98 |
| `1#0` | 0.11873 | 0.11873 | ≥ 0.98 |
| `1#2` | 0.00190 | 0.00190 | ≥ 0.98 |

Two independent implementations agreeing with each other and disagreeing with
the corpus says the **corpus** is the outlier. `86-1.7.2` reproduces it at
0.99998; nothing else tried does. See `EMBEDDING.md` §5 — this is that defect
measured from the other side, and it means the corpus is welded to one pinned
image, on CPU **and on GPU**.

An earlier version of this section said the transformers path "works" and that
"the gap is memory, not speed". It runs, and it cannot serve this corpus at any
memory budget. Quantizing to fit 1,280 MB would not have helped.

Corpus embedding on CPU is separately not viable at any budget — 1,364 ms/chunk
is 135 hours for 355K chunks.

Everything else in this document is measured on the target profile and holds
without a GPU. This is the one component that does not.

---

## 2. Memory

### The whole stack, cold start to steady state

Re-measured 2026-09-20 with `python -X utf8 dev_tools/eval/hardware_profile.py`,
which stops the stack, starts it, drives it through named phases and samples
every container on an interval. Everything pinned to **2 cores**, engine capped
at 512 MB, embedder at 1,280 MB, Postgres at 6,336 MB — an 8 GB box.

Run at **both corpus sizes**, same hardware, back to back:

| phase | db | embed | engine | stack | | db | embed | engine | stack |
|---|---|---|---|---|---|---|---|---|---|
| | **355K** | | | | | **5.1M** | | | |
| all stopped | 0 | 0 | 0 | **0** | | 0 | 0 | 0 | **0** |
| starting | 60 | 374 | 6 | 440 | | 94 | 365 | 7 | 466 |
| idle after start | 63 | 376 | **9** | 448 | | 102 | 366 | **39** | 507 |
| first query (cold) | 64 | 375 | 9 | 447 | | 1,518 | 366 | 39 | 1,922 |
| 44 sequential | 1,713 | 376 | 13 | 2,102 | | 5,907 | 367 | 41 | **6,316** |
| 12 concurrent | 1,800 | 392 | **14** | **2,206** | | 5,713 | 378 | **42** | 6,133 |
| idle after load | 1,846 | 393 | 14 | 2,253 | | 4,946 | 378 | 40 | 5,364 |

! **The engine is NOT flat in corpus size, and this document said it was.**
14 MB → 42 MB for 14× the rows, and the jump is at **startup** — 9 MB → 39 MB
before a single query is served. The cause is the **centroid table held hot**:
177 clusters become 2,478, and `k ∝ n` by construction
(`cluster_maint/kmeans.py:117`). §6 already called routing "linear in `k`"; that
applies to memory as well as to CPU, and the earlier claim of "~13 MB flat,
independent of corpus size" is **withdrawn**.

What *is* flat is the **per-request** working set: the engine's figure barely
moves between idle and a 12-way concurrent burst at either size (39 → 42 MB at
5.1M). The arms still return `(id, score)` rows under a `LIMIT` and no vector
enters the process. The fixed cost is O(k); the variable cost is not.

Extrapolated: ~50,000 clusters at 100M is ~200 MB of centroids as f32, so the
engine stays inside its 512 MB limit — but it is a term, ✗ a constant, and
hierarchical routing (§6) would remove it.

**db is the ceiling, not the engine.** At 5.1M Postgres peaks at **5,907 MB of a
6,336 MB cap** — it wants every byte of cache available. That is the number that
decides how large a corpus a given box serves.

**The embedder is genuinely flat**: 393 MB against 378 MB across a 14× corpus.
It does not touch the corpus at all.

**Nothing leaks.** Idle-after-load is at or below peak-under-load at both sizes
(2,253 vs 2,206 at 355K; 5,364 vs 6,316 at 5.1M): what the db holds is page cache
it already had.

! `docker stats` reports the cgroup's `memory.current`, which **includes
reclaimable page cache**. Most of the db column is that. Split by
`/sys/fs/cgroup/memory.stat` the db's `anon` is tens of MB; the rest is cache.

! The 355K figures here are higher than the 1,495 MB this section reported
earlier the same day. Nothing regressed: that run followed a `DROP DATABASE` and
measured an unusually cold page cache. 2,253 MB is the honest steady state for a
box that has been doing work, and is what a deployment will look like.

### The embedder's dtype is the single largest memory decision

| `DTYPE` | `anon` | Recall@5 | Recall@10 | MRR |
|---|---|---|---|---|
| `float32` (the default) | **2,553 MB** | 63.6% | 70.5% | 0.557 |
| `bfloat16` | **296 MB** | 65.9% | 72.7% | 0.562 |

**8.6× less memory, and recall is not worse** — +2.3 points is one query on
n=44, so read it as no measurable difference. The startup canary accepts it:
the corpus was embedded at float32 and a bfloat16 query still reproduces the
space above `CANARY_MIN_COSINE`.

! **At `float32` this stack cannot start on its own documented budget.** The
embedder sits at 1,242 MB of a 1,280 MB limit at idle — 97% full — and is
SIGKILLed (exit 137) by the first request. The 1,060 MB this section used to
quote was measured on the old TEI container, which loaded fp16; the reference
implementation adopted on 2026-09-19 defaults to `float32` and needs 2.4× that.
Every figure above is at `DTYPE=bfloat16`.

! `corpus_meta` records model, width, pooling and normalization — **but not
dtype**, although `dev_tools/pre_embed/server.py` calls dtype part of the
contract. The canary catches a mismatch empirically through cosine; nothing
*declares* it. Closing that gap starts in Ravel (`README.md`).

! That server was **recovered into the repository on 2026-09-20**. Until then it
existed only as a local image on one machine, while `docker-compose.yml` still
started the TEI image the corpus had been re-embedded away from — so
`docker compose up` produced a stack whose canary refused to serve, and the
vector space this corpus lives in was not reproducible anywhere else.
`docker/Dockerfile.embed` builds it now.

### Per-request working set

**Measured, 2026-09-19 — and it refuted what this section used to say.**
Engine peak RSS, from the container's own `memory.peak`, at three settings:

| `CLUSTER_BATCH` | 1 | 2 | 5 |
|---|---|---|---|
| peak engine RSS | 14 MB | 12 MB | 14 MB |

**Flat in the batch.** It does not move with the window, and batch=5 would have
had to hold five clusters — some 21 MB more — if the old formula were right.
(It is not flat in *corpus size*; see §2.)

! The previous text here read **"Peak engine RAM = 8.3 MB + (MAX_CONCURRENCY ×
23.4 MB)", giving 102 MB at the defaults. That number was never measured**, and
it is ~7× the real one. It put a cluster-sized working set inside the engine,
which is not where it lives.

The engine never receives a vector. The arms return `(id, score)` rows capped by
`LIMIT k × OVERFETCH`, so Postgres does the scan and hands back a few hundred
short rows. This is the cluster cost, and it is **Postgres's**, ✗ the engine's:

```
cluster size    median 2,030 rows    max 11,448 rows
x 1024 dims x 2 bytes (halfvec)
                median   4.2 MB      worst  23.4 MB    ← in the DATABASE
```

What the engine holds is enumerable: 177 centroids held hot (177 × 1024 × f32 ≈
725 KB), the candidate pool's metadata, and the snippets being returned. None is
a function of `n`, which is why the resident set is flat.

! The **database** side is unmeasured per setting. `memory.peak` is cumulative
from container start and this one had served a full day, so an honest figure
needs a db restart per row. Postgres's bound is configured (`shared_buffers`,
`work_mem`, the container limit) rather than demonstrated.

Tested in `crates/mcp/src/main.rs`: `the_ceiling_is_a_ceiling`,
`waiting_is_bounded_by_the_wait_ceiling`.

### The 4 GB budget — VALIDATED

`docker-compose.vps.yml`, measured with the full stack up and the eval running
through the deployed HTTP server:

| container | limit | measured | |
|---|---|---|---|
| `vera-db` | 1,792 MB | 1,156 MB | 65% |
| `vera-embed` | 1,280 MB | **341 MB** | 27% · at `DTYPE=bfloat16` |
| `vera-mcp` | 512 MB | **12 MB** | 2.3% |
| **total** | **3,584 MB** | **2,253 MB** | leaves ~1.3 GB for the OS |

Re-measured 2026-09-20 by `hardware_profile.py` at **355K**, replacing the
2,640 MB this table used to report. The embedder moved because of `DTYPE`; the
engine because 8.5 MB was always an idle figure and 14 MB is the peak under a
concurrent burst.

! **At 5.1M rows this budget does not hold.** Postgres alone peaks at 5,907 MB
and the stack at 6,316 MB — well past 3,584 MB. The 4 GB profile serves a
355K-row corpus; it does not serve a 5M one, and §2's two-scale table is what
says so.

! **At the default `DTYPE=float32` this profile does not start.** The embedder
needs 2,553 MB against the 1,280 MB budgeted here and is SIGKILLed on the first
request. `bfloat16` is not an optimisation for this profile, it is a
prerequisite — and it is not set anywhere in `docker-compose.vps.yml`.

! The equivalence claim — constrained scoring the same as unconstrained — is
**still not re-established** on the post-re-embed corpus. The memory numbers
above are current; that claim is not.

! `shared_buffers` drops 1500 → 512 MB. On a 4 GB box with a 2.4 GB database,
page cache beats a large private pool: every arm is a scan, and Postgres
ring-buffers large sequential reads specifically to avoid evicting
`shared_buffers` with them.

! db and embed sit near 88% of their limits. That is deliberate for db — most of
it is reclaimable page cache — but it means **the embedder has ~190 MB of
headroom and no elastic component**. A larger embedding model does not fit this
profile without taking the memory from Postgres.

Reproduce:

```bash
docker compose -f docker-compose.yml -f docker-compose.vps.yml up -d
VERA_HTTP=http://localhost:8081 python dev_tools/eval/e2e.py
docker stats --no-stream vera-db vera-embed vera-mcp
```

---

## 3. Latency

44 real queries through the deployed HTTP server. **Read the profile with the
number** — these differ by 15× across configurations that all call themselves
"the stack", and the difference is which cores the embedder gets.

| profile | p50 | p95 | max |
|---|---|---|---|
| dev box, embedder unpinned | **866 ms** | 1,459 ms | 1,701 ms |
| **2 cores pinned · 355K** | **10,251 ms** | 11,790 ms | 14,036 ms |
| **2 cores pinned · 5.1M** | **18,527 ms** | 24,604 ms | 27,541 ms |
| sub-1 GB Postgres, embedder unpinned | 5,908 ms | 7,664 ms | — |
| refused by domain gate · 355K / 5.1M | 6,953 / 8,651 ms | | |
| first query after start · 355K / 5.1M | 13,219 / 23,190 ms | | |
| startup: db + embedder load + canary | 39–44 s | | |

**14× the corpus costs 1.8× the latency.** That is far better than the per-arm
figures predict (§6 measures 19.8× on the SQL alone) and it has two causes: the
worst-scaling arm was replaced by an index (`pg_search`, migration 0004), and on
2 pinned cores the embedder is a fixed ~7 s floor that database growth hides
behind. Extrapolating, 10M lands near 25–30 s on this hardware.

! The 10,251 ms at 355K reproduces the 10,007 ms measured earlier the same day
on a separate cold-start run, so the 2-core figure is stable to ~2%.

! **The 2-core row is the honest one for a 2 vCPU VPS**, and it is the only row
measured with the embedder on the same cores as everything else
(`hardware_profile.py`, 2026-09-20). Every other latency figure in this document
— including the per-arm table below — was taken with the embedder free to use
the dev box's remaining 12 cores. Those per-arm numbers are still correct *per
arm*; they do not sum to what a 2-vCPU box delivers.

! The 1,059 ms figure this table used to publish for "2 vCPU / 4 GB" was
measured under the VPS overlay, which constrains **memory** but left the
embedder unpinned. It is withdrawn, ✗ corrected: it was never a 2-core
measurement.

### CPU, not memory, is the constraint on 2 cores

Peak CPU by phase, where 2 cores = 200%:

| phase | | db | embed | engine |
|---|---|---|---|---|
| 44 sequential | 355K | 135% | 147% | 0% |
| | 5.1M | **199%** | 146% | 0% |
| 12 concurrent | 355K | 47% | **201%** | 0% |
| | 5.1M | **200%** | **202%** | 0% |

**Both cores are saturated, and which process owns them shifts with scale.** At
355K the embedder dominates; at 5.1M Postgres pins a full 200% and the two
together demand roughly twice what the box has. The engine is 0–1% throughout.

This is a CPU wall, ✗ a memory one: §2 shows 2,253 MB of a 3,584 MB budget at
355K while latency is already 10 s.

So the largest lever available is the embedder: quantization, fewer torch
threads, or moving embedding off the box. None of the retrieval work in §6
touches the term that actually dominates this profile.

! Concurrency behaves differently when each query costs 10 s: 12 concurrent
gives **4 served, 8 refused**, against the 8/4 recorded in §4. Same
`MAX_CONCURRENCY=4` and same 2,000 ms queue ceiling — the queue simply times out
more often when the work takes longer. Both are correct for their profile.

### Where it goes

| stage | p50 | share |
|---|---|---|
| tsv · OR semantics (RUM) | 404 ms | 47% |
| dense · 5 clusters, `CLUSTER_BATCH=1` | 268 ms | 31% |
| sparse · global scan | 165 ms | 19% |
| embed query | 85 ms | 10% |

**Routing works.** Dense over the whole corpus is 637 ms; over 5 probed clusters,
81 ms — 8×. Windowing at the default of 1 costs 268 ms against 81 ms for a single
statement over the same five, and raising `CLUSTER_BATCH` recovers **182 ms of
it end to end** (§6a) at no measurable cost in engine memory.

**The text arm dominates.** OR semantics matches a median of 219,792 rows (62% of
the corpus) because a natural question ANDed together matches nothing — measured
at zero rows for 42 of 44 queries. `ts_rank` cannot be served from a GIN index,
so all 220K are scored and sorted. RUM stores ranking data in the index and turns
it into one ordered index scan: **750 ms → 404 ms at p50, 1,481 → 807 at p95,
Recall@5 unchanged.**

! RUM is optional. `SearchOps::has_rum()` probes and falls back to `ts_rank`.
Without it, add ~350 ms to p50.

### Rejected optimisations

| tried | result |
|---|---|
| drop low-IDF terms from the OR query | 14× faster, Recall@5 40.9% → 22.7%. No. |
| AND first, OR as fallback | no gain: AND returns zero rows for 42/44 queries |

---

## 4. Concurrency

On the dev box, `MAX_CONCURRENCY=2`, `QUEUE_WAIT_MS=1500`:

```
2 concurrent  ->  2 served,  0 refused
8 concurrent  ->  4 served,  4 refused
                  503 at 1,506 ms · retry-after: 1
```

On the target profile, deployed, `MAX_CONCURRENCY=4`, `QUEUE_WAIT_MS=2000`:

```
 4 concurrent  ->  4 served,  0 refused · served p50 2,089 ms
12 concurrent  ->  8 served,  4 refused · served p50 2,850 ms
```

! Latency degrades, the server does not. At 3× its ceiling it still answers two
thirds of the burst and refuses the rest with a retry — it does not queue them,
and it does not fall over.

! Invariant 6 needs a bound on **time** as well as queue depth. A bounded queue
alone still lets a caller block indefinitely behind a full one.

A refusal is an answer, not a failure: it states the request was never attempted,
which is what makes retrying safe. That is why it is `503` + `Retry-After` and
never `500`.

! None of this is testable on stdio, which reads one line, answers it, and only
then reads the next — the semaphore never has two callers to arbitrate. Every
concurrency claim here was unfalsifiable before the HTTP transport existed.

---

## 5. Disk

Re-measured 2026-09-19, after the re-embed and a `VACUUM (FULL, ANALYZE)`:

```
database        1,801 MB    chunks 1,749 = heap 552 · TOAST 1,042 · indexes 154
model weights   1,200 MB
engine binary       5.3 MB
------------------------
                ~3.0 GB
```

Where the table's bytes are, by column (logical size, TOAST included):

| | size | |
|---|---|---|
| `dense` | 696 MB | 1024 × halfvec. Mostly TOAST: 2,048 B exceeds the inline threshold |
| `tsv` | 228 MB | generated, `STORED` |
| `body` | 139 MB | the text itself |
| `sparse` | 115 MB | BM25 `sparsevec` |
| `chunks_tsv_rum` | 137 MB | the only index that costs anything |

! **A bulk `UPDATE` bloats the indexes too, and `VACUUM FULL` is not optional
after one.** Re-embedding rewrote every `dense` value; the table went to 3,246 MB
and RUM alone to **349 MB**. The rewrite took them to 1,749 MB and **137 MB** — RUM
2.5× smaller for having been rebuilt. The 134 MB this section used to quote was
right for a fresh index and had silently stopped being true.

! `chunks_tsv_idx` (GIN) is **not** in these figures: it was dropped on this box,
for the reason in the note below. The shipped schema still creates it. It last
measured 110 MB on the bloated table; a freshly built one would be smaller and has
not been measured.

! The database still does **not** fit a sub-1 GB Postgres profile — 1,801 MB against
it — so every arm pages from disk. That is what sets latency on that profile
(§3), and no amount of vacuuming changes it; only a narrower `dense` would, which
is what `EMBEDDING.md` §5e is about.

RUM costs 137 MB and 25 s to build. Both it and GIN are kept, but ✗ for the reason
this section used to give.

! Where RUM is installed, GIN serves **nothing**. Measured on the live corpus:
`chunks_tsv_rum` 368 scans, `chunks_tsv_idx` **0**, counters never reset
(`pg_stat_user_indexes`). RUM answers the `@@` match as well as the ordering, so
the earlier claim that GIN carried the match was wrong.

Its actual job is the one `migrations/0003_rum_text_index.sql` describes: RUM is
an optional extension not present in stock Postgres, `SearchOps::has_rum` probes
for it, and a deployment without it falls back to `ts_rank` — which needs GIN.
So GIN is insurance, ✗ a working index, and that is why the schema keeps it
(`dev_tools/pre_embed/schema.sql`, generated from Ravel).

An operator who has confirmed RUM is installed can reclaim it on that box alone:

```sql
DROP INDEX chunks_tsv_idx;   -- recreate: CREATE INDEX chunks_tsv_idx ON chunks USING GIN (tsv);
```

! Box-local, ✗ a schema change. Removing it upstream would delete the fallback
for every deployment that never built RUM, and the schema is Ravel's to define
in any case (`README.md`).

---

## 6. Scaling · measured at 14x, ✗ extrapolated

The corpus was replicated to **5.0M indexable rows / 23 GB** in a separate
database and every arm timed against it on the same hardware
(`dev_tools/eval/scale_probe.py --factor 14`, then
`dev_tools/eval/arm_latency.py`). 2 cores, Postgres capped at 6,336 MB — an 8 GB
box after the OS, embedder and engine. 12 queries, median of 3 after a warm run.

| arm | 355,621 rows | 4,978,694 rows | ratio | |
|---|---|---|---|---|
| layer-2 routing | 6 ms | 85 ms | 14.2× | linear in `k`, and `k ∝ n` |
| **dense** | 46 ms | **57 ms** | **1.24×** | **flat — routing works** |
| sparse | 150 ms | 6,282 ms | **41.9×** | worse than linear |
| text | 446 ms | 6,428 ms | 14.4× | linear |
| **total (sequential)** | **649 ms** | **12,853 ms** | 19.8× | |

**The central claim is confirmed.** 14× the corpus moved the dense arm by 11 ms.
Layers 1-3 do what `ARCHITECTURE.md` §2 says they do, and it is no longer an
argument from how the SQL is written.

! **Sparse is worse than this document used to predict.** It was described as
"linear · global scan", which implied 14× — it is **41.9×**. At 355K the `sparse`
column (115 MB) sits in cache; at 5M it is 1.6 GB and is read from disk on every
query, so it pays linear growth *plus* a cache-miss penalty. Any extrapolation
that assumed linearity for this arm was optimistic by ~3×.

! **Routing is not free at scale either.** `k ∝ n` means 2,478 centroids at 5M
and ~50,000 at 100M; the scan over them is linear in `k`. 85 ms is nothing, and
1.7 s at 100M would not be. Hierarchical routing — centroids over centroids —
is the fix, and is neither designed nor built.

### What this predicts for 10M

Doubling from 5M, where both global arms are already out of cache and their
marginal cost is disk-bandwidth-bound, so linear should hold from here:

| | 10M, 8 GB box |
|---|---|
| sequential, as `pipeline.rs` runs the arms today | **~26 s** |
| with the arms run concurrently | **~13 s** |
| disk | ~46 GB |

! **Extrapolated, ✗ measured.** 5M is measured; 10M is this table doubled. The
assumption is that sparse stops super-scaling once it is fully out of cache,
which is plausible and unverified. Run `scale_probe.py --factor 28` to settle it.

! **The arms run sequentially** (`pipeline.rs`, `search_arms`) — three awaits in
a row over three independent read-only queries. Sparse and text are within 3% of
each other and together are 99% of the time, so `tokio::try_join!` would take the
total from `sum` to `max`: roughly half. That is the largest single latency win
available at scale, and it is the reason a 4-core box currently buys almost
nothing per query.

### What grows with the corpus, and what does not

| | scales with n | measured |
|---|---|---|
| engine RAM, per request | **no** | flat under load and across `CLUSTER_BATCH` (§2) |
| engine RAM, floor | **yes**, linear in `k` | 9 MB → 39 MB for 14× the corpus: hot centroids (§2) |
| dense arm latency | **no** | 1.24× for 14× rows |
| layer-2 routing | yes, linear in `k` | 14.2× |
| sparse arm latency | **worse than linear** | 41.9× |
| text arm latency | yes, linear | 14.4× |
| disk | yes | ~4.8 KB per chunk |

! The global arms are the wall, and now the wall has a number. The fix is the one
already applied to dense — make them touch only what is relevant — at a recall
cost that is still unmeasured. The text arm has the right structure already (RUM)
and asks it badly: OR semantics matches a median of 62% of the corpus (§3). The
sparse arm has **no index at all** and is a sequential scan of a `sparsevec`
column; giving BM25 a posting-list index is a schema change, so it starts in
Ravel (`README.md`).

! The replicated corpus is valid for **latency only**. Every answer chunk exists
14 times and the centroids are copies, so recall and routing accuracy from it are
meaningless — `scale_probe.py` says so in its own docstring.

---

## 6c. What `CLUSTERS_PROBED` buys · and why 5 is right

§6b showed routing giving up 10.3 points @20 against a flat scan. The obvious
next move is to probe wider, so this sweeps the dial over the eval set
(`dev_tools/eval/probe_sweep.py`, dense arm only — the other two arms are global
and do not move with probe width, so any change here is routing's alone).

| probed | corpus touched | @5 | @20 | @50 |
|---|---|---|---|---|
| 1 | 0.6% | 30.8% | 35.9% | 38.5% |
| 3 | 1.7% | 56.4% | 61.5% | 69.2% |
| **5** (shipped) | **2.8%** | **64.1%** | **71.8%** | **79.5%** |
| 8 | 4.4% | 64.1% | 71.8% | 79.5% |
| 12 | 6.6% | 61.5% | 74.4% | 82.1% |
| 20 | 11.4% | 64.1% | 76.9% | 84.6% |
| 30 | 17.0% | 64.1% | 76.9% | 84.6% |
| 45 | 25.4% | 66.7% | 79.5% | 87.2% |

No cliff and no free lunch: recall climbs roughly with corpus touched, and
**5 → 8 buys literally nothing**. The whole end-to-end cost of the two settings
worth comparing (`arm_latency.py`, 10 queries, all three arms):

| | @20 | @50 | total |
|---|---|---|---|
| `CLUSTERS_PROBED=5` | 71.8% | 79.5% | **639 ms** |
| `CLUSTERS_PROBED=20` | 76.9% | 84.6% | 4,187 ms |

**+5.1 points for 6.5× the latency.** `CLUSTERS_PROBED=5` stays.

### Widening the probe costs cache locality, ✗ just rows

This is the part worth remembering, because the arithmetic suggests otherwise.
20 clusters is 3.5× the rows of 5, and on a **warm cache over the same clusters**
it costs exactly that — 229 ms against 65 ms, measured with `EXPLAIN ANALYZE`
repeated. The plans are identical, both `Bitmap Index Scan on chunks_cluster_idx`;
there is no planner flip.

But across a **stream of different queries**, each routing to its own clusters,
the dense arm goes 53 ms → 3,604 ms — **68×, not 3.5×**. At probe 5 the union of
working sets over many queries stays small enough to cache. At probe 20 it does
not, and every query pays disk.

! So the cost of probe width is superlinear in production and linear in a
benchmark that repeats one query. A measurement that reuses the same query
vector will report 3.5× and recommend widening. Ours did, until it was run over
the whole eval set.

! The `p50 ms` column of `probe_sweep.py` is **not** usable as a latency figure:
it sweeps all widths back to back per query, so the 25%-of-corpus probe evicts
the cache the narrow ones need and every row is polluted by the next. Its recall
columns are sound — those are deterministic. Latency comes from `arm_latency.py`,
one width per process.

---

## 6b. Routing against a dedicated vector index

The design's central bet — route to a few clusters rather than carry a global
index — had never been measured against anyone. ParadeDB's `pg_search` builds a
vector index alongside its BM25 one, so the same corpus and the same 39 labelled
queries can answer it (`dev_tools/eval/pdb_vector_compare.py`), both databases
capped at 6,336 MB on 2 cores.

| | p50 | @5 | @20 | @50 | index |
|---|---|---|---|---|---|
| **Vera, routed (probe 5)** | **49 ms** | 64.1% | 71.8% | 79.5% | **none** |
| Vera, flat scan | 948 ms | 69.2% | 82.1% | 89.7% | none |
| ParadeDB IVF | 2,409 ms | 69.2% | 82.1% | 89.7% | 2,621 MB |

**It is the same architecture.** Their build log says `ivf_build … centroids=910
vectors=84945` and `paradedb.vector_info` reports type `ivf` — centroids plus
posting lists, partitioned ~20× finer than Vera's (≈97 vectors per centroid
against ≈2,000).

**Their index is slower than no index.** 2,409 ms against a 948 ms flat scan for
*identical* recall. On this hardware it is not competitive with brute force.

**Routing is 49× faster and 10 points worse.** That is the trade this design
makes, stated plainly: 2.8% of the corpus touched, −10.3 points @20 against the
flat ceiling. It is an operating point, ✗ a defeat — and §6c is about whether it
is the right one.

! Their opclasses cover `vector` only — there is no `halfvec` entry in
`pg_opclass`. At 1024 dims that is 1,392 MB against Vera's 696 MB, and the gap
**compounds with width**: at the `halfvec(4096)` production target named in
`dev_tools/pre_embed/schema.sql` it is 5.6 GB against 2.8 GB, with an index
scaling past the whole 6,336 MB Postgres budget. The comparison above is at the
width most favourable to them.

! Measured on 39 queries. A 2.6% step is one query.

---

## 6a. What `CLUSTER_BATCH` costs and buys

44 eval queries × 2 repeats per setting, through the deployed server, each setting
in a freshly restarted container (`python dev_tools/eval/cluster_batch_sweep.py`):

| `CLUSTER_BATCH` | p50 | p95 | vs. batch=1 | peak engine RSS |
|---|---|---|---|---|
| **1** (default) | 5,908 ms | 7,664 ms | — | 14 MB |
| 2 | 5,909 ms | 7,965 ms | +7 ms | 12 MB |
| 5 | 5,720 ms | 7,225 ms | **−182 ms** | 14 MB |

**It buys 182 ms and costs nothing measurable.** The saving matches the 187 ms the
arm-level figures predict (81 ms batched against 268 ms sequential, §3), so the
latency half of the model holds. The memory half does not: see §2.

! **−182 ms is 3.1% of a 5,908 ms request.** On this profile the knob is close to
irrelevant — the request is dominated by the global text and sparse arms paging
from a database that does not fit its cache. It would matter proportionally more
on a box where the database fits, which is exactly where the memory it was
supposed to cost would have been affordable anyway.

! batch=2 is +7 ms, i.e. indistinguishable from batch=1. There is no gradient to
tune along here; the useful settings are 1 and `≥ CLUSTERS_PROBED`.

! Measured at concurrency 1. The `MAX_CONCURRENCY` multiplier is untested — but
the `CLUSTER_BATCH` multiplier needs no concurrency to appear, and did not.

! These absolute latencies belong to the **sub-1 GB Postgres profile**, ✗ the
profile in §3. See §3's own note.

---

## 7. Reproducing

```bash
# latency, end to end, through the real server
DATABASE_URL=... BM25_VOCAB=... python dev_tools/eval/e2e.py

# under the target profile
docker compose -f docker-compose.yml -f docker-compose.vps.yml up -d

# memory, as the kernel sees it
docker exec vera-db sh -c 'grep -E "^(anon|file|shmem|slab) " /sys/fs/cgroup/memory.stat'
```
