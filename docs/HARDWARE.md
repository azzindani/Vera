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

### Measured, on the dev box

Read from the container cgroups (`/sys/fs/cgroup/memory.stat`), so `anon` is
memory that must be resident and `file` is cache the kernel can reclaim.

| component | measured | notes |
|---|---|---|
| engine (`mcp`) | **8.3 MB** RSS | idle and under load; vectors never enter it |
| Postgres `anon` | 18.8 MB | backend private memory |
| Postgres `shmem` | 1,624 MB | this *is* `shared_buffers=1500MB` |
| Postgres `slab` | 75.6 MB | kernel structures |
| Postgres page cache | 4,105 MB | **reclaimable** — not a requirement |
| embedder `anon` | 1,060 MB | model weights |

### Per-request working set

**Measured, 2026-09-19 — and it refuted what this section used to say.**
Engine peak RSS, from the container's own `memory.peak`, at three settings:

| `CLUSTER_BATCH` | 1 | 2 | 5 |
|---|---|---|---|
| peak engine RSS | 14 MB | 12 MB | 14 MB |

**~13 MB, flat.** It does not move with the batch, and batch=5 would have had to
hold five clusters — some 21 MB more — if the old formula were right.

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
| `vera-db` | 1,792 MB | 1,544 MB | 88% |
| `vera-embed` | 1,280 MB | 1,088 MB | 87% |
| `vera-mcp` | 512 MB | **8.5 MB** | 1.7% |
| **total** | **3,584 MB** | **2,640 MB** | leaves ~1.4 GB for the OS |

Quality under the limits was **identical** to unconstrained when measured:
Recall@5 50.0%, domain gate 6/6, 0/44 false refusals.

! Those figures predate the 2026-09-19 re-embed (`EMBEDDING.md` §5d) and the
memory numbers above have **not** been re-measured since. Memory is unlikely to
have moved — the vectors are the same width and count — but the retrieval figures
are superseded by `EVAL.md` §4, and the *equivalence claim* (constrained ==
unconstrained) has not been re-established on the new corpus.

! The engine holds 8.5 MB inside a 512 MB limit. The limit exists for the
concurrency term, not the resident one — see the peak calculation above.

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

Measured end-to-end through the MCP server over stdio, 44 real queries,
reproduced across three runs at ±3%.

| | p50 | p95 | max |
|---|---|---|---|
| `search_knowledge` · dev box | **866 ms** | 1,459 ms | 1,701 ms |
| `search_knowledge` · **2 vCPU / 4 GB** | **1,059 ms** | 1,638 ms | 2,005 ms |
| refused by domain gate | 357 ms | | |
| exact identifier (routing bypassed) | 444 ms | | |
| `explain_routing` | 68 ms | | |
| `read_chunk`, `list_domains` | ~0 ms | | |
| startup (centroids + canary) | 146–938 ms | | |

! **These belong to the 1,792 MB Postgres profile** (`docker-compose.vps.yml`,
`shared_buffers=512MB`), where the database mostly fits its cache. They do **not**
describe the sub-1 GB profile.

On a **1 GB** Postgres cap with `shared_buffers=320MB`, the same 44 queries measure
**p50 5,908 ms · p95 7,664 ms** (§6a) — 5.6× slower. The database is 1,801 MB (§5)
against a 1 GB cap, so every arm pages from disk and no amount of tuning closes it.
Only a smaller database would: `EMBEDDING.md` §5e is the 347 MB on the table.

! Both figures are real; neither is "the" latency. Quote the profile with the
number, or the number means nothing.

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
| engine RAM | **no** | ~13 MB flat (§2) |
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
