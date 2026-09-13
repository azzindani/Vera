# HARDWARE.md — budgets, and what they are measured against

> Every number here was measured, on the corpus named below, with the commands
> given. Where something is budgeted rather than measured it says so. A budget
> nobody has tested is a guess with a table around it.

**Corpus under test:** `spike-02` · 355,621 indexable chunks · 1024 dims
(Qwen3-Embedding-0.6B, last-token pooling) · 177 clusters · 2,417 MB database.

---

## 1. The target

**2 vCPU / 4 GB, no GPU.**

Tightened from 2 vCPU / 8 GB. The 8 GB figure predated any measurement; the
numbers in §2 say the working set does not need it.

! The target has no GPU, and the embedding container currently pins a CUDA image
(`text-embeddings-inference:86-1.7.2`, compute 8.6). **That gap is open.** TEI
ships CPU images and 0.6B is small, but CPU embedding latency has not been
measured, so nothing here claims it works yet.

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

Clusters are scanned **one at a time** (`CLAUDE.md` §5.4), so a request holds one
cluster's vectors, never five:

```
cluster size    median 2,030 rows    max 11,448 rows
x 1024 dims x 2 bytes (halfvec)
                median   4.2 MB      worst  23.4 MB
```

**Peak engine RAM = 8.3 MB + (MAX_CONCURRENCY × 23.4 MB).**
At the default 4: **102 MB**. Both terms are bounded, so the total is.

! This is the whole OOM guarantee, and it is the reason for the sequential scan.
One `WHERE cluster_id = ANY($1)` is 3× faster (81 ms vs 268 ms, §3) and makes
peak RAM a function of `clusters_probed`. The speed is the price paid for the
bound.

Tested in `crates/mcp/src/main.rs`: `the_ceiling_is_a_ceiling`,
`waiting_is_bounded_by_the_wait_ceiling`.

### The 4 GB budget — BUDGETED, not yet validated

`docker-compose.vps.yml`:

```
embed   1280 MB   measured 1060 MB anon + runtime headroom
db      1792 MB   shared_buffers 512 + work_mem + page cache
engine   512 MB   measured 102 MB peak + headroom
------------------
total   3584 MB   leaving ~512 MB for the OS
```

! `shared_buffers` drops 1500 → 512 MB. On a 4 GB box with a 2.4 GB database,
page cache beats a large private pool: every arm is a scan, and Postgres
ring-buffers large sequential reads specifically to avoid evicting
`shared_buffers` with them.

**Not yet run end-to-end under this overlay.** Until it is, §1 is a target and
this table is arithmetic.

---

## 3. Latency

Measured end-to-end through the MCP server over stdio, 44 real queries,
reproduced across three runs at ±3%.

| | p50 | p95 | max |
|---|---|---|---|
| `search_knowledge` | **866 ms** | 1,459 ms | 1,701 ms |
| refused by domain gate | 357 ms | | |
| exact identifier (routing bypassed) | 444 ms | | |
| `explain_routing` | 68 ms | | |
| `read_chunk`, `list_domains` | ~0 ms | | |
| startup (centroids + canary) | 146–938 ms | | |

### Where it goes

| stage | p50 | share |
|---|---|---|
| tsv · OR semantics (RUM) | 404 ms | 47% |
| dense · 5 clusters, sequential | 268 ms | 31% |
| sparse · global scan | 165 ms | 19% |
| embed query | 85 ms | 10% |

**Routing works.** Dense over the whole corpus is 637 ms; over 5 probed clusters,
81 ms — 8×. Sequential loading costs 268 ms against 81 ms combined; see §2.

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

Measured over HTTP with `MAX_CONCURRENCY=2`, `QUEUE_WAIT_MS=1500`:

```
2 concurrent  ->  2 served,  0 refused
8 concurrent  ->  4 served,  4 refused
                  503 at 1,506 ms · retry-after: 1
```

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

```
database        2,417 MB    chunks 1,069 · RUM 134 · GIN 121 · TOAST the rest
model weights   1,200 MB
engine binary       5.3 MB
------------------------
                ~3.6 GB
```

RUM costs 134 MB and 25 s to build — 2 MB larger than the GIN index it sits
beside. Both are kept: GIN serves the `@@` match and the fallback ranking.

---

## 6. Scaling

What grows with the corpus, and what does not:

| | scales with n | why |
|---|---|---|
| engine RAM | **no** | one cluster at a time, and clusters are held near 2,000 rows |
| dense arm latency | **no** | routing probes 5 clusters regardless of corpus size |
| sparse arm latency | yes | global scan |
| text arm latency | yes | global scan |
| disk | yes | ~6.8 KB per chunk |

! The global arms are the wall, not memory. At 10 M chunks the text arm is
~28× today's row count; RUM bought roughly a 2× head start, not immunity. The
fix when it arrives is the one already applied to dense — partition it — at a
recall cost that has not been measured.

---

## 7. Reproducing

```bash
# latency, end to end, through the real server
DATABASE_URL=... BM25_VOCAB=... python eval/e2e.py

# under the target profile
docker compose -f docker-compose.yml -f docker-compose.vps.yml up -d

# memory, as the kernel sees it
docker exec vera-db sh -c 'grep -E "^(anon|file|shmem|slab) " /sys/fs/cgroup/memory.stat'
```
