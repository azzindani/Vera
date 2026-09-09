# METRICS.md — Vera

What "working" means, as numbers. Every target here is **falsifiable**: it has a value,
a way to measure it, and a source that justifies it.

This document exists so that "is it fast enough?" and "is it accurate enough?" stop
being judgement calls. If a change moves a number the wrong way, the change is wrong —
not the number.

---

! **`FACTORS.md` is the companion**: this document is *what must be achieved*, that one
is *what determines whether you achieve it* — the ~36 variables involved, which of them
differ per data source, and why a number measured on one source does not transfer to
another.

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

**Global BM25 latency at 100M is unknown, and it is the dominant stage.**

This section previously blamed the fixture: a 50-word vocabulary, so an 8-term OR
matched a large share of the corpus, and real Zipfian text would be selective. The
fixture is now Zipfian (20K terms, exponent 1.0, `vera-bench synth --vocab/--zipf`) and
**BM25 is still ~82% of query time.** The diagnosis was wrong, and the corrected one is
worse, because it is about the engine rather than the fixture.

#### The real mechanism: query-term reach, not vocabulary size

The number that predicts BM25 cost is not how many distinct terms a corpus has. It is
**how many rows the terms in an actual query reach.** Those are different, and on
Zipfian text they point in opposite directions:

| Measure | Zipfian fixture, 20K rows | Reads as |
|---|---|---|
| vocabulary | 16,388 terms | large ✅ |
| Zipf slope | −0.79 | natural ✅ |
| mean IDF **per term type** | 8.68 | "average term reaches 0.02% of rows" ✅ |
| **rows an 8-term query actually reaches** | **20,928 of 20,000** | **the whole corpus** ❌ |

Both are true. The type-level average is dominated by the rare tail, and **a query never
draws from the tail** — it draws from a document's words, which are mostly head terms.
The commonest term in that corpus appears in 16,499 of 20,000 rows. One such term in the
query puts nearly every row in the union, however selective the other seven are.

! `CorpusProfile` therefore reports **occurrence-weighted** IDF as the headline and
`expected_query_reach(8)` as the verdict. An earlier version of that check used the type
average and would have certified this corpus as representative — the precise corpus it
exists to reject.

#### What this means for the design

`ARCHITECTURE.md` §5 argues the keyword half needs no routing because "BM25 **is** an
index" and an inverted index already prunes to the matching postings. That argument
holds **per term**. It does not survive `fts_match_expression` ORing *every* query token:
the union of the postings lists is bounded below by the commonest term in the query, so
cost tracks corpus size no matter how good the index is.

So the ≤ 40 ms budget in §2.1 is not reachable at 100M by the current query construction.
The three candidate responses, in increasing order of cost:

1. **Selectivity-aware term capping** — drop query terms whose `df` exceeds a threshold,
   keeping at least one. Data-driven stopwords. `CorpusProfile::head_terms` stores what
   this needs. Cheap, and it is where to start.
2. **Score-only inclusion** — keep every term for scoring but restrict the *candidate*
   set to the selective ones. More faithful, more work.
3. **A different keyword backend** with block-max WAND or similar early termination.

! **None of them is implemented, deliberately.** All three change which documents come
back, `EVAL.md` §5 rejects any change that lowers recall@k, and there is no labelled set
to measure that against yet (§8 blocker 2). Making retrieval faster by silently dropping
query terms, with no way to detect the recall it costs, is the trade this project exists
to refuse. **Term capping is the first thing to build once the eval set exists, and must
not land before it.**

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
| **Exact-match recall** (identifier queries) | **100%** — no exceptions | 100% | ⚠️ *synthetic · see below* |
| **Recall@10** vs exhaustive | ≥ 95% at probe=5 | 99.8% | ✅ *synthetic* |
| **Routing recall** (route/D) at probe=5 | ≥ 95% | 99.8% | ✅ *synthetic* |
| **Layer-1 false rejection** | **≤ 1%** | 0% | ✅ *synthetic* |
| **top-1 agreement** | ≥ 95% | 100% | ✅ *synthetic* |
| **MRR** | ≥ 0.90 | 1.000 | ✅ *synthetic* |
| **nDCG@10** | ≥ 0.85 | 0.986 at probe=5 | ✅ *synthetic* |
| **Candidate-cap loss** | ≤ 1% | 0.0% at `per_cluster_top_k` 50 | ✅ *synthetic* · §3.1 |

! **Exact-match recall is measured against a deliberately unrelated query vector.** A
named regulation has to come back *because it was named*, so the harness routes each
identifier query with a random direction — the worst case for layer 1. Measured with a
well-aimed vector it would pass even with the bypass fully gated by routing, which is a
regression this project has shipped once (`CLAUDE.md` §8 finding, `LOOPHOLES.md` §1).
It reads 100% on the synthetic corpus, which proves the *plumbing*, not the grammar: the
fixture's identifiers are already in canonical form, so extraction is never tested
against how a person writes a citation. That still needs the labelled set.

! **Exact-match recall is the only one with no tolerance.** A named regulation that
exists and is not returned is the failure this whole architecture is shaped around
(`LOOPHOLES.md` §1). 99% is not a pass.

! **Layer-1 false rejection is a silent failure.** A rejected query returns
`success: true` with zero results, indistinguishable from an empty corpus. It is
therefore capped tighter than recall, and measured separately (`L1-rej` in the sweep)
rather than folded into a recall figure where it would hide.

### 3.1 ✅ Candidate-cap loss — now measured, and what it took

**Status: built.** `vera-bench run` prints a recall-loss decomposition, and
`--per-cluster-top-k` makes the dial sweepable. The rest of this section is kept because
the reasoning is what makes the numbers readable — and because getting the *reference*
right took two attempts, both of which produced clean-looking output.

`LOOPHOLES.md` §7: the leaf scan keeps only `per_cluster_top_k` (50) candidates per
cluster. **If the correct chunk ranks 51st inside its own cluster, it is discarded
before fusion ever sees it** — and raising `max_results` cannot recover it, because the
loss happened two stages earlier.

! **No current metric detects this**, which is worse than the bug itself:

- `recall@10` compares against a **fused exhaustive** baseline that applies *the same*
  `per_cluster_top_k`. Both sides drop the same row, so recall reads 100% while the true
  answer was thrown away by both.
- `route/D` asks only "was it in a **probed cluster**". A row that was in a probed
  cluster and then cut by the cap scores as a routing *success*.

So a cap set too low would show up as **nothing at all** in the sweep — the same class
of silent failure as the layer-1 over-rejection (finding 7), and found the same way:
by decomposing a number that was hiding two causes.

**The fix — decompose recall loss by the stage that dropped the row.** The engine now
reports which candidates reached fusion, so each miss can be charged to one stage:

| Cause | Test | Dial |
|---|---|---|
| **routing** | never reached fusion, in no probed cluster | `clusters_probed` |
| **cap** | probed and scanned, cut before fusion | `per_cluster_top_k`, `candidate_cap` |
| **fusion** | reached fusion, ranked out — *and the exhaustive search returned it* | `rrf_k`, weights |
| **by design** | reached fusion, ranked out — *and the exhaustive search dropped it too* | none · not a defect |

#### The reference is the hard part

! **Cap loss is only visible against an *uncapped* dense top-k.** The obvious reference —
what this same engine returns probing every cluster — applies the same `per_cluster_top_k`
to the same clusters, so any row it keeps the routed run keeps too. Measured that way the
cap column is **structurally pinned at 0.0%**: turning the dial from 50 to 1 moved
*routing* instead and left *cap* at zero. That is the identical blind spot recall@k has,
reproduced inside the metric built to fix it, and it looked entirely healthy.

! **The fourth column exists because three were not enough.** Scored against dense-only
truth, two thirds of the set reads as "fusion loss" while end-to-end recall against the
fused baseline is 98%. Those rows are outranked by keyword hits at *every* probe count
including exhaustive — RRF working as specified. Reporting that as loss would argue for
retuning `rrf_k` to fix nothing. So `fusion` is split by whether the exhaustive fused run
returned the row: if it did, fewer candidates cost the rank and the loss is real; if it
did not, it is `by design`.

Verified on the 20K fixture at probe=5 — the cap column moves with its own dial while
`route/D` cannot see it at all:

| `per_cluster_top_k` | found | cap | recall@10 | route/D |
|---|---|---|---|---|
| 50 | 33.0% | **0.0%** | 98.2% | 99.2% |
| 5 | 53.0% | **9.2%** | 97.2% | 99.2% |
| 1 | 32.8% | **62.8%** | 85.5% | 99.2% |

! `route/D` is flat at 99.2% across a recall collapse from 98.2% to 85.5%. That is the
blindness this section claimed, demonstrated.

! The cap is **not a pure recall/cost dial**. `found` above is non-monotonic because RRF
scores by rank *within each list*, so a longer dense list gives documents present in both
halves a second contribution and pushes dense-only documents down. `per_cluster_top_k`
and `rrf_k` are therefore coupled, and the one-dial-per-cause table above is an
approximation. End-to-end recall is monotonic in the cap; dense faithfulness is not.

**Do this before tuning any retrieval dial on real data.** Tuning against a metric that
cannot observe one of the failure modes will drive the wrong knob.

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
| **Offline build peak RSS** | `train_sample × dim × 4` + vocabulary + one row | ✅ measured · 89 MB / 29 MB / 13 MB at train-samples of 20K / 5K / 1K on a 20K × 1024 import |
| Loader working set independent of corpus size | required | ✅ the corpus is streamed twice, never held (`PRE_EMBEDDING.md` §2b) |
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
| Latency | ❌ no real corpus | ⚠️ scan 11× over budget; **BM25 over budget by construction** (§2.3) |
| Recall | ❌ no labelled set | ❌ |
| Memory | ✅ synthetic | ❌ not measured |
| Disk | — | ⚠️ f32 not halfvec |
| Invariants | ✅ all guarded | ✅ |
| **Instrumentation** | ✅ **complete** | ✅ |

! The instrumentation row is new and is the point of this round: recall loss decomposes
by dial, the corpus reports its own lexical shape, exact-match recall runs adversarially,
and every keyword number now arrives with a machine-checked caveat about whether it
transfers. Nothing about retrieval got faster; what changed is that a wrong number can
no longer look right.

---

## 8. What blocks each unmeasured number

Ordered by what unblocks the most.

1. **A real corpus** → unblocks every latency and recall figure. *Blocked on: the
   dataset.* ! `huggingface.co` is refused by this environment's egress policy, so the
   planned source cannot be fetched from here; the import path (`vera-ingest`) is built
   and waiting.
2. **A labelled eval set** (50–100 queries with expert-confirmed answers, `EVAL.md` §2)
   → unblocks the real knee, honest exact-match recall against how people actually write
   citations, and **every keyword optimisation in §2.3**. Without it, recall is measured
   against an exhaustive scan of the same retrieval, which cannot detect a systematic
   error. *Blocked on: domain expertise, not engineering.* **This is now the top
   engineering-adjacent blocker**, because §2.3 has a named fix that must not ship
   without it.
3. ~~**Recall-loss decomposition**~~ → **done** (§3.1). Alongside it: the corpus profile,
   nDCG@10, exact-match recall, a Zipfian fixture, and `--per-cluster-top-k`.
4. **Contiguous per-cluster storage + halfvec** → the only path to the §2.2 target.
   *Blocked on: nothing. It is a schema change, so it belongs with the source/metadata
   work (`MULTI_DOMAIN.md` §12).*
5. **A 2 vCPU / 8 GB box** → unblocks every T2 resource number. *Blocked on:
   provisioning. Constrained-cgroup runs are a partial substitute.*

! The ordering changed once (3) landed. **(2) is now first**, because §2.3 turned from an
unknown into a known problem with a known fix that cannot be validated without it —
every candidate response changes which documents come back, and `EVAL.md` §5 rejects any
change that lowers recall@k. **(4) is the largest single lever on latency**, but it is a
schema change paid for in re-ingest, so it lands *with* the source-model work.
