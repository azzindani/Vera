# ROADMAP.md — v0.2.0 · accuracy on regulations

> v0.1.0 proved the hardware story (20M rows on 2 vCPU / 8 GB). v0.2.0 is about being
> **right** on the corpus that demands it most: Indonesian regulation. A wrong article
> cited with confidence is worse than no answer.
>
> Every target is checked by `dev_tools/eval/e2e.py` through the deployed engine on the
> real 1.7M corpus. The baseline is the v0.1.0 tag. A stage counts as reached only when it
> is measured on the **holdout**.

## Baseline · v0.1.0, 196 in-domain + 46 off-topic cases

| measure | v0.1.0 |
|---|---|
| Recall@5 / @10 / MRR | 57.1% / 68.4% / 0.438 (±6.9 at 95%) |
| answer inside the top 50 | 85.2% |
| routing recall (answer's cluster probed) | 86.4% |
| `exact_ref` (query names the regulation and article) | **1 / 5** in top 5 |
| weakest types, Recall@5 | exception 9/20 · conceptual 15/31 · procedural 15/29 · numeric 25/41 |
| real questions refused / off-topic refused | 0 / 196 · 35 / 46 |

## Stage 0 · an eval that can tell us we improved

The precondition. No engine change is claimed until this is done: at ±6.9 points a real
5-point gain is invisible.

| target | why | status 2026-11-14 |
|---|---|---|
| ≥ 600 in-domain cases (±4 points on Recall@5) | to see a 5-point change | **496** (±4.3), 604 with off-topic |
| every query type ≥ 40 cases, `exact_ref` ≥ 60 | per-type numbers that mean something | 8 of 12 types; short: obligation 37, sanction 36, hard_negative 22, multi_tier 17; `exact_ref` 51 |
| 30% frozen holdout, never used to fit anything | so a fit cannot grade itself | **27%** (134); raising it means moving cases the v0.1.0 fit already saw |
| ≥ 100 cases reviewed by a domain reviewer, disagreement rate published | labels are the ceiling of every number here | sheet ready (`dev_tools/eval/review/review-001.csv`, 120 cases), **0 reviewed** |
| two hit levels per case: right **regulation**, right **article** | a correct law with the wrong pasal is a different failure from a wrong law | **done** (`e2e.py`) |

### Baseline on the Stage 0 set · v0.1.0 engine, unchanged

| measure | all (496) | holdout (134) |
|---|---|---|
| Recall@5 / @10 / MRR | 57.1% / 67.9% / 0.425 (±4.3) | 51.5% / 61.9% / 0.388 (±8.3) |
| right regulation in top 5 | 72.4% | 68.7% |
| right law, wrong article | 15.3% | 17.2% |
| `exact_ref`: article at rank 1 · named regulation at rank 1 | 31.4% · 54.9% (n 51) | 40.0% · 66.7% (n 15) |
| real questions refused | 0 / 496 | 0 / 134 |
| off-topic refused | 62 / 106 (the 60 new ones add foreign law and near-domain traps) | 10 / 20 |

Every later stage is measured against this table, on the holdout.

### Found while building it

* **Explanatory notes labelled as articles.** About 48K of the corpus's 1.01M `Pasal`
  chunks (4.8%, at least 5,740 documents) are *penjelasan* text: "Ayat (1) ... Cukup
  jelas". They carry the article number they explain, so the engine can cite the
  explanation of Pasal 6 as Pasal 6. The drafters skipped 4 clauses for this, and the
  `multi_tier` drafter rejected 29 of 60 pairs, largely because the cited "parent"
  article was explanation or the wrong text. ~~This is a Ravel labelling defect (no
  `penjelasan` flag on the chunk)~~ — **corrected 2026-11-15:** Ravel does flag it. 92,498
  `Pasal` chunks carry chapter `PENJELASAN - ...`; only ~2.3K (0.2%) open like an
  explanation without it. The defects were Vera's: the eval counted an article's
  explanation as its answer, and the named lookup could promote it (Stage 3).
* **Regional boilerplate is the corpus's majority.** About half of the sampled regional
  clauses were skipped because hundreds of regions enact the same text, so no single label
  would be fair. That is why `underspecified` scores any near-identical twin as a pass.

## Stage 1 · exactness · what a legal user cannot forgive

| target | v0.1.0 |
|---|---|
| `exact_ref` Recall@1 ≥ **95%** | 1/5 at @5 |
| named-regulation queries: the named regulation is rank 1 ≥ **95%** | not measured |
| a revoked regulation never above its revoker: **100%** (kept as a test) | holds |
| real questions refused: **0%** | 0 / 196 |
| off-topic refused ≥ ~~90%~~ **70%**, foreign-law questions included (revised, below) | 76% |

**Status (Stage 1 engine, 604-case set; holdout in brackets).** Measured with
`dev_tools/eval/e2e.py` against the 1.7M corpus on the `docker-compose.vps.yml` profile.

| target | Stage 0 baseline | now |
|---|---|---|
| `exact_ref`: named article at rank 1 | 31.4% | **100%** (51/51) [15/15] |
| named regulation at rank 1 | 54.9% | **100%** (51/51) [15/15] |
| revoked never above its revoker | holds | holds (`naming_a_revoked_regulation_does_not_reorder_it`) |
| real questions refused | 0 / 496 | **0 / 496** [0 / 134] |
| off-topic refused | 62 / 106 (58%) | **74 / 106 (70%)** [13 / 20] — meets the revised 70% |

Side effects, all types: Recall@5 57.1 → 60.7% [51.5 → 55.2%], MRR 0.425 → 0.481, right
regulation in top 5 72.4 → 74.2%. No type moved down. BEIR SciFact `literal` 0.7175,
unchanged. p50 not re-measured; the named lookup itself is 0.8 ms (index on type, number,
year).

How:
* **Named references.** The query's regulation, tier, region and article are parsed
  (`crates/mcp/src/identifier.rs`); the named article — or the named regulation's chunks
  nearest the query — is promoted to rank 1, routing bypassed (`SearchOps::named`).
* **Foreign jurisdictions.** A query naming a jurisdiction the corpus declares foreign
  (`corpus_meta.gate.foreign`, `migrations/0006_gate_foreign.sql`) is refused unless a
  retrieved text names it too, so Indonesia's own treaties stay answerable.

Why 90% is not reached without refusing real questions. The 32 off-topic cases still
served are near-domain or everyday questions (tax in general, a contract between friends)
and foreign questions whose country appears in Indonesian treaty text ("Inggris" is also
the language). The centroid gate separates in-domain from off-topic at AUC 0.971, but the
floor that refuses no real question stops at ~60%; adding BM25 evidence reached 81% on
train and refused a real question in the holdout, so it was rejected. Separating the rest
needs a model reading the question, which principle 1 excludes. **Revised (accepted):**
off-topic refused ≥ 70% at 0 real refused, with the remainder left to the agent, which
sees the cited evidence and can decline.

## Stage 2 · the retrieval ceiling

Ranking cannot surface what retrieval never found.

| target | v0.1.0 |
|---|---|
| answer inside the candidate pool (60) ≥ **92%** | 85.2% (top 50) |
| routing recall ≥ **93%** | 86.4% |

**Status (604-case set, 2026-11-15).** ✗ Not reached. Nothing in the engine changed: every
lever that raises the ceiling lowers what ships, and Stage 2 does not trade Recall@10 for a
ceiling. Measured with `e2e.py` (`TOP_K` raised to the pool so "anywhere in the reply" is
pool recall) and `dev_tools/eval/pool_loss.py`:

| configuration | pool recall | Recall@5 | Recall@10 |
|---|---|---|---|
| shipped: 5 probes, `PER_ARM_K` 20 | 85.9% | 60.7% | 70.4% |
| 8 probes | 86.1% | 61.5% | 70.6% |
| 12 probes | 86.3% | 60.7% | 71.2% |
| `PER_ARM_K` 60, pool 120 | 89.7% | 59.7% | **64.9%** |
| `PER_ARM_K` 60, 8 probes, pool 120 | **90.9%** | 60.9% | **67.3%** |
| same, scoring pool cut back to 60 | 79.4% | 61.5% | 67.5% |

* **Routing recall on the larger set is 84.0%** at 5 probes (86.4% was the 191-case set,
  `routing_levels.py`), 89.2% at 8, 91.7% at 12, 93.5% at 20. Wider probing alone barely
  moves the pool (+0.4 at 12 probes), because the global sparse arm already covers most
  routing misses (principle 3 doing its job). `routing_recall.py` embedded queries without
  the corpus's query instruction, unlike the engine and `routing_levels.py`; fixed.
* **Where the 70 pool misses are** (`pool_loss.py`, exact_ref excluded):
  routing 15 · dense rank 21–100 in a probed cluster 20 · sparse rank 21–100 9 ·
  **no arm within 100: 26** · lost in fusion 0. The 26 cannot be reached by any depth; a
  third of them name a region the answer's regulation belongs to (Sleman, Kulon Progo,
  Samarinda …) and no arm uses that.
* **Why depth costs ranking.** With factors off, deeper arms leave Recall@10 flat; with them
  on it drops 3–5 points. The weights were fitted on a 20-deep pool, and a deeper pool hands
  the bounded prior more low-relevance candidates to lift. A deeper pool is only worth it
  with weights refitted on it — which is Stage 3's work, so the ceiling moves there.
* **Tried and dropped:** a region-scoped dense arm. The right region's chunks are scattered
  across clusters, and a scan of Yogyakarta's 110K rows took 30–41 s on cold reads.

**Carried to Stage 3:** refit the factor weights on a 60-deep, 8-probe pool
(`fit_pool.py`, which ranks the engine's own pool, dense arm included), then adopt that
depth only if Recall@10 holds. Region as a ranking signal over the pool, ✗ as an arm.

## Stage 3 · ranking · the v0.2.0 headline

| target | step a | v0.2.0 | v0.1.0 |
|---|---|---|---|
| Recall@5 | 65% | **70%** | 57.1% |
| Recall@10 | 75% | **80%** | 68.4% |
| MRR | 0.50 | **0.55** | 0.438 |
| right regulation in top 5 | | **≥ 85%** | not measured |
| no query type below | | **50%** Recall@5 | exception 45% |

**Status (604-case set, 2026-11-15).** ✗ Not reached, and nothing shipped changed.
Reweighting is exhausted: every fusion and prior setting tried ranks within about a point
of what ships, through the engine and in the fit. Measured with `fit_pool.py fuse` over
the engine's own 120-deep pool (8 probes, `PER_ARM_K` 60, per-arm ranks from the reply),
which reproduces the engine's Recall@5 and @10 exactly at both `k` 60 and `k` 2, and
confirmed with `e2e.py`:

| configuration (through the engine, all 496) | Recall@5 | Recall@10 | MRR |
|---|---|---|---|
| shipped: 5 probes, 20 per arm, `k` 60, dense:sparse 2:1, `balanced` | 60.7% | 70.4% | 0.481 |
| same, `k` 2, 1.5:1, floor 0.2, no prior | 60.1% | 70.8% | 0.490 |
| 8 probes, 60 per arm, pool 120, `k` 60, `balanced` | 60.9% | 67.3% | 0.495 |
| same, `k` 2, 1.5:1, floor 0.2, no prior | 60.7% | 70.8% | 0.497 |

* **Fit, chosen on train, scored once on the holdout:** `k` 60, 3:1, floor 0.2,
  authority 0.5, region 0.5: holdout Recall@5 55.2%, @10 67.2%, MRR 0.462. Within noise of
  what ships. The grid's top 8 on train span 0.515–0.520 MRR across `k` 2 to 60.
* **Region** (`enacting_body` named in the query as a whole phrase, 1.0 or 0): on the 140
  cases whose pool holds such a chunk it lifts holdout MRR 0.356 → 0.397 and Recall@5 by
  one case of 35; on all 496, MRR +0.008 and Recall@5 −0.2. Real in direction, too small to
  carry a new variable, a new transform and an algorithms file into the deployment.
* **`RRF_K` is now configuration** (it was compiled in), default 60, validated at start.
* **A false gain, caught.** The first fit showed `k` 2 +5 points on the holdout. Its mirror
  skipped the currency rule whenever every prior was 0, though the engine treats a floor
  alone as ranking, so revoked answers rose above their revokers. Fixed, the gain is gone.

### Explanations are not articles · 2026-11-15

`resolve_targets` matched labels on `article` alone, so the PENJELASAN of a labelled
article counted as its answer in **162 of 496 cases**. Now only operative text counts,
unless the article exists solely as explanation (7 cases). The honest baseline drops:

| engine (all 496; holdout in brackets) | Recall@5 | Recall@10 | MRR | right law, wrong article |
|---|---|---|---|---|
| shipped, old targets | 60.7% | 70.4% | 0.481 | 13.3% |
| shipped, strict targets | 57.9% | 67.9% | 0.460 | 16.3% |
| + named lookup puts operative text first | 57.9% [50.0%] | 67.9% [59.7%] | 0.462 [0.411] | 16.3% [20.9%] |
| + an explanation never above its own article | **59.3%** [**51.5%**] | **68.8%** [**60.4%**] | **0.473** [**0.425**] | **14.9%** [19.4%] |

* **Named lookup** (`SearchOps::named`): an article's explanation was the chunk promoted to
  rank 1 for 2 of 51 `exact_ref` queries (100% → 96.1% under strict targets). Hits are now
  ordered by the corpus's structural ladder first, distance second: back to 51/51.
* **Structure rule** (`engine::structure`): when an annex chunk of a document's article
  ranks above that article's operative chunk, the operative one moves to just above it. An
  ordering rule like currency, ✗ a prior; it acts only when both were retrieved, and not
  with factors off. No type moved down, overall or on the holdout. BEIR is untouched by
  construction (its chunks carry no article label), ✗ re-measured; p50 not re-measured.

* **The ~2.3K Ravel missed, relabelled** (Ravel `id_regulation_md@1.3`, `ravel bundle
  relabel` + `ravel db relabel`): 3,077 chunks in 537 documents now read `PENJELASAN`,
  re-embedded (the chapter is in the embedded header), corpus `add1aa61` → `9ff77c4d`,
  published as revision 2 of `Azzindani/ID_REG_RAG_2611`. Recall@5 59.3 → 59.1% [51.5 →
  50.7%], Recall@10 68.8 → 68.1% [60.4 → 58.2%]. The drop is the eval getting stricter: 13
  cases (6 holdout) labelled an article whose chunks included one of those explanations,
  which no longer counts as the answer.

Every Stage 3 number above this section used the old targets and stays as recorded; from
here the strict row, on the relabelled corpus, is the baseline: **Recall@5 59.1%
[50.7%], Recall@10 68.1% [58.2%], MRR 0.472 [0.415].**

**What is left is not a weight.** The answer is in the pool for 86–91% of questions and in
the top 10 for 70%; no ordering of today's signals closes that. The measured defects that
plausibly do: explanatory notes (the section above, +1.4 points), regional
boilerplate twins (`underspecified` 16–25 of 54),
and unreviewed labels (`review-001.csv`, 0 of 120). The first is a Ravel re-chunk.

## Guardrails · every change, every stage

* The holdout does not get worse beyond its confidence interval, overall or for any type.
* BEIR (`docs/BEIR.md`) with `literal` does not drop more than 1 point on any set. A
  gain that only works on our own labels is a fit, not an improvement.
* p50 ≤ 1 s on 2 vCPU / 8 GB at 1.7M rows. Speed only has to stay decent; it is not the
  target.
* The engine still never calls a model to rank (principle 1). Accuracy has to come from
  retrieval, structure and metadata.
