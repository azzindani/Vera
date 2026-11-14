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
  article was explanation or the wrong text. This is a Ravel labelling defect (no
  `penjelasan` flag on the chunk), and it bears on Stage 1.
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
| off-topic refused ≥ **90%**, foreign-law questions included | 76% |

## Stage 2 · the retrieval ceiling

Ranking cannot surface what retrieval never found.

| target | v0.1.0 |
|---|---|
| answer inside the candidate pool (60) ≥ **92%** | 85.2% (top 50) |
| routing recall ≥ **93%** | 86.4% |

## Stage 3 · ranking · the v0.2.0 headline

| target | step a | v0.2.0 | v0.1.0 |
|---|---|---|---|
| Recall@5 | 65% | **70%** | 57.1% |
| Recall@10 | 75% | **80%** | 68.4% |
| MRR | 0.50 | **0.55** | 0.438 |
| right regulation in top 5 | | **≥ 85%** | not measured |
| no query type below | | **50%** Recall@5 | exception 45% |

## Guardrails · every change, every stage

* The holdout does not get worse beyond its confidence interval, overall or for any type.
* BEIR (`docs/BEIR.md`) with `literal` does not drop more than 1 point on any set. A
  gain that only works on our own labels is a fit, not an improvement.
* p50 ≤ 1 s on 2 vCPU / 8 GB at 1.7M rows. Speed only has to stay decent; it is not the
  target.
* The engine still never calls a model to rank (principle 1). Accuracy has to come from
  retrieval, structure and metadata.
