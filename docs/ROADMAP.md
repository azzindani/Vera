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

| target | why |
|---|---|
| ≥ 600 in-domain cases (±4 points on Recall@5) | to see a 5-point change |
| every query type ≥ 40 cases, `exact_ref` ≥ 60 | per-type numbers that mean something |
| 30% frozen holdout, never used to fit anything | so a fit cannot grade itself |
| ≥ 100 cases reviewed by a domain reviewer, disagreement rate published | labels are the ceiling of every number here |
| two hit levels per case: right **regulation**, right **article** | a correct law with the wrong pasal is a different failure from a wrong law |

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
