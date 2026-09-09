# EVAL.md — Vera

The eval harness is the single artifact that turns "should work" into "measured to
work." It gates the model choice, tunes the dials, and triggers re-clustering. Build it
early; it is not optional for legal/regulatory retrieval where a wrong clause is a real
error.

---

## 1. Why it gates everything

Vera has several dials that can only be set with evidence, not intuition:

- truncation dimension (here: full 4096, but the eval confirms it earns its cost),
- `clusters_probed` (recall vs. latency vs. RAM),
- per-cluster top-k (the candidate cap),
- chunk size / boundary strategy,
- RRF fusion weighting (dense vs. keyword),
- the routing-confidence and `confidence:low` thresholds.

Without an eval set, every one of these is a guess. The benchmark rank of Qwen3-8B is a
prior; this eval is the proof for **Indonesian regulation text specifically**.

---

## 2. The dataset

A labeled set of **real queries → known-correct chunks/documents**:

- 50–100 queries to start, growing over time.
- Cover the query *types* that matter: conceptual ("sanctions for late tax filing"),
  exact-reference ("UU 28/2007 Pasal 9"), cross-reference, defined-term lookups, and
  paraphrases of the same need.
- For each query, record the **ground-truth** chunk id(s) / document(s) a domain expert
  confirms is correct.
- Include known **hard negatives** (similar-looking but wrong regulations) to catch
  confident wrong-domain/wrong-clause answers.

Store under `eval/` with the queries, the labels, and the scoring script.

---

## 3. Metrics

All of these are implemented in `vera-bench run`. What is missing is the **labelled
set** (§2), not the harness.

- **Recall@k** — is the correct chunk in the top-k? *The primary metric* for "cannot
  miss a regulation."
- **Routing recall** — was the correct chunk in a *probed cluster* at all? Separates
  routing failures (`LOOPHOLES.md` §1) from ranking failures. Scored against a
  **dense-only** exhaustive baseline: part of the fused result is not dense-reachable by
  construction, so scoring routing against a target no probe count can hit understates
  it. ! It is **blind to the candidate cap** — a row that was probed and then discarded
  scores as a routing success — which is why the decomposition below exists.
- **Recall-loss decomposition** — every miss charged to the stage that dropped it:
  routing / cap / fusion / by-design. This is the metric that says *which dial to turn*,
  and each column has a different one. `METRICS.md` §3.1 has the reference-choice
  argument, which is subtler than it looks.
- **Exact-match recall** — for exact-reference queries, did the global keyword path
  return the right regulation? Should be 100%; anything less means the routing-bypass
  net has a hole. ! Run with a **deliberately unrelated query vector**, so the
  regulation has to come back because it was *named*. A well-aimed vector would pass
  even with the bypass fully gated by routing.
- **MRR / nDCG@k** — ranking quality among retrieved results. Recall is a set measure
  and cannot tell rank 1 from rank 10; a caller that reads the first result can.
- **Latency p50/p95/p99** — alongside accuracy, so a dial change that helps recall but
  blows latency is visible.
- **Corpus profile** — not a retrieval metric but a precondition for reading one: the
  vocabulary, occurrence-weighted IDF, Zipf slope, document lengths and identifier
  density of the corpus under test, measured at ingest and stored with it. A keyword
  measurement taken on a corpus whose queries reach every row describes a full scan, and
  the profile is what says so out loud rather than leaving it to be noticed later.

---

## 4. What the eval drives

| Decision | Eval signal |
|---|---|
| Keep 4096 vs. truncate / rescore | Recall@k delta vs. dimension |
| `clusters_probed` (default ~5) | smallest value holding Recall@k |
| per-cluster top-k | the **cap** column of the loss decomposition · sweep with `--per-cluster-top-k` |
| keyword term capping (`METRICS.md` §2.3) | Recall@k must not fall · **this eval set is the gate that change waits on** |
| chunk strategy | Recall@k across query types |
| fusion weight | Recall@k / nDCG for keyword-heavy vs. conceptual queries |
| **re-cluster trigger** | drop in routing recall over time → schedule Tier-3 rebuild (`CLUSTER_MAINTENANCE.md`) |

---

## 5. Cadence

- **Before launch:** pass a recall threshold on the full eval set; confirm the cosine
  round-trip preflight (`EMBEDDING.md` §4) before trusting any numbers.
- **On every dial/config change:** re-run; a change that lowers Recall@k is rejected.
- **Continuously in production:** run the eval periodically; a routing-recall drop is
  the **trigger** for full re-clustering — drift is detected by measurement, not guessed
  by calendar.
