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

- **Recall@k** — is the correct chunk in the top-k? *The primary metric* for "cannot
  miss a regulation."
- **Routing recall** — was the correct chunk in a *probed cluster* at all? Separates
  routing failures (`LOOPHOLES.md` §1) from ranking failures. If routing recall is high
  but Recall@k is low, fix fusion/top-k; if routing recall itself is low, fix clustering
  or raise `clusters_probed`.
- **Exact-match recall** — for exact-reference queries, did the global keyword path
  return the right regulation? Should be ~100%; anything less means the routing-bypass
  net has a hole.
- **MRR / nDCG@k** — ranking quality among retrieved results.
- **Latency p50/p95** — alongside accuracy, so a dial change that helps recall but
  blows latency is visible.

---

## 4. What the eval drives

| Decision | Eval signal |
|---|---|
| Keep 4096 vs. truncate / rescore | Recall@k delta vs. dimension |
| `clusters_probed` (default ~5) | smallest value holding Recall@k |
| per-cluster top-k | Recall@k vs. candidate-cap misses |
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
