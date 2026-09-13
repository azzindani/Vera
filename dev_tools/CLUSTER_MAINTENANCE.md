# CLUSTER_MAINTENANCE.md

How layer-2 clusters are built, how they go stale, and the design for keeping them
healthy.

! **What exists today is the initial clustering** (`cluster_maint/kmeans.py`) and a
routing-recall check (`cluster_maint/eval_routing.py`). Incremental assignment,
size-triggered splits and the atomic version swap described in §2–§3 are **designed and
not implemented**. They are written down here because the day they are needed is the day
someone has already inserted into a live corpus, and improvising then is how a query
reads a half-updated index.

---

## 1. The drift problem

Clusters are k-means over the corpus vectors. As documents are added:

- new chunks attach to the nearest *existing* centroid,
- clusters grow lopsided,
- centroids drift away from their members,
- routing quality degrades — silently.

Routing quality is recall for the dense arm: a query routed to the wrong clusters
misses, with no error. Drift is therefore a correctness problem, not hygiene.

The measurement that detects it is **routing recall** — was the labelled article in a
probed cluster at all — which `eval_routing.py` reports and `../docs/EVAL.md` gates.

---

## 2. Three tiers by cost

### Tier 1 — incremental assign (cheap, every insert)

New chunk → embed → nearest existing centroid → append. No re-clustering. This is
layer-2 routing applied at write time.

### Tier 2 — split on size (medium, triggered)

When a cluster exceeds its row cap, run k-means with k=2 over **that cluster's members
only** and replace its centroid with two. Bounds cluster size without touching anything
else.

! The cap is a config value. Around 2,000 rows per cluster is what makes routing
meaningful at the current corpus size: probing 5 touches ~6% of the corpus. Let clusters
grow an order of magnitude and probing 5 touches 29%, at which point routing is not
pruning anything.

### Tier 3 — full re-cluster (expensive, rare)

Recompute every centroid and assignment on the GPU box, then publish atomically (§3).

! Triggered **by the eval set, not the calendar.** When routing recall over the labelled
queries drops below threshold, that is the signal drift has accumulated enough to
justify a rebuild. A monthly rebuild on a corpus that has not changed is wasted GPU; a
quarterly one on a corpus that has doubled is silent recall loss.

---

## 3. Atomic version swap

Live queries must never read a half-rebuilt index, so updates are copy-on-write:

1. Build the new clustering as a new **version**, alongside the live one.
2. Validate it against the eval set — a rebuild that lowers routing recall is not
   published.
3. Swap the active version pointer atomically.
4. The engine reloads centroids, its only hot state.
5. Drop the old version after a grace period.

A query pins one version for its lifetime, so a split or re-cluster can never corrupt a
query in progress.

---

## 4. What to monitor

| Signal | Threshold crossed → |
|---|---|
| cluster size | Tier 2 split |
| intra-cluster spread | the centroid no longer represents its members |
| routing recall on the eval set | Tier 3 rebuild |

This is what turns "clusters drift" from an eventual silent failure into a scheduled,
measured action.

---

## 5. Where each tier runs

| Tier | Where | When |
|---|---|---|
| 1 — incremental assign | alongside the engine (light) | every insert |
| 2 — split | alongside the engine (k-means on one cluster) | on size trigger |
| 3 — full re-cluster | GPU box | on eval-recall trigger |

Only Tier 3 needs a GPU.
