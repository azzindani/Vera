# CLUSTER_MAINTENANCE.md — Vera

How layer-2 clusters stay healthy as the corpus changes. k-means centroids drift; this
is the plan to keep routing accurate without ever disrupting live queries.

---

## 1. The drift problem

Layer-2 clusters are k-means over the corpus embeddings. As documents are added:

- new chunks get assigned to the nearest *existing* centroid,
- clusters grow lopsided,
- centroids drift away from their members,
- routing quality silently degrades (queries land in the wrong cluster).

Because routing quality *is* recall (a query routed to the wrong cluster silently
misses; see `LOOPHOLES.md` §1), drift is a correctness issue, not just hygiene.

---

## 2. Three tiers by cost (you almost never pay the expensive one)

### Tier 1 — Incremental assign (cheap, every insert)
New chunk → embed → nearest existing centroid → append to that cluster partition. No
re-clustering. This is just layer-2 routing applied at write time. The default path.

### Tier 2 — Cluster split (medium, triggered by size)
When a cluster exceeds its row cap (~10K), run local k-means (k=2) on **only that
cluster's members**, replace its centroid with two. Bounds cluster size without
touching any other cluster. This is also the **scale hook**: splitting is how clusters
stay ~10K instead of ballooning toward the 100K/1M sizes a production deployment might
allow.

### Tier 3 — Periodic full re-cluster (expensive, rare)
On the GPU box, re-run minibatch / FAISS k-means over the corpus, recompute all
centroids and assignments, then publish via **atomic version swap** (§3). Triggered
**by the eval set, not the calendar**: when recall over known regulation queries drops
below threshold, that is the signal drift has accumulated enough to justify a rebuild.
Monthly-ish at most.

---

## 3. Atomic version swap (never serve a half-updated index)

Live queries must never read a half-rebuilt index. So updates are **copy-on-write**:

1. Build the new clustering as a new **cluster-version** (new partitions / new centroid
   set), alongside the live one.
2. Validate the new version against the eval set.
3. **Atomically swap** the active cluster-version pointer.
4. The engine reloads the new centroids (its only hot state) on swap; in-flight queries
   finish against the version they started on.
5. Drop the old version after a grace period.

A single query **pins one cluster-version for its lifetime**, so splits and re-clusters
can never corrupt a query in progress.

---

## 4. Drift monitor (drives Tier 2/3)

Track per cluster:
- **size** — too big → flag for split (Tier 2),
- **intra-cluster spread** — too dispersed → centroid no longer representative,
- **aggregate routing recall** (from the eval set) — below threshold → schedule a full
  re-cluster (Tier 3).

The monitor is what turns "clusters drift" from an eventual silent failure into a
scheduled, measured maintenance action.

---

## 5. Environments and cadence

| Tier | Where | When |
|---|---|---|
| 1 — incremental assign | VPS (light) | every insert |
| 2 — split | VPS (local k-means on one cluster) | on size trigger |
| 3 — full re-cluster | GPU box | on eval-recall trigger (rare) |

Only Tier 3 needs the GPU; Tiers 1–2 are cheap enough to run on the VPS alongside
serving.
