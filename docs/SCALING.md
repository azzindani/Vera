# SCALING.md — from 20M rows to 100M across several databases

> **Status: designed, ✗ built.** Every number below is either measured (named) or
> labelled as an estimate. What is measured stops at 20.6M rows on 2 vCPU / 8 GB
> (`HARDWARE.md` §9).

---

## 1. What already scales, and what will not

Measured, 1.7M → 20.6M rows (12×), same box, same engine (`HARDWARE.md` §9.2):

| term | 1.7M | 20.6M | growth | at 100M (estimate) |
|---|---|---|---|---|
| dense arm, cold, 5 clusters | 54 ms | flat | **none**: rows per cluster are constant | flat |
| BM25 arm, warm p50 | 35-52 ms | 123 ms | ~linear in posting length | ~0.6 s, index ~31 GB |
| engine RSS (centroids hot) | 30 MB | 184 MB | linear in `k` | **~0.9 GB** at 50K clusters |
| centroid scan in the engine | <1 ms | ~5 ms (est.) | linear in `k` | ~25 ms |
| `chunk_vectors` on disk | 4.6 GB | 54 GB | linear | ~270 GB |
| page working set per query | ~1.8 GB | ~3.5 GB | sub-linear | unknown |

Three things break before 100M, in this order:

1. **The engine's centroid set** outgrows its 512 MB container limit at roughly
   ~30K clusters (60M rows). Not latency first: memory.
2. **BM25's index** stops fitting the database's page cache on an 8 GB box somewhere
   past ~30M rows (6.3 GB at 20.6M); then every lexical query pays disk.
3. **Disk.** 100M rows at today's layout is ~450 GB of database. Not a design problem,
   a provisioning one, but it decides whether one box can hold it at all.

The design answers these three and nothing else; what already scales is left alone.

---

## 2. Two-level routing · the engine holds the top level only

```
query ─► top level:  ~√k "super-centroids", held hot in the engine   (k=50K → ~224)
            │        pick the S nearest (S ≈ 4)
            ▼
         leaf level:  the leaf centroids of those S supers, read from Postgres
            │        (~S × 224 ≈ 900 rows of 2 KB = 1.8 MB, one indexed read)
            │        pick the 5 nearest leaves, exactly as today
            ▼
         chunk_vectors scan of 5 clusters                                  (unchanged)
```

- **Engine memory becomes O(√k).** 224 × 1024 × f32 ≈ 0.9 MB, against ~200 MB flat
  today at 10K clusters. The constant term returns to what §5.4 of `CLAUDE.md` promises.
- **Leaf centroids live in `clusters`, sorted by `parent_id`**, the same cluster-order
  trick as `chunk_vectors`, so a super's leaves are one contiguous read.
- **Recall cost is measurable before it is built.** `routing_recall.py` already reports
  where the answer's cluster ranks; a super-level miss is the new failure, and S is the
  dial. On the 1.7M corpus 5-of-846 probing keeps the dense arm within 93% of a flat
  scan (`EVAL.md` §4d at 355K, to be re-measured at 1.7M); the two-level version must be
  measured against that number, not argued.
- **Who builds it:** Ravel's `db prepare` already fits leaf centroids (`place`); it fits
  the supers over the leaves (k-means on ~50K points, seconds) and writes `parent_id`.
  Clustering stays a deployment decision (`Ravel/CLAUDE.md` §5.17).

Rejected: an HNSW index over the 50K centroids. Small and fast, but it puts an ANN
structure, and its own recall loss, in front of the exact leaf ranking, and makes the
leaf choice non-deterministic under concurrent inserts. Two exact levels are
deterministic and explainable (`explain_routing` shows both).

## 3. The lexical arm past the cache

BM25 is the one global arm left (`HARDWARE.md` §9.1). Two options, in order:

1. **Partition the BM25 index by super-cluster** and search the S chosen partitions
   plus a small global "head" index of short, high-authority documents (UU, PP). The
   global guarantee becomes "global over the head, routed over the tail", and its loss
   is measurable with the same `routing_recall.py`.
2. **Keep it global and give it the memory.** At ~31 GB of index for 100M this is a
   bigger-box decision, ✗ an 8 GB one.

Decision rule: build option 1 only when a real corpus past ~30M rows shows BM25's cold
p95 above the 1 s budget. Until then it is a design, not a need.

## 4. Several databases · one engine, a domain per database

The 100M target is several corpora, ✗ one. Each corpus is a database prepared by Ravel
on its own (`ravel db prepare`), with its own `corpus_meta`, canary, vocabulary and
clusters. The engine:

- **opens one pool per database** listed in its configuration (a list of DSNs,
  validated at startup like every other limit), each with its own startup canary;
  a database whose canary fails is not served, the others are;
- **holds one domain anchor set per database** (its corpus centroid plus its top-level
  supers). Layer 1 stops being a yes/no gate and becomes the **selector** it was
  scaffolded to be: score the query against every database's anchors, search the top
  D (D ≈ 1-2) whose scores clear the floor, refuse when none does;
- **fuses across databases by rank**, as it already fuses arms: each database returns
  its pool, RRF merges them, factors apply per row with the vocabulary of the corpus
  the row came from (`corpus_meta.scoring_vocabulary` is per database already);
- **embeds once per vector space**, ✗ once per database. Two databases declaring the
  same model, pooling and instruction share one embedding; a database declaring another
  space gets its own embedder endpoint, and is skipped (with a `progress` line) when
  that endpoint is down, because §7.7 forbids an unvalidated fallback.

A request still never names a domain (§7.13). It can only be narrowed by the gate.

## 5. Order of work

| step | needs | proves |
|---|---|---|
| 1 | a real corpus of ≥5M rows (not a replica) | routing recall and BM25 at scale with real vocabulary |
| 2 | two-level routing (§2) on the 1.7M and that corpus | engine RSS O(√k), recall loss vs one level |
| 3 | a second database (any second corpus) | multi-database selection, cross-database fusion |
| 4 | §3 option 1, only if step 1 shows BM25 over budget | lexical arm past the cache |

Step 1 is the gate on everything else: no design here should be built against a replica,
because a replica cannot show the one thing each of them risks, which is recall.
