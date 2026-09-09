# PRE_EMBEDDING.md — Vera

The offline pipeline that turns source documents into the populated database the VPS
serves. Runs on a rented GPU box (transient), never on the query path.

---

## 1. Stages

```
ingest → extract text → structure-aware chunk → embed (GPU) → stage → bulk-load → index keyword → initial cluster
```

1. **Ingest** the source files (PDF regulations, contracts).
2. **Extract text.** Regulations/contracts are almost always real text, not scanned
   images — extract directly. (Scanned/figure-heavy pages are a future vision-embedding
   path, deliberately deferred; see §6.)
3. **Structure-aware chunk** (see §2) — split on legal units, carry provenance.
4. **Embed on GPU** with Qwen3-8B, applying the **document** instruction (not the
   query one), batched hard for throughput, full 4096, output as halfvec.
5. **Stage** vectors + text + metadata to parquet/JSONL on disk.
6. **Bulk-load** into Postgres with `COPY` (far faster than row inserts; also lets the
   DB be rebuilt without re-embedding).
7. **Build the keyword side** — `tsvector` + GIN index (or pg_search) for BM25.
8. **Initial k-means** to create layer-2 clusters and assign chunks (hands off to
   `CLUSTER_MAINTENANCE.md`).

---

## 2. Chunking: the highest-leverage decision for legal text

Regulations and contracts have **structure that is the meaning** — clause numbers,
cross-references ("subject to Pasal 4"), defined terms. Naive fixed-size chunking
shreds that.

- Split on the **legal unit** (pasal / ayat / section / clause), not a token count.
- Store the **heading path** in metadata (e.g. `UU 28/2007 › Bab II › Pasal 9 › ayat (3)`).
- Store **provenance at chunk creation**: `source_url`, `page`, and `section`/clause.
  Immutable thereafter — this is what `fetch(depth="provenance")` returns.
- Qwen3-8B's long context allows large chunks, but bigger chunks blur retrieval
  precision — chunk to the unit, not to the maximum.

---

## 2b. The load streams · it never holds the corpus

Stages 5–8 are `vera-ingest import` (`vera_index::stream`). The constraint that shapes
them is simple arithmetic: at 100M × 4096 the corpus is **1.6 TB of vectors and ~100 GB
of bodies**, so anything the loader keeps resident it keeps at that size.

The earlier loader kept four such things. Each is now bounded:

| held | at 100M × 4096 | now |
|---|---|---|
| every `IngestRow` (bodies + provenance) | ~100 GB | one row · written as it arrives |
| every vector | **1.6 TB** | a bounded training sample |
| one anchor cosine per row, sorted for percentiles | 400 MB | a fixed-width histogram |
| one SQL transaction over every insert | a ~1.6 TB WAL | batched commits + a completeness marker |

Peak RSS is now `train_sample × dim × 4` plus the vocabulary plus one row. Measured on a
20K × 1024 import: **89 MB** training on everything, **29 MB** at 5,000, **13 MB** at
1,000.

### Two passes, and why not one

k-means must finish before any row can be assigned, and the domain anchor must be known
before any row-to-anchor cosine can be measured. So pass 1 collects the training sample,
the anchor sum, the lexical profile and the row count; pass 2 assigns, writes, and
measures. ! There is no single-pass version that is not a guess — estimating the anchor
from a sample too would make the **layer-1 threshold** depend on a random subset, and
that threshold is the difference between a working corpus and one that silently rejects
97% of queries (`CLAUDE.md` §8 finding 1).

The source is therefore **streamed twice** and must be replayable in a stable order. A
source that cannot be replayed (a pipe, a network response) has to be staged to disk
first — which stage 5 already does.

### Training on a sample

`--train-sample` is the dial that sets peak RAM. This is the **standard IVF
construction**, not a shortcut: FAISS trains a coarse quantizer on ~30–256 vectors per
centroid and then adds the full corpus. It defaults to `40 × k`, floored at 50,000.

! **Assignment stays exact.** Every row is scored against every centroid in pass 2. Only
the centroid *positions* come from a sample.

! **The sample is uniform, not a prefix.** Sources are ordered — by document, by date, by
id — so taking the first N rows would train the quantizer on whichever slice came first.
On a regulation corpus ordered by year that is a quantizer for the 1990s.

Measured cost on a corpus with real cluster structure (50K × 512, 100 topics):

| training rows | per centroid | cluster tightness |
|---|---|---|
| 50,000 (all) | 224 | 0.8814 |
| 8,000 | 36 | 0.8791 *(−0.3%)* |
| 2,000 | 9 | 0.8709 *(−1.2%)* |

Within the FAISS range the cost is negligible; below it the centroids start describing
the sample. `mean_similarity` in the build report is the number that says which side of
that line a build landed on.

And tightness is the *pessimistic* view — what matters is retrieval, which is unmoved.
Swept at probe=5 on the same corpus, training on 8,000 rows versus all 50,000:

| | recall@10 | route/D | nDCG@10 |
|---|---|---|---|
| full training | 100.0% | 100.0% | 1.000 |
| 8,000 sampled | 100.0% | 100.0% | 1.000 |

! Both are on a *synthetic* corpus, so this shows the mechanism is sound, ✗ that the
ratio transfers. Re-check `mean_similarity` on the real corpus, where cluster structure
is whatever the embedding model produced rather than something the fixture arranged.

---

## 3. Non-negotiables at 100M scale

1. **Resumable.** Checkpoint progress by chunk id / content hash. A GPU dying at row
   60M resumes from 60M, never restarts.
   ! Not built. The loader is *interruptible* but not resumable: batched commits mean an
   interrupted import leaves a partial corpus, and it marks itself `build_state =
   in_progress` so the store refuses to open it (`LOOPHOLES.md` §10). That converts a
   silent half-corpus into a loud restart — the safe failure, not yet the cheap one.
2. **Idempotent.** Re-running never double-inserts (key on content hash).
3. **Consistency capture.** Record the pinned model + version, the exact document and
   query instruction strings, and pooling/normalization into metadata. These are what
   the engine's startup canary and the query side must match (`EMBEDDING.md` §4).
4. **Preflight before the big run.** Pass the cosine round-trip test (GPU vs. pinned
   OpenRouter provider, ≥ ~0.999) on a sample **before** spending GPU-days on 100M.

---

## 4. Throughput and cost (planning)

~100M chunks × ~500 tokens ≈ 50B tokens. An 8B embedding model in batch on a rented
A100 is a multi-GPU-day job (~$200–400 range). Doing the same via the API would be
~$500 *and* hit rate limits. The real reasons to self-host the bulk are (a) no
rate-limit throttling on a 50B-token job and (b) full control of serving config =
your consistency guarantee. Plan for it as a known operational event, not a surprise.

---

## 5. Re-embedding is a planned event, not a crisis

If the pinned model is deprecated, or the truncation dimension changes, the corpus
must be re-embedded. Because (a) staged vectors are kept and (b) the pipeline is
resumable/idempotent, a re-embed is a scheduled GPU job that publishes via the same
atomic swap as a re-cluster (`CLUSTER_MAINTENANCE.md` §3). Budget for it; do not be
surprised by it.

---

## 6. Deferred: vision embeddings (v2)

Scanned regulations, stamped/signed contract pages, and complex tables are where text
extraction fails. A vision-language embedding model (e.g. Qwen3-VL-Embedding) would
handle those — but VL vectors live in a **different space** than text vectors, so they
cannot be searched with one query vector. That makes vision a **separate index/column
with its own query embedding**, i.e. a real second system. Deferred until a real
document forces it.
