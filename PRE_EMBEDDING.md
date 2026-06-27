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
  Immutable thereafter — this is what `get_provenance` returns.
- Qwen3-8B's long context allows large chunks, but bigger chunks blur retrieval
  precision — chunk to the unit, not to the maximum.

---

## 3. Non-negotiables at 100M scale

1. **Resumable.** Checkpoint progress by chunk id / content hash. A GPU dying at row
   60M resumes from 60M, never restarts.
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
