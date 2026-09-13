# PRE_EMBEDDING.md

The offline pipeline that turns source documents into the database the engine serves.
Runs on the operator's GPU box, never on the query path.

---

## 1. Stages

```
ingest → extract text → structure-aware chunk → embed (GPU)
       → sparse-vectorize → load → index → cluster
```

1. **Ingest** the source documents.
2. **Extract text.** Regulations and contracts are almost always real text rather than
   scanned images, so extraction is direct. Scanned and figure-heavy pages are a
   separate problem; see §5.
3. **Chunk on structure** — §2.
4. **Embed on GPU**, applying whatever instruction convention the corpus will record,
   batched for throughput, stored as `halfvec`.
5. **Sparse-vectorize** with BM25 against a vocabulary fitted on this corpus.
6. **Load** into Postgres, streaming and committing per batch.
7. **Index** — `tsvector` + GIN, the RUM index for the text arm's ordering, and the
   sparse index.
8. **Cluster** — hands off to `CLUSTER_MAINTENANCE.md`.

! Text, dense vector and sparse vector are produced in **one pass over one text**. The
corpus this project inherited had them produced by separate runs, which is exactly why
its vectors could not be reproduced from its text. Doing all three here makes that class
of drift structurally impossible.

---

## 2. Chunking is the highest-leverage decision

Regulations and contracts have structure that *is* the meaning — clause numbers,
cross-references, defined terms. Fixed-size chunking shreds it.

- Split on the legal unit (pasal / ayat / section / clause), not a token count.
- Store the heading path in metadata.
- Store provenance at chunk creation: source, page, section. Immutable thereafter —
  this is what `get_provenance` returns, and the engine never synthesizes it.

This is not a stylistic preference; it is measurable. The text arm scored **0.0%** on an
unchunked corpus where a whole 32,000-character article was one `tsvector`, and **40.9%**
after chunking — from worst arm to best, on the same model and the same queries.

! Re-chunking changes every chunk id, which is why the eval labels are article-level.
It also changes the arms' relative strength, which is why the fusion weights must be
refitted afterwards. Both of those are documented failures, not hypotheticals
(`../docs/FAILURE_MODES.md` §3).

---

## 3. Non-negotiables

1. **Resumable.** Progress is checkpointed per batch. A run that dies at 90% continues.
2. **Idempotent.** Re-running never double-inserts; content hash is the key.
3. **The recipe is recorded.** Model, dimension, pooling, normalization, dtype,
   instruction, tokenizer hash and vocabulary hash all go into `corpus_meta`. The engine
   reads them at startup and builds itself to match. A corpus that does not record what
   built it cannot be served safely, because nothing downstream can check.
4. **Verify the space before the big run.** Embed a sample, compare against the
   implementation you intend to serve queries with, and require the cosine to clear the
   canary floor *before* spending GPU-days.

! Point 4 is the one that was skipped, and it cost the dense arm entirely. The corpus
was embedded through a backend whose output does not match the model's reference
implementation; the mismatch is invisible to the startup canary because both sides end
up in the same wrong space. `../docs/EMBEDDING.md` §5 has the measurement. Verifying
against an **independent** implementation is the check that catches it, and it is cheap
before the run and very expensive after.

---

## 4. Re-embedding is a planned event

If the model changes, the dimension changes, or the space turns out to be wrong, the
corpus is re-embedded. Because the pipeline is resumable and idempotent, that is a
scheduled job, not a crisis. Budget for it.

A re-embed invalidates the stored vectors, the cluster centroids and the canary
baseline, and it requires refitting the fusion weights afterwards. All four move
together.

---

## 5. Deferred: vision embeddings

Scanned regulations, stamped contract pages and complex tables are where text extraction
fails. A vision-language embedding model would handle them — but VL vectors live in a
different space than text vectors, so they cannot be searched with one query vector.
That makes vision a separate index with its own query embedding: a second system, not a
column. Deferred until a real document forces it.
