# Changelog

## v0.1.1 · 2026-11-18 · ranking inside the regulation, a faster embedder

A point release. The v0.2.0 accuracy targets (`docs/ROADMAP.md` Stage 3) are **not**
reached, so this is not v0.2.0. The Stage 4 performance targets are reached against
revised thresholds, with the originals kept in brackets in the roadmap. Same model
(Qwen3-Embedding-0.6B), same corpus vectors, no re-embed.

### Changed

- **Exact references.** A named article and a named regulation land at rank 1 (the
  identifier path's hits are now ordered by the corpus's structural ladder). A
  foreign-jurisdiction gate refuses questions about another country's law
  (`migrations/0006_gate_foreign.sql`).
- **Region arm.** BM25 within the region a question names (`migrations/0007`); off for
  questions that name none.
- **Ordering inside a regulation,** grouped by `doc_id`:
  - an explanation never ranks above its own article, and it pulls in that article's
    operative text;
  - operative text comes first;
  - the definitions article goes after the law's other operative text, unless the
    question asks what something means. The corpus declares this in
    `scoring_vocabulary.definitions`.

  These are ordering rules, ✗ priors.
- **"One regulation" is `doc_id`,** ✗ `source_title`: 912 titles are shared by about
  2,100 regulations.
- **`RRF_K` is configuration** (default 60), validated at start.
- **Embedder** (`dev_tools/pre_embed/server.py`), each change behind a switch:
  - `PREFIX_CACHE` reuses the instruction's keys and values (cosine 1.000000 against the
    full pass; canary 1.00000);
  - `NARROW_EMBEDDINGS` keeps the token table bf16, bit-identical because the checkpoint
    is bf16;
  - `QUERY_CACHE` is a bounded LRU of query vectors (1024).
- **Eval.** 604 answerable cases plus 106 off-topic, with a 168-case frozen holdout.
  Only operative text counts as an answer. Review sheets for domain and alternative
  answers are in `dev_tools/eval/review/`.

### Measured

Accuracy on the real 1.7M corpus (`vera.sh e2e defs1`, `defs1_h`; `docs/ROADMAP.md`
Stage 3). The v0.1.0 column is the same engine on the 496-case Stage 0 set:

| | v0.1.0 · all [holdout] | v0.1.1 · 604 [168] |
|---|---|---|
| Recall@5 | 57.1% [51.5%] | **67.2%** [**64.9%**] |
| Recall@10 | 67.9% [61.9%] | **75.8%** [**72.0%**] |
| MRR | 0.425 [0.388] | **0.528** [**0.519**] |
| right regulation in top 5 | 72.4% [68.7%] | **78.3%** [**74.4%**] |
| `exact_ref`: named article at rank 1 | 31.4% | **100%** (63/63) |
| real questions refused | 0 | 0 |
| off-topic refused | 62 / 106 | **74 / 106** |

Performance on two pinned cores (`docker-compose.vps.yml`, `load_test.py`, cache misses;
`docs/ROADMAP.md` Stage 4):

| | v0.1.0 | v0.1.1 |
|---|---|---|
| warm p50 / p95, 1 client | 430 / 569 ms | **321–349 / 390–518 ms** |
| cold p50 / p95 | 556 / 833 ms | **492 / 875 ms** |
| 4 clients | 2.73 q/s, p50 1,432 ms | **3.67–3.84 q/s**, p50 1,007–1,068 ms |
| CPU per query | 0.73 core-s | **0.52–0.54 core-s** |
| 8 clients refused | 3 of 64 | **0 of 64** |
| engine resident at 1.7M | 30 MB | **22–24 MB** |
| embedder resident | 2.0 GB | 2.29–2.38 GB |

### Measured and dropped

Each of these is recorded in `docs/ROADMAP.md` with the run that dropped it:
- refitting fusion and the priors;
- twin collapse;
- sibling expansion;
- regulation-level evidence;
- deeper arms (k 30 / 40, 8 probes);
- int8 quantization of the embedder (cosine 0.58–0.98, below the canary);
- serialized and batched forward passes;
- a second, instruction-free query embedding.

### Known limits · not proven by this release

- **v0.2.0 accuracy is not reached.** The holdout is short of 70% / 80% / 0.55 and of 85%
  right-regulation. `underspecified` sits at 44% Recall@5.
- **The labels are not yet reviewed by a lawyer.** `review-001.csv` and `review-002.csv`
  are open, and some misses may be equally valid answers.
- **Scale rows (5.1M–20.6M) were not re-measured** for this release. The v0.1.0 table
  stands for them.
- **BEIR was not re-measured.** The new rules act only on article-labelled chunks, so
  BEIR is untouched by construction.

## v0.1.0 · 2026-11-14 · the concept, proven

The first tagged release. **Target:** show that a RAG database built by
[Ravel](https://github.com/azzindani/Ravel) can be served with cited hybrid search on
very limited hardware (2 vCPU / 8 GB, no GPU on the query path) up to 20M rows, and fix
the numbers every later version is measured against.

### What it is

An MCP retrieval server (stdio and HTTP) over a Ravel-prepared PostgreSQL database:

- **Routing:** a domain gate, then k-means clusters (~2,000 rows each), then a flat
  `halfvec` scan of 5 clusters over `chunk_vectors` (stored in cluster order). Past
  8,192 clusters the engine holds only the ~√k parent centroids (two-level routing).
- **Arms:** dense (routed), BM25 via `pg_search` (global), exact identifier (bypasses
  routing). The tsvector arm exists and is off by default; an arm at weight 0 is not run.
- **Ranking:** Reciprocal Rank Fusion over ranks, then a bounded metadata prior
  (authority, structure, subject; `1 + Σw ≤ 2`), a relevance floor that demotes, and a
  rule that a revoked regulation never outranks the one that revoked it.
- **The corpus declares itself:** model, width, pooling, instruction, scoring vocabulary
  and gate floors are read from `corpus_meta`; a startup canary refuses to serve a
  vector space it cannot reproduce. Nothing about a corpus is compiled in.
- **Never calls an LLM.** Results are snippets and addresses with provenance; full text
  only through the bounded `read_chunk`.

### Measured

Whole stack on two pinned cores and 8 GB (`docker-compose.vps.yml`), `docs/HARDWARE.md`
§9, `docs/SCALING.md` §2. Rows above 1.7M are replicas: latency and memory only.

| rows | p50 cold / warm | p95 cold | 4 clients | engine memory |
|---|---|---|---|---|
| 1.7M (real) | 556 / 430 ms | 833 ms | 2.73 q/s | 30 MB |
| 5.1M | 530 / 430 ms | 751 ms | 2.73 q/s | 56 MB |
| 10.3M | 604 / 480 ms | 918 ms | 2.67 q/s | 105 MB |
| 20.6M | 708 / 629 ms | 1,158 ms | 2.29 q/s | 15 MB |

On the real 1.7M corpus, 196 labelled questions (`docs/EVAL.md` §2b, §4c-ter;
`docs/SCORING.md` §9b): Recall@5 **57.1%**, Recall@10 **68.4%**, MRR **0.438**, the answer
inside the top 50 for **85.2%**; 0 of 196 real questions refused, 35 of 46 off-topic
refused. Nothing OOM-killed at any size; overload is refused with `503` + `Retry-After`.

### Requirements

- A database prepared by `ravel db prepare` (schema, clusters, `chunk_vectors`, parent
  level, indexes), optionally `ravel db calibrate` for the gate floor.
- The corpus's embedding model on disk, served by `docker/Dockerfile.embed` (float32).
- The corpus's BM25 vocabulary (`BM25_VOCAB`).

### Known limits · not proven by this release

- **Recall above 1.7M rows.** No real corpus that large exists yet; replicas duplicate
  every answer, so they cannot measure it. Two-level routing's recall cost is measured
  only at 846 clusters (−2.6 points of routing recall at 8 of 29 parents, one case end to
  end).
- **Ranking is the weak layer.** The answer is retrieved for 85% of questions and ranked
  into the top 5 for 57%; neither factor weights nor fusion tuning closes that gap.
- **Labels are not domain-reviewed.** 192 of the 244 eval cases were written by a
  retrieval engineer or drafted from the corpus and machine-checked, not checked by a
  lawyer.
- **The gate cannot see jurisdiction.** Questions about another country's law are
  admitted as legal questions.
- **The embedder sets the throughput ceiling** (~2.5 q/s on two cores); the database sits
  at 30-70% CPU.
- **100M rows across several databases** is designed (`docs/SCALING.md` §3-§4), ✗ built.
