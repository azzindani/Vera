# Changelog

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
