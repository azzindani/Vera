# Vera

**Truth from sources.** An MCP retrieval server: routed, hybrid search over a large
document corpus that returns ranked, **source-cited** results for an AI agent to
summarize and a human to independently verify.

Runs on **2 vCPU / 8 GB**, no GPU on the query path, up to **20M rows** measured. Measured, not budgeted — see
[docs/HARDWARE.md](docs/HARDWARE.md).

---

## What it does

Given a query, Vera embeds it, routes it through three layers (domain → cluster →
leaf), runs three retrieval arms over only the relevant slice of the corpus, fuses
them, and returns a small set of results — each with a snippet, component scores, and
**provenance** (source document + page/section).

The calling agent writes the summary. Vera supplies the evidence and the citation
block. The agent never picks a domain — the engine detects it from the query, and
returns nothing rather than guessing.

## What it is not

- Not an assistant. It never calls an LLM, never summarizes, never reranks with a model.
- Not a generative system. It surfaces what exists and returns the receipt.

## Current state · v0.1.0

Measured 2026-11-14 on a Ravel-prepared corpus of Indonesian regulations, the whole stack
pinned to two cores and 8 GB (`docker-compose.vps.yml`). Rows above 1.7M are replicas of
the real corpus: their latency and memory are real, their recall is not measured.

| | 1.7M (real) | 5.1M | 10.3M | 20.6M |
|---|---|---|---|---|
| p50 cold / warm | 556 / 430 ms | 530 / 430 ms | 604 / 480 ms | 708 / 629 ms |
| p95 cold | 833 ms | 751 ms | 918 ms | 1,158 ms |
| throughput, 4 clients | 2.7 q/s | 2.7 q/s | 2.7 q/s | 2.3 q/s |
| engine memory | 30 MB | 56 MB | 105 MB | **15 MB** (two-level routing) |

Quality on the real 1.7M corpus, 196 labelled questions (`docs/EVAL.md`):

| | |
|---|---|
| Recall@5 / @10 / MRR | **57.1% / 68.4% / 0.438** (±6.9 points at 95%) |
| answer inside the top 50 | **85.2%** |
| real questions refused | **0 / 196** |
| off-topic questions refused | **35 / 46** (the 11 admitted are mostly foreign-law questions) |

Public benchmark, BEIR nDCG@10, same embedder and chunks for every engine (`docs/BEIR.md`):

| | SciFact | NFCorpus | FiQA |
|---|---|---|---|
| Vera, hybrid | **0.718** | **0.368** | 0.428 |
| pgvector HNSW / Qdrant, dense | 0.692 / 0.694 | 0.365 / 0.368 | 0.463 / **0.470** |
| Qdrant hybrid (RRF) | 0.709 | 0.359 | 0.411 |

Nothing was OOM-killed at any size; past four clients extra load is refused with `503` +
`Retry-After`, never queued. What this release does **not** prove is listed in
[CHANGELOG.md](CHANGELOG.md).

## Run it

```bash
cp .env.example .env      # fill in DATABASE_URL, EMBED_ENDPOINT, BM25_VOCAB
docker compose up -d
curl -s localhost:8081/health
```

Under the target hardware profile:

```bash
docker compose -f docker-compose.yml -f docker-compose.vps.yml up -d
```

Configuration is entirely environment-driven, with no defaults for anything the
process cannot safely guess. See [docs/CONFIGURATION.md](docs/CONFIGURATION.md).

## Architecture at a glance

- **Engine** — stateless Rust MCP server, stdio and HTTP transports. 15-105 MB resident from 1.7M to 20.6M rows.
- **Store** — PostgreSQL + pgvector (`halfvec` dense, `sparsevec` BM25) + RUM for the
  text arm. No global ANN index: routing does the pruning, and an ANN graph would have
  to be resident over all `n` rows — reintroducing the term this design removes.
- **Embedding** — one model, both ends. The corpus declares its vector space and the
  engine refuses to serve a provider that does not reproduce it.
- **Fusion** — Reciprocal Rank Fusion over ranks. No reranker, no model in the loop.
- **Scoring** — retrieval finds what is relevant; metadata factors (authority,
  structure, subject) reorder inside a bounded prior, and a revoked regulation never
  outranks the one that revoked it. [docs/SCORING.md](docs/SCORING.md).

## Layout

```
crates/          the engine · contract, embed, store, engine, mcp
migrations/      schema and indexes
docker/          images for the engine and the RUM-enabled database
docs/            how the running system works
dev_tools/       offline, operator-side: embedding, clustering, eval, fixtures
```

`dev_tools/` never runs on the query path. See [dev_tools/README.md](dev_tools/README.md).

## Docs

| Doc | Contents |
|---|---|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | the three routing layers, the three arms, why there is no ANN index |
| [docs/MCP_ENGINE.md](docs/MCP_ENGINE.md) | tool surface, request path, concurrency, the OOM guarantee |
| [docs/TOOL_SURFACE.md](docs/TOOL_SURFACE.md) | what an agent may steer per request, and what the engine keeps |
| [docs/SCORING.md](docs/SCORING.md) | the multi-factor model, the algorithm registry, pool expansion — and why viewpoints and consensus are dropped |
| [docs/OUTPUT_CONTRACT.md](docs/OUTPUT_CONTRACT.md) | response schema, provenance, confidence |
| [docs/CONFIGURATION.md](docs/CONFIGURATION.md) | every environment variable and what it costs |
| [docs/OPERATIONS.md](docs/OPERATIONS.md) | deploying, health, what each refusal means |
| [docs/EMBEDDING.md](docs/EMBEDDING.md) | the vector space and how it is enforced |
| [docs/HARDWARE.md](docs/HARDWARE.md) | measured memory, latency, concurrency, disk, scaling |
| [docs/EVAL.md](docs/EVAL.md) | the harness that gates retrieval quality |
| [docs/FAILURE_MODES.md](docs/FAILURE_MODES.md) | what breaks silently, and what stops it |
| [docs/BEIR.md](docs/BEIR.md) | public benchmark: Vera against pgvector and Qdrant on three BEIR sets |
| [docs/SCALING.md](docs/SCALING.md) | from 20M rows to 100M across databases: two-level routing built, the rest designed |
| [docs/STANDARDS_COMPLIANCE.md](docs/STANDARDS_COMPLIANCE.md) | mapping to the upstream standard |

## Where the corpus comes from — [Ravel](https://github.com/azzindani/Ravel)

Vera serves a corpus; it does not build one. That half is
**[azzindani/Ravel](https://github.com/azzindani/Ravel)**, the offline corpus compiler.

```
Ravel (offline, GPU, transient)          Vera (online, VPS, always-on)
────────────────────────────────         ────────────────────────────────
documents → bundle → Postgres  ────────►  query → route → search → cite
    owns the schema                           reads the schema
```

The seam is the **`corpus_meta` row**. Ravel generates the schema, and stamps the
bundle's manifest — model, width, pooling, instruction strings, chunker version, source
hashes — into that table before the first chunk is loaded. Vera reads it at startup
(`crates/store/src/search.rs`) and builds its embedder to match: nothing about the vector
space is compiled into the Rust, and the startup canary refuses to serve if the space it
reproduces is not the one the row declares.

**Ravel takes precedence on the corpus schema.** Ravel writes it, Vera reads it, and
readers do not define formats. A schema change starts there and propagates here.

Shared between the two repositories:

| Thing | Here | In Ravel | Rule |
|---|---|---|---|
| Labelled eval queries | [`dev_tools/eval/queries.json`](dev_tools/eval/queries.json) — 244 cases, the origin | [`eval/id_legal/queries@v1.jsonl`](https://github.com/azzindani/Ravel/blob/main/eval/id_legal) | Vera's copy is the origin; Ravel's is imported by [`tools/import_vera_queries.py`](https://github.com/azzindani/Ravel/blob/main/tools/import_vera_queries.py) and drops the chunk ids, which cannot survive a re-chunk |
| Byte-aware batching | [`dev_tools/pre_embed/batching.py`](dev_tools/pre_embed/batching.py) — frozen | [`src/runtime/batching.py`](https://github.com/azzindani/Ravel/blob/main/src/runtime/batching.py) | Ravel owns it now; Vera's copy is held only until `pre_embed`'s remaining scripts follow it |
| Corpus schema | [`migrations/*.sql`](migrations) — applied by hand | [`src/bundle/schema.py`](https://github.com/azzindani/Ravel/blob/main/src/bundle/schema.py) — generated | Generated from the manifest there, never hand-written here |

Family: Folio · Pipeline · Sift · **Vera** · Ravel.

## Licence

MIT.
