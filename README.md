# Vera

**Truth from sources.** An MCP retrieval server: routed, hybrid search over a large
document corpus that returns ranked, **source-cited** results for an AI agent to
summarize and a human to independently verify.

Runs on **2 vCPU / 4 GB**, no GPU on the query path. Measured, not budgeted — see
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

## Current state

| | |
|---|---|
| Corpus under test | 355,621 chunks · 177 clusters · 2.4 GB |
| Recall@5 | **50.0%** through the deployed server |
| Latency p50 / p95 | **1,059 ms / 1,638 ms** on 2 vCPU / 4 GB |
| Memory, full stack | **2,640 MB** measured against a 3,584 MB budget |
| Domain gate | 6/6 out-of-domain queries refused, 0/44 false refusals |
| Concurrency | 12 concurrent → 8 served, 4 refused with `503` + `Retry-After` |

The dense arm currently contributes nothing (weight 0.0): the corpus vectors were
produced by a backend whose embedding space does not match the model's reference
implementation. Retrieval runs on the sparse and text arms.
[docs/EMBEDDING.md](docs/EMBEDDING.md) §5 has the measurement.

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

- **Engine** — stateless Rust MCP server, stdio and HTTP transports. 8.5 MB resident.
- **Store** — PostgreSQL + pgvector (`halfvec` dense, `sparsevec` BM25) + RUM for the
  text arm. No global ANN index: routing does the pruning.
- **Embedding** — one model, both ends. The corpus declares its vector space and the
  engine refuses to serve a provider that does not reproduce it.
- **Fusion** — Reciprocal Rank Fusion over ranks. No reranker, no model in the loop.
- **Scoring** — retrieval finds what is relevant; metadata factors (authority,
  recency, structure) decide what matters. Designed, not yet built — see
  [docs/SCORING.md](docs/SCORING.md).

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
| [docs/SCORING.md](docs/SCORING.md) | the multi-factor model, viewpoints, consensus, tiered effort, pool expansion |
| [docs/OUTPUT_CONTRACT.md](docs/OUTPUT_CONTRACT.md) | response schema, provenance, confidence |
| [docs/CONFIGURATION.md](docs/CONFIGURATION.md) | every environment variable and what it costs |
| [docs/OPERATIONS.md](docs/OPERATIONS.md) | deploying, health, what each refusal means |
| [docs/EMBEDDING.md](docs/EMBEDDING.md) | the vector space and how it is enforced |
| [docs/HARDWARE.md](docs/HARDWARE.md) | measured memory, latency, concurrency, disk, scaling |
| [docs/EVAL.md](docs/EVAL.md) | the harness that gates retrieval quality |
| [docs/FAILURE_MODES.md](docs/FAILURE_MODES.md) | what breaks silently, and what stops it |
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
| Labelled eval queries | [`dev_tools/eval/queries.json`](dev_tools/eval/queries.json) — 50 cases, the origin | [`eval/id_legal/queries@v1.jsonl`](https://github.com/azzindani/Ravel/blob/main/eval/id_legal) | Vera's copy is the origin; Ravel's is imported by [`tools/import_vera_queries.py`](https://github.com/azzindani/Ravel/blob/main/tools/import_vera_queries.py) and drops the chunk ids, which cannot survive a re-chunk |
| Byte-aware batching | [`dev_tools/pre_embed/batching.py`](dev_tools/pre_embed/batching.py) — frozen | [`src/runtime/batching.py`](https://github.com/azzindani/Ravel/blob/main/src/runtime/batching.py) | Ravel owns it now; Vera's copy is held only until `pre_embed`'s remaining scripts follow it |
| Corpus schema | [`migrations/*.sql`](migrations) — applied by hand | [`src/bundle/schema.py`](https://github.com/azzindani/Ravel/blob/main/src/bundle/schema.py) — generated | Generated from the manifest there, never hand-written here |

Family: Folio · Pipeline · Sift · **Vera** · Ravel.

## Licence

MIT.
