# Vera

**Truth from sources.** An agentic retrieval MCP server: routed, hybrid search over very
large document corpora that returns ranked, **source-cited** results for an AI agent to
summarize and for a human to independently verify.

Built to run on a **2 vCPU / 8 GB VPS** over **100M+ rows**, and to scale out unchanged.

---

## What it does

Given a query, Vera embeds it, routes it through three layers (domain → cluster → leaf),
runs hybrid search (semantic + keyword) over only the relevant slice of the corpus, and
returns a small set of results — each with a snippet, score, and **provenance** (source
link + page/section). The calling agent writes the summary; Vera supplies the verifiable
evidence and the citation block. The agent never picks a domain — the engine detects it
from the query.

Vera is **composable infrastructure** — a tool an agent calls, not an app a person uses
directly. Family: Folio · Pipeline · Sift · Vera.

## What it is not

- Not an assistant — it never calls an LLM, never summarizes, never reranks with a model.
- Not a generative system — it surfaces what exists and returns the receipt.

## Architecture at a glance

- **Engine:** stateless **Rust** MCP server (stdio + streamable-http).
- **Store:** **PostgreSQL + pgvector**, vectors as `halfvec`, partitioned by cluster, no
  global ANN index (routing does the pruning).
- **Embedding:** **Qwen3-Embedding-8B** (4096-dim, open weights). Corpus pre-embedded
  offline on rented GPU; queries embedded at runtime via **OpenRouter** (pinned).
- **Hybrid:** dense (Qwen3) + Postgres BM25, fused with RRF. No reranker.
- **Two containers:** stateless engine ↔ Postgres. Replicate the engine for QPS; shard
  the DB for size.

## Design docs

| Doc | Contents |
|---|---|
| [CLAUDE.md](CLAUDE.md) | anchor context for AI coding agents; principles; never-do list |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | full system design, routing, data flow |
| [docs/MCP_ENGINE.md](docs/MCP_ENGINE.md) | tools, query flow, concurrency, OOM guarantee |
| [docs/OUTPUT_CONTRACT.md](docs/OUTPUT_CONTRACT.md) | result + provenance schema; agent-summarizes |
| [docs/EMBEDDING.md](docs/EMBEDDING.md) | Qwen3-8B, 4096, OpenRouter pinning, consistency |
| [docs/PRE_EMBEDDING.md](docs/PRE_EMBEDDING.md) | offline GPU ingestion pipeline |
| [docs/CLUSTER_MAINTENANCE.md](docs/CLUSTER_MAINTENANCE.md) | layer-2 drift, atomic swap |
| [docs/HARDWARE.md](docs/HARDWARE.md) | RAM/disk/latency budgets; scale path |
| [docs/LOOPHOLES.md](docs/LOOPHOLES.md) | known failure modes + solutions |
| [docs/STANDARDS_COMPLIANCE.md](docs/STANDARDS_COMPLIANCE.md) | mapping to local_mcp standard; divergences |
| [docs/EVAL.md](docs/EVAL.md) | the eval harness that gates quality |

## Standard

Follows the discipline of
[`azzindani/Standards/local_mcp/STANDARDS.md`](https://github.com/azzindani/Standards/blob/main/local_mcp/STANDARDS.md),
with documented divergences for **agentic cloud AI use on a VPS** (cloud query
embedding, agent-driven rather than local-model-driven, Rust engine, read-only query
path). See [docs/STANDARDS_COMPLIANCE.md](docs/STANDARDS_COMPLIANCE.md).
