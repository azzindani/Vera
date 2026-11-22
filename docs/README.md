# Vera docs — index

Every document, by what it is for. A doc is named the same everywhere, so a reference by
name alone (`SCORING.md` §3) is unambiguous; this page says where it lives.

## Start here

| | |
|---|---|
| [`../CLAUDE.md`](../CLAUDE.md) | the rules: what Vera is, its principles, what it must never do |
| [`design/ARCHITECTURE.md`](design/ARCHITECTURE.md) | how Vera is built; §0 is the whole engine in one screen |
| [`operations/OPERATIONS.md`](operations/OPERATIONS.md) | bringing it up, reading its health, what each refusal means |
| [`eval/EVAL.md`](eval/EVAL.md) | how quality is measured; §1 runs a corpus's eval in one command |
| [`project/ROADMAP.md`](project/ROADMAP.md) | the accuracy targets every change is held to |

## design/ — what the engine is

| doc | what it fixes |
|---|---|
| [`ARCHITECTURE.md`](design/ARCHITECTURE.md) | the routing layers, the arms, fusion, the containers |
| [`MCP_ENGINE.md`](design/MCP_ENGINE.md) | the six tools, how a request travels, what bounds it |
| [`TOOL_SURFACE.md`](design/TOOL_SURFACE.md) | what a calling agent may steer per request, and what it may not |
| [`OUTPUT_CONTRACT.md`](design/OUTPUT_CONTRACT.md) | what `search_knowledge` returns, and why none of it is prose |
| [`SCORING.md`](design/SCORING.md) | how Vera ranks: the relevance floor, the factors, named algorithms |
| [`EMBEDDING.md`](design/EMBEDDING.md) | the vector space, declared by the corpus; the startup canary |
| [`FAILURE_MODES.md`](design/FAILURE_MODES.md) | the ways the design can be wrong without saying so |

## operations/ — running it

| doc | |
|---|---|
| [`OPERATIONS.md`](operations/OPERATIONS.md) | bring-up, health, refusals |
| [`CONFIGURATION.md`](operations/CONFIGURATION.md) | every tunable, where it is read, what moving it costs |
| [`HARDWARE.md`](operations/HARDWARE.md) | budgets and what they are measured against (2 vCPU / 4 GB and up) |
| [`SCALING.md`](operations/SCALING.md) | 20M → 100M rows across databases: two-level routing built, the rest designed |

## eval/ — how it is judged

| doc | |
|---|---|
| [`EVAL.md`](eval/EVAL.md) | the question sets, the metrics, the current results |
| [`BEIR.md`](eval/BEIR.md) | public benchmarks, against pgvector and Qdrant |

The per-corpus serving and question sets are in `dev_tools/eval/` (`corpora.json`,
`queries_*.json`).

## project/ — direction and standards

| doc | |
|---|---|
| [`ROADMAP.md`](project/ROADMAP.md) | v0.2.0: accuracy on regulations, staged |
| [`STANDARDS_COMPLIANCE.md`](project/STANDARDS_COMPLIANCE.md) | how Vera maps to the upstream local-MCP standard, and where it diverges |

## Beside the code

| doc | |
|---|---|
| [`../dev_tools/README.md`](../dev_tools/README.md) | the offline tools: what each does, none on the query path |
| [`../dev_tools/PRE_EMBEDDING.md`](../dev_tools/PRE_EMBEDDING.md) | the original ingest → embed → load path, superseded by Ravel |
| [`../dev_tools/CLUSTER_MAINTENANCE.md`](../dev_tools/CLUSTER_MAINTENANCE.md) | k-means and routing recall |
| [`../migrations/README.md`](../migrations/README.md) | the post-ingest SQL, applied by hand |

The corpus itself is built by Ravel: its `docs/README.md` indexes how.
