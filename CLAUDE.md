# CLAUDE.md — Vera

> Context file for AI coding agents working in this repository.
> Read this first. It defines what Vera is, the rules you must follow, and the
> deliberate ways it diverges from the upstream
> `azzindani/Standards/local_mcp/STANDARDS.md`.

---

## 1. What Vera is

**Vera** is an MCP retrieval server: a routed, hybrid search engine over a large
document corpus (current domain: Indonesian regulations) that returns **ranked,
source-cited results** for an AI agent to reason over and a human to independently
verify.

The name means *truth from sources*. Vera never invents knowledge; it surfaces what
exists and hands back the receipt. Every result is traceable to its origin — document
plus page or section.

**One-line purpose:** turn a query against a large corpus into a handful of verifiable,
cited results — fast enough to feel interactive, light enough to run on 2 vCPU / 4 GB,
and structured so the calling agent can summarize and the human can double-check.

Vera is **composable infrastructure**: a tool an agent calls, not an application a
person uses directly.

---

## 2. The core mental model

Vera is a **deterministic retrieval function**, not an assistant.

```
Agent's job:   understand intent, choose tools, compose the summary + citations
Vera's job:    embed → gate → route → search → fuse → SCORE → return cited evidence
```

Retrieval finds what is relevant; **scoring decides what matters**. Legal results
cannot be ordered on text similarity alone — a district regulation and the national
law it implements both match the query, and only metadata separates them. The
multi-factor model, viewpoints and consensus are in `docs/SCORING.md`.

Vera **never calls an LLM**. It does not summarize, rerank with a model, or interpret.
It returns structured evidence; the agent does the thinking. This single rule is what
keeps the engine stateless, light and replicable.

---

## 3. Architecture in one screen

Full detail in `docs/ARCHITECTURE.md`.

```
LAYER 1  domain gate     → centroid similarity + lexical evidence; serve or refuse
LAYER 2  k-means clusters → pick the 5 nearest centroids (held hot in the engine)
LAYER 3  per-cluster scan → load ONE cluster, flat halfvec scan, keep top-k, drop
                          ⊕ global sparse (BM25) ⊕ global text (tsvector/RUM)
                          ⊕ global exact-identifier path (routing bypassed)
                          → RRF over RANKS → candidate pool
SCORING  multi-factor     → relevance gate, then authority / temporal / structural /
                            completeness · several viewpoints · consensus sets
                            confidence and triggers more work  [designed, ✗ built]
```

- **Embedding:** one model, both ends. The **corpus declares** its model, width,
  pooling and instruction in `corpus_meta`; the engine builds itself to match and
  refuses to serve if a startup canary cannot reproduce the space. Nothing is compiled
  in. `docs/EMBEDDING.md`.
- **Store:** PostgreSQL + pgvector. Dense as `halfvec`, sparse as `sparsevec`, text as
  `tsvector` ordered by a RUM index. **No global ANN index** — routing does the pruning.
- **Fusion:** RRF over ranks, never over scores. No model reranker. Fusion yields a
  **candidate pool**, ✗ a final ranking.
- **Scoring:** six factor families, only two of which need retrieval. Weights are
  fitted per query type against the eval set, never chosen. `docs/SCORING.md`.
- **Containers:** stateless Rust engine · Postgres/pgvector+RUM · embedding server.
- **Offline tools** live in `dev_tools/` and never touch the query path.

---

## 4. Repository structure

```
vera/
├── CLAUDE.md                   ← you are here
├── README.md
├── .env.example                ← every knob, with working values
├── docker-compose.yml          ← the stack; overlay docker-compose.vps.yml for the target profile
│
├── crates/
│   ├── contract/               ← domain types + the wire contract. Zero I/O, zero sibling deps.
│   ├── embed/                  ← the embedding provider trait, HTTP client, space validation
│   ├── store/                  ← Postgres access: corpus meta, centroids, the three arms
│   ├── engine/                 ← pure logic: routing, RRF fusion, trust
│   └── mcp/                    ← the binary: settings, transports, tool dispatch
│
├── migrations/                 ← post-ingest SQL, applied BY HAND. ✗ the schema:
│                               dev_tools/pre_embed/schema.sql is the source of truth
├── docker/                     ← Dockerfile.engine, Dockerfile.db (pgvector + RUM)
│
├── docs/                       ← how the running system works
│   ├── ARCHITECTURE.md  MCP_ENGINE.md  OUTPUT_CONTRACT.md
│   ├── CONFIGURATION.md  OPERATIONS.md  HARDWARE.md  SCORING.md
│   ├── EMBEDDING.md  EVAL.md  FAILURE_MODES.md  STANDARDS_COMPLIANCE.md
│
└── dev_tools/                  ← offline, operator-side. Never on the query path.
    ├── pre_embed/              ← ingest → chunk → embed → load
    ├── cluster_maint/          ← k-means, routing recall
    ├── eval/                   ← the harness that gates quality
    └── fixtures/               ← a tiny corpus so CI can run integration tests
```

Language split follows the upstream rule "libraries dictate language": the engine is
Rust (tiny stateless footprint, tokio concurrency); the offline tools are Python
(transformers, GPU, k-means).

---

## 5. Architecture principles (do not violate)

1. **The engine never calls an LLM.** No summarization, no model rerank. Return
   evidence; the agent composes prose.
2. **The corpus declares the vector space, and the engine matches it.** Model, width,
   pooling, instruction — all read from `corpus_meta`, never compiled in. A startup
   canary re-embeds a stored chunk and refuses to serve if the space does not reproduce.
3. **Routing accelerates; the global arms guarantee.** Semantic routing may miss, so the
   sparse and text arms scan globally and an exact identifier bypasses routing entirely.
4. **OOM is impossible by construction.** Peak RAM = fixed cost + (concurrency ceiling ×
   one cluster). Sequential cluster loading is what makes the second term independent of
   probe width.
5. **Provenance is captured at ingestion and immutable.** Never synthesize a source link
   at query time; omit what ingestion did not record. Immutability is a database
   trigger (`migrations/0002_provenance_immutable.sql`), ✗ a convention — and it is only true for a corpus the
   trigger has been applied to.
6. **The query path is read-only.** Enforced, not promised: the engine container runs
   with a read-only root filesystem.
7. **Design for 2 vCPU / 4 GB; let bigger hardware benefit automatically.** Never
   hardcode a limit — read it from the environment.
8. **Measure, then claim.** Every number in `docs/` names the command that produced it,
   and anything designed but not built says so.
9. **Factors rank, but only after relevance.** A relevance floor precedes every
   metadata prior. Authority without relevance ranks the most prestigious document in
   the corpus first for every query, and it looks correct.
10. **Effort is escalated on measured disagreement, never by default.** A query
   answered confidently in one round must not be made to spend thirty seconds.

---

## 6. Tool surface (agent-facing, read-only, ≤ 8 tools)

Schemas in `docs/MCP_ENGINE.md`. The read-only analog of the upstream
LOCATE→INSPECT→PATCH→VERIFY loop is **ROUTE → SEARCH → READ → VERIFY**.

| Tool | Role | Returns |
|---|---|---|
| `list_domains` | introspection (not a router input) | domain ids + descriptions, zero content |
| `search_knowledge` | the workhorse: routed hybrid search (**query only**) | ranked results: id, snippet, scores, provenance |
| `read_chunk` | bounded surgical read of one chunk | full text of one chunk, size-capped |
| `get_provenance` | verification bundle for result ids | source + locator, id-capped |
| `explain_routing` | debug / transparency | detected domain, probed clusters, scores |

The agent never passes a `domain` — it is detected inside the engine. If nothing clears
the gate, the engine returns empty results rather than guessing.

Every tool returns an object with `success` first, plus `token_estimate`, `progress`,
and on failure `error` + `hint`.

---

## 7. What you must NEVER do

1. **Never call an LLM from inside the engine.** Not to summarize, not to rerank.
2. **Never compile in a model, dimension, or instruction.** The corpus declares them.
3. **Never serve if the startup canary fails.**
4. **Never let exact-identifier search be gated by cluster routing.**
5. **Never load all probed clusters into memory at once.** Sequential only — this is the
   OOM guarantee.
6. **Never use an unbounded request queue.** Bounded + backpressure + wait-timeout, and
   the wait ceiling is on *time*, not just depth.
7. **Never fall back to an unvalidated embedding provider.**
8. **Never synthesize or guess a provenance link.**
9. **Never mutate a live cluster in place.** Copy-on-write + atomic version swap.
10. **Never write to stdout.** stdout is the MCP channel; all logs go to stderr.
11. **Never return raw vectors or full documents from search.** Snippets and addresses;
    full text only through the bounded `read_chunk`.
12. **Never hardcode a limit.** Read it from the environment, validate it at startup,
    and fail loudly on a value that cannot work.
13. **Never accept a domain from the agent.**
14. **Never log the connection string.** It carries a password.
15. **Never claim a number you did not measure.** If it is budgeted rather than
    measured, say so.

---

*Upstream standard: `https://github.com/azzindani/Standards/blob/main/local_mcp/STANDARDS.md`.*
*Where this file or `docs/STANDARDS_COMPLIANCE.md` conflicts with it, this project takes
precedence, per the standard's own precedence rule.*
