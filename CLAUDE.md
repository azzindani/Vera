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
multi-factor model is in `docs/design/SCORING.md`, and the algorithms a caller may
select from are declared in a file, ✗ compiled in (`config/algorithms.json`).

Vera **never calls an LLM**. It does not summarize, rerank with a model, or interpret.
It returns structured evidence; the agent does the thinking. This single rule is what
keeps the engine stateless, light and replicable.

---

## 3. Architecture in one screen

The whole engine, with what is built and what was dropped, is `ARCHITECTURE.md` §0. In
short:

- **Gate → route → search → fuse → score.** A domain gate serves or refuses; k-means
  routing picks the clusters the dense arm scans; global sparse, exact-identifier and
  region arms run beside it; RRF over ranks makes a candidate pool; the multi-factor model
  orders it, after a relevance floor, by a **named algorithm** declared in `config/`.
- **Embedding:** one model, both ends, declared by the corpus in `corpus_meta`; a startup
  canary refuses to serve a space it cannot reproduce. (`EMBEDDING.md`)
- **Store:** PostgreSQL + pgvector. Dense as `halfvec`, sparse as `sparsevec`, text as
  `tsvector` ordered by RUM. **No global ANN index**: routing does the pruning.
- **Containers:** stateless Rust engine · Postgres/pgvector+RUM · embedding server.
  Offline tools live in `dev_tools/` and never touch the query path.

---

## 4. Where things are

| path | holds |
|---|---|
| `crates/` | `contract/` (domain types + wire contract; zero I/O), `embed/` (provider trait, space validation), `store/` (Postgres: corpus meta, centroids, the arms), `engine/` (routing, fusion, factors), `mcp/` (the binary: settings, transports, tools) |
| `config/` | the algorithms (`algorithms.json`, `algorithms.no_prior.json`) and scoring vocabularies, mounted read-only |
| `migrations/` | post-ingest SQL, applied **by hand**; ✗ the schema, which Ravel generates (`migrations/README.md`) |
| `docker/`, `docker-compose*.yml` | the images; the stack, its VPS profile, and the eval overlay |
| `dev_tools/` | offline: `pre_embed/`, `cluster_maint/`, `eval/` (the harness that gates quality, `corpora.json`), `fixtures/` |
| `docs/` | `design/`, `operations/`, `eval/`, `project/`: **[`docs/README.md`](docs/README.md) is the index** |
| `.env.example` | every knob, with working values |

A doc is named the same everywhere, so `SCORING.md` §3 means `docs/design/SCORING.md` §3.
Sections here are cited across the code as `CLAUDE.md` §5.n (a principle) and §7.n (a
never-rule): keep their numbers when editing.

Language split follows the upstream rule "libraries dictate language": the engine is
Rust (tiny stateless footprint, tokio concurrency); the offline tools are Python
(transformers, GPU, k-means). The corpus is built by Ravel, which writes the schema
Vera reads.

---

## 5. Architecture principles (do not violate)

1. **The engine never calls an LLM.** No summarization, no model rerank. Return
   evidence; the agent composes prose.
2. **The corpus declares the vector space, and the engine matches it.** Model, width,
   pooling, instruction — all read from `corpus_meta`, never compiled in. A startup
   canary re-embeds a stored chunk and refuses to serve if the space does not reproduce.
3. **Routing accelerates; the global arms guarantee.** Semantic routing may miss, so the
   sparse and text arms scan globally and an exact identifier bypasses routing entirely.
   Measured at 14× the corpus: the dense arm goes 46 → 57 ms while sparse goes 150 →
   6,282 ms. Routing is proven; **the global arms are what does not scale**, and that is
   the accepted cost of the guarantee, ✗ an oversight (`docs/operations/HARDWARE.md` §6).
4. **OOM is impossible by construction, and the reason is measured.** The engine's
   **per-request** cost is flat — it never materialises the corpus, every arm returns
   `(id, score)` rows under a `LIMIT`, so Postgres does the scanning and the
   cluster-sized working set is the **database's**. Measured 14/12/14 MB at
   `CLUSTER_BATCH` 1/2/5, and 39 → 42 MB from idle to a 12-way burst at 5.1M rows.
   ! Its **fixed** cost is O(k), ✗ constant: centroids are held hot and `k ∝ n`, so
   the engine's floor is 9 MB at 355K and 39 MB at 5.1M, before any query.
   `docs/operations/HARDWARE.md` §2.
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
11. **Everything a measurement needs is in git.** A result is reproduced by
   `dev_tools/eval/eval_corpus.py` from `corpora.json` and `eval.env`, never from the
   operator's `.env` or a shell history (`EVAL.md` §1). Ravel holds the same rule for
   building a corpus (Ravel `CLAUDE.md` §5.19).

---

## 6. Tool surface (agent-facing, read-only, 6 of ≤ 8 tools)

Schemas in `docs/design/MCP_ENGINE.md`; per-request arguments in `docs/design/TOOL_SURFACE.md`. The read-only analog of the upstream
LOCATE→INSPECT→PATCH→VERIFY loop is **ROUTE → SEARCH → READ → VERIFY**.

| Tool | Role | Returns |
|---|---|---|
| `list_domains` | introspection (not a router input) | domain ids + descriptions, zero content |
| `list_algorithms` | introspection: what each way of ranking is for | names, what they scored, which are unfitted |
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
5. **Never return a vector from an arm, and never scan a cluster in the engine.** This
   is what actually holds the resident set flat — measured, ✗ argued. Arms return ranked
   addresses under a `LIMIT`; the scan belongs to Postgres. `CLUSTER_BATCH` windows the
   loop (default 1) and is a **latency** knob: it must stay configuration, since
   hardcoding it would violate §7.12, but do not attach a memory claim to it.
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
*Where this file or `docs/project/STANDARDS_COMPLIANCE.md` conflicts with it, this project takes
precedence, per the standard's own precedence rule.*
