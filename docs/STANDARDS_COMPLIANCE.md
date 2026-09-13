# STANDARDS_COMPLIANCE.md

How Vera maps to `github.com/azzindani/Standards/blob/main/local_mcp/STANDARDS.md`, and
where it deliberately diverges. The upstream document's own precedence rule applies:
where this project conflicts with the standard, the project takes precedence — provided
the divergence is documented here.

The standard targets **local MCP servers driven by a local LLM under sovereignty
constraints**. Vera is a **retrieval server driven by an agentic model**. That one
difference drives every divergence below.

---

## Complied with

| Standard | Vera |
|---|---|
| §2 deterministic function, not assistant | Strictly. The engine never calls an LLM; it returns evidence and the agent decides. Enforced structurally by a test that scans every query-path crate for completion-shaped API surfaces. |
| §3 tools-first | Everything is a tool. No prompts. |
| §4 self-hosted execution | **Fully.** The query path is entirely local: local embedding container, local Postgres, no external API and no API key anywhere in it. |
| §8 tool-count discipline (≤8) | Five read-only tools. |
| §10 surgical read protocol | `search_knowledge` returns snippets and addresses; `read_chunk` is the bounded read. Never full documents, never raw vectors. |
| §11 schema design | Short docstrings, `verb_noun` naming, primitive types, `additionalProperties: false`. |
| §12 tool annotations | Present. `openWorldHint` is false throughout — nothing reaches outside the deployment. |
| §16 return contract | `success` first, plus `token_estimate`, `progress`, and `hint` on failure. Minus write-only fields — divergence C. |
| §17 error handling | No raised exceptions across the tool boundary; `error` + actionable `hint`. |
| §18 security | No secrets in responses, and the connection string is never logged. Engine runs non-root with a read-only root filesystem and all capabilities dropped; both the engine and the database bind to loopback. |
| §20 token discipline | Followed and, if anything, stricter — see divergence A. |
| §21 CPU-first | The engine is CPU-only. GPU is used by the offline pipeline in `dev_tools/`, which the standard explicitly permits. See the caveat in `HARDWARE.md` §1 about CPU embedding. |
| §30 transports | stdio and HTTP. |
| §32 naming | Followed. |
| §36 never-do list | Followed, with Vera's own additions in `CLAUDE.md` §7. |

---

## Divergences

### A. No local LLM, no VRAM budget — §8 hard limits, §20 VRAM chain, §21 model table

**Standard:** sized around a local model on an 8 GB VRAM budget;
`MCP_CONSTRAINED_MODE` governs VRAM-driven response sizes.

**Vera:** called by an agentic model that lives elsewhere. There is no local VRAM
constraint on the query path.

**Reframe, not abandonment.** The spirit of §20 still binds — the constraint is simply
the *agent's context window* and the box's RAM rather than local VRAM. Token discipline
is stricter here, because agentic loops re-issue calls and pay for every returned token.
The constrained-mode idea is repurposed as hard caps on result count, snippet size,
`read_chunk` length and `get_provenance` ids, all configurable. `HARDWARE.md` is this
project's analog of the standard's VRAM table, and every figure in it is measured.

### B. Language and runtime — §5

**Standard:** Python plus FastMCP is the default; Rust is listed for single-binary
tools.

**Vera:** the engine is Rust; the offline tools are Python.

**Rationale** — and it is the standard's own rule, "libraries dictate language". The
engine's job is a tiny stateless footprint and bounded concurrency, which is what Rust
and tokio are for; the result is 8.5 MB resident and a compile-time-enforced bound on
peak memory. The offline tools need transformers, GPU and k-means, which is Python. The
FastMCP-specific rules apply to `dev_tools/`; the engine uses the Rust equivalents
(`cargo fmt`, clippy at pedantic with `-D warnings`, `cargo test`), all gated in CI on
three platforms.

### C. Write-tool machinery — §9, §13, §19, §25, §26

**Standard:** assumes tools that mutate data, hence snapshot-before-write, `dry_run`,
`restore_version`, receipt logs.

**Vera:** the query path is read-only and never mutates the corpus, so none of it
applies to the engine. The container enforces this rather than merely promising it: the
root filesystem is mounted read-only.

**Where it does apply:** the offline tools in `dev_tools/` are the write side, and they
honour the spirit through resumable idempotent ingestion and — designed, not yet
implemented — atomic cluster-version swaps.

The four-tool `LOCATE → INSPECT → PATCH → VERIFY` loop becomes the read-only
`ROUTE → SEARCH → READ → VERIFY` loop.

### D. Distribution — §29, §31

**Standard:** clone-and-run, with a self-updating `mcp.json` in a local AI client's
config.

**Vera:** deployed as a service — containers reached over HTTP by agents, or launched
over stdio by a harness. It is infrastructure, not a per-user local tool, so it ships as
images and a compose file with an explicit deploy step. A stdio entry point is still
provided, and the release workflow publishes a standalone binary for exactly that case.

---

## Net statement

Vera follows the standard's discipline — deterministic tools, surgical reads, tight
schemas, token economy, error and return contracts, the never-do rules — and now meets
its sovereignty constraint as well: the query path has no external dependency. What
remains divergent is the *caller* (an agentic model rather than a local one) and the
*language* (Rust for the engine), both of which the standard's own rules accommodate.
