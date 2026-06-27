# STANDARDS_COMPLIANCE.md — Vera

How Vera maps to the upstream standard
`github.com/azzindani/Standards/blob/main/local_mcp/STANDARDS.md`, and the deliberate
divergences. The upstream document's own precedence rule applies: *where the project's
CLAUDE.md/docs conflict with the standard, the project takes precedence* — provided the
divergence is documented here.

The upstream standard targets **local MCP servers driven by a local LLM under
sovereignty constraints** (no cloud APIs, CPU-first, 8 GB VRAM budget). Vera is a
**VPS-hosted retrieval server driven by an agentic cloud model**. That difference
drives every divergence below.

---

## Comply (followed as-is)

| Standard | Vera |
|---|---|
| §2 Core mental model (deterministic function, not assistant) | **Followed, strictly.** Vera never calls an LLM; it returns evidence, the agent decides. |
| §3 Primitives (tools-first) | **Followed.** Everything is a tool; no prompts. |
| §8 Tool-count discipline (≤8, sharp tools) | **Followed.** 5 read-only tools. |
| §10 Surgical read protocol | **Followed.** `search_knowledge` returns snippets+addresses; `read_chunk` is the bounded read; never returns full documents or raw vectors. |
| §11 Tool schema design (≤80-char docstrings, snake_case verb_noun, primitive types) | **Followed.** See `MCP_ENGINE.md`. |
| §12 Tool annotations | **Followed**, with `openWorldHint=True` on tools that call OpenRouter. |
| §16 Return value contract (dict, `success` first, `token_estimate`, `progress`, `hint`) | **Followed**, minus write-only fields (see divergence D). |
| §17 Error handling (no raised exceptions; `error`+`hint`) | **Followed.** |
| §18 Security (path/subprocess/expression hygiene, no secrets in responses) | **Followed.** Plus: OpenRouter key never in responses; auth token generated at deploy. |
| §20 Token budget discipline | **Followed, reframed** (see divergence B): the budget protects the *agent's* context, and matters more under token-expensive agentic loops. |
| §21 CPU-first execution | **Followed for the engine.** The query engine is CPU-only on the VPS. GPU is used only by the offline pre-embed pipeline — a documented domain exception the standard explicitly permits. |
| §30 Transport modes (stdio + http) | **Followed.** stdio for harnesses, streamable-http for Claude.ai / remote agents. |
| §32 Naming conventions | **Followed.** |
| §35 CLAUDE.md required | **Followed.** Root `CLAUDE.md` present with overview, structure, principles, never-do list, progress tracker. |
| §36 What to never do (stdout, dict returns, swallowed exceptions, magic numbers, etc.) | **Followed.** Vera adds its own never-do list in `CLAUDE.md` §7. |

---

## Diverge (deliberate, documented)

### A. Self-hosted execution / sovereignty — §4, Constraint 2
**Standard:** every tool must complete its primary operation offline; no cloud APIs as
primary execution; no API keys.
**Vera:** the query path calls **OpenRouter** (cloud) to embed the query. This is a
genuine divergence.
**Rationale:** the corpus — the heavy part — is embedded **locally/offline** on rented
GPU; only the tiny per-query embedding is external. This keeps the engine stateless and
~500 MB–1 GB and avoids hosting an 8B model on the VPS. The trade (a ~120 ms network
dependency per query, and an API key requirement) is accepted deliberately.
**Mitigations** for the risks this introduces: pinned provider + version, startup
canary check, validated-secondary fail-over, retry/queue (see `EMBEDDING.md`,
`LOOPHOLES.md` §3–4). Everything *except* query embedding (storage, search, fusion,
provenance) is fully local on the VPS.

### B. Local LLM / VRAM budget — §8 hard limits, §20 VRAM chain, §21 model table
**Standard:** sized around a local 9B model on 8 GB VRAM; `MCP_CONSTRAINED_MODE`
governs VRAM-driven response sizes.
**Vera:** driven by an **agentic cloud model** (Claude.ai, harnesses). There is no local
VRAM constraint.
**Rationale & reframe:** the *spirit* of §20 still binds — token economy — but the
constraint is the **agent's context window and the VPS's RAM**, not local VRAM. Token
discipline is if anything *stricter*, because agentic loops re-issue calls and pay for
every returned token. `MCP_CONSTRAINED_MODE` is repurposed to cap **result counts and
snippet sizes** for context economy, and Vera adds its own RAM-driven limits
(concurrency, queue, clusters-probed) in config. `HARDWARE.md` is the VPS-RAM analog of
the standard's VRAM table.

### C. Language / runtime — §5
**Standard:** Python + FastMCP is the default; Rust is listed only for filesystem/single-
binary tools.
**Vera:** the engine is **Rust**; the pipelines are **Python**.
**Rationale:** consistent with the standard's actual rule ("libraries dictate language").
The engine's job is tiny stateless footprint + high-concurrency I/O orchestration —
Rust + tokio is the right fit and directly serves the OOM/concurrency guarantees. The
pipelines need transformers/GPU and PDF/k-means libraries — Python. Vera therefore uses
a Rust MCP framework rather than FastMCP; the FastMCP-specific rules (§5 pins, §34
pyright/ruff CI) apply to the Python pipelines, with Rust equivalents (clippy, cargo
test) for the engine.

### D. Write-tool machinery — §9 PATCH, §13 patch protocol, §19 snapshot/version,
§25 receipt log, §26 output generation
**Standard:** assumes tools that mutate data — hence snapshot-before-write, `dry_run`,
`restore_version`, receipt logs.
**Vera:** the **query path is read-only** — it never mutates the corpus. So
snapshot/dry_run/restore/receipt **do not apply to the query engine**.
**Where they DO apply:** the offline **pipelines** (pre-embed, cluster maintenance) are
the write side, and they honor the *spirit* of §19/§25 via **atomic version swaps** and
an ingestion/cluster-version log (`CLUSTER_MAINTENANCE.md` §3). The four-tool
LOCATE→INSPECT→PATCH→VERIFY loop becomes the read-only **ROUTE→SEARCH→READ→VERIFY** loop
(`MCP_ENGINE.md` §2).

### E. Installation / distribution — §31 self-updating mcp.json, §29 client config
**Standard:** clone-and-run `mcp.json` into a local AI client's config (LM Studio etc.),
JIT `uv sync` on launch.
**Vera:** deployed as a **service** (two containers on a VPS), reached over
streamable-http by agents, or launched via stdio by a harness. The self-updating local
clone pattern is not the primary install path.
**Rationale:** Vera is infrastructure, not a per-user local tool. It ships as container
images with a deploy step, not a client `mcp.json` bootstrap. A stdio entry is still
provided for harnesses that launch it locally.

---

## Net statement

Vera follows the upstream standard's **discipline** (deterministic tools, surgical
reads, tight schemas, token economy, error/return contracts, never-do rules) and
diverges only where the standard's two founding constraints — *local LLM* and
*offline sovereignty* — do not hold, because Vera is explicitly built for **agentic
cloud AI use on a VPS**, not for a local model. Each divergence is bounded and
mitigated above.
