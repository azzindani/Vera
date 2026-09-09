# EMBEDDING.md — Vera

The embedding model, the dimension decision, and the consistency rules that keep the
corpus and query in the same vector space.

---

## 1. Model

**Qwen3-Embedding-8B.** Open weights (Apache-2.0). Multilingual (100+ languages,
strong for Indonesian), 32K context, MRL (custom dimensions 32–4096), instruction-aware
(queries take an instruction; documents do not). Ranked #1 on the MTEB multilingual
leaderboard at release.

It is the only Qwen3-Embedding variant that reaches **4096 dimensions** (0.6B → 1024,
4B → 2560, 8B → 4096), which is why 8B is the chosen variant given the 4096 decision.

> Benchmark rank is a prior, not proof. The decisive test is the eval set on real
> Indonesian regulation queries (`EVAL.md`), not the leaderboard.

---

## 2. The split: self-embed corpus, API for queries

- **Corpus:** embedded **offline on rented GPU** running the open Qwen3-8B weights
  (cheaper at bulk, no rate limits, full config control). See `PRE_EMBEDDING.md`.
- **Queries:** embedded **at runtime via OpenRouter** serving the *same* model
  (`qwen/qwen3-embedding-8b`, OpenAI-compatible embeddings endpoint).

This split is only valid because both ends run the **same open-weight model**. The
hard rule: **same model, same version, same instruction, same pooling, same
normalization** — otherwise the query vector and the corpus vectors are not in the
same space and similarity is meaningless.

---

## 3. The dimension decision: full 4096

Kept at full **4096** for maximum ranking precision ("cannot miss a regulation").
Consequences, all handled elsewhere:

- pgvector cannot ANN-index 4096 → **no global index; routing + per-cluster flat
  scan** instead (`ARCHITECTURE.md` §3).
- Storage ~0.82 TB (halfvec) for 100M rows → **disk, not RAM** is the cost; accepted.
- Per-cluster working set ~82 MB → fine under **sequential loading** (`MCP_ENGINE.md`).

Stored as `halfvec` (16-bit): halves storage and per-query I/O for ~1–2% recall loss.

> Alternative on record (not chosen): the **rescore pattern** — index a truncated
> 1024-dim vector for cheap candidate retrieval, keep full 4096 for rescoring only the
> finalists. Same final accuracy, smaller working set. Revisit if hardware tightens.

---

## 4. Consistency rules (the silent-failure class)

A space mismatch produces no error — just quietly worse results. Defend with:

1. **Pin the provider.** OpenRouter routes across hosts by default (Balanced/Nitro);
   different hosts can serve with subtly different config. Use **Exacto (one fixed
   provider)**. Treat the provider id as part of the pinned config.

   ! The pin lives in **two places, and they are not the same thing.** The corpus records
   `validated_providers` — the hosts proven by the §4.5 round-trip to reproduce its
   space, a property of the data that survives every deployment. The deployment records
   `[provider] primary` and `fallback` — which of those hosts answers today, a property
   that changes during an outage. Merging them would make failover look like a corpus
   change.

   ! **Provider id is deliberately not part of `assert_matches`.** §2 has the corpus
   embedded on a rented GPU and queries served through OpenRouter, so the two ids never
   match by design; comparing them would reject the architecture. Space equality is about
   *geometry* — model, width, normalization. The provider check is a separate,
   one-directional permission: is this host on the corpus's allowlist?
2. **Pin the model version.** Store the exact model + version string in each
   partition's metadata. A floating "latest" alias can silently update and rot the
   index.
3. **Match the instruction.** Qwen3 is instruction-aware: apply the query instruction
   only to queries, never to documents, and use the identical wording the corpus was
   built against. (Worth +1–5% when correct; a silent loss when not.)
4. **Match pooling + normalization** (last-token pooling, L2 normalize) between the
   GPU pre-embed and the OpenRouter serving.
5. **Cosine round-trip preflight.** Before indexing 100M rows, embed the same sample
   sentences on the GPU and via the pinned OpenRouter provider; require cosine ≥ ~0.999.
   Do not index until it passes.
6. **Startup canary check.** The engine embeds a known reference string at boot and
   compares to a stored reference vector; if cosine < threshold it **refuses to serve**
   and alerts. This catches provider/model drift in production automatically.

---

## 5. Availability vs. consistency

Pinning one provider for consistency creates a single point of failure. Resolution:

- Validate a **second** OpenRouter provider offline against the same reference vector.
- Pin primary; **fail over only to the validated secondary**.
- Through transient blips, **queue + retry with backoff** rather than switching.
- **Never** fall back to an unvalidated host — a wrong-space embedding is worse than a
  brief delay.

**How it is enforced.** The corpus carries the allowlist; `ProviderConfig` carries the
choice; `assert_all_validated` checks **every** candidate at startup, primary and
fallback alike, and refuses to boot on an unlisted one.

! Checking the fallback at *failover* would be too late in the only way that matters: it
passes boot, looks configured, and fails during the outage it was added for. A permission
check that only runs on the unhappy path has never been run.

! An empty allowlist is permitted and warned about, not refused. A corpus built before
validation existed records none, and hard-failing those would break every existing index
to guard against a risk the operator has not opted into. Recording one closes the list.

The retry/backoff half and the HTTP client itself are still unbuilt (`CLAUDE.md` §8).

---

## 6. Reranker

Not used. RRF fusion is the ranking mechanism (`ARCHITECTURE.md` §5). If the eval set
later shows fusion is not sharp enough, the matching open-weight **Qwen3-Reranker**
(also on OpenRouter) is the drop-in option — but it would add a network round-trip and
the same provider-consistency caveats, so it stays out until evidence demands it.
