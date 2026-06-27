# LOOPHOLES.md — Vera

Known failure modes of the design, each with its solution. These are the things that
break silently or catastrophically if not handled — review before and during build.

---

## 1. Silent zero-recall on routing miss  *(highest priority)*

**Hole:** Layer-2 routes the query to the wrong clusters → the right regulation sits in
an unprobed cluster → it is never scored → it is **silently absent**. No error, no
signal. Directly violates the "cannot miss a regulation" goal.

**Solution:** a **global exact-identifier keyword path that bypasses cluster routing.**
Exact references (regulation numbers, citation strings, defined terms) are searched
against the whole-corpus BM25 index, not just the routed clusters. Semantic routing
accelerates conceptual search; the global keyword net guarantees an exact reference is
never lost. Surface these as `exact_matches` in the response so the agent can see they
came from the high-trust global path. Secondary mitigation: `clusters_probed` > 1
(insurance against near-miss routing) and a `confidence: low` flag when top scores are
weak/clustered.

---

## 2. Who writes the summary?

**Hole:** The output must include a summary. If the engine summarizes, it needs an LLM
→ statelessness, lightness, and the no-model-dependency design all break.

**Solution:** the **calling agent** writes the summary. Vera returns structured results
+ `summary_payload` + `citation_block`. Because Vera is built for agentic AI use, the
LLM already exists on the caller side — summarization is free where it belongs. The
engine never calls a model. See `OUTPUT_CONTRACT.md`.

---

## 3. Embedding space drift

**Hole:** OpenRouter routes across providers, or the model is silently updated → query
vectors drift out of the corpus's space → quality collapses with **no error**.

**Solution:** pin one provider (Exacto) + pin model version in metadata + **startup
canary check** (embed a known string, compare to a stored reference vector, refuse to
serve if cosine < threshold) + cosine round-trip preflight before indexing. See
`EMBEDDING.md` §4.

---

## 4. Pinned provider = single point of failure

**Hole:** Pinning one provider for consistency means an outage kills the query side,
and you cannot fall back without breaking the embedding space.

**Solution:** pre-validate a **second** provider offline against the reference vector;
fail over only to that validated provider; queue + retry through transient blips; never
fall back to an unvalidated host. See `EMBEDDING.md` §5.

---

## 5. Unbounded queue → OOM via waiters

**Hole:** The concurrency semaphore caps *executing* requests, but an unbounded wait
queue still grows memory under a flood — an OOM vector hiding behind the concurrency
limit.

**Solution:** **bounded queue** (cap ~64) + 503 backpressure when full + a **queue-wait
timeout** (reject if waited > T) which also bounds tail latency. The queue holds tiny
request descriptors, not working sets. See `MCP_ENGINE.md` §4.

---

## 6. Live cluster mutation mid-query

**Hole:** Incremental inserts, splits, or re-clusters running while queries serve could
let a query read a half-updated cluster.

**Solution:** **copy-on-write + atomic version swap**; a query **pins one
cluster-version for its lifetime**. Splits/re-clusters publish a new version and swap
atomically. See `CLUSTER_MAINTENANCE.md` §3.

---

## 7. Fixed candidate cap drops the true answer

**Hole:** top-50 per cluster × ~5 clusters = ~250 candidates. If the right regulation
ranks 51st within its cluster, it is cut before fusion.

**Solution:** tune per-cluster top-k from the eval set; the **global exact-identifier
path (§1)** bypasses this cap for exact references; emit `confidence: low` so the agent
can widen the search. Make top-k and `clusters_probed` config, not constants.

---

## 8. Provenance integrity (the verify link is wrong/stale)

**Hole:** The entire value proposition is the human verifying via the source link +
locator. If provenance is wrong or synthesized, trust collapses.

**Solution:** capture provenance **at ingestion**, store it immutably with the chunk,
and never synthesize or guess a link at query time. `url` is the original source. Store
a source hash so a moved/changed source can be detected. See `OUTPUT_CONTRACT.md` §3.

---

## 9. Wrong domain — guessed, or supplied by the agent

**Hole:** Two ways to land in the wrong knowledge base. (a) If the agent supplies a
domain string, it can pass one that doesn't exist in the engine, or the wrong one — the
model's choice is unreliable. (b) Even with engine-side detection, a query that fits no
domain could be forced into the nearest one and confidently return an irrelevant
regulation.

**Solution:** the engine **owns domain detection** — `search_knowledge` takes only
`query`; there is no `domain` argument. Detection is an anchor match on the query vector
against pre-embedded domain anchors. A **confidence threshold** gates it: below
threshold the engine returns empty results with `detected_domain: null` and
`confidence: "none"` rather than guessing. As domains grow, an ambiguous match can fan
out to the top-N domains instead of forcing one. `explain_routing` exposes the decision
for tuning. A confidently wrong domain is worse than an honest "nothing matched."

---

## 10. Disk exhaustion during ingestion / split

**Hole:** At ~1.3 TB, running out of disk mid-ingest or mid-split can corrupt state.

**Solution:** pre-flight free-space check before bulk-load and before a split/re-cluster
writes a new version; fail fast with a clear error; never partially apply. (Mirrors the
upstream resource-check-before-start rule.)

---

## Review rule

Any change to routing, the candidate cap, the provider client, the queue, or the
cluster-update path must be checked against the relevant loophole above before merge.
