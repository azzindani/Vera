# TOOL_SURFACE.md

How a calling agent steers Vera, and what it is deliberately not allowed to steer.

`MCP_ENGINE.md` §2 defines the five tools and their wire shapes. This document
defines the **arguments** — the knobs an agent may turn per request, the ones the
server keeps, and the reasoning for each side of that line.

> **Status.** §3 (retrieval controls) and §5 (profiles) are **built**. §4 (effort),
> §6 (expansion and the citation graph) and §7 (consensus) are **designed, not
> implemented**. §8 is the ledger.

---

## 1. The principle: richness in arguments, ✗ in tools

The obvious way to offer more capability is more tools — `semantic_search`,
`keyword_search`, `deep_research`, `authority_search`. Vera does not do that, for
two reasons.

**An agent picks a tool before it knows anything.** It has the query and nothing
else — no idea whether the corpus covers it, whether the answer is one clause or
twelve, whether the arms agree. Tool choice forces a commitment at the moment of
least information. An argument on one tool lets the agent call, look, and adjust.

**Every tool is permanent surface.** `CLAUDE.md` §6 caps this engine at eight, and
the cap is doing work: five tools with rich arguments compose into one loop an
agent can learn (`ROUTE → SEARCH → READ → VERIFY`); twelve tools is a menu nobody
reads to the end.

So: **the tool count stays at five. Capability arrives as arguments.**

---

## 2. What the agent may steer, and what it may not

The line is not about trust. It is about **what the caller can possibly know**.

| The agent decides | Vera decides |
|---|---|
| what to ask | which domain the question belongs to |
| how many results it wants | which clusters to probe |
| how wide to look before ranking | what "relevant" means |
| which arms to use | what the factor weights are |
| how much effort it is willing to spend | whether that effort is needed |
| what it is optimising *for* | what numbers express that |

Everything in the left column is a property of the **caller's situation** — its
latency budget, its context window, what it is doing with the answer. Vera cannot
know any of it.

Everything in the right column is a property of the **corpus**, fitted against the
eval set (`EVAL.md`). An agent has no fitting signal for these and no way to check
whether a value it invented was better or worse than the one it replaced.

! **Three rules govern every argument below.**
>
> 1. **A caller may narrow a server limit, never widen it.** Every numeric argument
>    is clamped to the configured ceiling. This is why `read_chunk.max_chars`
>    already works the way it does (`MCP_ENGINE.md` §2), and every new knob follows
>    it. A request is a preference, ✗ an override.
> 2. **Anything that changed the ranking is echoed in the response.** A result that
>    cannot be reproduced cannot be verified, and "measure, then claim"
>    (invariant 15) is unenforceable if no two calls are comparable.
> 3. **Every argument is optional and every default is the measured one.** A caller
>    that passes nothing gets exactly today's behaviour.

---

## 3. Retrieval controls  · built

```
search_knowledge(
    query: str,
    mode: "hybrid" | "keyword" | "semantic" = "hybrid",
    top_k: int = <TOP_K>,
    candidate_pool: int = <CANDIDATE_POOL>,
) -> dict
```

### `mode`

Which arms run. `hybrid` fuses all three; `keyword` runs sparse + text only;
`semantic` runs the dense arm alone.

! **`semantic` is measurably broken on this corpus and says so.** The dense arm
scores **0.0% Recall@5** alone and ships at weight 0.0 (`EVAL.md` §4,
`EMBEDDING.md` §5). The mode exists because the argument surface should not be
reshaped when the embedding space is fixed — but requesting it returns results
with an explicit `hint` that this arm is not currently contributing. Offering a
capability we measured as absent, silently, would be the worse option.

### `top_k` and `candidate_pool`

`top_k` is what comes back. `candidate_pool` is how many candidates are ranked
before the cut — the single most valuable dial in the engine, per `SCORING.md` §7:

| pool depth | labelled article present |
|---|---|
| 5 | 40.0% |
| 60 | **80.0%** |

Both clamp to the server's configured ceiling, and `candidate_pool` is additionally
floored at `top_k` — a pool smaller than the answer silently caps the reply.

---

## 4. Effort  · designed, ✗ built

```
    max_rounds: int = 1,
    budget_ms: int = <derived>,
```

This resolves the question `SCORING.md` §6 left open — *"either the tier is
requested by the caller, or Vera picks one"*. **The caller sets a ceiling; Vera
decides whether to spend it.**

! `max_rounds`, ✗ `rounds`. Invariant 10: effort is escalated on **measured
disagreement**, never by default. If the argument reads as a target, every caller
will set it high, because more looks better — and a query answered confidently in
one round will be made to spend thirty seconds proving it.

! **The concurrency conflict is unresolved and is the reason this is not built
yet.** `MAX_CONCURRENCY=4` with `QUEUE_WAIT_MS=2000` was sized for ~1 s requests.
Four callers each spending 30 s occupy the server for half a minute and the fifth
is refused after two. Peak RAM is `ceiling × working set`, and a longer request
holds its working set longer. Shipping `max_rounds` before that is resolved turns
a documented design tension into an outage.

---

## 5. Scoring: profiles, ✗ raw weights  · built

```
    profile: "balanced" | "authority_first" | "operative_only" = "balanced",
```

This is the part of the proposal that changed in review, and the measurement is
the reason.

Fitting the factor weights took **625 weight combinations against 40 labelled
cases**. The in-sample best was +17.5 points; leave-one-out was **+7.5**
(`SCORING.md` §2, `dev_tools/eval/fit_factors.py`). Ten points of that apparent
gain was the fitting procedure flattering itself.

**An agent setting `authority=0.8` has no fitting signal at all.** It cannot run
the eval, cannot see Recall@5, and will pick a number that sounds right. It will
also pick a different one next time, and then the same query returns a different
ranking for reasons nothing in the response explains.

So the agent selects **intent**; Vera keeps the **numbers**:

| profile | fitted for | weights |
|---|---|---|
| `balanced` | the general case — the fitted default | authority 0.5 · structural 0.25 · completeness 0.25 |
| `authority_first` | "what is the governing rule?" — leans on the hierarchy | *to be fitted* |
| `operative_only` | obligations and sanctions; suppresses annex material hard | *to be fitted* |

! **A profile is not a preset someone chose.** Each is fitted against the eval
subset for the query shapes it targets, and a profile that does not beat
`balanced` on its own shape does not ship. Until a profile is fitted, it is not
listed in the tool schema — an unfitted profile is a raw weight vector with a
friendly name.

### The escape hatch

```
    factor_weights: {authority: float, ...}   # experimental
```

Raw weights remain reachable for experimentation, and carry two conditions: the
response sets `experimental: true`, and the **effective weights are echoed back**
whether or not the caller passed any. A ranking that cannot be reproduced is not
evidence.

---

## 6. Expansion and the citation graph  · designed, ✗ built

```
    expand: ["siblings", "citations"] = [],
```

`SCORING.md` §7 measures the case for `siblings` — in 10% of labelled cases the
engine held the right regulation and returned the wrong clause of it, and the
`chunks_identifier_idx` that serves the exact-identifier path already answers the
query (11,750 rows in 24 ms).

`citations` is the knowledge-graph axis, and the corpus supports it **today,
without a re-embed**:

| | |
|---|---|
| indexable chunks carrying an explicit citation | **45,683** (12.8%) |
| distinct cited regulations | 4,676 |
| **resolvable inside this corpus** | **3,424 (73%)** |

```sql
-- chunks citing a regulation by type, number and year
SELECT count(*) FROM chunks WHERE indexable AND body ~*
  '(undang-undang|peraturan pemerintah|peraturan presiden|peraturan daerah)\s+(republik indonesia\s+)?nomor\s+[0-9]+\s+tahun\s+[0-9]{4}';
```

! `identifier::extract` **already parses exactly this form**. It is applied to
queries today, and it is plain text in, structured identifier out — so the edge
table is a pass over `body` with code that exists, no model and no GPU.

! Both expansions are **admitted, ✗ merged**. An expanded chunk arrives with no
retrieval score — nothing matched it, it came in as a neighbour — and any score
invented for it is fiction. Admission is the §3 relevance gate reused:
`bm25::evidence(query, [body])` against the same floor. One definition of
"relevant" for the whole engine.

! What Vera does **not** copy from `06_ID_Legal`: a hierarchy walk that links on
`|tier diff| ≤ 1` **and** `|year diff| ≤ 5` without ever checking subject. On this
corpus that admits thousands of topically unrelated chunks (`SCORING.md` §7).

---

## 7. Consensus  · designed, ✗ built

```
    viewpoints: int = 1,
```

Several weight vectors over **one** candidate pool (`SCORING.md` §5). Agreement
across viewpoints becomes `confidence`, and disagreement is the signal that
triggers §4's extra rounds.

! One pool, ✗ one retrieval per viewpoint. `06_ID_Legal` runs a separate search
per persona, which makes "this viewpoint ranked it low" and "this viewpoint never
retrieved it" indistinguishable, and pays N× for candidates that are mostly the
same. Sharing the pool is what keeps the vote meaningful *and* cheap.

---

## 8. What ships today

| Argument | Status |
|---|---|
| `query` | ✔ |
| `mode` | ✔ — `semantic` hinted as non-contributing |
| `top_k`, `candidate_pool` | ✔ clamped to server ceilings |
| `profile` | ✔ `balanced` only; others unfitted, so unlisted |
| `factor_weights` | ✔ experimental, echoed back |
| `max_rounds`, `budget_ms` | ✗ blocked on the §4 concurrency conflict |
| `expand` | ✗ measured and specified, not built |
| `viewpoints` | ✗ |

! **No argument here has been measured through the engine yet.** The factor layer
itself is fitted offline against one arm; `e2e.py` is the only thing that scores
what ships (`EVAL.md` §1), and every default above is the value that was measured
without these arguments existing. Adding knobs does not change the number — it
changes what can be attributed when the number moves.
