# SCORING.md

How Vera ranks. Retrieval finds candidates; this decides their order.

> **Status.** The factor model, viewpoints and consensus described here are
> **designed, not implemented**. What ships today is §7. Sections 1–6 are the
> target, written down so the build has something to be measured against.

---

## 1. Why two arms is not enough

Vera currently ranks on semantic similarity, term overlap and exact wording.
Three signals, all of them *retrieval* signals — each answers "does this text
look like the query?" and nothing else.

That question has a known failure, visible in Vera's own output today. Asked
about tax penalties, it returns a district `PERATURAN BUPATI` above the national
`UNDANG-UNDANG` that governs it. Both texts match the query. One of them is the
law and the other implements it locally, and nothing in a text-similarity score
can tell them apart.

The information needed to rank them is already in the corpus — `regulation_type`,
`enacting_body`, `year`, `chapter`, `article`, `about` — and is thrown away at rank
time.

**Multi-factor scoring is the correction:** retrieval finds what is *relevant*,
factors decide what is *authoritative, current, and structurally right*.

---

## 2. The factors

Six families. Only the first two require retrieval; the rest are properties of
the chunk, computable at ingest and free at query time.

| Factor | Source | Answers |
|---|---|---|
| `relevance` | dense + sparse + text arms | does this text address the query? |
| `exactness` | `identifier` match | did the query name this regulation? |
| `authority` | `regulation_type`, `enacting_body` | how binding is this instrument? |
| `temporal` | `year` | is this current, or superseded? |
| `structural` | `chapter`, `article` | is this an operative clause or an annex? |
| `topical` | `about` | is the instrument *about* what was asked? |
| `completeness` | `length(body)`, legal-term density | is this a whole provision or a fragment? |

### authority

Indonesian regulation is a strict hierarchy, so this is a lookup, not a model:

```
UUD 10  ·  TAP MPR 9  ·  UU 8  ·  PERPU 7  ·  PP 6
PERPRES 5  ·  PERMEN 4  ·  PERDA PROVINSI 3  ·  PERDA KAB/KOTA 2
authority = (hierarchy / 10) * 0.7 + (enacting_body_level / 5) * 0.3
```

### structural

Vera returns `LAMPIRAN / LAMPIRAN` hits above `Pasal` hits today. An annex is
rarely the answer to a question about obligations; an article usually is, and
`chapter` / `article` already record which is which — `locator_of` in
`pipeline.rs` reads exactly these two columns to build the citation.

---

## 3. The relevance gate — the factor model's load-bearing safety rule

! Factors are applied **after** a relevance floor, never instead of one.

```
relevance = mean(dense, sparse, text)
if relevance < RELEVANCE_FLOOR:  drop the candidate entirely
```

Without this, authority weighting ranks the most prestigious document in the
corpus first for *every* query — a banking law for a tax question, because it
scores high on authority and was never required to be relevant. High authority
and zero relevance is the specific failure the gate exists to prevent, and it is
a plausible-looking failure: the result is a real law, correctly cited.

---

## 4. Weights are per query type, and measured

A question naming `Pasal 9` needs exact wording. A question about who holds a
power needs authority. One weight vector cannot serve both.

| query type | leans on |
|---|---|
| `specific_article` | exactness, relevance |
| `definitional` | relevance, authority |
| `sanction` | relevance, structural |
| `conceptual` | relevance, topical |
| `authority` | authority, structural |
| `procedural` | relevance, completeness |
| `numeric` | exactness, structural |

! These are **fitted against `dev_tools/eval/queries.json`, ✗ chosen.** Every
case in that set already carries its `type`, so per-type weights are directly
measurable — and a weight that does not improve Recall@5 for its type does not
ship. This is the same discipline that governs the fusion weights, and for the
same reason: they are properties of the corpus, not of the engine.

---

## 5. Viewpoints and consensus

Several weight vectors are applied to **one** candidate pool. Each is a
viewpoint — the analogue of asking several paralegals with different instincts
to rank the same shortlist.

! Cheap by construction. Retrieval is the expensive half; scoring is a weighted
sum over a pool already in memory. Eight viewpoints cost what one does, **as
long as they share a pool.** A design that re-runs retrieval per viewpoint pays
eight times for the same candidates and is the thing to avoid.

```
agreement(chunk) = viewpoints ranking it top-k / total viewpoints
score(chunk)     = mean_score × (0.7 + 0.3 × agreement)
```

Agreement is a **modifier, ✗ a gate**. A chunk one viewpoint loves and seven
ignore is probably an artefact; a chunk all eight rank highly is probably the
answer. Neither is certain enough to hard-filter on.

### Consensus is where `confidence` comes from

Vera's output contract already carries `confidence: high | medium | low | none`,
currently set by a heuristic over score spread. Consensus is a better source:

| agreement across viewpoints | confidence |
|---|---|
| high, low score variance | `high` |
| split | `medium` |
| scattered | `low` — and the signal to do more work |

One mechanism, two jobs: it orders the results *and* it says how much to trust
the ordering.

---

## 6. Tiered effort — do the cheap thing first

! The budget is a ceiling, ✗ a target. A query that is answered confidently in
one round must not be made to spend thirty seconds proving it.

```
tier 1   routed dense + global arms → factor score → viewpoints     ~1s
         └─ consensus strong?  return.
tier 2   widen probe, relax thresholds, re-score the larger pool    ~3s
         └─ still contested?
tier 3   expand: cross-references, parent and implementing rules    ~10s+
```

Escalation is triggered by **measured disagreement**, not by query length or a
guess at difficulty. The system knows when it is uncertain, which is precisely
what §5 computes.

! **This conflicts with the current concurrency model** and the conflict is
unresolved. `MAX_CONCURRENCY=4` with `QUEUE_WAIT_MS=2000` was sized for ~1s
requests; a 30s tier-3 request means four callers occupy the server for half a
minute and the fifth is refused after two seconds. Peak RAM is also
`ceiling × working set`, and a longer request holds its working set longer.
Interactive retrieval and deep research are different budgets. Either the tier
is requested by the caller, or Vera picks one.

---

## 7. What the code does today, and what has to move

Read from the source, ✗ assumed. Each is a concrete blocker or enabler.

### The blocker: the pool is cut before metadata is fetched

`crates/mcp/src/pipeline.rs`:

```rust
let top: Vec<_> = fused.into_iter().take(self.cfg.top_k).collect();  // ~60 → 10
let rows = self.ops.chunks_by_id(&ids).await?;                       // metadata for 10
```

Factors would only re-rank the ten candidates that already won. A chunk at rank
15 carrying the governing law can never be rescued, which is precisely the case
the factor model exists for.

! **`take(top_k)` must move after factor scoring**, and `chunks_by_id` must run
over the whole fused pool. That is the one structural change the design needs;
everything else is additive.

### Already there

| | |
|---|---|
| `bm25::evidence(query, texts)` | matched IDF-mass over query IDF-mass — **this is the §3 relevance gate**, already written, already used by the domain gate |
| `ChunkRow` | carries `regulation_type`, `regulation_number`, `year`, `chapter`, `article`, `body` |
| `Fused.contributions` | `(arm, rank)` per candidate — the per-arm input a viewpoint needs |
| `identifier::extract` | the `exactness` factor's input, with the regulation-tier table |

### Needs adding

- `enacting_body` and `about` are in the schema and **not selected** by
  `chunks_by_id`. One line each.
- `ComponentScores` publishes `dense` and `bm25` only — **the text arm, the
  best performer at 40.9%, is invisible in the output.** Factors will need their
  own scores published alongside, so this changes anyway.

### Will be replaced

`engine::trust_from` thresholds confidence on absolute RRF scores (`0.030` /
`0.020`). Those values depend on the arm weights and on `k`, so changing
`TEXT_WEIGHT` silently redefines what "high confidence" means. Consensus (§5)
replaces it; until then it is a latent bug, ✗ a design.

### Dead

`engine::routing::{detect_domain, route, DomainAnchor, Route}` are exported and
never called — the pipeline implements its own gate. `routing.rs` still
describes domain detection as "anchor match on the query vector", which is not
what the engine does.

### Not reachable from this corpus

`Locator.page` is hardcoded `None` in `to_results`, and there is **no page
column in the schema**. `OUTPUT_CONTRACT.md` §2 shows `"page": 14` in its
example; no result from this corpus can carry one.

---

## 8. What actually ships today

| | |
|---|---|
| arms | dense (weight 0.0), sparse, text |
| fusion | RRF over ranks, no factors |
| exact identifiers | retrieved globally, fused normally |
| confidence | heuristic over score spread |
| effort | single round, always |
| Recall@5 | **50.0%** |

Everything above is the plan. The number is the baseline it has to beat.
