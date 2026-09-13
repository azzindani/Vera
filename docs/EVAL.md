# EVAL.md

The harness that turns "should work" into "measured to work". Every number in
`README.md`, `HARDWARE.md` and `CONFIGURATION.md` comes from here.

The scripts live in `dev_tools/eval/`. They are operator-side and never run on the
query path.

---

## 1. Run it

Against the deployed server over HTTP:

```bash
VERA_HTTP=http://localhost:8081 python dev_tools/eval/e2e.py
```

Against a local binary over stdio:

```bash
VERA_EXE=./target/release/vera-mcp python dev_tools/eval/e2e.py
```

To compare scoring configurations against each other, one process, weights
varied per request:

```bash
python dev_tools/eval/e2e_sweep.py
```

! `e2e.py` scores **through the shipped binary**. This is not a convenience — an
offline harness that reimplements routing and fusion is measuring itself. The previous
harness (`run.py`) fused with equal weights of its own and reported 50.0% for a server
that was delivering 38.6%. `run.py` is kept for arm-level experiments; it is not the
gate.

---

## 2. The dataset

`dev_tools/eval/queries.json` — 50 labelled cases across 12 question shapes.

Each label was written by opening the clause in the corpus and phrasing the question
the way a person would ask it, deliberately avoiding the clause's distinctive wording
so nothing leaks into the retrieval it is meant to test.

| Shape | n | What it tests |
|---|---|---|
| `numeric` | 9 | a deadline, percentage or amount — one right number |
| `procedural` | 6 | how something is done, or what happens if it is not |
| `out_of_domain` | 6 | the corpus cannot answer this; correct behaviour is **empty** |
| `exact_ref` | 5 | the query names a regulation — exercises the routing bypass |
| `sanction` | 5 | what penalty attaches to a breach |
| `obligation` | 4 | "is X required to …" |
| `conceptual` | 4 | natural question, no reference given |
| `authority` | 3 | "who has the power to …" — the commonest real question |
| `exception` | 3 | the carve-out, not the rule |
| `multi_tier` | 2 | answered by **both** a parent law and its implementing rule |
| `hard_negative` | 2 | a near-identical clause about a different subject must not outrank |
| `underspecified` | 1 | several equally correct answers; any one is a pass |

! **Labels are article-level, not chunk-level.** Chunk ids are an artefact of how the
corpus was split and re-chunking changes every one of them. An article is a property of
the law itself and survives. Any chunk of a labelled article counts as a hit.

! The `out_of_domain` set includes Indonesian-language cases on purpose. They separate
"detects the domain" from "detects the language" — a distinction a language check would
conflate, and the reason the domain gate needs a lexical half as well as a semantic one.

---

## 3. Metrics

| Metric | Why |
|---|---|
| **Recall@5** | the primary gate: is the right article in the top 5? |
| Routing recall | was it in a probed cluster at all? Separates a routing failure from a ranking failure. |
| Exact-match recall | did the global identifier path return the named regulation? |
| MRR | ranking quality among what was retrieved |
| Domain-gate accuracy | out-of-domain refused; in-domain not refused |
| Latency p50 / p95 | so a dial that helps recall and wrecks latency is visible |

Routing recall is the diagnostic that matters most when Recall@5 moves: high routing
recall with low Recall@5 means fusion or the candidate caps; low routing recall means
clustering or `CLUSTERS_PROBED`.

---

## 4. Current results

Through the real server, 44 retrievable cases, `dense=0 sparse=1 text=1`:

| | | |
|---|---|---|
| | factors off | **shipped** |
| Recall@5 | 50.0% | **54.5%** |
| Recall@10 | 52.3% | **59.1%** |
| MRR | 0.360 | **0.454** |

| | |
|---|---|
| Domain gate | 6/6 out-of-domain refused, 0/44 false refusals |
| Latency p50 / p90 | 691 ms / 1,046 ms |

Both columns come from **one binary** (`e2e_sweep.py`), with `factor_weights`
varied per request, so the difference is the scoring layer and nothing else —
rebuilding between configurations would vary the binary too. Identical under the
2 vCPU / 4 GB memory limits and without them — the constraints cost latency, not
quality.

### These numbers were not reproducible until they were

! Every result above is the mean of nothing: it is **one run, repeated three
times, identical**. That is worth stating because it was not true before
`2026-09-13`. `ORDER BY <distance> LIMIT k` returns an arbitrary member of any
tie group straddling the limit, so **7 of 44 queries returned different results
across three runs of one server process**, and two identical evaluations of the
same binary scored 52.3% and 54.5%.

Every comparison made against a corpus this size has to survive that first. A
2.5-point difference between two configurations, on 40 to 44 cases, is one case
— which is also the size of the noise the engine was generating on its own.

The fix is in `store::search`: each arm reads `OVERFETCH × k` and breaks ties on
id in memory (`settle`). Appending `, id` to the SQL instead costs **8×** — it
defeats the RUM index's early termination and sorts all 286,199 matching rows
(765 ms → 6,069 ms). Reading deeper costs nothing measurable: 922 ms at
`LIMIT 20`, 944 ms at `LIMIT 60` in Postgres, and p50 732 ms → 691 ms end to end.

! **The window is narrowed, ✗ closed.** A tie group straddling `OVERFETCH × k`
is still resolved by the database. At `OVERFETCH = 3` the residual is 0 of 44
queries over three runs; that is a measurement, not a guarantee, and it should
be re-run whenever `PER_ARM_K` changes.

Per-arm, measured independently:

| arm | Recall@5 alone |
|---|---|
| text (`tsvector`) | 40.9% |
| sparse (BM25) | 38.6% |
| dense | 0.0% — see `EMBEDDING.md` §5 |

### Where the answer is, as opposed to where it ranks

Recall@5 cannot distinguish "never retrieved" from "retrieved and buried", and
those have different fixes. `pool_depth.py` separates them by running the text
arm alone to increasing depth:

```bash
python dev_tools/eval/pool_depth.py
```

| pool depth | labelled article present | right regulation, wrong clause | regulation absent |
|---|---|---|---|
| 5 | 40.0% | 15.0% | 45.0% |
| 10 | 60.0% | 10.0% | 30.0% |
| 20 | 70.0% | 15.0% | 15.0% |
| 60 | **80.0%** | 10.0% | **10.0%** |

! **Retrieval is not the limiting factor; ranking is.** One arm alone has the
answer in a 60-candidate pool 80% of the time. The engine returns it in the top
5 half the time. The gap between those two numbers is what `SCORING.md` exists
to close, and it is larger than the gap any new arm could close.

The middle column is the case for sibling expansion (`SCORING.md` §7): the
right law was retrieved and the wrong clause of it was returned.

! **Denominator: 40, not 44.** This measures against `answer_articles`, and 4
of the 5 `exact_ref` cases are labelled with a regulation only — correctly, since
any chunk of the named regulation is a pass for them. They cannot be sorted into
article-level buckets, so they are excluded here and included in Recall@5.

---

## 5. What the eval decides

| Decision | Signal |
|---|---|
| fusion weights | Recall@5 / MRR per weight combination — refitted after every re-chunk |
| `CLUSTERS_PROBED` | the smallest value that holds Recall@5 |
| candidate caps | Recall@5 against cap-induced misses |
| `CANDIDATE_POOL` | `pool_depth.py` — the depth at which "regulation absent" stops falling |
| factor weights + `RELEVANCE_FLOOR` | `fit_factors.py` — leave-one-out Recall@5, ✗ the in-sample peak, and fitted **jointly** so no factor is credited for the floor's work |
| chunk strategy | Recall@5 across question shapes |
| domain-gate floors | out-of-domain rejected without rejecting real queries |
| re-cluster trigger | a drop in routing recall over time |

Optimisations this harness has **rejected**:

| Tried | Result |
|---|---|
| drop low-IDF terms from the text query | 14× faster, Recall@5 40.9% → 22.7%. No. |
| AND-first, OR as fallback | no gain: AND returns zero rows for 42 of 44 queries |
| `temporal` factor (recency) | nothing at any weight, under any floor. A 1999 statute still governs unless repealed. |
| `completeness` factor (body length) | +2.5 **before** a relevance floor existed, **0.0 after** — it was serving as a crude relevance proxy, not measuring completeness. It returns at 1.0 in any fit that does not bound `Σw`, which is the clearest evidence the bound is doing work. |
| an unbounded metadata prior | the best unconstrained fit scores the **highest Recall@5 measured on this corpus (56.8%)** and is the worst answer on the list: Recall@10 is also 56.8% — positions 6–10 find nothing new — and MRR falls to 0.382 against 0.454. Its bound is 3.75× against a pool spanning 2.56×, so metadata decides the order. Recall@5 alone cannot see this; `SCORING.md` §3 is the rule that rejects it. |
| the relevance floor at 0.4 | fitted against the **text arm**, which the engine does not rank. On the fused pool it costs 2.5 points; refitting it against IDF-weighted `bm25::evidence` does not rescue it. Kept at 0.3, where it is free. |
| `enacting_body` in the authority score | the column is unusable: `PERATURAN BUPATI` / `MA` (10,395 rows), `PERATURAN DAERAH KABUPATEN` / `RI` (7,770). Extraction artefacts, not enacting bodies. |
| fine-grained regulation hierarchy | **unmeasurable at n=40.** The legally-correct ordering and an inverted one score identically (57.5% / 47.5% LOO), and a coarse national-vs-local split does at least as well. |
| TurboQuant / TurboVec vector compression | wrong bottleneck: the dense arm is 53 ms of a 1,059 ms query and returns 0.0%. Compressing it 16× buys ~1% of latency on the one arm that contributes nothing. |

Both looked obviously good on paper. That is what the harness is for.

---

## 6. Cadence and the open gap

- On every dial or config change: re-run. A change that lowers Recall@5 is rejected.
- After every re-chunk: refit the fusion weights, then re-run.
- Periodically in production: a routing-recall drop is the trigger for re-clustering —
  drift detected by measurement, not guessed by calendar.

### What the fitting harness has to match, and why

`fit_factors.py` reranks a pool it builds itself, so every difference between
that pool and the engine's fits weights against something nobody serves. Four
such differences have been found, all of them silent — the harness produced a
number either way:

| divergence | how it ended |
|---|---|
| fitted the **text arm**, engine ranks the **fused** pool | `--pool fused` is now the default. This is the one that mattered: it is why the floor was 0.4. |
| harness **renumbered** survivors `1/(k + rank)`, engine keeps the fused score | fixed — the pool's own score is carried through. It had been measured as agreeing 40/40 at the then-shipped config, which is agreement by luck, not by rule. |
| harness's `HIERARCHY` table was the **inverted** one | fixed, and pinned: `factors::tests::hierarchy_matches_the_engine` parses the Python table and asserts it against `tier()`. |
| harness took SQL `LIMIT` at face value | fixed — it mirrors `OVERFETCH` and `settle`. |

! A cross-language lookup table needs a test that the two copies agree, not a
comment saying they should. The hierarchy diverged for two commits without
moving any number.

! **38 of the 50 labels have not been reviewed by a domain expert.** Whether a clause
genuinely *answers* a question is a lawyer's judgement, not a retrieval engineer's, and
a wrong label is worse than no label: it silently moves every dial this document gates.
Treat 54.5% as a number measured against labels of known-imperfect provenance.
