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

Through the deployed server, target hardware profile:

| | |
|---|---|
| Recall@5 | **50.0%** (44 retrievable cases) |
| MRR | 0.360 |
| Domain gate | 6/6 out-of-domain refused, 0/44 false refusals |
| Latency p50 / p95 | 1,059 ms / 1,638 ms |

Identical under the 2 vCPU / 4 GB memory limits and without them — the constraints cost
latency, not quality.

Per-arm, measured independently:

| arm | Recall@5 alone |
|---|---|
| text (`tsvector`) | 40.9% |
| sparse (BM25) | 38.6% |
| dense | 0.0% — see `EMBEDDING.md` §5 |

---

## 5. What the eval decides

| Decision | Signal |
|---|---|
| fusion weights | Recall@5 / MRR per weight combination — refitted after every re-chunk |
| `CLUSTERS_PROBED` | the smallest value that holds Recall@5 |
| candidate caps | Recall@5 against cap-induced misses |
| chunk strategy | Recall@5 across question shapes |
| domain-gate floors | out-of-domain rejected without rejecting real queries |
| re-cluster trigger | a drop in routing recall over time |

Optimisations this harness has **rejected**:

| Tried | Result |
|---|---|
| drop low-IDF terms from the text query | 14× faster, Recall@5 40.9% → 22.7%. No. |
| AND-first, OR as fallback | no gain: AND returns zero rows for 42 of 44 queries |

Both looked obviously good on paper. That is what the harness is for.

---

## 6. Cadence and the open gap

- On every dial or config change: re-run. A change that lowers Recall@5 is rejected.
- After every re-chunk or re-embed: refit the fusion weights, then re-run.
- Periodically in production: a routing-recall drop is the trigger for re-clustering —
  drift detected by measurement, not guessed by calendar.

! **38 of the 50 labels have not been reviewed by a domain expert.** Whether a clause
genuinely *answers* a question is a lawyer's judgement, not a retrieval engineer's, and
a wrong label is worse than no label: it silently moves every dial this document gates.
Treat 50.0% as a number measured against labels of known-imperfect provenance.
