# dev_tools/eval — index

Every file here, by what it is for. The files stay flat because the scripts import each
other by name (`from run import ...`) and every number in `docs/` names its command by
path; this page is the grouping instead. `dev_tools/check_docs.py` fails CI on a file
that is not listed.

## The measurement: one command

| | |
|---|---|
| `eval_corpus.py` | serve one corpus through the engine and score it (`EVAL.md` §1) |
| `corpora.json` | how each corpus is served for an eval: bundle, database, port |
| `eval.env` | the engine settings an eval runs with, ✗ the operator's `.env` |
| `e2e.py` | the gate: score a question set through the real MCP server, stdio or HTTP |
| `e2e_sweep.py` | several factor configurations through one server process |
| `recall_at_k.py` | recall at the width a caller is handed, not just k=5 |
| `depth_recall.py` | recall at every cut-off, 5 through the whole pool |
| `algorithm_scenarios.py` | scenario tests for algorithm selection and the scoring vocabulary |

## Question sets

| | |
|---|---|
| `queries.json` | the Indonesian set, 604 cases with out-of-domain ones (its `_README`) |
| `queries_fresh.json` | the Indonesian fresh set, drafted after the levers were chosen; nothing is tuned on it (`EVAL.md` §2d) |
| `queries_ca_laws.json` | Canada, citation-labelled |
| `queries_us_cfr.json` | the US CFR, citation-labelled; 115 + 16, gate calibrated on its training split (floor 0.48) |
| `review/` | the legal reviewers' sheets and their instructions (`review/README.md`) |
| `draft/` | drafting a set: `sample.py` / `sample_corpus.py` (+ `corpus_*.json`, one per corpus) sample clauses, `check.py` / `check_citation.py` label and merge, `review.py` and `alternatives.py` export and import domain review |

## Agent-side inputs to the simulations

Texts an agent wrote per question, kept so a simulation can be rerun unchanged:
`hyde_train100.json`, `hyde_holdout.json`, `hyde2_holdout.json`, `hyde_fresh.json`,
`hyde_fresh_blind.json` (HyDE drafts), `loop_train100.json`, `loop_holdout.json` (round-2
drafts), `rewrites_holdout.json` (rewordings). Each states its own provenance in
`_README` where it has one; `ROADMAP.md` gives the results.

## Fitting the factors

| | |
|---|---|
| `fit_factors.py` | the `SCORING.md` weights, offline, against the shipped pool |
| `fit_pool.py` | the weights against the pool the engine builds |
| `fit_engine.py` | the `balanced` weights through the live engine |
| `run.py` | the arms scored with fusion reimplemented in Python: arm experiments, ✗ the gate |
| `pipeline_mirror.py` | a read-only Python mirror of the query path, sparse arm switchable |

## Diagnostics: where an answer is lost

| | |
|---|---|
| `miss_diag.py` | where a missed answer ranks against the whole corpus, per signal |
| `miss_map.py` | where the top-5 misses are, through the deployed engine |
| `article_diag.py` | right law, wrong article: what outranks the right one |
| `twins_diag.py` | what regional twins cost |
| `pool_depth.py` | was the answer retrieved at all (text arm, database only) |
| `pool_loss.py` | where a labelled answer is lost before ranking |
| `scoped_oracle.py` | oracle: would searching only the named region find it |
| `gate_signals.py` | every signal the domain gate could use, per query |
| `arm_overlap.py` | the sparse arm's overlap and marginal recall |
| `routing_levels.py` | two-level routing against one level, in routing recall |

## Offline simulations (`*_sim.py`): would a rule help, before it is built

`article_sim.py` (ordering inside a regulation), `definition_sim.py` (definitions
articles), `doc_rescore_sim.py` and `doc_support_sim.py` (regulation-level evidence),
`phrase_sim.py` (adjacency), `bridge_sim.py` (a declared vocabulary bridge),
`header_sim.py` (a richer embedded header), `dense_variants_sim.py` (the query side),
`hyde_sim.py`, `loop_sim.py`, `rounds_sim.py` (agent rounds), `swarm_sim.py`,
`query_swarm_sim.py` (many small searches), `tail_sim.py` (deeper arms).

## Arms, knobs and other indexes

| | |
|---|---|
| `arm_latency.py` | each arm timed separately |
| `probe_sweep.py` | what `CLUSTERS_PROBED` buys, at every width |
| `cluster_batch_sweep.py` | what `CLUSTER_BATCH` costs in memory and buys in latency |
| `cluster_layout.py` | does row order decide the cold cost of the dense arm |
| `lexical_prune.py` | the global lexical arms with fewer query terms |
| `mrl_recall.py` | Matryoshka truncation: 1024 vs 512 vs 256 |
| `bm25_compare.py`, `fused_bm25.py` | ParadeDB `pg_search` BM25 against the lexical arms |
| `pdb_vector_compare.py` | ParadeDB's IVF index against Vera's routing |

## Scale and load

| | |
|---|---|
| `scale_clone.py` | an N-times copy of a Ravel-prepared corpus, in the engine's layout |
| `scale_probe.py` | a synthetic corpus N times the real one |
| `scale_validate.py` | the scoring layer at 355K and 5.1M |
| `load_test.py` | concurrency and latency through the deployed HTTP server |
| `hardware_profile.py` | cold start to steady state, the whole stack |

## Public benchmarks

`beir/`: `sources.py` turns a BEIR corpus into Ravel's input, `bench.py` runs Vera,
pgvector and Qdrant on it with `algorithms.json`, and `results/` holds what they scored
(`BEIR.md`).
