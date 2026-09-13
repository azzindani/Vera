# dev_tools

Everything that runs **off** the query path: building a corpus, clustering it, scoring
it, and seeding fixtures.

! None of this ships. The deployed system is the engine and the database; these are the
operator's tools for producing what the engine serves. They are Python because the
libraries are — GPU transformers, PDF extraction, k-means — and they run on the
operator's own hardware, not on the VPS.

---

## Layout

```
pre_embed/       source documents → chunks → vectors → Postgres
cluster_maint/   k-means over the dense vectors; layer-2 centroids
eval/            the harness that gates retrieval quality
fixtures/        a ~30-row corpus so integration tests can run in CI
```

| Doc | |
|---|---|
| [PRE_EMBEDDING.md](PRE_EMBEDDING.md) | the offline ingestion pipeline |
| [CLUSTER_MAINTENANCE.md](CLUSTER_MAINTENANCE.md) | cluster drift, and the design for handling it |
| [../docs/EVAL.md](../docs/EVAL.md) | what the eval measures and what it gates |

---

## Building a corpus

```bash
python pre_embed/ingest.py --manifest <manifest.json> --run-id <run-id>
python cluster_maint/kmeans.py
```

`ingest.py` produces text, dense vector and sparse vector in **one pass over one text**.
That is the load-bearing property: the corpus this project inherited had them produced
by separate runs, which is why its vectors could not be reproduced from its text.

It is streaming and resumable. Each batch is embedded, written and committed, so
`ingest_progress` makes a re-run continue rather than restart — a crash at 90% does not
throw away the GPU time.

The run records its own recipe into `corpus_meta`: model, dimension, pooling,
normalization, dtype, instruction, tokenizer hash, and the BM25 vocabulary hash. The
engine reads that at startup and builds itself to match, rather than asserting a model
the corpus may not share.

! **The vocabulary is part of the corpus.** Sparse vectors are indexed by vocabulary
position. Keep the `.bm25.json` artifact with the database it was built from, and point
`BM25_VOCAB` at that one — another run's vocabulary compares unrelated dimensions and
returns plausible nonsense rather than failing.

---

## Clustering

```bash
python cluster_maint/kmeans.py
python cluster_maint/eval_routing.py
```

Spherical k-means, not Euclidean: the vectors are L2-normalised, so cosine similarity
*is* the dot product and centroids must be renormalised after each update. Plain
Lloyd's on normalised vectors quietly optimises the wrong objective.

Cluster count is a config value, not a constant. Around 2,000 rows per cluster is what
makes routing meaningful at this corpus size — probing 5 touches ~6% of the corpus, the
same pruning ratio a much larger deployment would target. Leave clusters an order of
magnitude larger and probing 5 touches 29%, at which point routing proves nothing.

numpy only, no sklearn. The recipe is short enough to own outright, and owning it means
the assignment rule is inspectable rather than a library default.

---

## Scoring

```bash
VERA_HTTP=http://localhost:8081 python eval/e2e.py    # the deployed server
VERA_EXE=./target/release/vera-mcp python eval/e2e.py # a local binary over stdio
```

`e2e.py` is the gate. It drives the shipped binary, so it measures the same weights, the
same domain gate and the same contract a client gets.

`run.py` scores the **arms** — it reimplements fusion in Python with equal weights. It
is useful for arm-level experiments and it is not the gate. The two disagreeing is the
bug `e2e.py` exists to catch, and it has caught one: `run.py` reported 50.0% while the
server delivered 38.6%, because the engine's text weight was 0.0 and nothing was
measuring the engine.

```bash
python eval/pool_depth.py            # needs only the database
```

`pool_depth.py` answers a different question from either: not "did the answer rank?" but
"was the answer retrieved at all?". It runs the text arm alone straight against Postgres,
so it needs **no embedder and no GPU**, and it is the cheapest check in the repo —
useful whenever a change to chunking or the corpus might have moved what is reachable.
Results in [../docs/EVAL.md](../docs/EVAL.md) §4.

---

## Fixtures

```bash
python fixtures/seed.py
```

Builds a ~33-row corpus in seconds so the `#[ignore]`d integration tests can run in CI.
"Needs a database" stops meaning "never runs".

```bash
export DSN="host=localhost port=5432 dbname=vera_fx user=vera password=vera"

DATABASE_URL="$DSN" python fixtures/seed.py          # seed (writes seed.dense.json)
DATABASE_URL="$DSN" cargo test -p store -- --ignored     # the SQL layer   · 14 tests
VERA_FX_DSN="$DSN" cargo test -p vera-mcp -- --ignored   # the pipeline    · 16 tests
```

! **The whole pipeline is testable without a GPU, and the startup canary still runs
for real.** The fixture's dense vectors used to come from `random.Random(SEED)` in
sequence, which no other language can reproduce — so the canary, which re-embeds a
stored chunk and checks it lands where ingestion put it, could only be satisfied by a
real embedding server. Everything above the SQL layer was therefore untestable in CI:
routing, the domain gate, fusion, factor scoring, the option surface, the response
contract.

They are now a pure function of `(theme, body)`:

```
dense = normalize(0.95 · unit(theme) + 0.05 · unit(body))
unit(s) = normalize(fnv_vector(s))       # mirrors embed::StubProvider::vector_for
```

`seed.dense.json` records the body → theme map, which is the only thing a provider
cannot derive from the text it is handed. `FixtureProvider` in `crates/mcp/src/pipeline.rs`
reproduces the formula, so `Pipeline::new` passes its canary at cosine 1.000 — **not
bypassed**. A test constructor that skipped the canary would be a hole in invariant 3,
and these tests exist partly to prove the canary works.

! These test **mechanics, ✗ retrieval quality.** The vectors are derived from hashes
and say nothing about meaning. `eval/e2e.py` is the only thing that scores quality.
