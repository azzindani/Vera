# CONFIGURATION.md

Every tunable Vera has, where it is read, and what moving it costs.

Nothing in this table is compiled into the binary. All of it is resolved once at
startup by `Settings::from_env` (`crates/mcp/src/main.rs`), and a value that is
missing or unparseable stops the process there — never on the first request, where
the orchestrator has already been told the server is healthy.

`.env.example` is a working set. `docker compose` reads it directly.

---

## 1. Required

No defaults, deliberately. Each names something outside the process that only the
operator knows, and each fails *silently* if guessed wrong rather than loudly.

| Variable | What it is | Why there is no default |
|---|---|---|
| `DATABASE_URL` | libpq connection string | a default pointing at localhost is how a container ends up serving an empty corpus and saying nothing |
| `EMBED_ENDPOINT` | embedding server base URL | the corpus's vector space lives behind this URL; the wrong one is caught by the canary, but only if it is a *different* model |
| `BM25_VOCAB` | path to this corpus's vocabulary artifact | sparse vectors are indexed by vocabulary **position**; another run's vocabulary compares unrelated dimensions and returns plausible nonsense |

---

## 2. Transport

| Variable | Default | Notes |
|---|---|---|
| `TRANSPORT` | `stdio` | `stdio` or `http`. Not case-folded — the documented spelling is the accepted one. |
| `HTTP_ADDR` | `0.0.0.0:8081` | bind address, `http` only |

`stdio` is serial: one line read, handled, answered, then the next. Nothing can
queue, so `QUEUE_WAIT_MS` has no effect there.

---

## 3. Concurrency — the OOM guarantee

```
Peak RAM = fixed cost + (MAX_CONCURRENCY × one cluster)
```

| Variable | Default | Notes |
|---|---|---|
| `MAX_CONCURRENCY` | `4` | the hard ceiling on in-flight requests |
| `QUEUE_WAIT_MS` | `2000` | how long a call may wait for a slot before being refused |
| `STATEMENT_TIMEOUT_MS` | `15000` | how long a query may **run** before Postgres cancels it; `0` disables |

! `MAX_CONCURRENCY` and the container's memory limit are **one decision, not two**.
Raising it raises peak RAM linearly; the worst measured cluster is 23.4 MB.
`0` is rejected at startup — it parses fine and would produce a server that reports
healthy and refuses every request.

! `QUEUE_WAIT_MS` is the half of invariant 6 that people forget. A bounded queue
alone still lets a caller block indefinitely behind a full one; the bound has to be
on **time** as well as depth.

! `STATEMENT_TIMEOUT_MS` is the third bound, and the one that was missing. The
semaphore caps what runs, `QUEUE_WAIT_MS` caps what waits, and this caps how **long**.
Without it a slow query holds its permit for its whole duration — four of those and
every later caller is refused while `permits_available` sits at zero. It is set as a
connection option, so it costs no round trip and cannot be skipped by a code path that
forgets to issue it; any `options` already in `DATABASE_URL` are appended to.

15 s is ~9× the measured p95 of 1,638 ms. Raise it only alongside the effort tiers in
`SCORING.md` §6, and remember it bounds *each statement*, not the whole request.

A refusal is `-32000` over JSON-RPC and `503` + `Retry-After: 1` over HTTP — never
`500`. The distinction is the point: the request was never attempted, so a retry is
safe.

---

## 4. Retrieval dials

| Variable | Default | Effect |
|---|---|---|
| `CLUSTERS_PROBED` | `5` | layer-2 probe width. Recall vs latency — **not** vs RAM, because clusters load one at a time. |
| `PER_CLUSTER_K` | `20` | candidates kept per cluster per arm |
| `PER_ARM_K` | `20` | candidates each global arm contributes |
| `TOP_K` | `10` | results returned |
| `SNIPPET_CHARS` | `280` | preview length per result |

---

## 5. Fusion weights

| Variable | Default | |
|---|---|---|
| `DENSE_WEIGHT` | `0.0` | |
| `SPARSE_WEIGHT` | `1.0` | |
| `TEXT_WEIGHT` | `1.0` | |

! These are properties of the **corpus**, not of the engine, and a corpus is
re-chunked far more often than the engine is rebuilt. Refitting must not require a
release — which is exactly what happened once: the defaults were fitted on an
unchunked corpus where the text arm scored 0.0%, chunking took that arm to 40.9% on
its own, and the weights were never refitted. The engine shipped at 38.6% Recall@5
while the offline harness reported 50.0%, because the harness fused with its own
weights. Stale weights are silent: every arm still runs and the results still look
reasonable.

Measured through the deployed server on the current corpus, 44 retrievable cases:

| dense | sparse | text | Recall@5 | MRR |
|---|---|---|---|---|
| 0 | 1 | 0 | 38.6% | 0.344 |
| 0 | 1 | 1 | **50.0%** | 0.360 |
| 1 | 1 | 1 | 50.0% | 0.360 |

Recall@5 is flat for any text weight in 0.3–1.0 and MRR wanders 0.360–0.383 with no
trend, so that spread is noise on n=44. Equal weight is the RRF paper's default and
claims no precision the measurement supports.

All three at zero is rejected at startup: it fuses nothing and returns nothing, which
is indistinguishable from a corpus that simply has no match.

---

## 6. Gates

| Variable | Default | Effect |
|---|---|---|
| `CANARY_MIN_COSINE` | `0.98` | startup vector-space check |
| `DOMAIN_FLOOR` | `0.45` | nearest-centroid similarity below which results carry a weak-match hint |
| `DOMAIN_LEXICAL_FLOOR` | `0.40` | minimum IDF-mass of the query the evidence must account for |
| `GATE_SAMPLE` | `5` | how many sparse hits the lexical evidence pools over |

! Loosening the canary is a deliberate, visible act. There is no code path that
quietly skips it. `halfvec` storage puts an exact round-trip near 0.999 rather than
1.0 — that gap is fp16 rounding, not drift, and a genuinely different model scores
far below the floor.

### Vocabulary coverage — measured, and not a problem

The BM25 vocabulary keeps the 20,000 **most frequent** terms, which sounds backwards:
frequency selects for the least discriminative words and drops the rare ones that
identify a document. Measured against the 44 in-domain eval queries it costs almost
nothing — **9 of 422 query terms are out of vocabulary, 2.1%** — and the queries
carrying OOV terms are not the queries that miss: four of the six worst rank in the
top four.

! Worth knowing anyway, because the pattern is systematic. Every OOV term is a
conversational question-word — `bisakah`, `bolehkah`, "can it", "is it allowed" — which
by construction never appears in regulation text. An unknown term is charged
`unknown_idf`, the **maximum IDF in the vocabulary** (9.98 against a median of 8.75),
so a naturally-phrased question puts the largest possible weight in the gate's
denominator on words that could never be matched. On the current set that causes 0/44
false refusals, so it is a sharp edge, ✗ a live bug. It would bite first on a corpus
with a narrower vocabulary or a gate floor raised much above 0.40.

Both halves of the domain gate are load-bearing, measured on the 50-case eval set
with identifier queries exempt (invariant 4): together they reject 6 of 6
out-of-domain queries and 0 of 39 real ones. Centroid similarity alone cannot reject
a plumbing question asked in Indonesian (0.717, above the in-domain mean) because it
tracks language, not subject. Lexical evidence alone cannot reject an English
general-knowledge question (0.789), because English function words do occur in this
corpus. Each covers the other's blind spot.

---

## 7. Response caps

| Variable | Default | |
|---|---|---|
| `READ_CHUNK_CHARS` | `4000` | ceiling on `read_chunk`; a caller's `max_chars` may lower it, never raise it |
| `MAX_PROVENANCE_IDS` | `50` | ids honoured per `get_provenance` call |

! `MAX_PROVENANCE_IDS` is the one place a read-only server can still be made to
allocate without limit. An agent pasting a whole result set back is the normal case.

---

## 8. Database and image settings

Read by `docker-compose.yml`, not by the engine.

| Variable | Default | |
|---|---|---|
| `POSTGRES_USER` / `POSTGRES_PASSWORD` / `POSTGRES_DB` | — | required; no fallback values in the compose file |
| `POSTGRES_PORT` | `5432` | published on `127.0.0.1` only |
| `PG_SHARED_BUFFERS` | `1500MB` | the VPS overlay drops this to 512MB |
| `PG_EFFECTIVE_CACHE_SIZE` | `4GB` | |
| `PG_MAINTENANCE_WORK_MEM` | `512MB` | |
| `PG_WORK_MEM` | `32MB` | |
| `PG_MAX_CONNECTIONS` | `32` | |
| `MODEL_DIR` | — | required; model weights, mounted read-only |
| `VOCAB_DIR` | — | required; directory holding `BM25_VOCAB` |
| `EMBED_IMAGE` | TEI `86-1.7.2` | pinned, ✗ `:latest` — this container defines the vector space |
| `EMBED_POOLING` | `last-token` | must match the model's `1_Pooling/config.json` |
| `EMBED_DTYPE` | `float16` | |
| `EMBED_MAX_BATCH_TOKENS` | `32768` | the model's full context; lower silently truncates long chunks |
| `EMBED_PORT` | `8080` | published on `127.0.0.1` only |
| `ENGINE_BIND` / `ENGINE_PORT` | `127.0.0.1` / `8081` | |
| `VPS_CPUSET` | `0,1` | overlay only; db and engine share the same pair on purpose |

! `random_page_cost` is pinned to 1.1 in both profiles. The default 4.0 tells the
planner random reads are expensive and pushes it to sequentially scan a partition the
cluster index should have pruned.

! `locale=C` at initdb. Collation changes `ORDER BY` on text and BM25 tie-breaks, and
a ranking that differs between CI and the deployment box is worse than no ranking
test at all.
