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
engine   14 MB @ 355K, 42 MB @ 5.1M   the delta is the hot centroid table,
         which is O(k) and k grows with the corpus. Per-REQUEST it is flat:
         14/12/14 MB at CLUSTER_BATCH 1/2/5, and 39->42 MB under a burst.
postgres bounded by shared_buffers + work_mem + its container limit
```

The engine never materialises the corpus — arms return `(id, score)` under a `LIMIT`
— so nothing here is a memory dial for it. `HARDWARE.md` §2 and §6a.

| Variable | Default | Notes |
|---|---|---|
| `MAX_CONCURRENCY` | `4` | the hard ceiling on in-flight requests |
| `CLUSTER_BATCH` | `1` | clusters per statement. A **latency** dial (−182 ms at 5), ✗ a memory one — see §4 and the note below |
| `QUEUE_WAIT_MS` | `2000` | how long a call may wait for a slot before being refused |
| `STATEMENT_TIMEOUT_MS` | `15000` | how long a query may **run** before Postgres cancels it; `0` disables |

! **Neither `MAX_CONCURRENCY` nor `CLUSTER_BATCH` measurably moves engine RAM.** An
earlier version of this file said they multiplied into a 102 MB budget; that was
arithmetic and the measurement refutes it (`ARCHITECTURE.md` §4). `MAX_CONCURRENCY`
remains a bound on in-flight work and on load reaching Postgres — which is the
process that actually holds the pages.

! Both are still rejected at `0` at startup. They parse fine and would produce a
server that reports healthy and returns nothing.

! `CLUSTERS_PROBED` costs latency and recall, not memory — which is true for a
simpler reason than this file used to give: nothing the engine probes is resident in
the engine at all.

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
| `CLUSTERS_PROBED` | `5` | layer-2 probe width. Recall vs latency — **not** vs RAM, because clusters load `CLUSTER_BATCH` at a time and this is not that number. |
| `CLUSTER_BATCH` | `1` | How many probed clusters are resident at once. The second term of the OOM guarantee: `peak RAM = fixed + (MAX_CONCURRENCY × CLUSTER_BATCH × one cluster)`, worst case 23.4 MB per cluster measured. `1` reproduces the original one-at-a-time scan and is the 2 vCPU / 4 GB number; higher buys back the 187 ms that separate statements cost. **Raise the container's memory limit with it** — they are one decision (`ARCHITECTURE.md` §4). |
| `PER_CLUSTER_K` | `20` | candidates kept per cluster per arm |
| `PER_ARM_K` | `20` | candidates each global arm contributes |
| `TOP_K` | `10` | results returned |
| `EXPAND_SEEDS` | `5` | Top-ranked candidates sibling expansion walks from, de-duplicated by regulation. Only consulted when a caller passes `expand: ["siblings"]`. Seeds, ✗ the whole pool: expanding 60 candidates at up to 1,313 chunks each is a different query. |
| `EXPAND_PER_SEED` | `40` | Sibling rows read per seed, before the relevance gate admits any. p50 is 22 chunks per regulation and p95 is 210, so this takes the common case whole and bounds the tail. Measured cost of the widest possible walk — 10 seeds from the 10 largest regulations — is 11,750 rows in 24 ms. |
| `CANDIDATE_POOL` | `60` | candidates carried into scoring. **Metadata is fetched for this many, not for `TOP_K`** — a candidate whose metadata was never loaded cannot be reordered. Must be ≥ `TOP_K`; a smaller pool would silently cap the reply. 60 is where the "regulation absent" bucket stops falling (`SCORING.md` §7). |
| `SNIPPET_CHARS` | `280` | preview length per result |

---

## 5. Fusion weights

| Variable | Default | |
|---|---|---|
| `DENSE_WEIGHT` | `2.0` | fitted 2026-09-19 · `EVAL.md` §4b |
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

Measured through the deployed server, 44 retrievable cases, **re-measured
2026-09-19 after the re-embed** (`EMBEDDING.md` §5d):

| dense | sparse | text | Recall@5 | Recall@10 | MRR |
|---|---|---|---|---|---|
| 0 | 1 | 1 | 50.0% | 52.3% | 0.349 |
| 1 | 1 | 1 | 54.5% | 63.6% | 0.411 |
| **2** | **1** | **1** | **56.8%** | **65.9%** | 0.428 |
| 4 | 1 | 1 | 54.5% | 63.6% | 0.450 |
| 8 | 1 | 1 | 59.1% | 63.6% | 0.470 |

! **Fit this on MRR and Recall@10, ✗ on Recall@5.** At n=44 one query is 2.3 points,
so every gap in the Recall@5 column is one or two cases and it moves
non-monotonically. MRR rises monotonically across the whole range and Recall@10
moves +13.6 points, which is 6 cases and clear of the noise.

! **And do not fit it at k=5 at all.** `EVAL.md` §4b measures the same corpus at the
width a caller actually receives: fusion *loses* to dense-alone at k=5 and *wins*
from k=20 up (82.1% at k=20, 94.9% at k=100). A k=5 fit therefore over-weights
dense for the way the engine is used. `2.0` is shipped rather than `8.0` for that
reason.

~~Recall@5 is flat for any text weight in 0.3–1.0~~ — that observation was made when
the dense arm contributed nothing and has not been re-measured.

All three at zero is rejected at startup: it fuses nothing and returns nothing, which
is indistinguishable from a corpus that simply has no match.

---

## 5b. Factor weights

Fusion decides what is *relevant*; these decide what *matters* among the relevant
(`SCORING.md` §2). They are a bounded multiplicative prior on the fused score, so
no weight here can promote a candidate that retrieval did not find.

| Variable | Default | |
|---|---|---|
| `RELEVANCE_FLOOR` | `0.4` | **a floor, ✗ a weight.** Share of the query's content terms a candidate must contain to be ranked at all. Worth more than every weight below combined. |
| `FACTOR_AUTHORITY` | `1.0` | how binding the instrument is — the published hierarchy |
| `FACTOR_STRUCTURAL` | `0.5` | operative clause vs annex |
| `FACTOR_TOPICAL` | `0.25` | subject match — earns its weight once the floor exists |
| `FACTOR_COMPLETENESS` | `0.0` | **measured to add nothing** once a real floor exists |
| `FACTOR_TEMPORAL` | `0.0` | recency — **measured to add nothing** |

Fitted, not chosen: `python dev_tools/eval/fit_factors.py` reranks the text arm over
a 60-candidate pool on the 40 article-labelled cases.

| | Recall@5 |
|---|---|
| text arm, no floor, no factors | 40.0% |
| **floor alone** | **52.5%** |
| best in-sample | 65.0% |
| **leave-one-out** | **57.5%** |

! **+17.5 points, not +25.** Taking the best of 3,125 configurations on 40 cases
overfits; leave-one-out is the number that survives out of sample. The gain is real
rather than a lucky peak on two grounds: 2,939 of the 3,125 configurations (94%) beat
the baseline, and leave-one-out chose exactly this configuration in 37 of 40 folds.

! Floor and weights are fitted **jointly**. Fitting them separately credits a factor
for work the floor was doing — which is what happened the first time: `completeness`
took 0.25 as a crude relevance proxy, and drops to 0.0 once a real floor exists.

Setting all five to `0` is exactly the identity on the fused order — the layer can be
turned off in production without a rebuild, which is the point of it being config.

! These numbers come from reranking **one arm** offline. The engine fuses three, and
only `e2e.py` scores what ships.

---

## 5c. The lexical arm has two backends

| | with `pg_search` | without |
|---|---|---|
| mechanism | Tantivy BM25, indexed | `sparsevec` inner product, **no index** |
| 355K | 86 ms | 170 ms |
| 5.1M | 1,858 ms | 6,282 ms |
| needs | migration 0004 + the extension | `BM25_VOCAB` |

Chosen at runtime by `SearchOps::has_bm25`, probed once and cached. Nothing is
configured: an engine pointed at a database with the index uses it.

! `BM25_VOCAB` stays **required**. It feeds the fallback, and a deployment that
loses `pg_search` — a stock image, a volume restored elsewhere — must still
answer. This is the same arrangement as GIN backing RUM (`HARDWARE.md` §5).

! The vocabulary must still be the one the corpus was built with. The fallback
compares vocabulary **positions**, so a mismatched file scores unrelated
dimensions and returns plausible nonsense rather than failing.

---

## 6. Gates

| Variable | Default | Effect |
|---|---|---|
| `CANARY_MIN_COSINE` | `0.98` | startup vector-space check |
| `DOMAIN_FLOOR` | `0.39` | nearest-centroid similarity below which results carry a weak-match hint · lowered 2026-09-19, `EVAL.md` §4c |
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
