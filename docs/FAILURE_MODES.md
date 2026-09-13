# FAILURE_MODES.md

The ways this design can be wrong without saying so, and what stops each one.

A failure that raises an error is a small problem. Everything in this file is a
failure that returns a plausible answer — those are the ones worth a document.

---

## 1. Silent zero-recall on a routing miss

Layer-2 routes to the wrong clusters, the right chunk sits in an unprobed one, and it
is never scored. No error, no signal.

**Stopped by** the global arms. Only the dense arm is routed; sparse and text scan the
whole corpus, so a chunk missed by routing can still be retrieved on its wording. On
top of that, a query naming a regulation identifier bypasses routing entirely and is
reported separately as `exact_matches`, so the agent can tell a high-trust global hit
from a routed one.

**Residual risk.** A conceptual query whose answer shares no vocabulary with it depends
on the dense arm alone — which currently carries zero weight (`EMBEDDING.md` §5). This
is the sharpest open edge in the system.

---

## 2. Embedding-space drift

The query lands in a different space than the corpus. Vectors still normalize, rankings
still look reasonable, results are wrong.

**Stopped by** the corpus declaring its own space, boundary validation of every returned
vector, and a startup canary that re-embeds a stored chunk and refuses to serve below
threshold. `EMBEDDING.md` §3.

**Residual risk.** The canary verifies that query and corpus agree — not that either is
*correct*. Two sides in the same wrong space pass it, which is precisely what happened
(`EMBEDDING.md` §5). Catching that requires comparing against an independent
implementation, which is an offline check, not a startup one.

---

## 3. Stale fusion weights

The arms' weights are fitted on one corpus and the corpus is re-chunked. Every arm
still runs, results still look reasonable, and recall silently drops.

**Happened.** The defaults were fitted when the text arm scored 0.0% because the corpus
was unchunked and a whole 32,000-character article was one `tsvector`. Chunking took
that arm to 40.9% on its own — the best of the three — and nobody refitted. The engine
shipped at 38.6% Recall@5 while the offline harness reported 50.0%.

**Stopped by** two changes. The weights are read from the environment, so refitting does
not require a release. And `dev_tools/eval/e2e.py` scores through the shipped binary
rather than reimplementing fusion — the old harness reported 50.0% for a server
delivering 38.6% because it fused with its own weights.

! An eval harness that reimplements the thing it measures is measuring itself.

---

## 4. Wrong domain, guessed or supplied

Two ways to land in the wrong knowledge base: the agent passes one that is wrong or
does not exist, or the engine forces a query that fits nothing into the nearest domain.

**Stopped by** removing the argument and gating the detection. `search_knowledge` takes
a query and nothing else, with `additionalProperties: false`. Below either floor of the
domain gate the engine returns `detected_domain: null` and an empty result set.

The gate needs both halves. Measured on the 50-case set, with identifier queries exempt,
it rejects 6 of 6 out-of-domain queries and 0 of 39 real ones. Centroid similarity alone
cannot reject a plumbing question asked in Indonesian (0.717, above the in-domain mean)
because it tracks language, not subject. Lexical evidence alone cannot reject an English
general-knowledge question (0.789), because English function words do occur in this
corpus.

---

## 5. Unbounded queue, OOM via waiters

The concurrency semaphore caps what *executes*; an unbounded wait queue still grows
under load. An OOM vector hiding behind the concurrency limit.

**Stopped by** bounding time as well as depth. `QUEUE_WAIT_MS` refuses a request that
has waited too long, with `503` + `Retry-After` — never `500`, because the request was
never attempted and a client needs to know a retry is safe.

Tested without a database: `the_ceiling_is_a_ceiling`,
`waiting_is_bounded_by_the_wait_ceiling`. A guarantee that needs a corpus to test is a
guarantee nobody tests.

---

## 6. Peak RAM as a function of probe width

Loading all probed clusters at once makes peak memory scale with `CLUSTERS_PROBED` — a
dial an operator would reasonably raise for recall, with no visible connection to an
OOM two weeks later.

**Stopped by** sequential loading: load one cluster, scan, keep top-k, drop, load next.
Per-request working set is one cluster regardless of probe width.

! It costs 187 ms per request (268 ms against 81 ms for a single batched query). That is
the price of the bound, paid deliberately.

---

## 7. Wrong or synthesized provenance

The whole value is a human verifying through the link and locator. Provenance that is
wrong, or invented at query time, collapses that.

**Stopped by** capturing provenance at ingestion. The engine never synthesizes a link,
and where ingestion recorded no URL the field is omitted rather than filled with an
empty string that would render as a link leading nowhere.

! **"Immutably" was a claim, ✗ an enforcement.** The trigger meant to forbid UPDATEs to
provenance named `locator_page` and `locator_section` — columns this schema does not
have — so it could never have been applied, and nothing in the repo applies migrations
anyway. It now guards the real columns (`source_url`, `source_title`, `chapter`,
`article`) and must be applied by hand:

```bash
psql "$DATABASE_URL" -f migrations/0002_provenance_immutable.sql
```

Until it is run against a given corpus, provenance in that corpus is mutable.

---

## 8. Wrong vocabulary against the right corpus

Sparse vectors are indexed by vocabulary **position**. Serving a corpus against another
run's vocabulary compares unrelated dimensions — and returns results, plausibly ranked.

**Stopped by** requiring `BM25_VOCAB` explicitly, with no default, and mounting it
beside the corpus it was built from. `corpus_meta.sparse_vocab_sha256` records which
vocabulary the corpus was built with.

**Residual risk.** The engine does not currently verify that hash against the loaded
file. It is a one-line check and it is not written.

---

## 9. A configuration typo that parses

`MAX_CONCURRENCY=four` silently falling back to the default is a memory bound nobody
chose. `MAX_CONCURRENCY=0` parses perfectly and builds a server that reports healthy and
refuses every request.

**Stopped by** treating both as fatal. Unparseable values stop the process and echo the
bad value back; values that parse but cannot work are caught by a range check. Both
happen at startup, before the orchestrator has been told anything is healthy.

---

## 10. A cluster mutated while a query reads it

Incremental inserts, splits or re-clusters running during a query could let it read a
half-updated cluster.

**Not currently reachable**: cluster maintenance is offline and manual, and nothing
writes while the engine serves. The design for when it becomes reachable —
copy-on-write with an atomic version swap, a query pinning one version for its lifetime
— is in `dev_tools/CLUSTER_MAINTENANCE.md` and is **not implemented**.

---

## 11. A signal the contract promises and never sends

The output contract exists so an agent can act on more than the results themselves.
Four of its fields do not carry the information they claim, and every one of them fails
*quietly* — the agent reads a plausible value and draws a wrong conclusion.

| Field | Promised | Actual |
|---|---|---|
| `truncated` (search) | the result set was cut | always `false`; `TOP_K` drops candidates silently |
| `token_estimate` | measured response size | hardcoded 60 / 120 on `list_domains` and `explain_routing` |
| `detected_domain: null` | the corpus does not cover this | *also* returned when the domain matched and fusion simply found nothing |
| `explain_routing.detected_domain` | what the gate decided | the corpus id, unconditionally — it never runs the gate |

The third is the sharpest: an agent told `detected_domain: null` will stop asking. It
cannot distinguish "wrong knowledge base" from "right knowledge base, no match", and
the response can even carry `exact_matches` beside a null domain.

The fourth makes the transparency tool disagree with the tool it explains. Asked about
a query `search_knowledge` would refuse, `explain_routing` reports a matched domain —
so the one tool built to make routing falsifiable cannot falsify the gate.

**Not stopped.** All four are recorded, none is fixed. They are grouped here because
they share a cause: a field was added to the contract before the code that fills it,
and nothing failed in between.

---

## 12. A slow query holds its permit forever

`QUEUE_WAIT_MS` bounds how long a request may **wait**. Nothing bounds how long one may
**run**.

```
admission   QUEUE_WAIT_MS = 2000 ms      bounded
embed       reqwest timeout = 30 s       bounded
every SQL query                          UNBOUNDED
```

No `statement_timeout` is set by the engine, by `docker-compose.yml`, or by the VPS
overlay. A query that takes minutes holds its semaphore permit for minutes. Four of
those and the server is wedged: `permits_available: 0` indefinitely, and every
subsequent caller is refused after two seconds.

! **Reachable, ✗ hypothetical.** The text arm ORs every lexeme of its input; a query
built from a whole chunk body matched 98.7% of the corpus and took **48 seconds**,
measured. `ts_rank` cannot be served from an index, so all 351K matches are scored and
sorted. A long enough user query walks toward the same plan.

`OPERATIONS.md` §2 already tells an operator that `permits_available: 0` with nothing
completing means wedged — so the state is diagnosable, and nothing prevents it.

**Stopped by** `STATEMENT_TIMEOUT_MS` (default 15,000), applied as a connection
**option** rather than a `SET` per checkout — libpq installs it when the session is
established, so it costs no round trip and no code path can forget to issue it. `0`
disables it. An operator's existing `options` in the connection string are appended to,
never replaced.

The concurrency story is now complete: bounded queue, bounded wait, bounded memory,
**bounded duration**. Duration is what `SCORING.md` §6 proposes to start spending
deliberately, so the bound had to exist before the tiers do.

Tested against a live corpus — `a_runaway_query_is_killed_rather_than_held` asserts a
10-second sleep dies under a 250 ms timeout, and on SQLSTATE `57014` rather than the
error text, since tokio-postgres renders a server error as the bare string "db error".
`zero_disables_the_timeout` holds the escape hatch honest.

---

## 13. CI tests a different text arm than production runs

`SearchOps::text` has two SQL paths, chosen by probing for the RUM index:

```
RUM present   ORDER BY tsv <=> q.tq        one ordered index scan   ← what ships
RUM absent    ORDER BY ts_rank(...) DESC   score and sort every match
```

CI runs `pgvector/pgvector:pg16`, which has no RUM extension, so **every integration
test exercises the fallback**. The deployment runs `vera-db:pg16-rum`. The SQL that
actually serves queries is never executed by the gate.

The two paths are believed equivalent — measured at 98.9% top-20 overlap and identical
rank-1 on 44/44 eval queries — but that was a manual measurement taken once, ✗ a check
that runs. A change to the RUM query, or a RUM version whose operator behaves
differently, passes CI green.

! Neither container image is built in CI either. `docker/Dockerfile.db` and
`docker/Dockerfile.engine` are only built at deploy time, so a broken image is
discovered by the deployment rather than by the pull request.

**Stopped by** a second integration job, `integration-rum`, which builds
`docker/Dockerfile.db` and runs the same `--ignored` tests against it.

! It asserts RUM is actually present before running them. A job that silently fell back
would pass green while testing exactly the path this one exists to stop testing — the
same failure, one level up.

A third job builds `docker/Dockerfile.engine` and checks the binary **refuses to start
without configuration**, because an image that builds can still produce something that
cannot run.

---

## Review rule

A change to routing, the candidate caps, the provider, the queue, or the cluster-update
path is checked against the relevant section above before it merges.
