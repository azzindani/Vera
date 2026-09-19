# OPERATIONS.md

Running Vera: bringing it up, reading its health, and what each refusal means.

---

## 1. Bring it up

```bash
cp .env.example .env      # fill in at minimum DATABASE_URL, EMBED_ENDPOINT, BM25_VOCAB
docker compose up -d
```

Under the target hardware profile — 2 vCPU / 4 GB, with db and engine pinned to the
same two cores:

```bash
docker compose -f docker-compose.yml -f docker-compose.vps.yml up -d
```

! The overlay is not decoration. Testing on 14 cores proves nothing about the target
box: the OOM guarantee and every latency number are claims about 2 vCPU / 4 GB, and
they are only under test with it on.

Startup order is enforced by healthchecks, not by hope. The embedding container accepts
connections roughly 40 seconds before it can embed anything; `depends_on:
service_healthy` is what stops the engine's canary from hitting a live socket that
refuses work, correctly refusing to serve, and restart-looping until the model warms.

---

## 2. Is it working?

```bash
curl -s localhost:8081/health
```

```json
{ "success": true, "clusters": 177, "permits_available": 4, "version": "0.1.0" }
```

| Reading | Means |
|---|---|
| `permits_available` = ceiling | idle |
| `permits_available` = 0, requests completing | at capacity, working |
| `permits_available` = 0, nothing completing | wedged — check the database |
| no response at all | still in startup, or refused to serve; read the logs |

All logs go to **stderr**, one line per event. Startup prints the corpus id, model,
width, pooling, cluster count and concurrency ceiling. The connection string is never
logged: it carries a password and an engine log is the one thing people paste into
tickets without thinking.

---

## 3. What the refusals mean

Vera fails closed by design. Each of these is the system working, not breaking.

| At startup | Cause | Fix |
|---|---|---|
| `DATABASE_URL is not set` | required variable missing | see `.env.example` |
| `MAX_CONCURRENCY="four" is not a valid positive integer` | unparseable value | the bad value is echoed back; fix it |
| `corpus … engine …` mismatch | the endpoint serves a different model than the corpus declares | point at the right endpoint, or re-embed |
| canary below threshold | the endpoint serves *different weights* under the same name | check pooling, dtype and the mounted model |
| vocabulary load failure | `BM25_VOCAB` missing or from another run | mount the vocabulary built with this corpus |

| While serving | Status | Means |
|---|---|---|
| `503` + `Retry-After: 1` | at capacity | never attempted — retry is safe |
| `200` with `success: false` | a tool-level error | the `hint` says what to do |
| `200` with empty `results`, `detected_domain: null` | below the domain gate | the query does not belong to this corpus |

! The last one is not a failure. A confidently wrong domain is worse than an honest
"nothing matched", and the difference between them is invisible downstream if the
engine guesses.

---

## 4. Exposing it

The engine has no auth, no TLS and no CORS, and `docker-compose.yml` binds it to
`127.0.0.1`. That is deliberate: it is infrastructure an agent calls over a private
network.

To reach it from elsewhere, put a reverse proxy in front that terminates TLS and
authenticates, and leave the engine on loopback. Do not change `ENGINE_BIND` to
`0.0.0.0` as a shortcut — the corpus and every provenance link in it become public the
moment you do.

The database is bound to loopback for the same reason.

---

## 5. Changing a dial

Everything in `CONFIGURATION.md` is an environment variable, so a change is a restart,
not a rebuild:

```bash
docker compose up -d --force-recreate engine
```

Two of them change **what the server returns** and must be re-measured, not guessed:

- `CLUSTERS_PROBED` — recall against latency.
- the three fusion weights — properties of the corpus, refitted after every re-chunk.

`CLUSTER_BATCH` is the exception: it changes latency and peak RAM but **not** what the
server returns, because windowing changes how many clusters one statement covers, never
which ones are probed. Re-measure memory when you change it, ✗ recall — and change the
container's memory limit in the same edit (`ARCHITECTURE.md` §4).

! That neutrality is a property of the code, not of the idea. One `LIMIT` cannot be
per-partition, so a window of `m` clusters has to ask for `m × PER_CLUSTER_K` or it
returns a fraction of what those clusters yield separately — a memory dial quietly
becoming a recall dial. At the shipped defaults it would not show, because `PER_ARM_K`
equals `PER_CLUSTER_K` and both spellings truncate to the same rows; it appears as soon
as someone raises `PER_ARM_K`. So it is asserted rather than assumed:

```bash
cargo test -p store -- --ignored batching_clusters_does_not_change_which_rows_win
```

Run `dev_tools/eval/e2e.py` against the running server before and after. It scores
through the shipped binary, which is the only way to know what the *server* does — an
offline harness that reimplements fusion measures itself.

```bash
VERA_HTTP=http://localhost:8081 python dev_tools/eval/e2e.py
```

---

## 6. Upgrading the corpus

A new corpus is a new set of vectors, a new vocabulary and new centroids, and all three
move together:

1. Build it offline (`dev_tools/`).
2. Point `BM25_VOCAB` at the vocabulary built **with that corpus**. Sparse vectors are
   indexed by vocabulary position; another run's vocabulary compares unrelated
   dimensions and returns plausible nonsense rather than failing.
3. Refit the fusion weights against the new corpus.
4. Restart. The canary re-verifies the vector space on the way up.

---

## 7. Backups

The only state is the `pgdata` volume. Everything else — engine, embedder, centroids in
memory — is reconstructible from it plus the images.

```bash
docker compose stop engine
docker exec vera-db pg_dump -U "$POSTGRES_USER" -Fc "$POSTGRES_DB" > vera.dump
docker compose start engine
```

Stopping the engine first is not required for correctness — the query path never
writes — but it keeps the dump from competing with live scans for the two cores.

Back up the BM25 vocabulary alongside the dump. A corpus without its vocabulary cannot
serve the sparse arm, and it is a small file that lives outside the database.
