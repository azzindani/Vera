# SCALING.md — from 20M rows to 100M across several databases

> **Status (2026-11-14):** §2 two-level routing is **built and measured**; §3 and §4 are
> designed, ✗ built. Every number below is measured (named) or labelled as an estimate.
> What is measured stops at 20.6M rows on 2 vCPU / 8 GB (`HARDWARE.md` §9).

---

## 1. What already scales, and what will not

Measured, 1.7M → 20.6M rows (12×), same box, same engine (`HARDWARE.md` §9.2):

| term | 1.7M | 20.6M | growth | at 100M (estimate) |
|---|---|---|---|---|
| dense arm, cold, 5 clusters | 54 ms | flat | **none**: rows per cluster are constant | flat |
| BM25 arm, warm p50 | 35-52 ms | 123 ms | ~linear in posting length | ~0.6 s, index ~31 GB |
| engine RSS (centroids hot) | 30 MB | 184 MB | linear in `k` | **~0.9 GB** at 50K clusters |
| centroid scan in the engine | <1 ms | ~5 ms (est.) | linear in `k` | ~25 ms |
| `chunk_vectors` on disk | 4.6 GB | 54 GB | linear | ~270 GB |
| page working set per query | ~1.8 GB | ~3.5 GB | sub-linear | unknown |

Three things break before 100M, in this order:

1. **The engine's centroid set** outgrows its 512 MB container limit at roughly
   ~30K clusters (60M rows). Not latency first: memory.
2. **BM25's index** stops fitting the database's page cache on an 8 GB box somewhere
   past ~30M rows (6.3 GB at 20.6M); then every lexical query pays disk.
3. **Disk.** 100M rows at today's layout is ~450 GB of database. Not a design problem,
   a provisioning one, but it decides whether one box can hold it at all.

The design answers these three and nothing else; what already scales is left alone.

---

## 2. Two-level routing · the engine holds the top level only

```
query ─► top level:  ~√k "super-centroids", held hot in the engine   (k=50K → ~224)
            │        pick the S nearest (S ≈ 4)
            ▼
         leaf level:  the leaf centroids of those S supers, read from Postgres
            │        (~S × 224 ≈ 900 rows of 2 KB = 1.8 MB, one indexed read)
            │        pick the 5 nearest leaves, exactly as today
            ▼
         chunk_vectors scan of 5 clusters                                  (unchanged)
```

- **Engine memory becomes O(√k).** 224 × 1024 × f32 ≈ 0.9 MB, against ~200 MB flat
  today at 10K clusters. The constant term returns to what §5.4 of `CLAUDE.md` promises.
- **Leaf centroids live in `clusters`, sorted by `parent_id`**, the same cluster-order
  trick as `chunk_vectors`, so a super's leaves are one contiguous read.
- **Recall cost is measurable before it is built.** `routing_recall.py` already reports
  where the answer's cluster ranks; a super-level miss is the new failure, and S is the
  dial. On the 1.7M corpus 5-of-846 probing keeps the dense arm within 93% of a flat
  scan (`EVAL.md` §4d at 355K, to be re-measured at 1.7M); the two-level version must be
  measured against that number, not argued.
- **Who builds it:** Ravel's `db prepare` already fits leaf centroids (`place`); it fits
  the supers over the leaves (k-means on ~50K points, seconds) and writes `parent_id`.
  Clustering stays a deployment decision (`Ravel/CLAUDE.md` §5.17).

### Built and measured · 2026-11-14

Ravel's `db prepare` fits the parents (`src/db/routing.py`, √k by default, seeded; also
`ravel db parents` for a database prepared before it); the engine uses them when the
database has them, `PARENTS_PROBED > 0` and the corpus has at least `TWO_LEVEL_FROM`
leaves (default 8,192, ~16M rows). Below that one level holds every leaf in a few tens of
MB and loses nothing.

**Routing recall**, 191 labelled questions on the real 1.7M corpus, 5 leaves probed
(`dev_tools/eval/routing_levels.py`): the share whose answer chunk lies in a probed cluster.

| | routing recall | leaves considered |
|---|---|---|
| one level, 846 leaves | **86.4%** | all |
| 29 parents (√k), 4 probed | 80.6% | ~14% |
| 29 parents, 8 probed | 83.8% | ~28% |
| 58 parents, 8 probed | 82.2% | ~14% |
| 58 parents, 16 probed | 85.3% | ~28% |

! The loss follows the **share of leaves the probed parents cover**, ✗ the parent count:
~28% coverage costs 1-2.6 points, ~14% costs 4-6. At 50K leaves √k parents and 8 probed
cover ~4%, so the default must grow with the corpus. That is the measurement a real
≥5M corpus has to make (§5); a replica cannot, because its centroids are copies.

**End to end** on the 1.7M corpus, two levels forced (`TWO_LEVEL_FROM=0`, 8 of 29):
Recall@5 56.6% against 57.1%, Recall@10 67.9% against 68.4%, MRR 0.432 against 0.438.
One case each: the global BM25 arm recovers most of what routing gives up, as invariant 3
intends.

**Memory and latency** on the 20.6M replica (10,152 leaves, 101 parents, 8 probed), same
box and profile as `HARDWARE.md` §9.2:

| | one level | two levels |
|---|---|---|
| engine RSS | 184 MB | **15 MB** |
| cold p50 / p95 | 676 / 1,035 ms | 708 / 1,158 ms |
| warm p50 / p95 | 613 / 833 ms | 629 / 895 ms |
| 4 clients | 2.41 q/s | 2.29 q/s |

The engine's resident set no longer grows with the corpus: 101 parents are 0.4 MB. The
leaf ranking costs ~20-30 ms in Postgres (808 leaves of 2 KB under 8 parents).

Rejected: an HNSW index over the 50K centroids. Small and fast, but it puts an ANN
structure, and its own recall loss, in front of the exact leaf ranking, and makes the
leaf choice non-deterministic under concurrent inserts. Two exact levels are
deterministic and explainable (`explain_routing` shows both).

## 3. The lexical arm past the cache

BM25 is the one global arm left (`HARDWARE.md` §9.1). Two options, in order:

1. **Partition the BM25 index by super-cluster** and search the S chosen partitions
   plus a small global "head" index of short, high-authority documents (UU, PP). The
   global guarantee becomes "global over the head, routed over the tail", and its loss
   is measurable with the same `routing_recall.py`.
2. **Keep it global and give it the memory.** At ~31 GB of index for 100M this is a
   bigger-box decision, ✗ an 8 GB one.

Decision rule: build option 1 only when a real corpus past ~30M rows shows BM25's cold
p95 above the 1 s budget. Until then it is a design, not a need.

## 4. Several databases · one engine, a domain per database

The 100M target is several corpora, ✗ one. Each corpus is a database prepared by Ravel
on its own (`ravel db prepare`), with its own `corpus_meta`, canary, vocabulary and
clusters. The engine:

- **opens one pool per database** listed in its configuration (a list of DSNs,
  validated at startup like every other limit), each with its own startup canary;
  a database whose canary fails is not served, the others are;
- **holds one domain anchor set per database** (its corpus centroid plus its top-level
  supers). Layer 1 stops being a yes/no gate and becomes the **selector** it was
  scaffolded to be: score the query against every database's anchors, search the top
  D (D ≈ 1-2) whose scores clear the floor, refuse when none does;
- **fuses across databases by rank**, as it already fuses arms: each database returns
  its pool, RRF merges them, factors apply per row with the vocabulary of the corpus
  the row came from (`corpus_meta.scoring_vocabulary` is per database already);
- **embeds once per vector space**, ✗ once per database. Two databases declaring the
  same model, pooling and instruction share one embedding; a database declaring another
  space gets its own embedder endpoint, and is skipped (with a `progress` line) when
  that endpoint is down, because §7.7 forbids an unvalidated fallback.

A request still never names a domain (§7.13). It can only be narrowed by the gate.

### 4a. Separated and unified: a domain is a schema · built 2026-11-22, measured 2026-11-23

Two deployments, one engine:

| | separated | unified |
|---|---|---|
| for | production: corpora on different machines, scaled and replaced apart | local and personal: every corpus on one box |
| storage | one database per corpus, its tables in `public` | one database, one **schema** per corpus (`id_reg`, `ca_laws`, `fr_legi` …) |
| containers | one Postgres per corpus, one engine over all | one Postgres, one volume, one engine |
| today | ✓ `vera-v6` (Indonesia, 15 GB), `vera-ca` (Canada, 1.5 GB) | ✓ database `unified`, six schemas (`id_reg`, `ca_laws`, `us_cfr`, `jp_egov`, `de_gii`, `fr_legi`), one engine (`config/domains.unified.json`): with two, each corpus scores as it does alone; with six, 86.6% at @25 over all (below) |

**A domain is `(database, schema)`.** Each domain keeps everything it has today, in its own
schema: `chunks`, `corpus_meta`, centroids, the sparse vocabulary, the scoring vocabulary,
the gate floors, its startup canary. Nothing is merged into shared tables, so a domain
served from a schema reads the same rows through the same indexes as from its own
database. Its results are then the same by construction, and that is checked, ✗ assumed
(§5, step 3b).

- **The engine opens one pool per domain**, its connections pinned to the domain's schema
  (`search_path` in the connection options). Its SQL is unqualified already (`chunks`,
  `corpus_meta`), so no query changes. ! The two column probes that read
  `information_schema` (`store/src/search.rs`, `has_currency` and its twin) match on
  `table_name` alone and must also match `table_schema = current_schema()`, or one
  domain's probe sees another's table.
- **One list configures both deployments**: `DOMAINS_FILE`, a JSON file mounted read-only
  beside `config/`, one entry per domain: its `schema`, its BM25 vocabulary file, its
  algorithms file, and `dsn_env`, the name of the environment variable holding its
  connection string (default `DATABASE_URL`). ! Names, ✗ connection strings: a DSN carries
  a password (§7.14) and the file is in git. Separated is N entries with N `dsn_env`; unified
  is N entries sharing `DATABASE_URL`, each with its schema; a mix is allowed. Validated at
  startup (§7.12): an entry whose canary fails, or that declares another vector space,
  refuses the start. Unset, the engine serves the one database `DATABASE_URL` names, by
  the same code path as before.
- **Built first: selection, ✗ fusion.** Each domain's own gate judges the query (its
  centroid similarity and lexical evidence, `explain_routing`'s two halves); the best
  domain that would serve it answers, with its reply unchanged, and the reply's first
  `progress` line says how every domain judged it. Fusing two domains' replies would
  rebuild their numbered citations and summary payload, and is left until a question set
  shows a question one domain cannot answer alone.
- **"Best" is the nearest centroid among the domains whose documents hold the question.**
  ! Similarity alone does not compare across corpora: by it, an Indonesian question on
  marine pollution (s462) went to Canada (0.656 against 0.644), whose top documents held 9%
  of it. ✗ similarity × lexical evidence: lexical evidence is lopsided across languages
  (English Canadian questions score 0.4-0.7 against the Indonesian corpus), and the product
  sent two Canadian questions (ca069, ca091) to Indonesia. The rule: of the domains that
  would serve, one whose lexical evidence is below `DOMAIN_SELECT_LEXICAL` (0.2) answers
  only if none reaches it. Chosen on the training questions (`queries.json`, Canada's
  calibration set, 670): every floor from 0.20 to 0.30 selected all of them correctly, and
  the lowest a right domain scored was 0.24; then scored once on the test sets (below).
- **A `Pipeline` is already a domain**: its store, centroids, gate floors, scoring
  vocabulary and BM25 vocabulary. The multi-domain engine holds one per entry and shares
  the embedding provider between them; a chunk id is looked up in each domain in turn for
  `read_chunk` and `get_provenance`, and one found in two domains is refused as ambiguous,
  ✗ guessed (§7.8).
- **Selection, fusion and factors are §4's**, unchanged: anchors per domain, the top D
  that clear their floor, RRF across domains, each row scored with its own domain's
  vocabulary and algorithm (`balanced` for Indonesia, `no_prior` where no prior is
  declared).
- **One embedding per vector space.** All six corpora Ravel has built declare the same
  space (Qwen3-Embedding-0.6B, fingerprint `2231c38c768a08c7`), so a unified box embeds
  each query once.
- **Ravel writes the schema** named by the DSN's `search_path`
  (`ravel db prepare <bundle> --dsn "…/unified?options=-csearch_path%3Dca_laws,public"`),
  into an existing database: a new corpus or a new edition is a schema built beside the
  live one and swapped by rename, the copy-on-write rule (§7.9) at domain scale.
  ! Then `ravel db calibrate` on that schema: the gate floor is measured on the schema's
  own clusters and recorded in its `corpus_meta`. Unmeasured, Canada served from a schema
  refused 5 real questions instead of 2 and lost 2.6 points at @25; calibrated (floor 0.48,
  as in its own database) it matches case for case.

**Measured** (2026-11-23, `python dev_tools/eval/eval_corpus.py <corpus>`; Indonesia on the
fresh split with the query rewrites, `--split fresh`, `EVAL_REWRITES`). Recall
@5/@10/@15/@20/@25:

| corpus | served | Recall | cases that differ from separated |
|---|---|---|---|
| Indonesia, 468 | separated (`id_reg_markdown`) | 79.5 / 88.5 / 92.1 / 93.6 / 94.9 | (reference run: 79.7 / 88.7 / 92.3 / 93.8 / 95.1; 1 case, s838 refused once) |
| | schema `id_reg` (`id_reg_unified`) | 79.7 / 88.7 / 92.1 / 93.6 / 95.1 | 7 rank shifts of 1-9 places, none across @5 or @25 |
| | one engine, both domains (`unified_id_reg`) | 79.7 / 88.7 / 92.1 / 93.6 / 95.1 | 1 from the schema run (s873, 49 → 50) |
| Canada, 114 | separated (`ca_laws`) | 71.9 / 81.6 / 84.2 / 85.1 / 88.6 | |
| | schema `ca_laws` (`ca_laws_unified`) | 71.9 / 81.6 / 84.2 / 85.1 / 88.6 | 0 |
| | one engine, both domains (`unified_ca_laws`) | 71.9 / 81.6 / 84.2 / 85.1 / 88.6 | 0 |

! Open: the 7 Indonesian rank shifts between a separated and a schema run. The data is the
same and the arms already break ties on id (`settle`); BM25 scores agree to 4 decimals on
the queries checked. Two separated runs differ by 1 case, so some is run-to-run noise in
the query embedding; the rest is not yet isolated.

Selection on the test sets: one engine picks Indonesia for every Indonesian question and
Canada for every Canadian one it serves. Off-topic questions refused: 30 of 55 on the
Indonesian set (33 by Indonesia alone): the 3 more answered are foreign-law questions in
English (US bankruptcy, UK inheritance tax, Sarbanes-Oxley) that Canada's gate serves, as it
does when alone, since Canada declares no foreign jurisdictions.

**Six corpora, one engine** (2026-11-28, `python dev_tools/eval/eval_corpus.py unified_all`,
Indonesia fresh with `EVAL_REWRITES=.test/keep/rw_fresh_AQ.json`): schemas `id_reg`,
`ca_laws`, `us_cfr`, `jp_egov`, `de_gii`, `fr_legi`, each gate calibrated on its own training
split, every set run against the same six-domain engine. Recall @5/@10/@15/@20/@25:

| set | n | Recall | MRR | right domain | off-topic refused | served elsewhere |
|---|---|---|---|---|---|---|
| Indonesia (fresh) | 468 | 79.7 / 88.7 / 92.1 / 93.6 / 95.1 | 0.670 | 468/468 | 24/49 | 1/6 |
| Canada | 114 | 51.8 / 57.9 / 60.5 / 61.4 / 64.0 | 0.416 | 77/114 | 9/11 | 5/5 |
| US CFR | 115 | 74.8 / 80.0 / 81.7 / 82.6 / 87.0 | 0.595 | 106/115 | 11/16 | 0/0 |
| Japan | 135 | 57.8 / 72.6 / 75.6 / 77.8 / 87.4 | 0.474 | 133/135 | 7/16 | 0/0 |
| Germany | 121 | 71.1 / 80.2 / 81.0 / 82.6 / 87.6 | 0.558 | 113/121 | 8/16 | 0/0 |
| France | 125 | 52.0 / 61.6 / 61.6 / 62.4 / 73.6 | 0.409 | 123/125 | 8/16 | 0/0 |
| **all** | **1078** | **69.3 / 78.4 / 80.8 / 82.2 / 86.6** | **0.567** | 1020/1078 | 67/124 | 6/11 |

"Served elsewhere": an off-topic question another loaded corpus holds (`served_by`), right
when that domain answers it. ! Open: Canada loses 24.6 points at @25 against its two-domain
run (88.6), and 37 of its 114 questions are answered by another domain, which is selection,
✗ retrieval: a second English corpus (US CFR) now competes for the same questions under the
lexical floor that was chosen with one English corpus loaded. Japan and France have no
separated run yet to compare against.

**What unified costs.** One Postgres holds every corpus: about 15 GB for Indonesia and
1.5 GB for Canada today, and its shared buffers are shared by all domains. The engine's
per-request cost stays flat (principle 4); its fixed cost is the sum of each domain's
centroids.

## 5. Order of work

| step | needs | proves |
|---|---|---|
| 1 | a real corpus of ≥5M rows (not a replica) | routing recall and BM25 at scale with real vocabulary |
| 2 | ~~two-level routing (§2)~~ **built**: 184 MB -> 15 MB at 20.6M; recall loss measured at 846 leaves only | the loss at 10K+ leaves, on step 1's corpus |
| 3 | a second database (any second corpus) | multi-database selection, cross-database fusion |
| 3a | ~~`DOMAINS`, one pool per `(database, schema)`; `ravel db prepare --schema`~~ **built**: `DOMAINS_FILE`, Ravel's schema from the DSN's `search_path` | one engine serving a corpus from a schema |
| 3b | ~~Indonesia served from a schema of a unified database~~ **measured** (§4a): same Recall at every cut; 7 rank shifts, open | its fresh scores equal `vera-v6`'s, case by case |
| 3c | ~~Canada beside it in the same database~~ **measured** (§4a): both unchanged through one engine; Canada case for case | each corpus's own score unchanged; each refuses the other's questions |
| 4 | §3 option 1, only if step 1 shows BM25 over budget | lexical arm past the cache |

Step 1 is the gate on everything else: no design here should be built against a replica,
because a replica cannot show the one thing each of them risks, which is recall.
