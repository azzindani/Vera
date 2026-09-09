# FACTORS.md — Vera

The variables that determine whether Vera performs well, which of them differ per data
source, and how to measure anything at all when there are more than thirty of them.

Companion to `METRICS.md`: that document says *what must be achieved*, this one says
*what determines whether you achieve it*.

---

## 0. The problem

A benchmark run produces one number under one setting of ~36 variables. Change any of
them and the number may not survive.

That is tolerable with one corpus and one workload. It is not tolerable under
`MULTI_DOMAIN.md`, where **every source has different values for the variables nobody
chose** — how tightly it clusters, how skewed its vocabulary is, how often a query names
a document outright. A recall figure measured on regulations says nothing about court
decisions, and a latency figure measured on either says nothing about an ontology.

So the goal is not to sweep 36 variables. It is to:

1. separate what is **given** from what is **chosen**,
2. **derive** every chosen value that can be derived from a given one,
3. leave a handful genuinely free, and sweep only those,
4. re-measure, per source, only the small **profile** that drives the derivations.

! This is the generalisation of something already built. The layer-1 threshold used to
be a constant, then became a value derived from a measured corpus statistic
(`AnchorStats`). **That pattern is the answer to this whole document** — it just has to
be applied to every other dial.

---

## 1. Taxonomy

### A · Corpus-intrinsic — **given, and different per source**

These are properties of the data. You cannot choose them; you can only measure them and
adapt. **Every one of these must be re-measured for every new source.**

| # | Factor | Affects | Cheap to measure? |
|---|---|---|---|
| A1 | `N` — row count | `k`, shard count, everything | trivial |
| A2 | `d` — embedding width | RAM, scan bandwidth | trivial |
| A3 | **Anisotropy** — width of the cone the vectors occupy | layer-1 threshold | ✅ done (`AnchorStats`) |
| A4 | **Intrinsic topic count** — how many real clusters exist | routing recall, useful `k` | moderate |
| A5 | **Cluster tightness** — mean cosine to own centroid | routing recall | ✅ measured at build |
| A6 | **Cluster size skew** — max/mean | per-request RAM, tail latency | ✅ measured at build |
| A7 | **Vocabulary size + Zipf slope** | BM25 selectivity **and latency** | ⚠️ not measured |
| A8 | **Document-frequency distribution** | whether stopwords are needed | ⚠️ not measured |
| A9 | Mean document length + variance | BM25 `b`, chunk strategy | easy |
| A10 | **Lexical–semantic correlation** — do the two halves agree? | whether hybrid helps *at all*, RRF weighting | ⚠️ not measured |
| A11 | **Identifier density** — share of rows with a citable id | value of the exact path | easy |
| A12 | Metadata cardinality / filter selectivity | the cardinality rule (`MULTI_DOMAIN.md` §7) | easy |
| A13 | Near-duplicate rate | wasted candidate slots, inflated recall | moderate |
| A14 | Language morphology | stemming, tokenizer choice | qualitative |
| A15 | Temporal churn rate | re-cluster cadence | needs history |

### B · Index-build — **chosen, offline, paid for in re-ingest**

| # | Parameter | Current | Derived from |
|---|---|---|---|
| B1 | `k` cluster count | `√N` | A1 ✅ |
| B2 | k-means iters / seed / init sample | 15 / fixed / 50K | — |
| B3 | Chunk size + boundary strategy | upstream of Vera | A9, A14 |
| B4 | Vector precision | f32 ⚠️ (target halfvec) | A2 |
| B5 | Storage layout | row-per-chunk ⚠️ (target: contiguous) | — |
| B6 | Tokenizer / stemmer / stopwords | `unicode61`, none ⚠️ | A7, A8, A14 |
| B7 | Which metadata is filterable | none yet | A12 |

### C · Query-time — **chosen, online, free to change**

| # | Parameter | Default | Derived from |
|---|---|---|---|
| C1 | `clusters_probed` | 5 | A4, A5 — **free dial** |
| C2 | `per_cluster_top_k` | 50 | **free dial** |
| C3 | `bm25_limit` | 50 | A7 |
| C4 | `candidate_cap` | `k×10` | C1×C2 |
| C5 | `max_results` | 10 | caller |
| C6 | `rrf_k` | 60 | A10 — **free dial** |
| C7 | `threshold_margin` | 1.0 | A3 ✅ |
| C8 | confidence thresholds | 0.5 / 0.3 | A10 |

### D · Query distribution — **given, and different per workload**

! Not a property of the corpus. Two teams querying the *same* corpus can need different
dials, and nothing measured at ingest reveals this.

| # | Factor | Affects |
|---|---|---|
| D1 | Query length | BM25 term count, latency |
| D2 | **Query-type mix** — conceptual / exact-reference / cross-reference / defined-term | which half carries recall |
| D3 | **Paraphrase gap** — query↔document vocabulary overlap | dense-vs-keyword balance |
| D4 | **Query→answer vector distance** | required `clusters_probed` |
| D5 | Cluster hit skew | cache warmth, real p50 vs cold p50 |

### E · Environment

| # | Factor | Current bench | Target |
|---|---|---|---|
| E1 | cores | 4 | **2** |
| E2 | RAM | 16 GB | **8 GB** |
| E3 | disk bandwidth | NVMe-ish | fast NVMe |
| E4 | concurrency | 1 | 4 |
| E5 | cache warmth | warm | mixed |

---

## 2. What actually has to be re-measured per source

Of ~36 factors, **the per-source set is group A** — and most of it is already computed
at build time or is trivial to add.

**The source profile**, computed once at ingest and stored in corpus metadata:

```
N, d                          → k = √N                       (B1)
anchor p1 / p50 / p95         → domain_threshold             (C7)   ✅ built
cluster tightness, size skew  → expected routing recall, RAM ceiling ✅ built
vocab size, Zipf slope, df    → bm25_limit, stopwords, BM25 latency  ⚠️ missing
doc length mean/var           → BM25 b, chunk sanity                 ⚠️ missing
identifier density            → is the exact path load-bearing here? ⚠️ missing
lexical–semantic correlation  → RRF weighting, does hybrid help?     ⚠️ missing
filter-field selectivity      → the cardinality rule                 ⚠️ missing
```

! **Three of the four missing ones are cheap** — vocabulary statistics, document lengths
and identifier density are all one pass over the corpus that ingest already makes.
Lexical–semantic correlation needs a sample of queries, so it belongs with the eval set.

**This profile is the thing that makes a new source configuration rather than
guesswork.** Without it, every new source is a fresh tuning exercise; with it, the dials
start in the right place and only the free ones (§3) need sweeping.

---

## 3. Derived vs free

After derivation, **only three dials are genuinely free**:

| Dial | Why it cannot be derived |
|---|---|
| `clusters_probed` (C1) | depends on D4, a property of the *workload*, not the corpus |
| `per_cluster_top_k` (C2) | depends on within-cluster rank distribution of true answers |
| `rrf_k` (C6) | depends on A10 × D2 — how much the two halves agree *for these queries* |

Three dials is a sweepable space. Thirty-six is not. **That reduction is the entire
point of the profile.**

---

## 4. Interactions — pairs that cannot be tuned independently

! Sweeping one dial at a time is wrong wherever these hold.

| Interaction | Consequence |
|---|---|
| `k` × `clusters_probed` | rows scanned = `nprobe · N/k`. **Cost depends only on the product; recall depends on both separately.** Halving `k` and doubling `nprobe` holds latency constant and changes recall. |
| `clusters_probed` × `per_cluster_top_k` | total candidates = their product, but they lose answers for *different reasons* — routing vs cap (`METRICS.md` §3.1) |
| cluster tightness (A5) × `clusters_probed` | tighter clusters need fewer probes; a `nprobe` tuned on a tight corpus under-probes a diffuse one |
| Zipf slope (A7) × `bm25_limit` | a flat vocabulary makes BM25 return near-arbitrary rankings — **observed: IDF 0.33, scores ~3e−6 on the current fixture** |
| A10 × `rrf_k` | if the halves agree strongly, fusion adds little; if they disagree, `rrf_k` decides who wins |
| cluster size skew (A6) × concurrency | the RAM ceiling is set by the **largest** cluster × concurrent requests, not the mean |

---

## 5. ⚠️ What this says about every measurement taken so far

The synthetic fixture has **pathological values for A7, A8 and A10**:

| Factor | Fixture | Real text |
|---|---|---|
| Vocabulary size | **50 words** | 10⁴–10⁵ |
| Document frequency | ~72% per term | Zipfian; most terms rare |
| IDF | **0.33** | 0 for stopwords, 5–10 for rare terms |
| BM25 score magnitude | **~3e−6** | meaningful separation |
| Lexical–semantic correlation | near 1 by construction | unknown, likely much weaker |

Consequences, stated plainly:

- **BM25's 72% latency share is unreliable.** Terms matching 72% of documents force
  enormous posting-list scans. Real selectivity would likely cut it sharply — or, at
  500× the rows, not.
- **BM25's recall contribution is unreliable.** With no IDF, its ranking was driven by
  term frequency, which the generator *correlates with topic* — so it acted as an
  accidental topic classifier, not a keyword matcher.
- **The fusion has never been tested with two genuinely independent halves.** A10 ≈ 1 by
  construction means RRF has never had to arbitrate real disagreement, which is the case
  it exists for.

**Fixing the fixture's vocabulary is therefore not cosmetic** — it is the precondition
for any BM25 or fusion number meaning anything, including the recall-loss decomposition
(`METRICS.md` §3.1), which cannot attribute fusion loss when one half is noise.

---

## 5b. Statistics, metadata, and the graph — three kinds of per-source variable

They are all "things that differ per source", which is why they feel like one topic.
They behave differently enough that conflating them causes real mistakes.

| | **Corpus statistics** (§1 group A) | **Row metadata** | **Graph / knowledge graph** |
|---|---|---|---|
| What it is | emergent properties of the whole corpus | declared fields on each row | typed relations between rows or entities |
| Example | Zipf slope 1.2; anchor p1 0.71 | `court = "MA"`, `year = 2020` | `cites`, `supersedes`, `parent_of` |
| Where it lives | corpus metadata, **one value per corpus** | a column/JSONB, **one value per row** | an edge table, **one row per relation** |
| Cardinality | O(1) | O(rows) | O(relations) |
| What it drives | **dial derivation** (§2) | **constraints** — filters that gate routing | **traversal**, and a fourth ranker |
| How it is obtained | measured at ingest | supplied by the source | **extracted** — a pipeline of its own |
| Failure mode | a mis-derived dial | a silently dropped filter | wrong or missing edges — invisible |

! **Zipf slope is not metadata on a chunk.** It is a property of the corpus, computed
once. Treating it as a per-row field would be a category error — and treating per-row
metadata as a tuning statistic is the same error mirrored.

### The connection that *is* real, and useful

**Corpus statistics should be computed per metadata partition, not only globally.**

Cluster tightness for `court = "Mahkamah Agung"` may differ sharply from `court =
"Pengadilan Negeri"`. Vocabulary skew for 1998 documents differs from 2020. Anchor
geometry for one document type differs from another.

That matters because a **filtered query is a query against a different corpus** — one
with its own tightness, its own selectivity, and therefore its own right answer for
`clusters_probed`. It is the same fact that produces the cardinality rule
(`MULTI_DOMAIN.md` §7) seen from the statistics side: filters and routing interact
because the filter changes the distribution routing was tuned against.

So the profile (§2) is not one row of numbers per source. It is **one row per source,
plus one row per high-cardinality filter value worth splitting on** — which is also the
signal for where to shard (`HARDWARE.md` §5) and where to align cluster boundaries.

### Where a knowledge graph fits

A KG is not a new architectural layer. It is the concrete realisation of **two things
already in the design**:

- the `graph` **ranker** (`MULTI_DOMAIN.md` §5) — "what cites this", "what supersedes
  this" as a ranked retrieval signal;
- `traverse` **edges** (`MCP_ENGINE.md` §2) — the tool primitive already exists, and its
  `Edge` vocabulary is deliberately closed and small precisely so real edges can be
  added without a new tool.

! **The exact-identifier path is already a degenerate knowledge graph.** "This chunk
mentions `UU 28/2007`" is an entity-mention edge with one entity type, extracted by a
hand-written grammar, stored in a column. A real KG generalises exactly that: more
entity types, resolved rather than string-matched, with typed relations between them.

Three cautions, in order of how likely they are to bite:

1. **Extraction is a pipeline, not a field.** Entities and relations must be extracted,
   and for most sources that means a model. This is permitted — offline, in the
   pipelines — and **forbidden on the query path** (`CLAUDE.md` §5 rule 1). The boundary
   holds, but the cost lands in ingest, and the KG then needs its own maintenance
   cadence alongside cluster maintenance.
2. **Entity resolution is the hard part.** "Mahkamah Agung", "MA", and a typo are one
   entity. Getting this wrong produces edges that are confidently wrong, and a wrong
   edge is worse than a missing one because `traverse` presents it as fact.
3. **Its value must be measured, not assumed.** The honest question is factor A10
   extended: **does the graph half surface answers dense and BM25 both miss?** If the
   eval set says no for a given source, the KG is cost without recall — and that answer
   will differ per source. Citation graphs are load-bearing for case law and nearly
   worthless for a drug label.

**Ordering:** the graph ranker is the *last* of the retrieval work in
`MULTI_DOMAIN.md` §12, and deliberately so. It depends on the metadata model (which
does not exist yet), on an extraction pipeline (which does not exist), and on an eval
set to justify it (which does not exist). Building it earlier means building it blind.

---

## 6. Experiment design

Given the taxonomy, the disciplined procedure:

1. **Fix E entirely.** Report the profile; never compare across hardware.
2. **Measure the A-profile** at ingest. It is cheap and it is what transfers.
3. **Derive B and most of C** from A.
4. **Sweep only C1, C2, C6** — and sweep C1×C2 as a *grid*, not separately (§4).
5. **Stratify every quality metric by D2** (query type). An aggregate recall number over
   a mixed workload hides that conceptual queries and exact-reference queries are served
   by different halves of the engine and fail for different reasons.
6. **Re-measure A per new source.** Never assume a dial transfers.

! Step 5 is the one most likely to be skipped and the most costly to skip. Vera has
three retrieval paths; a single blended recall figure can look healthy while one path is
entirely broken — which is exactly how the layer-1 over-rejection and the fused-baseline
error both hid.

---

## 7. What to build, in order

1. **Zipfian fixture vocabulary** — unblocks every BM25 and fusion measurement. (§5)
2. **Recall-loss decomposition** — routing / cap / fusion. (`METRICS.md` §3.1)
3. **Corpus profile at ingest** — vocabulary stats, doc lengths, identifier density.
   Cheap, one pass, and it is what makes a second source configuration.
4. **Query-type stratification** in the eval set — needs labels. (§6 step 5)
5. **Lexical–semantic correlation** — needs the eval set; decides whether hybrid earns
   its cost on a given source.

Items 1–3 need no real data and no labels.
