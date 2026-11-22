# BEIR.md — Vera against pgvector and Qdrant on public benchmarks

> **Status (2026-11-14):** measured on three BEIR test sets. Every number below names the
> run that produced it (`dev_tools/eval/beir/results/*.json`). Small corpora (3.6K-57.6K
> documents): this compares **ranking quality** on public labels. It is ✗ a scale test;
> that is `HARDWARE.md` §9 and `SCALING.md`.

---

## 1. Result

nDCG@10 (BEIR's headline metric), test split, every judged query.

| system | SciFact | NFCorpus | FiQA-2018 |
|---|---|---|---|
| BM25 alone (`pg_search`, Vera's BM25 arm) | 0.661 | 0.307 | 0.239 |
| pgvector, exact scan | 0.694 | 0.368 | **0.470** |
| pgvector HNSW, `ef_search` 40 (default) | 0.692 | 0.365 | 0.463 |
| pgvector HNSW, `ef_search` 100 | | 0.368 | 0.467 |
| Qdrant dense, defaults | 0.694 | 0.368 | 0.470 |
| Qdrant hybrid, dense + BM25, RRF | 0.709 | 0.359 | 0.411 |
| **Vera `literal`**, hybrid (shipped fusion, no prior) | **0.718** | **0.368** | 0.428 |
| Vera `balanced`, generic prior (§4) | 0.677 | 0.359 | 0.386 |
| *Qwen3-Embedding-0.6B, published* | *0.6945* | *0.3689* | *0.4727* |

The published row is a third-party evaluation of the same embedder
(superlinked.com/models/qwen-qwen3-embedding-0-6b). pgvector's exact scan reproduces it
within 0.001 on SciFact and NFCorpus and 0.003 on FiQA. That agreement is the check
that the embedding pipeline here is the model's own, and that every system below is
being fed the right vectors.

**In one line:** Vera's hybrid is the best of the systems tested where the lexical signal
is strong (SciFact, +2.4 over dense, +0.9 over Qdrant's own hybrid), ties where it is
neutral (NFCorpus), and loses 4.2 points to plain dense search where BM25 is weak (FiQA).
No configuration was tuned on BEIR.

## 2. Where FiQA's 4.2 points go

| FiQA-2018 | nDCG@10 | loss |
|---|---|---|
| pgvector exact (every chunk scored) | 0.470 | |
| Vera dense arm only, 5 of 29 clusters (17% of the corpus) | 0.455 | **−1.5** routing |
| Vera hybrid (dense ⊕ BM25, RRF, `DENSE_WEIGHT` 2) | 0.428 | **−2.7** fusion |
| Vera BM25 arm only | 0.239 | equal to `pg_search` alone, as it must be |

* **Fusion:** RRF with fixed weights adds the BM25 list whether or not it is any good.
  On FiQA (forum answers, paraphrase-heavy) BM25 scores half of dense, and fusing it
  costs 2.7 points. Qdrant's hybrid loses more (−5.9) for the same reason, without the
  dense weighting. On SciFact the same fusion **adds** 2.4. A fixed weight cannot be right
  for both. Telling them apart at query time is what principle 10 (escalate on measured
  disagreement) is for. ✗ built.
* **Routing:** the price of never scanning more than 5 clusters.

### Routing, made as coarse as the 1.7M corpus

At 1.7M rows, 5 of 846 clusters cover 0.6% of the corpus. FiQA at the default 2,000 rows
a cluster has 29, so 5 cover 17%. The same bundle was prepared again at 200 rows a cluster
(`ravel db prepare --rows-per-cluster 200`: 288 clusters, 5 cover 1.7%):

| FiQA-2018, 288 clusters | nDCG@10 | vs exact dense |
|---|---|---|
| Vera dense arm only | 0.437 | −3.3 (−7% relative) |
| Vera `literal` hybrid | 0.410 | −6.0 |
| Vera `balanced`, generic prior | 0.372 | |

Routing to 1.7% of a corpus keeps 93% of exact dense nDCG@10. That is the same figure
`EVAL.md` §4d measured for the dense arm's recall on the legal corpus at 355K, now on a
public benchmark.

## 3. Latency, and what it does not compare

p50 per query, whole stack on two pinned cores (`docker-compose.vps.yml`), Qdrant in its
own container on the same two cores (`--cpuset-cpus 0,1 -m 2g`):

| | SciFact | NFCorpus | FiQA |
|---|---|---|---|
| Vera, whole request (embeds the query on CPU) | 393 ms | 290 ms | 370 ms |
| the query embedding alone, same server | 392 ms | 281 ms | |
| pgvector HNSW / exact | 3 / 17 ms | 3 / 13 ms | 4 / 110 ms |
| Qdrant dense | 8 ms | 9 ms | 10 ms |

pgvector and Qdrant are timed on the search call with a precomputed vector, so the two
rows do not measure the same work. Nearly all of Vera's time is the CPU embedder
(`HARDWARE.md` §9); its search is in the same tens of milliseconds as the others at these
sizes. What separates the engines is memory at 20M rows, not speed at 57K: an HNSW graph
wants the vectors in RAM (~42 GB of `halfvec(1024)` at 20M), while Vera's routed scan
serves 20.6M from 8 GB (`HARDWARE.md` §9.2). That difference is not measured here.

## 4. The `balanced` prior on a corpus it was not built for

A BEIR corpus has no authority table, so the shipped `balanced` (authority 0.5) is
**refused at startup**, by design: a weight with nothing to read would silently do
nothing. The run uses `dev_tools/eval/beir/algorithms.json`: `balanced` minus authority,
leaving the structural (0.25) and topical (0.25) priors that Ravel's generic profile
declares, behind the 0.3 relevance floor.

It costs 1-4 points on all three sets. The generic profile's structural rules (numbered
lines, `SECTION`, `CHAPTER`) and obligation terms describe regulations and manuals, not
abstracts or forum answers. The prior is a domain tool. On a corpus whose vocabulary does
not describe it, `literal` is the right choice, and nothing currently stops a deployment
from choosing otherwise.

## 5. Held equal

* **Units.** Every system indexes the `chunks` rows of the Vera database Ravel built from
  the BEIR corpus: `title text` as one paragraph (MTEB's join), one chunk per document
  (FiQA: one document of 57,598 in two). A document is ranked at its best chunk. Two documents with identical text are one Ravel document, and either
  BEIR id counts as found (NFCorpus 40, FiQA 2).
* **Embeddings.** Qwen3-Embedding-0.6B at the pinned revision. Documents were embedded
  through vLLM on the GPU, after `ravel verify-backend` agreed with the reference
  implementation (min cosine 1.00000). Queries go through Vera's own CPU embedding server
  (float32) with the instruction the corpus declares, which is Qwen3's MTEB instruction
  for each task, in Vera's template. Vera's startup canary passed at 1.00000 on all four
  databases.
* **Lexical.** `pg_search`'s default tokenizer, no stemming: 0.661 / 0.307 / 0.239
  against BEIR's published Elasticsearch BM25 of 0.665 / 0.325 / 0.236. Qdrant's hybrid
  uses the BM25 weights Ravel stored per chunk (`chunks.sparse`) and Vera's query rule
  (1.0 per known term), fused by Qdrant's RRF.
* **Engines at their defaults:**
  * pgvector HNSW `m` 16, `ef_construction` 64. pgvector returns at most `ef_search`
    rows, so Recall@100 needs `ef_search` ≥ 100.
  * Qdrant v1.15.0, defaults.
  * Vera as shipped: probe 5, RRF k 60, `DENSE_WEIGHT` 2, candidate pool 60, text arm
    off, top 20 requested. The gate is off (`DOMAIN_FLOOR` 0), because no BEIR corpus is
    calibrated, and every query was answered.

## 6. Not covered

* **Scale.** At most 57.6K documents. The memory argument (§3) is measured elsewhere,
  not against these engines.
* **Reranking.** Model rerankers usually add several points on BEIR. Vera excludes them by
  principle 1; the systems here are compared without one.
* **More sets.** Three of BEIR's 18, picked to fit a laptop GPU in minutes. ArguAna,
  TREC-COVID, NQ and others are the same commands.

## 7. Reproduce

```
python -I dev_tools/eval/beir/sources.py <beir>/<set> <work>/<set>           # docs + id map
ravel extract / chunk / embed / sparse / bundle build beir_<set>              # Ravel, corpora/beir_<set>.yaml
ravel db prepare <bundle> --dsn ... [--rows-per-cluster 200]
# engine: ALGORITHMS_PATH=dev_tools/eval/beir/algorithms.json, DOMAIN_FLOOR=0, TOP_K=20
python -I dev_tools/eval/beir/bench.py <set> --raw ... --src ... --dsn ... --vocab <bundle>/corpus.bm25.json --out ...
```

Ravel's `corpora/beir_*.yaml` specs exist on the build machine only; they are not yet
committed to Ravel.
