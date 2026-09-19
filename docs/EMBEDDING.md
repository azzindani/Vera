# EMBEDDING.md

The vector space, and the machinery that keeps the corpus and the query inside the
same one.

---

## 1. The rule

**Same model, same version, same pooling, same normalization, same instruction —
both ends.**

A space mismatch produces no error. The vectors still normalize, the rankings still
look plausible, and the results are quietly wrong. It is the only failure mode in Vera
that costs nothing to create and is expensive to detect, which is why three separate
mechanisms defend it.

---

## 2. The corpus declares the space

The engine does **not** compile in a model name or a dimension. `corpus_meta` carries
them, and the engine builds its provider to match:

| Column | Used for |
|---|---|
| `dense_model` | the model the provider must claim to serve |
| `dense_dim` | the width every returned vector is checked against |
| `dense_pooling` | recorded so a mismatch is diagnosable |
| `dense_normalize` | as above |
| `dense_instruction` | prepended to queries, if the corpus was built with one |

! `dense_instruction` is applied to **queries only**, and only if the corpus recorded
one. Qwen3-Embedding is instruction-aware: an instruction on one side and not the other
puts the two in different spaces. Only the corpus knows which convention it was built
under, so only the corpus gets to say. The current corpus recorded `NULL`, so no
instruction is applied.

This is why `embed::validate_against` takes the expected model and width as arguments
rather than reading a constant. A constant could only ever encode the model the author
expected; the rule that matters is that query and corpus agree.

---

## 3. Three defences, in order

**Name check.** `CorpusMeta::ensure_compatible` compares the declared model and width
against the provider's. Cheap, and catches a misconfigured endpoint.

**Width check.** Every vector the provider returns is validated at the boundary. A
truncated or wrong-width vector is rejected there, not deep inside routing.

**The canary.** At startup the engine picks a stored chunk, re-embeds its own text
through the configured endpoint, and compares against the vector ingestion stored for
it. Below `CANARY_MIN_COSINE` (default 0.98) it refuses to serve.

! Only the canary tests the actual space. The first two compare strings and integers;
an endpoint serving different weights under the same name passes both and fails here.
That is the whole reason it exists.

`halfvec` storage puts an exact round-trip near 0.999 rather than 1.0 — fp16 rounding,
not drift. A genuinely different model scores far below 0.98, so the threshold
separates "same space" from "different space" with room to spare. There is no code path
that skips it.

---

## 4. The model

**Qwen3-Embedding-0.6B**, 1024 dimensions, open weights, stored as `halfvec`.

Chosen because it runs on the target hardware. The corpus is embedded offline on a GPU;
queries are embedded at runtime against a local container holding the same weights.
There is no external embedding API in the deployed path, and no API key anywhere in the
query path.

The container is pinned by tag, never `:latest`. That container *defines* the vector
space — an image that updates underneath a loaded corpus is this document's opening
rule broken silently.

---

## 5. The dense arm carried zero weight · resolved 2026-09-19

**The corpus was re-embedded with the model's reference implementation on
2026-09-19. `DENSE_WEIGHT` now ships at `2.0` and the dense arm scores 61.4%
Recall@5 — the strongest of the three arms (`EVAL.md` §4a).** This section is kept
in full because the defect is instructive and the canary genuinely could not catch
it; §5d records how it was closed.

### 5a. What was wrong

`DENSE_WEIGHT` shipped at `0.0` for months. Not because the model is weak — because
the corpus vectors were produced by a backend whose output does not match the
model's reference implementation.

Measured head to head: 12 questions against a pool of 203 candidate chunks, the only
variable being which implementation produced the vectors.

| | median rank of the correct chunk | top-5 | top-1 |
|---|---|---|---|
| the backend that embedded this corpus | **32** | 3/12 | — |
| the model's reference implementation | **1** | 10/12 | 10/12 |
| random baseline | ~101 | | |

Ruled out as explanations: dtype (fp16 against fp32 agrees at 0.999986), pooling
position (−1, −2 and −3 all land around 0.12–0.16 against the corpus vectors), and
weight loading (595,776,512 parameters, no warnings).

### 5b. Why the canary could not catch it

! This is exactly the failure the canary cannot catch, and it is worth being precise
about why. The canary asks *is the query in the same space as the corpus?* The answer
was yes — consistently, at 0.99998. Both sides were in the same wrong space. Corpus
self-retrieval scored 8/8. Only question→clause retrieval failed, because only that
crosses from one kind of text to another, and only the head-to-head above compares the
space against an independent implementation of the same model.

~~The dense arm ships at weight **0.0** and contributes **0.0%** Recall@5~~ — superseded
by §5d; kept because the reasoning below is why the re-embed was not treated as urgent
at the time. As written then: sparse
and text carry retrieval to **54.5%** without it (`EVAL.md` §4). The defect above
is therefore documented, ✗ load-bearing for *ranking* — nothing in the query path
depends on the dense vectors today.

### 5c. It did constrain the deployment, though

The corpus is reproducible **only** by the pinned `86-1.7.2` image. Measured
against stored vectors (`HARDWARE.md` §1):

| provider | worst cosine over 5 chunks |
|---|---|
| TEI `86-1.7.2` (the image that embedded it) | **0.99998** |
| TEI `cpu-1.9.3` | 0.00190 |
| transformers reference (`docker/embed_cpu`) | 0.00190 |

The last two are **byte-identical to each other**. Two independent
implementations agreeing and both being orthogonal to the corpus is what turns
§5 from "the space looks wrong" into "the space is a property of one container
tag". Consequences, neither of them about CPU:

- **TEI cannot be upgraded**, on GPU or CPU. The startup canary refuses any
  other provider, which is the guard working correctly.
- A CPU-only deployment is impossible while that remains true, because no CPU
  build of 1.7.2 runs (it segfaults) and every build that does run is in the
  other space.

Both unblock the same way and only that way.

### 5d. How it was closed

The precondition set when the re-embed was previously closed was *a number, not a
better argument*: embed the chunks the labelled answers live in plus distractors and
score the dense arm on that subset. That number came back decisive —
`dev_tools/pre_embed/dense_probe.py --pool 5000 --hard 40`:

| space | Recall@5 | Recall@10 | median rank |
|---|---|---|---|
| stored (TEI `86-1.7.2`) | 2.6% | 2.6% | 1,316 |
| reference implementation | **79.5%** | **97.4%** | **1** |

`dev_tools/pre_embed/reembed.py` then wrote 355,623 vectors into `dense_v2`
alongside the live column, and `--swap` promoted them in one transaction. Both
columns were dumped to `.test/backup/` first; the old one verified at 355,621 rows.
Centroids and chunk assignments were rebuilt from the new vectors
(`dev_tools/cluster_maint/kmeans.py`), because the 177 centroids were means of the
old ones and routing would otherwise point into a space that no longer exists.

**The container lock-in is gone.** `docker/embed_cpu` — the transformers reference
server, which scored 0.002–0.19 against the old corpus and was correctly refused by
the canary — now reproduces the corpus at **0.999997**, and the engine's own startup
canary passes at **1.00000**. Both consequences above are lifted: TEI can be
upgraded, and a CPU-only deployment is possible.

! **The canary still cannot prove correctness, and that has not changed.** It asks
whether query and corpus occupy the same space; the old corpus passed at 0.99998
while ranking the right answer at median 857. Both sides agreed and both were wrong.
What establishes that *this* space is right is the head-to-head retrieval
measurement above, against an independent implementation — the same standard that
exposed the original defect.

! **Dense degrades with corpus size, measured rather than assumed.** Recall@5 over
the same 39 queries: **76.9%** against an 11k pool, **74.4%** at 103k, **69.2%** at
the full 355,621. Roughly 7.7 points across a 32× larger field. Worth knowing before
anyone extrapolates a small-pool probe to production.

! **The query instruction was tested and does not help.** `corpus_meta
.dense_instruction` is NULL and stays NULL: prefixing queries with Qwen3's
`Instruct: …` form scored identically at k=5 (76.9%) and *worse* at k=10 (92.3%
against 97.4%). Do not add one.

---

## 6. Reranker

Not used. RRF is the ranking mechanism. A cross-encoder reranker would put a model back
on the query path — a round-trip per request, a second set of weights to keep
consistent, and a dependency the lightweight design exists to avoid. It stays out until
the eval set shows fusion is the binding constraint.
