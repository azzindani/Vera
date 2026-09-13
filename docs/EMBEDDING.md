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

## 5. The dense arm carries zero weight, and why

`DENSE_WEIGHT` ships at `0.0`. Not because the model is weak — because the corpus
vectors were produced by a backend whose output does not match the model's reference
implementation.

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

! This is exactly the failure the canary cannot catch, and it is worth being precise
about why. The canary asks *is the query in the same space as the corpus?* The answer
was yes — consistently, at 0.99998. Both sides were in the same wrong space. Corpus
self-retrieval scored 8/8. Only question→clause retrieval failed, because only that
crosses from one kind of text to another, and only the head-to-head above compares the
space against an independent implementation of the same model.

The dense arm ships at weight **0.0** and contributes **0.0%** Recall@5; sparse
and text carry retrieval to **54.5%** without it (`EVAL.md` §4). The defect above
is therefore documented, ✗ load-bearing for *ranking* — nothing in the query path
depends on the dense vectors today.

### It does constrain the deployment, though

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

---

## 6. Reranker

Not used. RRF is the ranking mechanism. A cross-encoder reranker would put a model back
on the query path — a round-trip per request, a second set of weights to keep
consistent, and a dependency the lightweight design exists to avoid. It stays out until
the eval set shows fusion is the binding constraint.
