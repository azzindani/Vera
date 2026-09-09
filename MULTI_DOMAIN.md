# MULTI_DOMAIN.md — Vera

How Vera scales from *one domain, one data source* to **many domains, many sources**,
each with its own metadata, its own notion of provenance, and its own mix of retrieval
methods.

This is the design record for the foundation. It is deliberately written **before** the
code, because every decision here is a schema decision, and a schema decision made
wrongly is paid for in a full re-ingest of every row in the corpus.

---

## 0. Why this document exists

The predecessor system worked and was proven in production, but it was built for **one
domain with one data source**. Everything else — metadata fields, the search method,
the provenance shape — was allowed to be implicit, because there was only ever one of
each. Adding a second source means changing code, and adding a tenth means a rewrite.

Vera exists to make the second, tenth and fiftieth source **configuration**.

The immediate constraint: **only one data source is available at volume today.** That
does not weaken the foundation requirement — it changes how the foundation is
*validated*. See §11.

---

## 1. The problem, stated precisely

A domain is not a source. In law:

| Domain | Source | Provenance unit | Identifier grammar | Dominant retrieval |
|---|---|---|---|---|
| Legal | Regulations | article / clause (`Pasal 9 ayat (3)`) | `UU 28/2007` | identifier + semantic |
| Legal | Supreme Court | paragraph of judgment | `123 K/Pdt/2020` | semantic + citation graph |
| Legal | Contracts | clause (`12.3`) | internal contract id | structured filter + semantic |

And the same shape appears in a completely different domain:

| Domain | Source | Provenance unit | Identifier grammar | Dominant retrieval |
|---|---|---|---|---|
| Medical | Clinical guidelines | recommendation + strength of evidence | guideline id + version | temporal (version) + semantic |
| Medical | Drug labels | section (indications, contraindications) | ATC / registration no. | identifier + structured |
| Medical | Coding systems (ICD-10, SNOMED, MeSH) | node in a tree | code | **hierarchical** + keyword |
| Medical | Literature (PubMed) | abstract / section | DOI / PMID | semantic + citation graph |

Two things follow immediately:

1. **The variation is at the source level, not the domain level.** Regulations and ICD-10
   have less in common with each other than ICD-10 has with a legal code hierarchy.
   Designing "a legal mode" and "a medical mode" would be the same mistake at a larger
   scale.
2. **Nothing about the engine should know the word "legal" or "medical".** Domain is a
   label for grouping and routing. All behaviour comes from the *source*.

---

## 2. The reframe: three levels, not two

The engine today has two routing levels. The problem has three.

```
LEVEL 1   domain          grouping + coarse routing        ("legal", "medical")
LEVEL 2   SOURCE          ← the missing level
                          schema · provenance shape · identifier grammar
                          retrieval mix · validity semantics · chunking
LEVEL 3   cluster         k-means routing within a source
LEVEL 4   leaf            flat scan + lexical within a cluster
```

**The source is the unit of heterogeneity.** It owns:

- its metadata schema,
- what a locator *means*,
- how an identifier is written and normalized,
- which retrieval operators apply and with what weight,
- what "currently valid" means,
- how documents are chunked.

Clustering happens **within** a source, never across. Two sources never share a
centroid space, because their vectors mean different things even when the model is the
same.

---

## 3. Domain-agnostic by construction

The test for every proposed change to the engine:

> Could this code be read by someone who does not know whether the corpus is
> regulations or radiology reports?

If not, it belongs in a source manifest, not in the engine. Concretely, the following
must **leave** the engine core:

| Currently hardcoded | Belongs to |
|---|---|
| `Chunk { locator_page, locator_section, heading_path, identifier }` | source schema |
| `identifier::extract` Indonesian statute grammar (`UU`, `PP`, …) | source manifest |
| `Locator::render` → `"Pasal 9 ayat (3), p.14"` | source locator template |
| One global `SearchConfig` weighting | per-source scoring profile |
| `DEFAULT_QUERY_INSTRUCTION` mentioning "legal or regulatory question" | per-source (it is part of the embedding space) |

---

## 4. What actually varies per source

Worth stating plainly, because the answer is smaller than it first appears and that
changes how much work this is.

**Genuinely different (must be modelled):**

- **Metadata schema.** A judgment has court, panel, disposition, precedential status.
  A drug label has substance, ATC class, marketing authorisation, revision date. There
  is no useful common subset beyond an id and a body.
- **Provenance / locator shape.** Page+section is document-specific. A tree node has a
  path; a record has a timestamp; a media asset has an offset.
- **Identifier grammar.** `UU 28/2007`, `123 K/Pdt/2020`, `ICD-10 E11.9`, `PMID
  12345678`. Same *role*, incompatible syntax.
- **Validity semantics.** "In force" for a regulation, "current version" for a
  guideline, "not retracted" for a paper, "not superseded" for a code.
- **Chunking.** Statutes chunk by article; judgments by holding; guidelines by
  recommendation; ontologies are already atomic.

**Not actually different (do not build twice):**

- Dense retrieval, lexical retrieval, and RRF fusion are the right substrate for every
  source above. What differs is **weighting and which operators are enabled** — not the
  algorithms themselves.

**Genuinely new retrieval work, and only this:**

- **Hierarchical** operators (§5) — needed by ontologies and by structured legal codes.
- **Graph** operators (§5) — citations, "cited by", "supersedes", "overrules". This is
  where a knowledge graph lands: ✗ a new layer, but the concrete form of this operator
  plus the `traverse` edges. The exact-identifier path is already a degenerate case of
  it — one entity type, string-matched, stored in a column. See `FACTORS.md` §5b for
  what generalising it costs and how to decide whether it earns that cost.

That is the honest scope: two new operator families, one schema model, and a
configuration layer. Not N engines.

---

## 5. Retrieval primitives: rankers vs constraints

The critical distinction, and the one that most affects correctness.

**Rankers** produce an ordered candidate list with scores. They are fused with RRF.

| Ranker | What it is good at |
|---|---|
| `semantic` | paraphrase, concept, "sanctions for late filing" |
| `lexical` | exact terms, rare words, codes appearing in text |
| `identifier` | a named thing: `UU 28/2007`, `E11.9`, a PMID — **runs globally, bypasses routing** |
| `graph` | what cites / is cited by / supersedes this |

**Constraints** produce a predicate. They do **not** rank — they narrow the space that
rankers search.

| Constraint | Example |
|---|---|
| `structured` | `court = "MA" AND year >= 2020`, `atc_class = "A10B"` |
| `hierarchical` | "within Chapter IV", "descendants of E11" |
| `temporal` | in force on 2019-03-01; guideline version current as of today |

! **Conflating the two is the main way this design can go wrong.** A constraint applied
*after* routing produces silent zero-recall: the engine probes five clusters, filters
them to nothing, and returns a confident empty result while matching rows sit in
clusters it never opened. This is the same failure class as the exact-identifier
bypass, in new clothes — and equally invisible, because an empty result is
indistinguishable from "nothing exists".

Constraints must therefore be applied **before or during routing**, never after. See §7.

A source manifest declares which rankers are enabled, their fusion weights, and which
constraint fields exist. Multi-step search (§10) is the agent composing these across
calls.

---

## 6. The source manifest, and where code is still required

**A new source must be a manifest, not a pull request.** If integrating "Peraturan
Daerah" or "SNOMED CT" requires Rust, the original problem has been rebuilt at a larger
scale.

Declarative (the manifest):

- metadata schema (field names, types, which are filterable)
- locator template — how to render provenance for a human
- identifier patterns + normalization rules (a table, ✗ hand-written parsing)
- enabled rankers and their fusion weights
- constraint fields and their selectivity hints
- validity model — which fields mean "current", default temporal filter
- hierarchy field (path / parent) if the source has one
- embedding space (model, dim, instruction) — a source may legitimately differ

Code (a registered plugin, expected to be rare):

- chunking strategy for a new document *shape*
- a genuinely novel ranker (the citation graph is the first and possibly only one)

! The engine core must contain **no branch on source id.** It reads a manifest and
executes. A `match source_id { "regulations" => …, }` anywhere in the engine means the
abstraction has failed.

---

## 7. Constraints gate routing — the cardinality rule

Routing exists to prune. A selective constraint **has already pruned**, and often much
harder than routing could.

The rule:

```
estimate |candidate set| after constraints
   ├─ small  (< configured threshold, e.g. ~50K rows)
   │     → scan the constrained set directly · SKIP routing entirely
   │       (routing can only lose recall here, and saves nothing)
   └─ large
         → route within the constrained set, probing clusters as usual
```

This requires per-source **selectivity statistics** collected at ingest, so the
estimate is cheap and does not itself become a scan. The threshold is config, per the
"never hardcode a limit" rule.

Corollary: clustering should ideally be *aligned* with the most selective constraint
fields, so that a filter and a cluster boundary tend to agree rather than cut across
each other.

---

## 8. Source detection: fan-out, not argmax

Layer 1 today picks a **single** winner by cosine against a domain anchor. With several
sources inside one domain this is close to a coin flip, and there is measured evidence:
anchor similarity across the benchmark corpus spans only p1 0.71 → p95 0.75. That
spread is far narrower than the difference between, say, tax statutes and tax
judgments — both of which are *about tax*.

So:

- **Fan out to every source above threshold**, search them in parallel, fuse with RRF.
  Rank-based fusion works across independently-scored sources, which is precisely why
  RRF was chosen. `LOOPHOLES.md` §9 already anticipates this ("fan out to the top-N
  domains").
- **Use non-vector cues for source selection.** `Putusan 123 K/Pdt/2020` should select
  the judgments source by *grammar*, not geometry. Identifier patterns are strong
  source signals and cost nothing to check.
- **Never return "no source matched" when an identifier matched.** Same rule as §5.

---

## 9. Validity and time are first-class

! A repealed regulation is a perfect semantic match and a wrong answer. So is a
superseded clinical guideline, a withdrawn drug label, or a retracted paper. **Pure
similarity search cannot express this, and no amount of ranking fixes it.**

Every source therefore carries a validity model:

- `valid_from`, `valid_until`
- `status` (in force / repealed / superseded / draft / retracted)
- `supersedes` / `superseded_by`

Two distinct query modes follow, and they must not be conflated:

- **Current** (default) — only what is in force now. This is what a user almost always
  means, and it must be the default, not an opt-in.
- **As-of** — what was in force on a given date. Required for legal advice about past
  conduct and for reconstructing a clinical decision.

Both are constraints (§5), so both are subject to the cardinality rule (§7).

---

## 10. Bulk and multi-step search

**Bulk.** Many queries in one call. The argument is not convenience — it is that the
embedding provider is a network round-trip, and batching decomposed sub-queries into a
single request is already required by `MCP_ENGINE.md` §6.6. A `search_batch` tool
amortises that round-trip and lets one admission slot cover the whole set, which
matters under the concurrency ceiling.

**Multi-step.** "Find the statute → find judgments citing it → check it is still in
force" is a *sequence*, and the sequencing belongs to the **agent**, not the engine.
The engine stays a deterministic function; putting a planner inside it would require a
model and break every property this design exists to protect.

But the engine must make composition **possible**:

- stable, addressable ids that survive re-clustering,
- cheap follow-up primitives: hierarchy expand, graph traverse, `as_of` re-check,
- results carrying enough structure (source, ids, metadata) to be the input to the next
  step,
- continuation/cursor for walking a large result set rather than re-querying.

! The division of labour: **the engine exposes primitives and never plans; the agent
plans and never ranks.** Multi-step search is a property of the tool surface, not a
feature of the retrieval core.

---

## 10b. The operation surface: primitives, not verbs

Today: **5 tools** — `list_domains`, `search_knowledge`, `read_chunk`,
`get_provenance`, `explain_routing`. Each is a fixed verb over a fixed corpus shape.

Adding the capabilities in this document *as verbs* is the obvious path and the wrong
one: `search_by_filter`, `search_hierarchy`, `search_as_of`, `list_children`,
`get_ancestors`, `get_citations`, `get_cited_by`, `list_versions`, `search_batch`,
`list_sources`, `describe_schema`… ~15 tools, past the ≤8 budget, and every new source
type tempts another.

! **The unit of extension must be a declared parameter, not a new tool.** A capability
that arrives as a tool means the engine grew a verb; a capability that arrives as a
manifest field means the engine stayed the same size. Only the second scales to fifty
sources.

### The four primitives

| Op | Role | Replaces / absorbs |
|---|---|---|
| `describe` | what exists: sources, their filterable fields, identifier grammars, edge types, validity model | `list_domains`, and everything an agent needs before it can express a constraint |
| `search` | find candidates: query + constraints → ranked, cited results | `search_knowledge`, plus filtered / hierarchical / temporal / batch search |
| `fetch` | by id: provenance, snippet, or full text | `read_chunk` + `get_provenance` — the same operation at two depths |
| `traverse` | from id, follow a declared edge | hierarchy (`parent`, `children`, `ancestors`), graph (`cites`, `cited_by`), time (`versions`, `supersedes`) |

`explain_routing` folds into `search` as a dry-run flag: it is the same planning work
with execution suppressed, and keeping it as a separate tool duplicates the whole
constraint surface. *(Divergence from `MCP_ENGINE.md` §2, which lists it separately.)*

**Four tools cover strictly more capability than today's five**, and adding a source, an
edge type, a filterable field, or a validity model adds **zero** tools.

### The line that must not move

The agent expresses **intent**; the engine owns **strategy**.

| Agent may pass | Engine owns, always |
|---|---|
| query text | which source(s) — detected, never named by the agent (`CLAUDE.md` §7 rule 13) |
| constraints (filter / as-of / subtree) | which rankers run, and their fusion weights |
| `k`, fetch depth, edge name | which clusters are probed |
| dry-run | the fusion method |

! Constraints are safe to accept because they are *intent* ("only Supreme Court, only
in force in 2019"). Ranker selection and weighting are *strategy* and must never be
agent-supplied — that is the same hole as letting the agent name a domain, one level
down. An agent that can set fusion weights can silently disable the keyword half.

### Two rules that make this safe

1. **Bulk is a parameter shape, ✗ a tool.** `search` takes one query or many; `fetch`
   and `traverse` take one id or many. This amortises the embedding round-trip
   (`MCP_ENGINE.md` §6.6) and lets one admission slot cover a batch, without a
   `search_batch` verb. *(Supersedes the `search_batch` suggestion in §10.)*

2. **An unrecognised constraint fails loudly.** If the agent filters on `court` and the
   detected source has no `court` field, the engine returns an error naming the valid
   fields — it must **never** ignore the field and return unfiltered results the agent
   believes are filtered. Silently dropping a constraint is the same failure family as
   filtering after routing (§5): a confident answer to a question that was not asked.
   `describe` exists so the agent can avoid this rather than discover it.

### Why this is also the multi-step answer

Multi-step search (§10) is these four composed by the agent: `describe` → `search` →
`traverse` → `fetch`. The engine needs no planner because the primitives compose, and it
gains no planner because composition happens on the caller's side. Adding a fifth
primitive should be treated as evidence that something is missing from the *model*, not
from the tool list.

---

## 11. Validating a multi-source foundation with one source

The practical constraint: one source at volume, today.

**The key move is to separate the two axes, because they need different corpora:**

| Axis | Question | What it needs |
|---|---|---|
| **Scale** | does it stay fast and bounded at 750K–100M rows? | **one large corpus** — already available |
| **Generality** | does a second source cost a manifest or a rewrite? | **several tiny corpora** — cheap to obtain |

Volume does not test heterogeneity, and heterogeneity does not need volume. Conflating
them is why the previous system reached production before discovering it could not
scale sideways.

Concrete tactics, in order of value:

1. **No privileged path.** Integrate the one real source *through the manifest
   mechanism*, even though it is the only one. If source #1 is special-cased, source #2
   is a rewrite. This costs almost nothing now and is the single highest-value
   discipline.
2. **Split the real corpus into pseudo-sources.** Partition by a metadata field
   (document type, year, issuing body) and register each partition as a *separate
   source* with a deliberately different manifest — different locator template,
   different identifier grammar, different ranker weights. This exercises fan-out,
   cross-source fusion, and per-source scoring on real data at real volume, without a
   second dataset.
3. **Add one deliberately alien small source.** A few thousand rows whose metadata
   shares *no field* with the first — an ontology (ICD-10, MeSH) is ideal because it is
   small, public, hierarchical, and nothing like a statute corpus. This is the real test
   of whether the schema model is general or merely parameterised.
4. **Manifest conformance tests.** Two manifests over the *same* rows must produce
   correspondingly different behaviour (different locators, different identifier hits,
   different ranking). If they do not, the manifest is decorative.
5. **A schema the engine cannot read.** Assert that the engine compiles and runs with
   zero knowledge of any field name — the negative test for §3.

---

## 12. Foundation order of work

Ordered by cost-of-being-wrong, not by visibility.

1. **Data model** — source as a first-class entity; metadata as a validated per-source
   document; polymorphic provenance; validity fields; hierarchy and graph edges.
   *First, because changing it later re-ingests every row.*
2. **Storage layout** — contiguous per-cluster vector blobs, halfvec. Same re-ingest
   cost, so it is the *same decision* as (1), not a later optimisation. It also fixes
   the measured 5.3 µs/row leaf-scan bottleneck.
3. **Eval set with real labels** — before any routing work. Two confident claims about
   this project's own numbers were already wrong until the corresponding metric existed;
   multi-source routing has far more room to be confidently wrong.
4. **Manifest layer + fan-out routing + constraint/cardinality path.**
5. **Hierarchical and graph operators.**

Steps 1 and 2 are one decision, made once. Step 3 is what keeps steps 4–5 honest.

---

## 13. Open questions

- Is the available 750K-row corpus a single source, or already a mixture that could be
  partitioned per §11.2?
- Are structured filters needed at query time in the first release, or is the near-term
  need semantic + identifier only? This decides whether §7 is foundation or phase two.
- Expected source count in twelve months — ~5 or ~50? At 50, metadata schemas need
  their own registry and versioning story.
- Does the existing production RAG use Postgres? If its fan-out orchestration already
  works, Vera may be better as one *leaf* behind it than as the federating router.
- Will sources ever share an embedding space, or should each source pin its own model?
  (Mixed-model corpora are legal here — the space is per-source — but cross-source
  *dense* comparison then becomes meaningless and fusion must stay rank-based.)

---

## 14. The principles, short

1. **The source is the unit of heterogeneity.** Domain is a label; the source owns
   schema, provenance, identifiers, validity and retrieval mix.
2. **The engine never names a domain or a source.** Any `match` on a source id is a
   design failure.
3. **A new source is a manifest, not a pull request.**
4. **Rankers are fused; constraints gate routing.** A constraint applied after routing
   is silent zero-recall.
5. **Clusters live inside a source.** Never cluster across sources.
6. **Fan out across sources; never argmax.** Rank-based fusion is what makes this safe.
7. **Validity is a correctness property, not a filter option.** Current-only is the
   default.
8. **The engine exposes primitives and never plans; the agent plans and never ranks.**
9. **Scale and generality are separate experiments.** One big corpus proves the first;
   several tiny ones prove the second.
10. **Schema and storage layout are one decision, made once**, because both are paid for
    in re-ingest.
11. **Four primitives, not fifteen verbs.** `describe` · `search` · `fetch` ·
    `traverse`. Capability grows through manifest-declared parameters; a new tool means
    the model was wrong.
12. **The agent passes intent, never strategy.** Constraints yes; ranker weights never.
