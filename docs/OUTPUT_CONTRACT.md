# OUTPUT_CONTRACT.md

What `search_knowledge` returns, and why none of it is prose.

The types are in `crates/contract/src/contract.rs`. Field names there are the wire
format an agent parses: renaming one is a breaking change, not a refactor.

---

## 1. Vera returns evidence; the agent writes the prose

The user-facing answer is a summary plus links and locators a human can check. The
**calling agent** composes it. Vera returns everything needed to: ranked results with
snippets and component scores, complete provenance per result, and a ready-to-render
citation block.

Summarizing requires an LLM. An LLM inside the engine would break statelessness, add a
model round-trip to every call, and create the dependency this design exists to avoid.
Because Vera is called *by* an agent, the model already exists on the caller's side —
so summarization is free where it belongs and absent where it would be expensive.

Nothing in the response type holds a summary field, and nothing in it may ever hold
one.

---

## 2. Schema

```jsonc
{
  "success": true,
  "op": "search_knowledge",
  "query": "ketentuan sanksi keterlambatan pelaporan pajak",

  // Detected by the engine via the domain gate, ✗ supplied by the agent.
  // null when nothing matched above threshold — see §4.
  "detected_domain": "regulations",
  "domain_confidence": 0.88,
  "clusters_probed": 5,

  "results": [
    {
      "id": "reg::uu-28-2007::pasal-9::c3",
      "snippet": "Wajib Pajak yang terlambat ...",   // bounded preview, ✗ full text
      "score": 0.871,                                 // fused RRF score
      // ! Two of three arms. The text arm — the best performer at 40.9% — has
      // no field here, so a result it alone found shows all-zero components.
      "scores": { "dense": 0.83, "bm25": 0.61 },
      "source": {
        "title": "UU No. 28 Tahun 2007 — Ketentuan Umum Perpajakan",
        "url": "https://...",                         // omitted if ingest recorded none
        // ! page is reachable only for a corpus ingested with page numbers.
        // The current one has no page column, so every locator is section-only.
        "locator": { "section": "Pasal 9 ayat (3)" }
      }
    }
  ],

  "citation_block": [
    { "index": 1, "text": "UU No. 28 Tahun 2007, Pasal 9 ayat (3) — https://..." }
  ],

  "summary_payload": {
    "snippets": ["...", "..."],
    "sources":  ["[1]", "[2]"],
    "coverage": "5 clusters probed, 3 distinct documents, top score 0.87"
  },

  // Hits from the GLOBAL exact-identifier path, which bypasses routing.
  "exact_matches": [
    { "id": "reg::uu-28-2007::pasal-9", "matched_on": "UU 28/2007" }
  ],

  "confidence": "high",        // high | medium | low | none — §4
  "progress": ["embedded query (1024 dims)", "probed 5 of 177 clusters", "..."],
  "token_estimate": 412,
  // ! Always false. Nothing in the search path ever sets it — result count is
  // bounded by TOP_K without reporting that anything was dropped. read_chunk
  // has its own working `truncated`; this one is a promise, ✗ a signal.
  "truncated": false,
  "hint": null                 // present only when there is something to act on
}
```

`success` is first in the serialization, not by luck: `serde_json` runs with
`preserve_order` because without it keys sort alphabetically and that ordering silently
stops holding.

---

## 3. Provenance

The locator is the whole double-check promise.

| Field | Example | Verified by |
|---|---|---|
| `page` | `14` | opening the source at that page |
| `section` | `"Pasal 9 ayat (3)"` | reading that clause |

! `page` is **not populated by the current corpus** — there is no page column in
the schema and `to_results` sets it `None`. The field stays in the contract
because a future corpus ingested from paginated sources can fill it; until then
every locator is section-only, built from `chapter` and `article`.

Rules:

- Provenance is **captured at ingestion** and stored immutably with the chunk. The
  engine never synthesizes or guesses a link at query time.
- `url` is the **original** source a human clicks, never an internal path.
- Prefer the most precise locator available: clause > section > page.
- A missing `url` is **omitted**, not emptied. This corpus genuinely has chunks without
  one; an empty string would render as a link leading nowhere, which is worse than an
  honest absence.

`get_provenance` returns the same bundle for a set of ids, for an agent that wants to
verify without re-searching.

---

## 4. Confidence, and the honest empty answer

| Value | Means |
|---|---|
| `high` | strong top scores, an exact-identifier match, or agreement between arms |
| `medium` | moderate scores, single-arm support |
| `low` | top scores clustered and weak — the agent should widen or surface uncertainty |
| `none` | nothing matched the domain gate |

`exact_matches` is reported separately so the agent can see when a hit came from the
global identifier path rather than semantic routing. Those are the high-trust "this
regulation exists and here it is" results.

! **Two different outcomes share this response today.** `SearchResponse::no_matching_domain`
is returned both when the domain gate refuses *and* when fusion produced no candidates
at all — a query whose terms are entirely out of vocabulary, say. The second case sets
`detected_domain: null`, `clusters_probed: 0` and a hint reading "query matched no known
knowledge base", none of which is true: the domain matched and clusters were probed.
It can even carry non-empty `exact_matches` beside a null domain, which is
self-contradictory. "The corpus does not cover this" and "this corpus covers it but
nothing matched" are different answers and an agent should be able to tell them apart.

**No matching domain returns `success: true`.** Empty `results`, `detected_domain:
null`, `confidence: "none"`, and a hint saying the query did not match this knowledge
base. The engine worked correctly and found nothing — which is deliberately distinct
from an error, because a confidently wrong domain is worse than an honest "nothing
matched", and downstream nothing reveals the difference if the engine guesses.

---

## 5. Size discipline

- `search_knowledge` returns snippets and addresses, never documents. Full text comes
  from `read_chunk`, one chunk at a time, under a server cap a caller may lower but
  never raise.
- `get_provenance` honours at most `MAX_PROVENANCE_IDS` ids. An agent pasting a whole
  result set back is the normal case, and an uncapped id list is the one place a
  read-only server can still be made to allocate without limit.
- Every response carries `token_estimate` so the agent can budget its own context.
  Agentic loops are token-expensive, and the caller is the one that pays.
