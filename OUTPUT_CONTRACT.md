# OUTPUT_CONTRACT.md — Vera

What `search_knowledge` returns, why the agent (not Vera) writes the summary, and how
the double-check provenance works for both documents and media.

---

## 1. The principle: Vera returns evidence, the agent writes prose

The final user-facing output is a **summary + a list of source links + locators so the
human can independently verify** each claim. That output is composed by the **calling
agent**, not by Vera. Vera returns everything the agent needs to build it:

- ranked results with snippets and scores,
- complete provenance per result,
- a ready-to-render citation block.

**Why the engine does not summarize:** summarizing requires an LLM. Putting an LLM in
the engine would break statelessness, add latency and cost on every call, and create a
model dependency the lightweight design exists to avoid. Because Vera is built for
**agentic AI use**, the agent already is the LLM — so summarization is free where it
belongs and absent where it would be expensive. This is the resolution of the
"who summarizes?" loophole (`LOOPHOLES.md` §2).

---

## 2. `search_knowledge` response schema

```jsonc
{
  "success": true,
  "op": "search_knowledge",
  "query": "ketentuan sanksi keterlambatan pelaporan pajak",
  "detected_domain": "regulations",   // chosen by the engine via anchor match, not by the agent
  "domain_confidence": 0.88,           // anchor-match score; low → "no matching domain" (see §4)
  "clusters_probed": 5,

  "results": [
    {
      "id": "reg::uu-28-2007::pasal-9::c3",   // stable chunk id
      "snippet": "Wajib Pajak yang terlambat ...",   // bounded preview, not full text
      "score": 0.871,                          // fused RRF score
      "scores": { "dense": 0.83, "bm25": 0.61 }, // component scores (transparency)
      "source": {
        "title": "UU No. 28 Tahun 2007 — Ketentuan Umum Perpajakan",
        "url": "https://peraturan.example/uu-28-2007.pdf",
        "locator": {
          "page": 14,
          "section": "Pasal 9 ayat (3)"   // most precise unit available: clause > section > page
        }
      }
    }
    // ... up to max_results
  ],

  "citation_block": [
    // ready-to-render, ordered list the agent can drop into its answer
    "[1] UU No. 28 Tahun 2007, Pasal 9 ayat (3), p.14 — https://peraturan.example/uu-28-2007.pdf"
  ],

  "summary_payload": {
    // compact, de-duplicated material the agent uses to write the summary
    "snippets": ["...", "..."],
    "sources":  ["[1]", "[2]"],
    "coverage": "5 clusters probed, 3 distinct documents, top score 0.87"
  },

  "exact_matches": [
    // results found via the GLOBAL exact-identifier path (routing bypass)
    { "id": "reg::uu-28-2007::pasal-9", "matched_on": "UU 28/2007" }
  ],

  "confidence": "high",            // "high" | "medium" | "low" — see §4
  "progress": [ /* embed / route / scan / fuse steps */ ],
  "token_estimate": 412,
  "truncated": false
}
```

The agent's job: read `summary_payload` + `results`, write a short summary, and append
`citation_block` so the human can click through and verify.

---

## 3. Provenance (document-scoped)

The `locator` is the heart of the double-check promise. Vera's corpus is documents
(regulations, contracts), so the locator points to a precise place in a source document:

| Field | Example | Verify by |
|---|---|---|
| `page` | `14` | opening the PDF at that page |
| `section` | `"Pasal 9 ayat (3)"` | reading that clause/section |

Rules:
- Provenance is **captured at ingestion** and stored immutably with the chunk. The
  engine never synthesizes or guesses a link at query time (`LOOPHOLES.md` §8).
- `url` is the *original* source, not an internal path — it is what the human clicks.
- Prefer the most precise locator available: clause > section > page.

---

## 4. Confidence and the "did we miss it?" signal

Because the priority is *never miss a regulation*, the response carries an honest
confidence signal so the agent can decide whether to widen the search:

- `high` — strong top scores, exact-identifier match present, or tight agreement
  between dense and BM25.
- `medium` — moderate scores, single-half support.
- `low` — top scores clustered and low, no exact match → the agent should consider
  calling again with a larger `clusters_probed`, or surface uncertainty to the human.

`exact_matches` is always reported separately so the agent can see when a hit came from
the **global** exact-identifier path rather than semantic routing — these are the
high-trust, "this exact regulation number exists" results.

**No matching domain.** Domain is detected *by the engine* (anchor match on the query
vector), not chosen by the agent. If no domain anchor is matched above the confidence
threshold, the engine does **not** guess — it returns `success: true` with an empty
`results` array, `detected_domain: null`, and `confidence: "none"`, plus a `hint` that
the query did not match any known knowledge base. This is deliberate: a confidently
wrong domain is worse than an honest "nothing matched." See `MCP_ENGINE.md` §2 and
`LOOPHOLES.md` §9.

---

## 5. Size discipline

- `search_knowledge` returns **snippets and addresses**, never full documents. Full
  text is fetched per-chunk via `read_chunk` only for what the agent actually needs.
- Result count is bounded by `max_results` and the constrained-mode cap.
- Every response carries `token_estimate` so the agent can budget its own context —
  important because agentic loops are token-expensive (this is the upstream token
  discipline, reframed for an agent caller rather than a local model).
