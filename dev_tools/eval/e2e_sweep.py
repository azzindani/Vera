"""Score several factor configurations through ONE server process.

    BM25_VOCAB=... EMBED_ENDPOINT=... python dev_tools/eval/e2e_sweep.py

! Why one process: the alternative is editing `Weights::FITTED`, rebuilding and
running `e2e.py` per configuration, which varies the binary as well as the
weights. `search_knowledge` takes `factor_weights` per request, so the same
binary, the same corpus and the same vocabulary answer every row here and the
difference between rows IS the scoring layer.

! `factor_weights` is the EXPERIMENTAL surface (`docs/TOOL_SURFACE.md` §5) and
this is what it is for: measuring a configuration before it becomes a profile.
Nothing here should reach a caller as advice.

Rows are the configurations `docs/EVAL.md` §4 and `docs/SCORING.md` §3 quote.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import e2e  # noqa: E402
from run import resolve_targets  # noqa: E402

import psycopg  # noqa: E402


def w(floor, authority=0.0, structural=0.0, temporal=0.0,
      completeness=0.0, topical=0.0):
    return {
        "relevance_floor": floor, "authority": authority,
        "structural": structural, "temporal": temporal,
        "completeness": completeness, "topical": topical,
    }


# ! The shipped row is written out rather than read from the server, so this
# fails loudly if `Weights::FITTED` moves and nobody updated the docs.
CONFIGS = [
    ("factors off", w(0.0)),
    ("shipped", w(0.3, authority=0.5, structural=0.25, topical=0.25)),
    ("no floor", w(0.0, authority=0.5, structural=0.25, topical=0.25)),
    ("floor 0.4 · the text-arm fit", w(0.4, authority=1.0, structural=0.5, topical=0.25)),
    ("unconstrained · bound 3.75x",
     w(0.0, authority=1.0, structural=0.25, completeness=1.0, topical=0.5)),
]


def score(srv, cur, queries, weights):
    n = hit5 = hit10 = 0
    rr = []
    for q in queries:
        r = srv.call("tools/call", {
            "name": "search_knowledge",
            "arguments": {"query": q["query"], "factor_weights": weights},
        })
        ids = [x.get("id")
               for x in json.loads(r["result"]["content"][0]["text"]).get("results", [])]
        targets = resolve_targets(cur, q)
        if not targets:
            continue
        n += 1
        hit5 += any(x in targets for x in ids[:5])
        hit10 += any(x in targets for x in ids[:10])
        rr.append(next((1 / (i + 1) for i, x in enumerate(ids) if x in targets), 0.0))
    return n, hit5 / n, hit10 / n, sum(rr) / len(rr)


def main():
    srv = e2e.Server()
    cur = psycopg.connect(e2e.PG).cursor()
    spec = json.loads(
        (Path(__file__).parent / "queries.json").read_text(encoding="utf-8")
    )
    queries = [q for q in spec["queries"] if q["type"] != "out_of_domain"]

    print(f"{'configuration':<30}{'Recall@5':>10}{'Recall@10':>11}{'MRR':>8}")
    for label, weights in CONFIGS:
        n, r5, r10, mrr = score(srv, cur, queries, weights)
        print(f"{label:<30}{r5:>10.1%}{r10:>11.1%}{mrr:>8.3f}")
    srv.close()
    print(f"\nn={n} · one binary, one corpus, weights varied per request")


if __name__ == "__main__":
    sys.exit(main())
