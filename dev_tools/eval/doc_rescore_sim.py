"""Offline: re-rank inside each ranked regulation by dense similarity to the question.

    VERA_HTTP=... DATABASE_URL=... EMBED_ENDPOINT=... python dev_tools/eval/doc_rescore_sim.py

Against an engine started with `TOP_K` = `CANDIDATE_POOL` (the reply is the scored pool,
final order). A regulation keeps the SLOTS it holds in the ranking; only which of its
chunks fill them changes, operative chunks first (as the engine already does), then:

* `pool`   its chunks already in the pool, ordered by cosine to the question
* `doc@D`  its best chunks by cosine over the WHOLE document (Postgres computes the
           distances, `ORDER BY dense <=> q LIMIT`), for the first D regulations ranked

`shipped` is the engine's order. Recall@5 / @10 / MRR, all and holdout, plus wins/losses.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request
from pathlib import Path

import psycopg

sys.path.insert(0, str(Path(__file__).parent))
from article_sim import operative, pools  # noqa: E402


def embedder(cur):
    cur.execute("SELECT to_jsonb(corpus_meta) ->> 'dense_query_instruction' FROM corpus_meta"
                " ORDER BY created_at DESC LIMIT 1")
    instr = cur.fetchone()[0]
    prefix = f"Instruct: {instr}\nQuery: " if instr else ""
    url = os.environ.get("EMBED_ENDPOINT", "http://127.0.0.1:8080") + "/embed"

    def embed(text: str) -> str:
        body = json.dumps({"inputs": prefix + text, "truncate": False}).encode()
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            return "[" + ",".join(f"{x:.6f}" for x in json.load(r)[0]) + "]"
    return embed


def main() -> None:
    cur = psycopg.connect(os.environ["DATABASE_URL"]).cursor()
    cases = pools(cur)
    spec = {q["id"]: q for q in json.loads((Path(__file__).parent / "queries.json")
                                           .read_text("utf-8"))["queries"]}
    embed = embedder(cur)
    every = sorted({i for c in cases for i in c["ids"]})
    cur.execute("SELECT id, doc_id, chapter, article FROM chunks WHERE id = ANY(%s)", (every,))
    meta = {r[0]: r[1:] for r in cur.fetchall()}
    depth = [int(x) for x in os.environ.get("DOCS", "3,5").split(",")]
    per_doc = int(os.environ.get("PER_DOC", "6"))

    results: dict[str, list[list[str]]] = {"shipped": [], "pool": []}
    for d in depth:
        results[f"doc@{d}"] = []
    for c in cases:
        ids = c["ids"]
        v = embed(spec[c["id"]]["query"])
        cur.execute("SELECT id, dense <=> %s::halfvec FROM chunks WHERE id = ANY(%s)", (v, ids))
        dist = {r[0]: float(r[1]) for r in cur.fetchall()}
        order_docs: list[str] = []
        for i in ids:
            dd = meta.get(i, (None,))[0]
            if dd and dd not in order_docs:
                order_docs.append(dd)
        fetched: dict[str, list[str]] = {}
        if order_docs:
            cur.execute(
                "SELECT doc_id, id, chapter, article, d FROM (SELECT doc_id, id, chapter, article,"
                " dense <=> %s::halfvec AS d, row_number() OVER (PARTITION BY doc_id ORDER BY"
                " dense <=> %s::halfvec) AS n FROM chunks WHERE doc_id = ANY(%s) AND indexable) s"
                " WHERE n <= %s ORDER BY doc_id, d", (v, v, order_docs[: max(depth)], per_doc))
            for doc, i, ch, ar, dd in cur.fetchall():
                meta.setdefault(i, (doc, ch, ar))
                dist.setdefault(i, float(dd))
                fetched.setdefault(doc, []).append(i)

        def refill(source):
            out = list(ids)
            seen_docs = []
            for k, i in enumerate(ids):
                doc = meta.get(i, (None,))[0]
                if doc and doc not in seen_docs:
                    seen_docs.append(doc)
            for rank_doc, doc in enumerate(seen_docs):
                slots = [k for k, i in enumerate(ids) if meta.get(i, (None,))[0] == doc]
                members = source(doc, rank_doc, [ids[k] for k in slots])
                if members is None:
                    continue
                ordered = sorted(members, key=lambda i: (not operative(meta[i][1], meta[i][2]),
                                                         dist.get(i, 9.0)))
                for k, i in zip(slots, ordered):
                    out[k] = i
            # an id may now appear twice (pulled into one slot, still in another): keep first
            seen, dedup = set(), []
            for i in out:
                if i not in seen:
                    dedup.append(i)
                    seen.add(i)
            return dedup

        results["shipped"].append(ids)
        results["pool"].append(refill(lambda doc, r, members: members))
        for d in depth:
            results[f"doc@{d}"].append(refill(
                lambda doc, r, members, d=d: (list(dict.fromkeys(fetched.get(doc, []) + members))
                                              if r < d else None)))

    base = results["shipped"]
    for name, lists in results.items():
        for split in ("all", "holdout"):
            idx = [k for k, c in enumerate(cases) if split == "all" or c["holdout"]]
            h5 = h10 = 0
            mrr = 0.0
            win = loss = 0
            for k in idx:
                t = set(cases[k]["targets"])
                ids = lists[k]
                a = any(i in t for i in ids[:5])
                b = any(i in t for i in base[k][:5])
                h5 += a
                h10 += any(i in t for i in ids[:10])
                mrr += next((1 / (j + 1) for j, i in enumerate(ids) if i in t), 0.0)
                win += a and not b
                loss += b and not a
            n = len(idx)
            print(f"  {name:8} {split:8} n={n}  Recall@5 {h5 / n:.1%}  Recall@10 {h10 / n:.1%}"
                  f"  MRR {mrr / n:.3f}  wins {win} losses {loss}")


if __name__ == "__main__":
    main()
