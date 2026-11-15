"""Offline: should a law's definitions article give way unless the question asks for a meaning?

    VERA_HTTP=... DATABASE_URL=... python dev_tools/eval/definition_sim.py

Reuses `article_sim.py`'s pools (the engine's final order, `TOP_K` = `CANDIDATE_POOL`).
Within each regulation's slots (operative first, as the engine already orders):

* a question that asks for a meaning (`MEANING` below): definitions first
* any other question: the regulation's other operative chunks first, definitions after

A definitions chunk: `Pasal 1` whose text opens like one ("yang dimaksud dengan" within
its first 200 characters). `MODE`: `cue` (above, the default) · `never` (definitions
always last) · `only` (definitions first for a meaning question, nothing else moved).
Reports all / train / holdout with wins and losses.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import psycopg

sys.path.insert(0, str(Path(__file__).parent))
from article_sim import operative, pools  # noqa: E402

MEANING = re.compile(
    r"\b(apa itu|apa yang dimaksud|yang dimaksud|pengertian|definisi|maksudnya|artinya|"
    r"diartikan|apakah yang dimaksud)\b", re.I)


def main() -> None:
    cur = psycopg.connect(os.environ["DATABASE_URL"]).cursor()
    cases = pools(cur)
    spec = {q["id"]: q for q in json.loads((Path(__file__).parent / "queries.json")
                                           .read_text("utf-8"))["queries"]}
    every = sorted({i for c in cases for i in c["ids"]})
    cur.execute("SELECT id, doc_id, chapter, article, left(body, 200) FROM chunks"
                " WHERE id = ANY(%s)", (every,))
    meta = {r[0]: r[1:] for r in cur.fetchall()}

    def definitions(i: str) -> bool:
        _, ch, ar, head = meta[i]
        return (ar or "") == "Pasal 1" and operative(ch, ar) and \
            "yang dimaksud dengan" in (head or "").lower()

    def reorder(ids: list[str], meaning: bool) -> list[str]:
        out = list(ids)
        docs: dict[str, list[int]] = {}
        for k, i in enumerate(ids):
            if i in meta and meta[i][0]:
                docs.setdefault(meta[i][0], []).append(k)
        for slots in docs.values():
            members = [ids[k] for k in slots]

            def key(i, members=members):
                op = operative(meta[i][1], meta[i][2])
                d = definitions(i)
                tier = (0 if d else 1) if meaning else (1 if d else 0)
                return (not op, tier if op else 0, members.index(i))
            for k, i in zip(slots, sorted(members, key=key)):
                out[k] = i
        return out

    asked = sum(bool(MEANING.search(spec[c["id"]]["query"])) for c in cases)
    print(f"  questions asking for a meaning: {asked} of {len(cases)}")
    for split in ("train", "holdout", "all"):
        idx = [c for c in cases if split == "all" or (split == "holdout") == c["holdout"]]
        base5 = new5 = win = loss = 0
        mrr0 = mrr1 = 0.0
        for c in idx:
            t = set(c["targets"])
            mode = os.environ.get("MODE", "cue")
            asks = bool(MEANING.search(spec[c["id"]]["query"]))
            new = reorder(c["ids"], {"cue": asks, "never": False, "only": True}[mode])                 if mode != "only" or asks else list(c["ids"])
            a = any(i in t for i in new[:5])
            b = any(i in t for i in c["ids"][:5])
            new5 += a
            base5 += b
            win += a and not b
            loss += b and not a
            mrr0 += next((1 / (r + 1) for r, i in enumerate(c["ids"]) if i in t), 0.0)
            mrr1 += next((1 / (r + 1) for r, i in enumerate(new) if i in t), 0.0)
        n = len(idx)
        print(f"  {split:8} n={n}  Recall@5 {base5 / n:.1%} -> {new5 / n:.1%}  "
              f"MRR {mrr0 / n:.3f} -> {mrr1 / n:.3f}  wins {win} losses {loss}")


if __name__ == "__main__":
    main()
