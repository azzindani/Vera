"""Offline: does phrase evidence (adjacent query words found adjacent in the text) rank better?

    VERA_HTTP=... DATABASE_URL=... python dev_tools/eval/phrase_sim.py

Reuses the pools of `article_sim.py` (`POOLS`, the engine's final order at `TOP_K` =
`CANDIDATE_POOL`). For each candidate:

    phrase = share of the question's content bigrams that occur, adjacent, in the body
    score  = 1 / (k + rank) * (1 + w * phrase)

then re-sorted, operative-first within a regulation kept. Content words: lowercased
alphanumerics, the corpus's declared stopwords removed, longer than 3 characters. Reports
each `w`, all and holdout, wins and losses against the engine's order. TRAIN ONLY picks `w`.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import psycopg

sys.path.insert(0, str(Path(__file__).parent))
from article_sim import pools  # noqa: E402


def words(text: str, stop: set[str]) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 3 and t not in stop]


def bigrams(ws: list[str]) -> set[tuple[str, str]]:
    return set(zip(ws, ws[1:]))


def main() -> None:
    cur = psycopg.connect(os.environ["DATABASE_URL"]).cursor()
    cases = pools(cur)
    spec = {q["id"]: q for q in json.loads((Path(__file__).parent / "queries.json")
                                           .read_text("utf-8"))["queries"]}
    cur.execute("SELECT scoring_vocabulary FROM corpus_meta")
    v = cur.fetchone()[0]
    v = json.loads(v) if isinstance(v, str) else v
    stop = set(v.get("stopwords", []))
    every = sorted({i for c in cases for i in c["ids"]})
    cur.execute("SELECT id, body FROM chunks WHERE id = ANY(%s)", (every,))
    grams = {r[0]: bigrams(words(r[1], stop)) for r in cur.fetchall()}
    k = float(os.environ.get("RRF_K", "60"))
    for w in [0.0] + [float(x) for x in os.environ.get("W", "0.25,0.5,1,2,4").split(",")]:
        for split in ("train", "holdout", "all"):
            idx = [j for j, c in enumerate(cases)
                   if split == "all" or (split == "holdout") == c["holdout"]]
            h5 = h10 = win = loss = 0
            mrr = 0.0
            for j in idx:
                c = cases[j]
                qg = bigrams(words(spec[c["id"]]["query"], stop))
                ids = c["ids"]

                def phrase(i):
                    return len(qg & grams.get(i, set())) / len(qg) if qg else 0.0
                order = sorted(range(len(ids)),
                               key=lambda r: -(1 / (k + r + 1)) * (1 + w * phrase(ids[r])))
                new = [ids[r] for r in order]
                t = set(c["targets"])
                a = any(i in t for i in new[:5])
                b = any(i in t for i in ids[:5])
                h5 += a
                h10 += any(i in t for i in new[:10])
                mrr += next((1 / (r + 1) for r, i in enumerate(new) if i in t), 0.0)
                win += a and not b
                loss += b and not a
            n = len(idx)
            print(f"  w={w:<5} {split:8} n={n}  Recall@5 {h5 / n:.1%}  Recall@10 {h10 / n:.1%}"
                  f"  MRR {mrr / n:.3f}  wins {win} losses {loss}")


if __name__ == "__main__":
    main()
