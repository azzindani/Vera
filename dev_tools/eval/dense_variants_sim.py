"""Offline: can the SAME model, used differently on the query side, find more answers?

    DATABASE_URL=... POOLS2=pools.json EMBED_ENDPOINT=... python dev_tools/eval/dense_variants_sim.py

Reuses `doc_support_sim.py`'s cache (`POOLS2`: the engine's final order at `TOP_K` =
`CANDIDATE_POOL`, with scores). For each question, a dense arm over the 5 clusters nearest
its vector (one-level routing, the same for every variant so they compare fairly), top 20:

* `instr`    the question with the corpus's instruction, as the engine embeds it
* `plain`    the question alone, no instruction (a second embedding)
* `prf@a/n`  pseudo-relevance feedback: q + a * mean(the engine's top n), renormalised,
             then routed and searched again (no second embedding; one more dense search)

Reports, per variant: arm Recall@20, how many answers it finds that the engine's whole
pool lacks, and a crude fusion: the engine's order plus the variant as one more RRF list
at weight `w` (1 / (60 + rank) each), Recall@5 / MRR with wins and losses, train / holdout.
The fusion is an estimate of an arm's worth, ✗ the engine: the engine fuses before scoring.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request
from pathlib import Path

import numpy as np
import psycopg

sys.path.insert(0, str(Path(__file__).parent))
from doc_support_sim import pools  # noqa: E402

K = 60.0


def vec(text: str) -> np.ndarray:
    return np.array([float(x) for x in text.strip("[]").split(",")], dtype=np.float32)


def lit(v: np.ndarray) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in v) + "]"


def main() -> None:
    cur = psycopg.connect(os.environ["DATABASE_URL"]).cursor()
    cases = pools(cur)
    spec = {q["id"]: q for q in json.loads((Path(__file__).parent / "queries.json")
                                           .read_text("utf-8"))["queries"]}
    cur.execute("SELECT to_jsonb(corpus_meta) ->> 'dense_query_instruction' FROM corpus_meta")
    instr = cur.fetchone()[0]
    url = os.environ.get("EMBED_ENDPOINT", "http://127.0.0.1:8080") + "/embed"

    def embed(text: str) -> np.ndarray:
        req = urllib.request.Request(url, data=json.dumps({"inputs": text, "truncate": False})
                                     .encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            return np.array(json.load(r)[0], dtype=np.float32)

    def dense(v: np.ndarray) -> list[str]:
        cur.execute("SELECT id FROM clusters ORDER BY centroid <=> %s::halfvec LIMIT 5", (lit(v),))
        cl = [r[0] for r in cur.fetchall()]
        cur.execute("SELECT id FROM chunk_vectors WHERE cluster_id = ANY(%s)"
                    " ORDER BY dense <=> %s::halfvec LIMIT 20", (cl, lit(v)))
        return [r[0] for r in cur.fetchall()]

    variants = ["instr", "plain"] + [f"prf@{a}/{n}" for a in (0.3, 0.6, 1.0) for n in (3, 5)]
    arms: dict[str, list[list[str]]] = {v: [] for v in variants}
    for j, c in enumerate(cases):
        q = spec[c["id"]]["query"]
        qi = embed(f"Instruct: {instr}\nQuery: {q}")
        arms["instr"].append(dense(qi))
        arms["plain"].append(dense(embed(q)))
        top = c["ids"][:5]
        cur.execute("SELECT id, dense::text FROM chunks WHERE id = ANY(%s)", (top,))
        dv = {i: vec(t) for i, t in cur.fetchall()}
        for a in (0.3, 0.6, 1.0):
            for n in (3, 5):
                got = [dv[i] for i in top[:n] if i in dv]
                v = qi + a * np.mean(got, axis=0) if got else qi
                arms[f"prf@{a}/{n}"].append(dense(v / np.linalg.norm(v)))
        if j % 100 == 0:
            print(f"  {j}/{len(cases)}", file=sys.stderr, flush=True)

    for name in variants:
        found = new = 0
        for c, arm in zip(cases, arms[name]):
            t = set(c["targets"])
            found += any(i in t for i in arm)
            new += any(i in t for i in arm) and not any(i in t for i in c["ids"])
        line = [f"  {name:10} arm R@20 {found / len(cases):.1%}  new to the pool {new:3}"]
        for w in (0.5, 1.0):
            for split in ("train", "holdout"):
                idx = [k for k, c in enumerate(cases) if (split == "holdout") == c["holdout"]]
                h5 = win = loss = 0
                mrr = 0.0
                for k in idx:
                    c, arm = cases[k], arms[name][k]
                    s: dict[str, float] = {}
                    for r, i in enumerate(c["ids"]):
                        s[i] = s.get(i, 0.0) + 1 / (K + r + 1)
                    for r, i in enumerate(arm):
                        s[i] = s.get(i, 0.0) + w / (K + r + 1)
                    new_order = sorted(s, key=lambda i: -s[i])
                    t = set(c["targets"])
                    a = any(i in t for i in new_order[:5])
                    b = any(i in t for i in c["ids"][:5])
                    h5 += a
                    win += a and not b
                    loss += b and not a
                    mrr += next((1 / (r + 1) for r, i in enumerate(new_order) if i in t), 0.0)
                n = len(idx)
                line.append(f"w{w} {split[:5]} R@5 {h5 / n:.1%} MRR {mrr / n:.3f} +{win}/-{loss}")
        print("  |  ".join(line), flush=True)


if __name__ == "__main__":
    main()
