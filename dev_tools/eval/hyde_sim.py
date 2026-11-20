"""Offline: a hypothetical answer (HyDE) · the agent drafts the article it expects to find.

    POOLS=best_pools.json REWRITES=rewrites_holdout.json REPLIES=rewrite_replies.json \
    HYDE=hyde_holdout.json OUT=hyde_replies.json DATABASE_URL=... EMBED_ENDPOINT=... \
    VERA_HTTP=... python -I dev_tools/eval/hyde_sim.py

HYDE maps a case id to one passage written in the corpus's own style, from the question
alone, never from its label. It is used two ways, both fixed before any reply was read:

* `as query`     sent through the deployed engine like a rewording (the corpus's query
                 instruction, every arm, the same ranking), started like the one that
                 wrote POOLS (`TOP_K` = `CANDIDATE_POOL` = 60).
* `as document`  embedded WITHOUT the instruction, as a chunk is, then a dense arm over the
                 5 clusters nearest it, top 60: document-to-document similarity.

HYDE2 (optional) maps a case id to two more passages from other angles (the rule at
another legal level; the definitions style), each sent through the engine as a query.

Each list is fused with the original's (and the three rewordings', from REPLIES) by RRF,
equal weights, `k` 60, the method `query_swarm_sim.py` fixed. Reports Recall@5/10/25 and
MRR over the cases HYDE covers, with wins and losses at 10 and 25 against the original.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request
from pathlib import Path

import psycopg

sys.path.insert(0, str(Path(__file__).parent))
from e2e import HttpServer  # noqa: E402

K = 60.0


def rrf(lists: list[list[str]]) -> list[str]:
    score: dict[str, float] = {}
    for lst in lists:
        for rank, i in enumerate(lst, 1):
            score[i] = score.get(i, 0.0) + 1.0 / (K + rank)
    return sorted(score, key=lambda i: -score[i])


def main() -> None:
    cases = {c["id"]: c for c in json.loads(Path(os.environ["POOLS"]).read_text("utf-8"))}
    rewrites = json.loads(Path(os.environ["REWRITES"]).read_text("utf-8"))
    replies = json.loads(Path(os.environ["REPLIES"]).read_text("utf-8"))
    hyde = {k: v for k, v in json.loads(Path(os.environ["HYDE"]).read_text("utf-8")).items()
            if not k.startswith("_")}
    out = Path(os.environ.get("OUT", "hyde_replies.json"))
    got = json.loads(out.read_text("utf-8")) if out.exists() else {"query": {}, "doc": {}}
    got.setdefault("more", {})
    more = ({k: v for k, v in json.loads(Path(os.environ["HYDE2"]).read_text("utf-8")).items()
             if not k.startswith("_")} if os.environ.get("HYDE2") else {})

    srv = HttpServer(os.environ.get("VERA_HTTP", "http://127.0.0.1:8081"))
    cur = psycopg.connect(os.environ["DATABASE_URL"]).cursor()
    url = os.environ.get("EMBED_ENDPOINT", "http://127.0.0.1:8080") + "/embed"

    def embed(text: str) -> str:
        req = urllib.request.Request(url, data=json.dumps({"inputs": text, "truncate": True})
                                     .encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            return "[" + ",".join(f"{x:.6f}" for x in json.load(r)[0]) + "]"

    for cid, text in hyde.items():
        if cid not in got["query"]:
            got["query"][cid] = [r["id"] for r in srv.search(text).get("results", [])]
        if cid not in got["doc"]:
            v = embed(text)
            cur.execute("SELECT id FROM clusters ORDER BY centroid <=> %s::halfvec LIMIT 5", (v,))
            cl = [r[0] for r in cur.fetchall()]
            cur.execute("SELECT id FROM chunk_vectors WHERE cluster_id = ANY(%s)"
                        " ORDER BY dense <=> %s::halfvec LIMIT 60", (cl, v))
            got["doc"][cid] = [r[0] for r in cur.fetchall()]
        if cid in more and cid not in got["more"]:
            got["more"][cid] = [[r["id"] for r in srv.search(t).get("results", [])]
                                for t in more[cid]]
    out.write_text(json.dumps(got, ensure_ascii=False), "utf-8")

    ids = [c for c in hyde if c in cases and c in rewrites]
    orig = {c: cases[c]["ids"] for c in ids}
    rw = {c: [replies[w] for w in rewrites[c]] for c in ids}
    variants: dict[str, dict[str, list[str]]] = {
        "original": orig,
        "hyde as query, alone": {c: got["query"][c] for c in ids},
        "hyde as document, alone": {c: got["doc"][c] for c in ids},
        "original + hyde query": {c: rrf([orig[c], got["query"][c]]) for c in ids},
        "original + hyde document": {c: rrf([orig[c], got["doc"][c]]) for c in ids},
        "original + 3 rewordings": {c: rrf([orig[c]] + rw[c]) for c in ids},
        "  + hyde query": {c: rrf([orig[c]] + rw[c] + [got["query"][c]]) for c in ids},
        "  + hyde document": {c: rrf([orig[c]] + rw[c] + [got["doc"][c]]) for c in ids},
        "  + both": {c: rrf([orig[c]] + rw[c] + [got["query"][c], got["doc"][c]])
                     for c in ids},
    }
    if more:
        h = {c: [got["query"][c]] + got["more"][c] for c in ids}
        variants.update({
            "hyde B alone": {c: h[c][1] for c in ids},
            "hyde C alone": {c: h[c][2] for c in ids},
            "original + hyde A,B": {c: rrf([orig[c]] + h[c][:2]) for c in ids},
            "original + hyde A,B,C": {c: rrf([orig[c]] + h[c]) for c in ids},
            "original + 3 rw + hyde A,B": {c: rrf([orig[c]] + rw[c] + h[c][:2]) for c in ids},
            "original + 3 rw + hyde A,B,C": {c: rrf([orig[c]] + rw[c] + h[c]) for c in ids},
            "  + hyde as document too": {c: rrf([orig[c]] + rw[c] + h[c] + [got["doc"][c]])
                                  for c in ids},
            "original + rw 1 + hyde A,B,C": {c: rrf([orig[c], rw[c][0]] + h[c]) for c in ids},
        })

    def first(lst, t):
        return next((j + 1 for j, i in enumerate(lst) if i in t), None)

    base = {c: first(orig[c], set(cases[c]["targets"])) for c in ids}
    n = len(ids)
    print(f"cases {n} (no exact_ref)")
    for name, lists in variants.items():
        rs = {c: first(lists[c], set(cases[c]["targets"])) for c in ids}
        at = "  ".join(f"@{k} {100 * sum(1 for r in rs.values() if r and r <= k) / n:.1f}"
                       for k in (5, 10, 25))
        mrr = sum(1 / r for r in rs.values() if r and r <= 25) / n
        wl = []
        for k in (10, 25):
            def inside(x):
                return bool(x and x <= k)
            win = sum(1 for c in ids if inside(rs[c]) and not inside(base[c]))
            loss = sum(1 for c in ids if inside(base[c]) and not inside(rs[c]))
            wl.append(f"@{k} +{win} -{loss}")
        print(f"  {name:26} {at}  MRR {mrr:.3f}   {'   '.join(wl)}")


if __name__ == "__main__":
    main()
