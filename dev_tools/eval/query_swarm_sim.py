"""Offline: a swarm of QUESTIONS · the agent rewords, the engine fuses the replies.

    POOLS=best_pools.json REWRITES=rewrites.json OUT=replies.json VERA_HTTP=... \
        python -I dev_tools/eval/query_swarm_sim.py

REWRITES maps a case id to the agent's rewordings of its question, written from the
question alone, never from its label. Each rewording is searched through the deployed
engine (started like the one that wrote POOLS, `TOP_K` = `CANDIDATE_POOL`), and the
replies are fused with the original's by reciprocal rank fusion, equal weights, `k` 60.
That method was fixed before any reply was read, so the cases are scored once.

The engine never calls a model: a rewording is the agent's job (`CLAUDE.md` §2). What
this measures is whether the engine should accept several wordings in one request.

Reports Recall@5/10/25 and MRR over the cases REWRITES covers, for the original alone,
each single rewording alone, and the fusion, with wins and losses at 10 and 25.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

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
    out = Path(os.environ.get("OUT", "replies.json"))
    replies = json.loads(out.read_text("utf-8")) if out.exists() else {}
    srv = HttpServer(os.environ.get("VERA_HTTP", "http://127.0.0.1:8081"))
    for cid, words in rewrites.items():
        for w in words:
            if w not in replies:
                replies[w] = [r["id"] for r in srv.search(w).get("results", [])]
    out.write_text(json.dumps(replies, ensure_ascii=False), "utf-8")

    ids = [cid for cid in rewrites if cid in cases]
    variants: dict[str, dict[str, list[str]]] = {
        "original": {c: cases[c]["ids"] for c in ids},
        "rewording 1 alone": {c: replies[rewrites[c][0]] for c in ids},
        "rewording 2 alone": {c: replies[rewrites[c][1]] for c in ids},
        "rewording 3 alone": {c: replies[rewrites[c][2]] for c in ids},
        "fused, original + 3": {c: rrf([cases[c]["ids"]] + [replies[w] for w in rewrites[c]])
                                for c in ids},
        "fused, 3 rewordings only": {c: rrf([replies[w] for w in rewrites[c]]) for c in ids},
    }

    def first(lst, t):
        return next((j + 1 for j, i in enumerate(lst) if i in t), None)

    base = {c: first(cases[c]["ids"], set(cases[c]["targets"])) for c in ids}
    n = len(ids)
    print(f"cases {n} (no exact_ref: those bypass routing and are 100% already)")
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
