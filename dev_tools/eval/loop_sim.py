"""Offline: an agent loop · round 2 rewrites the hypothetical answer after reading round 1.

    POOLS=best_pools.json REWRITES=rewrites_holdout.json REPLIES=rewrite_replies.json \
    HYDE_REPLIES=hyde_replies.json LOOP=loop_holdout.json OUT=loop_replies.json \
    VERA_HTTP=... python -I dev_tools/eval/loop_sim.py

Round 1 is the original question fused with one hypothetical answer (`hyde_sim.py`). The
agent reads its top 4 (regulation, subject, article) and writes LOOP: one passage per
question that mirrors the law round 1 found for the point asked, or redirects where round
1 looked wrong. Each is searched through the deployed engine, started like the one that
wrote POOLS, and fused by RRF, equal weights, `k` 60, as before. LOOP's README names the
cases written after their labels were seen; the report repeats every line without them.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from e2e import HttpServer  # noqa: E402

K = 60.0
SEEN = {"s133", "s147", "s164", "n105", "s332", "s281"}


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
    hyde = json.loads(Path(os.environ["HYDE_REPLIES"]).read_text("utf-8"))
    loop = {k: v for k, v in json.loads(Path(os.environ["LOOP"]).read_text("utf-8")).items()
            if not k.startswith("_")}
    out = Path(os.environ.get("OUT", "loop_replies.json"))
    got = json.loads(out.read_text("utf-8")) if out.exists() else {}
    srv = HttpServer(os.environ.get("VERA_HTTP", "http://127.0.0.1:8081"))
    for cid, text in loop.items():
        if cid not in got:
            got[cid] = [r["id"] for r in srv.search(text).get("results", [])]
    out.write_text(json.dumps(got, ensure_ascii=False), "utf-8")

    ids = [c for c in loop if c in cases and c in rewrites]
    o = {c: cases[c]["ids"] for c in ids}
    a = {c: hyde["query"][c] for c in ids}
    rw = {c: [replies[w] for w in rewrites[c]] for c in ids}
    variants: dict[str, dict[str, list[str]]] = {
        "original": o,
        "round 2 passage alone": got,
        "round 1: original + A": {c: rrf([o[c], a[c]]) for c in ids},
        "round 2: original + A + L": {c: rrf([o[c], a[c], got[c]]) for c in ids},
        "original + L": {c: rrf([o[c], got[c]]) for c in ids},
        "original + 3 rw + A": {c: rrf([o[c]] + rw[c] + [a[c]]) for c in ids},
        "original + 3 rw + A + L": {c: rrf([o[c]] + rw[c] + [a[c], got[c]]) for c in ids},
        "  + A as document": {c: rrf([o[c]] + rw[c] + [a[c], got[c], hyde["doc"][c]])
                              for c in ids},
    }

    def first(lst, t):
        return next((j + 1 for j, i in enumerate(lst) if i in t), None)

    for subset, label in ((ids, "all"), ([c for c in ids if c not in SEEN], "unseen labels")):
        n = len(subset)
        base = {c: first(o[c], set(cases[c]["targets"])) for c in subset}
        print(f"cases {n} ({label})")
        for name, lists in variants.items():
            rs = {c: first(lists[c], set(cases[c]["targets"])) for c in subset}
            at = "  ".join(f"@{k} {100 * sum(1 for r in rs.values() if r and r <= k) / n:.1f}"
                           for k in (5, 10, 25))
            mrr = sum(1 / r for r in rs.values() if r and r <= 25) / n
            wl = []
            for k in (10, 25):
                def inside(x):
                    return bool(x and x <= k)
                win = sum(1 for c in subset if inside(rs[c]) and not inside(base[c]))
                loss = sum(1 for c in subset if inside(base[c]) and not inside(rs[c]))
                wl.append(f"@{k} +{win} -{loss}")
            print(f"  {name:28} {at}  MRR {mrr:.3f}   {'   '.join(wl)}")


if __name__ == "__main__":
    main()
