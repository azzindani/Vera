"""Recall at the cut-off a caller is handed: 5 through the whole pool.

    POOLS=pools.json python -I dev_tools/eval/depth_recall.py

Reads the pools `article_sim.py` writes (engine started with `TOP_K` = `CANDIDATE_POOL`,
so each reply is the whole scored pool in the engine's own final order). Reports
Recall@k for every k up to the pool, MRR over the pool, and the answers the pool never
holds, by type: no cut-off can rank those.

! The pools leave out `exact_ref`. Those cases are counted as hits at rank 1, the rate
`e2e.py` measures for them (63/63). Re-check that before reusing an old pools file.
! A pool can be shorter than 60 (fewer candidates cleared the floor), so Recall@k past
a short pool's length is a floor, ✗ an overcount.
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

CUTS = (5, 10, 15, 20, 25, 30, 40, 50, 60)


def first_rank(case: dict) -> int | None:
    targets = set(case["targets"])
    return next((i + 1 for i, x in enumerate(case["ids"]) if x in targets), None)


def report(label: str, cases: list[dict], exact: int) -> None:
    n = len(cases) + exact
    ranks = [first_rank(c) for c in cases]
    mrr = (sum(1 / r for r in ranks if r) + exact) / n
    cuts = "  ".join(
        f"@{k} {100 * (sum(1 for r in ranks if r and r <= k) + exact) / n:.1f}%" for k in CUTS
    )
    print(f"[{label}] n={n}  MRR over the pool {mrr:.3f}")
    print(f"  {cuts}")
    missing = [c for c in cases if first_rank(c) is None]
    by_type = Counter(c["type"] for c in missing).most_common()
    print(f"  never in the pool: {len(missing)}  ({' · '.join(f'{t} {k}' for t, k in by_type)})")


def main() -> None:
    pools = json.loads(Path(os.environ.get("POOLS", "article_pools.json")).read_text("utf-8"))
    spec = json.loads((Path(__file__).parent / "queries.json").read_text("utf-8"))
    exact = [q for q in spec["queries"] if q["type"] == "exact_ref"]
    depth = [len(c["ids"]) for c in pools]
    print(f"pool depth {min(depth)}-{max(depth)}", file=sys.stderr)
    report("all", pools, len(exact))
    report("holdout", [c for c in pools if c["holdout"]],
           sum(1 for q in exact if q.get("holdout")))


if __name__ == "__main__":
    main()
