"""Where the top-5 misses are, through the deployed engine.

    VERA_HTTP=... DATABASE_URL=... BM25_VOCAB=... python dev_tools/eval/miss_map.py

Against an engine started with `TOP_K` = `CANDIDATE_POOL`, so the reply is the whole scored
pool. Every answerable miss (exact_ref excluded) falls in exactly one bucket:

* `rank 6-10` / `rank 11+`   the answer was retrieved and ranked too low
* `not in pool, right law`   the labelled regulation is in the pool, the article is not
* `not in pool, wrong law`   neither the answer nor its regulation was retrieved

with the type mix and the share whose question names a region, per bucket.
"""

from __future__ import annotations

import collections
import json
import os
import sys
from pathlib import Path

import psycopg

sys.path.insert(0, str(Path(__file__).parent))
from e2e import HttpServer, regulation_hits  # noqa: E402
from run import resolve_targets  # noqa: E402


def main() -> None:
    cur = psycopg.connect(os.environ["DATABASE_URL"]).cursor()
    srv = HttpServer(os.environ.get("VERA_HTTP", "http://127.0.0.1:8081"))
    spec = json.loads((Path(__file__).parent / "queries.json").read_text("utf-8"))
    buckets: dict[str, list[dict]] = collections.defaultdict(list)
    n = 0
    for q in spec["queries"]:
        if q["type"] in ("out_of_domain", "exact_ref"):
            continue
        targets = resolve_targets(cur, q)
        if not targets:
            continue
        ids = [r["id"] for r in srv.search(q["query"]).get("results", [])]
        if q["type"] == "underspecified" and ids:
            cur.execute("SELECT r.id FROM chunks r, chunks t WHERE r.id = ANY(%s) AND t.id = ANY(%s)"
                        " AND r.dense <=> t.dense < 0.03", (ids, list(targets)))
            targets = targets | {r[0] for r in cur.fetchall()}
        n += 1
        rank = next((k for k, i in enumerate(ids) if i in targets), None)
        if rank is not None and rank < 5:
            continue
        regs = regulation_hits(cur, q, ids)
        cur.execute("SELECT enacting_body FROM chunks WHERE id = ANY(%s)", (list(targets),))
        bodies = {r[0] for r in cur.fetchall() if r[0] and len(r[0]) > 3}
        named = any(b.lower() in q["query"].lower() for b in bodies)
        key = ("rank 6-10" if rank is not None and rank < 10 else "rank 11+" if rank is not None
               else "not in pool, right law" if any(regs) else "not in pool, wrong law")
        buckets[key].append({"id": q["id"], "type": q["type"], "holdout": bool(q.get("holdout")),
                             "region": named, "reg5": any(regs[:5])})
    Path(os.environ.get("OUT", "miss_map.json")).write_text(json.dumps(buckets), "utf-8")
    misses = sum(len(v) for v in buckets.values())
    print(f"answerable {n} · top-5 misses {misses}")
    for k in ("rank 6-10", "rank 11+", "not in pool, right law", "not in pool, wrong law"):
        v = buckets.get(k, [])
        types = collections.Counter(x["type"] for x in v).most_common(5)
        print(f"  {k:24} {len(v):3}  holdout {sum(x['holdout'] for x in v):2}  "
              f"region-named {sum(x['region'] for x in v):2}  right law in top 5 "
              f"{sum(x['reg5'] for x in v):2}  · {dict(types)}")


if __name__ == "__main__":
    main()
