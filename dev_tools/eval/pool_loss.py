"""Where a labelled answer is lost before ranking -- the Stage 2 attribution.

`e2e.py` with TOP_K = CANDIDATE_POOL says HOW OFTEN the answer is missing from what
the engine ranked. This says WHERE, per case, by replaying each arm as the engine
builds it (`crates/store/src/search.rs`) at the engine's depths and deeper:

  routing     the answer is in the flat dense top 100, but in no probed cluster
  dense-depth in a probed cluster, below the arm's PER_ARM_K cut
  sparse-only the sparse arm has it past PER_ARM_K but within 100
  fusion      an arm returned it, the engine's reply did not (pool cap or floor)
  absent      no arm finds it within 100: the retrieval floor

It also prints routing recall (the answer's cluster among the probed ones) over every
case, the Stage 2 routing target.

    DATABASE_URL=... EMBED_ENDPOINT=... python pool_loss.py <cases.json from e2e.py>

! Reads the per-case file `e2e.py` writes with EVAL_CASES_OUT, from a run with
TOP_K = CANDIDATE_POOL, so "in the reply" means "in the scored pool".
! The flat dense scan reads the whole corpus, so it runs only for the misses.
"""

from __future__ import annotations

import collections
import json
import math
import os
import sys
import urllib.request
from pathlib import Path

import psycopg

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from e2e import resolve_targets  # noqa: E402  same labels, same twins rule

PG = os.environ["DATABASE_URL"]
EP = os.environ.get("EMBED_ENDPOINT", "http://localhost:8080").rstrip("/")
PROBED = int(os.environ.get("CLUSTERS_PROBED", "5"))
PER_ARM_K = int(os.environ.get("PER_ARM_K", "20"))
DEEP = 100
WIDTHS = (5, 8, 12, 20)


def query_prefix(cur) -> str:
    """The corpus's own query instruction, the rule `store::query_prefix` applies.
    ! Without it the query lands in a different space from the one the engine
    searches, and every dense number here describes a query the engine never runs."""
    cur.execute("SELECT to_jsonb(m) ->> 'dense_instruction_style',"
                " to_jsonb(m) ->> 'dense_query_instruction', to_jsonb(m) ->> 'dense_instruction'"
                " FROM corpus_meta m ORDER BY created_at DESC LIMIT 1")
    style, q, legacy = cur.fetchone()
    if style is None:
        return legacy or ""
    if style == "prefix":
        return f"Instruct: {q}\nQuery: " if q else ""
    sys.exit(f"instruction style {style!r} is not one this script reproduces")


def embed(text: str) -> list[float]:
    req = urllib.request.Request(
        f"{EP}/embed",
        data=json.dumps({"inputs": text, "truncate": False}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.load(r)[0]


def first(ids: list[str], targets: set[str]) -> int | None:
    return next((i for i, x in enumerate(ids, 1) if x in targets), None)


def main() -> None:
    per_case = {c["id"]: c for c in json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))}
    spec = json.loads((ROOT / "dev_tools/eval/queries.json").read_text(encoding="utf-8"))
    cases = [q for q in spec["queries"] if q["id"] in per_case and q["type"] != "exact_ref"]

    with psycopg.connect(PG) as pg, pg.cursor() as cur:
        prefix = query_prefix(cur)
        cur.execute("SELECT id, centroid::text FROM clusters ORDER BY id")
        cl = []
        for cid, c in cur.fetchall():
            v = [float(x) for x in c.strip("[]").split(",")]
            nrm = math.sqrt(sum(a * a for a in v)) or 1.0
            cl.append((cid, [a / nrm for a in v]))

        routed = collections.Counter()
        buckets = collections.defaultdict(list)
        n = 0
        for q in cases:
            tgt = resolve_targets(cur, q)
            if not tgt:
                continue
            n += 1
            cur.execute("SELECT DISTINCT cluster_id FROM chunks WHERE id = ANY(%s)", (sorted(tgt),))
            want = {r[0] for r in cur.fetchall()}
            qv = embed(prefix + q["query"])
            nrm = math.sqrt(sum(a * a for a in qv)) or 1.0
            order = [cid for _, cid in sorted(
                ((-sum(a * b for a, b in zip(qv, ct)) / nrm, cid) for cid, ct in cl))]
            pos = next((i for i, cid in enumerate(order, 1) if cid in want), None)
            for w in WIDTHS:
                routed[w] += 1 if pos and pos <= w else 0

            if per_case[q["id"]]["pool"]:
                continue
            lit = "[" + ",".join(f"{x:.6g}" for x in qv) + "]"
            # The engine's dense arm: the top PER_ARM_K across the probed clusters.
            cur.execute(
                "SELECT id FROM chunks WHERE indexable AND dense IS NOT NULL AND cluster_id = ANY(%s)"
                " ORDER BY dense <=> %s::text::halfvec LIMIT %s", (order[:PROBED], lit, DEEP))
            d_probed = first([r[0] for r in cur.fetchall()], tgt)
            d_engine = bool(d_probed and d_probed <= PER_ARM_K)
            cur.execute(
                "SELECT s.id FROM (SELECT id, paradedb.score(id) sc FROM chunks"
                " WHERE id @@@ paradedb.match('body', %s) ORDER BY paradedb.score(id) DESC"
                " LIMIT %s) s JOIN chunks c ON c.id = s.id WHERE c.indexable ORDER BY s.sc DESC",
                (q["query"], DEEP * 2))
            s_rank = first([r[0] for r in cur.fetchall()][:DEEP], tgt)
            cur.execute(
                "SELECT id FROM chunks WHERE indexable AND dense IS NOT NULL"
                " ORDER BY dense <=> %s::text::halfvec LIMIT %s", (lit, DEEP))
            d_flat = first([r[0] for r in cur.fetchall()], tgt)

            if d_engine or (s_rank and s_rank <= PER_ARM_K):
                b = "fusion"
            elif d_probed:
                b = "dense-depth"
            elif d_flat:
                b = "routing"
            elif s_rank:
                b = "sparse-only"
            else:
                b = "absent"
            buckets[b].append(f"{q['id']}({q['type']}, cluster#{pos}, dense {d_flat}, sparse {s_rank})")
            print(f"  {q['id']:>6} {b:<12} cluster rank {pos} · flat dense {d_flat} "
                  f"· probed dense {d_probed} · sparse {s_rank}", flush=True)

    print(f"\n{n} cases (exact_ref excluded: it bypasses routing)")
    print("routing recall: " + " · ".join(f"{w} probed {100 * routed[w] / n:.1f}%" for w in WIDTHS))
    miss = sum(len(v) for v in buckets.values())
    print(f"not in the scored pool: {miss}")
    for b in ("routing", "dense-depth", "sparse-only", "fusion", "absent"):
        print(f"  {b:<12} {len(buckets[b])}")


if __name__ == "__main__":
    main()
