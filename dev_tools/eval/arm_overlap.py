"""Is the sparse arm earning its keep? Overlap and marginal recall, per arm.

Two of the three arms are lexical. `sparse` is BM25 over a `sparsevec` column and
`text` is `tsvector` ordered by RUM -- different machinery, but both are matching
words, and `EVAL.md` reports them within 2.3 points of each other (38.6% against
40.9% Recall@5). If they find the same rows, one of them is costing a **global
scan** for nothing, and global scans are the wall this corpus hits at scale
(`HARDWARE.md` §6).

This asks two questions the per-arm table cannot:

  overlap        of what each arm returns, how much does the other already have?
                 High overlap means the second arm is confirming, ✗ contributing.
  marginal       what does FUSED recall lose if an arm is removed? This is the
  recall         number that decides, because an arm can overlap heavily and
                 still rescue the queries the others miss -- and those are the
                 only queries an arm has to justify itself on.

    DATABASE_URL=... EMBED_ENDPOINT=... BM25_VOCAB=... python arm_overlap.py

! Deleting an arm is a RETRIEVAL change, so it is judged at the width a caller is
handed (20-50), ✗ only at k=5 (`EVAL.md` §4b). An arm that looks redundant at 5
and rescues queries at 50 stays.

! Unweighted RRF, matching `run.py`. The engine's weighted fusion does better, so
treat every fused row here as a floor rather than as the shipped number.
"""

from __future__ import annotations

import itertools
import json
import math
import os
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "dev_tools" / "pre_embed"))

import psycopg  # noqa: E402
from sparse import Bm25Vectorizer  # noqa: E402

QUERIES = Path(__file__).parent / "queries.json"
PG = os.environ["DATABASE_URL"]
EP = os.environ.get("EMBED_ENDPOINT", "http://localhost:8080").rstrip("/")
if not os.environ.get("BM25_VOCAB"):
    sys.exit("set BM25_VOCAB to the vocabulary this corpus was built with")
VOCAB = Path(os.environ["BM25_VOCAB"])

KS = (5, 20, 50)
PER_ARM = 100
RRF_K = 60
PROBED = int(os.environ.get("CLUSTERS_PROBED", "5"))
ORQ = "array_to_string(tsvector_to_array(to_tsvector('indonesian', %s)), ' | ')::tsquery"
SKIP = {"out_of_domain", "exact_ref"}
ARMS = ("dense", "sparse", "text")


def embed(text: str) -> list[float]:
    req = urllib.request.Request(
        f"{EP}/embed",
        data=json.dumps({"inputs": text, "truncate": False}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.load(r)[0]


def fuse(*lists: list[str]) -> list[str]:
    f: dict[str, float] = defaultdict(float)
    for ids in lists:
        for i, cid in enumerate(ids, 1):
            f[cid] += 1.0 / (RRF_K + i)
    return [c for c, _ in sorted(f.items(), key=lambda kv: (-kv[1], kv[0]))]


def resolve_targets(cur, q) -> set[str]:
    ids = set(q.get("answer_chunks", []))
    for ref in q.get("answer_articles", []):
        cur.execute(
            "SELECT id FROM chunks WHERE regulation_type=%s AND regulation_number=%s"
            " AND year=%s AND about=%s AND article=%s",
            (ref["regulation_type"], ref["regulation_number"], ref["year"],
             ref["about"], ref["article"]),
        )
        ids |= {r[0] for r in cur.fetchall()}
    return ids


def main() -> None:
    cases = [c for c in json.loads(QUERIES.read_text(encoding="utf-8"))["queries"]
             if c["type"] not in SKIP]
    vz = Bm25Vectorizer.load(VOCAB)

    # Every non-empty subset of the arms, so "what does removing X cost" is read
    # off the table rather than argued.
    combos = [c for n in (1, 2, 3) for c in itertools.combinations(ARMS, n)]
    hits = {c: dict.fromkeys(KS, 0) for c in combos}
    # Jaccard of each unordered pair, averaged over queries.
    pair_j = {p: 0.0 for p in itertools.combinations(ARMS, 2)}
    # Queries where an arm supplies a target NO other arm has, at the widest k.
    unique_rescue = dict.fromkeys(ARMS, 0)
    scored = 0

    with psycopg.connect(PG) as pg, pg.cursor() as cur:
        cur.execute("SELECT id, centroid::text FROM clusters ORDER BY id")
        cl, cents = [], []
        for cid, c in cur.fetchall():
            v = [float(x) for x in c.strip("[]").split(",")]
            n = math.sqrt(sum(a * a for a in v)) or 1.0
            cl.append(cid)
            cents.append([a / n for a in v])

        for q in cases:
            tgt = resolve_targets(cur, q)
            if not tgt:
                continue
            scored += 1
            qv = embed(q["query"])
            lit = "[" + ",".join(f"{x:.6g}" for x in qv) + "]"
            nrm = math.sqrt(sum(a * a for a in qv)) or 1.0
            qn = [a / nrm for a in qv]
            order = [cid for _, cid in sorted(
                ((sum(a * b for a, b in zip(qn, ct)), cid)
                 for ct, cid in zip(cents, cl)), key=lambda t: -t[0])]

            cur.execute("SELECT id FROM chunks WHERE indexable AND dense IS NOT NULL"
                        " AND cluster_id = ANY(%s)"
                        " ORDER BY dense <=> %s::text::halfvec LIMIT %s",
                        (order[:PROBED], lit, PER_ARM))
            got = {"dense": [r[0] for r in cur.fetchall()]}

            cur.execute("SELECT id FROM chunks WHERE indexable AND sparse IS NOT NULL"
                        " ORDER BY sparse <#> %s::text::sparsevec LIMIT %s",
                        (vz.to_sparsevec(vz.query(q["query"])), PER_ARM))
            got["sparse"] = [r[0] for r in cur.fetchall()]

            cur.execute(f"SELECT id FROM chunks WHERE indexable AND tsv @@ {ORQ}"
                        f" ORDER BY ts_rank(tsv, {ORQ}) DESC LIMIT %s",
                        (q["query"], q["query"], PER_ARM))
            got["text"] = [r[0] for r in cur.fetchall()]

            for a, b in pair_j:
                sa, sb = set(got[a]), set(got[b])
                if sa or sb:
                    pair_j[(a, b)] += len(sa & sb) / len(sa | sb)

            widest = max(KS)
            for arm in ARMS:
                mine = set(got[arm][:widest]) & tgt
                others = set().union(
                    *(set(got[o][:widest]) for o in ARMS if o != arm)) & tgt
                if mine - others:
                    unique_rescue[arm] += 1

            for combo in combos:
                ranked = fuse(*(got[a] for a in combo))
                for k in KS:
                    hits[combo][k] += bool(set(ranked[:k]) & tgt)

    print(f"\n{scored} queries · probing {PROBED} clusters · per-arm depth {PER_ARM}\n")

    print("pairwise overlap (Jaccard of the top-100 each returns)")
    for (a, b), tot in pair_j.items():
        print(f"  {a:>6} vs {b:<6} {tot / scored:>6.1%}")

    print("\nrecall by combination")
    print(f"{'arms':<24}" + "".join(f"{'@' + str(k):>9}" for k in KS))
    for combo in combos:
        print(f"{' + '.join(combo):<24}"
              + "".join(f"{hits[combo][k] / scored:>8.1%}" for k in KS))

    print("\nwhat removing one arm costs (against all three)")
    full = hits[tuple(ARMS)]
    for arm in ARMS:
        rest = tuple(a for a in ARMS if a != arm)
        print(f"  drop {arm:<7}"
              + "  ".join(f"@{k} {(hits[rest][k] - full[k]) / scored:+.1%}"
                          for k in KS)
              + f"   · sole source of a target on {unique_rescue[arm]}/{scored}")

    print("\n! An arm is redundant only if dropping it costs ~nothing at the WIDE k"
          "\n  AND it is rarely the sole source of a target. Either one alone is not"
          "\n  evidence: heavy overlap with a few unique rescues is still an arm.")


if __name__ == "__main__":
    main()
