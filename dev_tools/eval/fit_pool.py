"""Fit the factor weights against the pool the ENGINE builds, ✗ one this file rebuilds.

    # 1. harvest: the engine at TOP_K=60 returns its whole fused pool (factors off) and
    #    its own ranking at the weights now shipped
    VERA_HTTP=http://127.0.0.1:8081 DATABASE_URL=... python fit_pool.py harvest pools.json
    # 2. prove the mirror, then fit on `train`, then score the winner once on `holdout`
    DATABASE_URL=... python fit_pool.py fit pools.json

Why a new harness. `fit_factors.py` rebuilds the pool from the text arm and the
sparsevec BM25 with no dense arm, which is not what the engine ranks any more: dense
carries weight 2.0, BM25 is pg_search, the text arm is off (`HARDWARE.md` §9).
`EVAL.md` §6 lists four silent divergences that harness had to have found and fixed;
asking the engine for its pool removes the class.

! The reranker mirrors `factors.rs` with the corpus's own vocabulary
(`corpus_meta.scoring_vocabulary`, mapped as `mcp::vocabulary` maps it). It is only
worth anything while it reproduces the engine's order at the shipped weights, so
`fit` checks that FIRST, per query, and prints how many agree. It also mirrors the
currency rule (`engine::rank_below_revokers`): a fully revoked chunk never sits above a
chunk of a document that revoked it. That rule is what the first version missed, and
it accounted for most of the queries the mirror then disagreed on.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import re
import sys
import urllib.request
from pathlib import Path

import psycopg

QUERIES = Path(__file__).with_name("queries.json")
sys.path.insert(0, str(Path(__file__).parent))
from run import resolve_targets  # noqa: E402

SHIPPED = {"relevance_floor": 0.3, "authority": 0.5, "structural": 0.25,
           "temporal": 0.0, "completeness": 0.0, "topical": 0.25}
OFF = {k: 0.0 for k in SHIPPED}


def call(base: str, query: str, weights: dict) -> dict:
    body = {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": "search_knowledge",
                       "arguments": {"query": query, "top_k": 60, "candidate_pool": 60,
                                     "factor_weights": weights}}}
    req = urllib.request.Request(base.rstrip("/") + "/mcp", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=300))
    return json.loads(r["result"]["content"][0]["text"])


def harvest(out: str) -> None:
    base = os.environ["VERA_HTTP"]
    spec = json.loads(QUERIES.read_text(encoding="utf-8"))
    cur = psycopg.connect(os.environ["DATABASE_URL"]).cursor()
    cases = []
    for q in spec["queries"]:
        if q["type"] == "out_of_domain":
            continue
        targets = resolve_targets(cur, q)
        if not targets:
            continue
        pool = call(base, q["query"], OFF)
        shipped = call(base, q["query"], SHIPPED)
        cases.append({
            "id": q["id"], "type": q["type"], "query": q["query"],
            "holdout": bool(q.get("holdout")), "near_duplicates": q.get("near_duplicates", 0),
            "targets": sorted(targets),
            "exact": bool(pool.get("exact_matches")),
            "pool": [(r["id"], r["score"]) for r in pool.get("results", [])],
            "shipped": [r["id"] for r in shipped.get("results", [])],
        })
        print(f"{q['id']} pool {len(cases[-1]['pool'])}", file=sys.stderr, flush=True)
    ids = sorted({i for c in cases for i, _ in c["pool"]})
    meta = load_meta(cur, ids)
    cur.execute("SELECT scoring_vocabulary FROM corpus_meta ORDER BY created_at DESC LIMIT 1")
    vocab = cur.fetchone()[0]
    Path(out).write_text(json.dumps({"cases": cases, "meta": meta,
                                     "vocabulary": vocab}, ensure_ascii=False), encoding="utf-8")
    print(f"{len(cases)} cases, {len(meta)} distinct chunks -> {out}", file=sys.stderr)


def load_meta(cur, ids: list[str]) -> dict:
    cur.execute(
        "SELECT c.id, c.regulation_type, c.article, c.chapter, c.about, c.body, c.year,"
        " c.doc_id, s.status, s.revoked_by FROM chunks c LEFT JOIN doc_status s"
        " ON s.corpus_id = c.corpus_id AND s.doc_id = c.doc_id WHERE c.id = ANY(%s)", (ids,))
    return {r[0]: {"regulation_type": r[1], "article": r[2], "chapter": r[3], "about": r[4],
                   "body": r[5], "year": r[6], "doc_id": r[7] or "",
                   "revoked": r[8] == "revoked", "revoked_by": list(r[9] or [])}
            for r in cur.fetchall()}


def below_revokers(order: list[str], meta: dict) -> list[str]:
    """engine::rank_below_revokers, with the pipeline's predicate (fully `revoked` only)."""
    n = len(order)

    def by(i: int, doc: str) -> bool:
        m = meta[order[i]]
        return m["revoked"] and doc in m["revoked_by"]

    docs = [meta[i]["doc_id"] for i in order]
    waiting = [[j for j in range(n) if docs[i] and j != i and docs[j] and docs[j] != docs[i]
                and by(i, docs[j]) and not by(j, docs[i])] for i in range(n)]
    done = [False] * n
    out: list[int] = []
    while len(out) < n:
        nxt = next((i for i in range(n) if not done[i] and all(done[j] for j in waiting[i])),
                   next(i for i in range(n) if not done[i]))
        done[nxt] = True
        out.append(nxt)
    return [order[i] for i in out]


# ---- factors.rs, over the corpus vocabulary as mcp::vocabulary builds it -----------------

class Vocab:
    def __init__(self, v: dict) -> None:
        if isinstance(v, str):
            v = json.loads(v)
        self.authority = {k.strip().upper(): t for k, t in v["authority"].items()}
        self.max_tier = float(v["authority_scale"])
        self.rules = ([(re.compile(rf"(?i)^\s*(?:{p})"), v["annex_score"]) for p in v["annex"]]
                      + [(re.compile(rf"(?i)^\s*(?:{p})"), v["operative_score"])
                         for p in v["operative"]])
        self.default = v["labelled_score"]
        self.stop = {s.strip().lower() for s in v["stopwords"]}
        self.min_chars = v["min_term_chars"]
        self._terms: dict[str, list[str]] = {}

    def tier(self, label: str | None) -> float:
        if not label:
            return 0.0
        want = label.strip().upper()
        hits = [(len(k), t) for k, t in self.authority.items() if want == k or want.startswith(k)]
        return max(hits)[1] / self.max_tier if hits else 0.0

    def structural(self, article: str | None, chapter: str | None) -> float:
        a, c = article or "", chapter or ""
        for pat, score in self.rules:
            if pat.search(a) or pat.search(c):
                return score
        return self.default

    def terms(self, text: str | None) -> list[str]:
        # Cached by text: the grid re-reads the same 7K bodies hundreds of times.
        key = text or ""
        hit = self._terms.get(key)
        if hit is None:
            hit = self._terms[key] = self._tokenize(key)
        return hit

    def _tokenize(self, text: str | None) -> list[str]:
        out: list[str] = []
        for w in re.split(r"[^\w]+", (text or "").lower()):
            if len(w) > self.min_chars and w not in self.stop and w not in out:
                out.append(w)
        return out


def rescore(case: dict, meta: dict, v: Vocab, w: dict) -> list[str]:
    pool = [(i, s) for i, s in case["pool"] if i in meta]
    if all(x == 0 for x in w.values()):
        return [i for i, _ in pool]
    qt = v.terms(case["query"])
    floor = 0.0 if case["exact"] else w["relevance_floor"]
    if floor > 0 and qt:
        kept = [(i, s) for i, s in pool
                if sum(t in set(v.terms(meta[i]["body"])) for t in qt) / len(qt) >= floor]
        pool = kept or pool[:1]
    years = [meta[i]["year"] for i, _ in pool if meta[i]["year"] is not None]
    lo, hi = (min(years), max(years)) if years else (0, 0)

    def prior(i: str) -> float:
        m = meta[i]
        temporal = ((m["year"] - lo) / (hi - lo) if (m["year"] is not None and hi > lo) else 0.5)
        about = set(v.terms(m["about"]))
        topical = sum(t in about for t in qt) / len(qt) if qt else 0.0
        completeness = 1.0 - math.exp(-len(m["body"] or "") / 400.0)
        return (w["authority"] * v.tier(m["regulation_type"])
                + w["structural"] * v.structural(m["article"], m["chapter"])
                + w["completeness"] * completeness + w["temporal"] * temporal
                + w["topical"] * topical)

    # ! Stable sort, ✗ a key that breaks ties differently: factors.rs sorts the pool
    # with `sort_by` over the incoming (fused) order.
    order = [i for i, _ in sorted(pool, key=lambda p: -(p[1] * (1.0 + prior(p[0]))))]
    # ! Currency only reorders when the floor applies (no named regulation) and ranking
    # is on, exactly the condition in `pipeline.rs::search_with`.
    return below_revokers(order, meta) if not case["exact"] else order


def score(cases: list[dict], meta: dict, v: Vocab, w: dict) -> tuple[float, float, float]:
    h5 = h10 = rr = 0.0
    for c in cases:
        order = rescore(c, meta, v, w)[:10]
        t = set(c["targets"])
        h5 += any(i in t for i in order[:5])
        h10 += any(i in t for i in order)
        rr += next((1 / (k + 1) for k, i in enumerate(order) if i in t), 0.0)
    n = len(cases)
    return h5 / n, h10 / n, rr / n


def fit(path: str) -> None:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    cases, meta, v = data["cases"], data["meta"], Vocab(data["vocabulary"])
    if "doc_id" not in next(iter(meta.values())):
        meta = load_meta(psycopg.connect(os.environ["DATABASE_URL"]).cursor(), sorted(meta))

    agree = sum(rescore(c, meta, v, SHIPPED)[:10] == c["shipped"][:10] for c in cases)
    print(f"mirror: python top-10 == engine top-10 at the shipped weights on "
          f"{agree}/{len(cases)} queries")
    if agree < 0.9 * len(cases):
        raise SystemExit("the mirror does not reproduce the engine; fix it before fitting")

    train = [c for c in cases if not c["holdout"]]
    hold = [c for c in cases if c["holdout"]]
    print(f"train {len(train)} · holdout {len(hold)}")
    for name, w in (("off", OFF), ("shipped", SHIPPED)):
        a, b = score(train, meta, v, w), score(hold, meta, v, w)
        print(f"  {name:8} train R@5 {a[0]:.3f} R@10 {a[1]:.3f} MRR {a[2]:.3f} · "
              f"holdout R@5 {b[0]:.3f} R@10 {b[1]:.3f} MRR {b[2]:.3f}")

    grid = []
    steps = [0.0, 0.25, 0.5, 0.75, 1.0]
    for fl, au, st, to, co in itertools.product([0.0, 0.1, 0.2, 0.3, 0.4], steps, steps,
                                                [0.0, 0.25, 0.5], [0.0, 0.25, 0.5]):
        # ! 1 + Σw ≤ 2.00, the startup guard of contract::algorithms (SCORING.md §3).
        if au + st + to + co > 1.0 + 1e-9:
            continue
        w = {"relevance_floor": fl, "authority": au, "structural": st,
             "temporal": 0.0, "completeness": co, "topical": to}
        grid.append((score(train, meta, v, w), w))
    # ! Ranked on MRR, ✗ Recall@5 alone: EVAL.md §3 and SCORING.md §3 record that a
    # Recall@5 maximum can be the worse ranker (positions 6-10 found nothing).
    grid.sort(key=lambda g: (-g[0][2], -g[0][1], -g[0][0]))
    print(f"grid {len(grid)} configurations, top 8 on train MRR:")
    for (r5, r10, mrr), w in grid[:8]:
        print(f"  R@5 {r5:.3f} R@10 {r10:.3f} MRR {mrr:.3f}  {w}")
    best = grid[0][1]
    b = score(hold, meta, v, best)
    print(f"best on holdout: R@5 {b[0]:.3f} R@10 {b[1]:.3f} MRR {b[2]:.3f}  {best}")
    print(json.dumps({"best": best}))


if __name__ == "__main__":
    {"harvest": harvest, "fit": fit}[sys.argv[1]](sys.argv[2])
