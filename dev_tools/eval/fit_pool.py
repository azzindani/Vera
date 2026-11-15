"""Fit the factor weights against the pool the ENGINE builds, ✗ one this file rebuilds.

    # 1. harvest: the engine at TOP_K=60 returns its whole fused pool (factors off) and
    #    its own ranking at the weights now shipped
    VERA_HTTP=http://127.0.0.1:8081 DATABASE_URL=... python fit_pool.py harvest pools.json
    # 2. prove the mirror, then fit on `train`, then score the winner once on `holdout`
    DATABASE_URL=... python fit_pool.py fit pools.json
    # 3. fusion and factors together, from the per-arm scores the reply carries
    DATABASE_URL=... python fit_pool.py fuse pools.json

FIT_DEPTH sets the pool harvested (default 60) and must equal the engine's CANDIDATE_POOL;
`fuse` needs FIT_DEPTH ≥ 2 × PER_ARM_K so both arms are in the pool whole.

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


# FIT_DEPTH: the pool to fit over. It must match the engine's CANDIDATE_POOL (and TOP_K
# at least that), or the fit ranks a pool the engine never builds.
DEPTH = int(os.environ.get("FIT_DEPTH", "60"))


def call(base: str, query: str, weights: dict) -> dict:
    body = {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": "search_knowledge",
                       "arguments": {"query": query, "top_k": DEPTH, "candidate_pool": DEPTH,
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
        ids = [r["id"] for r in pool.get("results", [])]
        # ! `underspecified` as e2e.py scores it: a pool chunk within cosine 0.03 of a
        # labelled one is the same clause in another region's regulation, and counts.
        if q["type"] == "underspecified" and ids:
            cur.execute("SELECT r.id FROM chunks r, chunks t WHERE r.id = ANY(%s)"
                        " AND t.id = ANY(%s) AND r.dense <=> t.dense < 0.03",
                        (ids, list(targets)))
            targets = targets | {r[0] for r in cur.fetchall()}
        cases.append({
            "id": q["id"], "type": q["type"], "query": q["query"],
            "holdout": bool(q.get("holdout")), "near_duplicates": q.get("near_duplicates", 0),
            "targets": sorted(targets),
            "exact": bool(pool.get("exact_matches")),
            "pool": [(r["id"], r["score"]) for r in pool.get("results", [])],
            "arms": [r.get("scores", {}) for r in pool.get("results", [])],
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
        " c.doc_id, s.status, s.revoked_by, c.enacting_body FROM chunks c LEFT JOIN doc_status s"
        " ON s.corpus_id = c.corpus_id AND s.doc_id = c.doc_id WHERE c.id = ANY(%s)", (ids,))
    return {r[0]: {"regulation_type": r[1], "article": r[2], "chapter": r[3], "about": r[4],
                   "body": r[5], "year": r[6], "doc_id": r[7] or "",
                   "revoked": r[8] == "revoked", "revoked_by": list(r[9] or []),
                   "enacting_body": r[10]}
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
    below: list = []
    if floor > 0 and qt:
        # ! Demotes, ✗ removes (factors.rs::rescore_composed, since 2026-11-14).
        cov = {i: sum(t in set(v.terms(meta[i]["body"])) for t in qt) / len(qt) for i, _ in pool}
        below = [(i, s) for i, s in pool if cov[i] < floor]
        pool = [(i, s) for i, s in pool if cov[i] >= floor]
    years = [meta[i]["year"] for i, _ in pool if meta[i]["year"] is not None]
    lo, hi = (min(years), max(years)) if years else (0, 0)

    def prior(i: str) -> float:
        m = meta[i]
        temporal = ((m["year"] - lo) / (hi - lo) if (m["year"] is not None and hi > lo) else 0.5)
        about = set(v.terms(m["about"]))
        topical = sum(t in about for t in qt) / len(qt) if qt else 0.0
        completeness = 1.0 - math.exp(-len(m["body"] or "") / 400.0)
        region = named_by_query(m.get("enacting_body"), case["query"])
        return (w.get("region", 0.0) * region
                + w["authority"] * v.tier(m["regulation_type"])
                + w["structural"] * v.structural(m["article"], m["chapter"])
                + w["completeness"] * completeness + w["temporal"] * temporal
                + w["topical"] * topical)

    # ! Stable sort, ✗ a key that breaks ties differently: factors.rs sorts the pool
    # with `sort_by` over the incoming (fused) order.
    order = [i for i, _ in sorted(pool, key=lambda p: -(p[1] * (1.0 + prior(p[0]))))]
    order += [i for i, _ in sorted(below, key=lambda p: -(p[1] * (1.0 + prior(p[0]))))]
    # ! Currency only reorders when the floor applies (no named regulation) and ranking
    # is on, exactly the condition in `pipeline.rs::search_with`.
    return below_revokers(order, meta) if not case["exact"] else order


def phrase(text: str | None) -> str:
    return " " + " ".join(w for w in re.split(r"[^\w]+", (text or "").lower()) if w) + " "


def named_by_query(value: str | None, query: str) -> float:
    """1.0 when the query contains the value as a whole phrase ("kulon progo"), else 0.
    ! A phrase, ✗ a share of terms: a share lifts every DAERAH ISTIMEWA YOGYAKARTA chunk by
    a third for any "peraturan daerah" question, and a term subset lets OKU TIMUR match
    "sebelah timur" once "oku" falls under the minimum term length."""
    v = phrase(value)
    return 1.0 if v.strip() and v in phrase(query) else 0.0


FACTORS = ("authority", "structural", "completeness", "topical", "region")


def features(case: dict, meta: dict, v: Vocab) -> dict:
    """Every factor value of every pool member, once · the grid then only multiplies.
    ! `temporal` is left out: it is pool-relative after the floor and the grid holds it at 0."""
    pool = [(i, s) for i, s in case["pool"] if i in meta]
    qt = v.terms(case["query"])
    rows = []
    for i, s in pool:
        m = meta[i]
        body = set(v.terms(m["body"]))
        rows.append((i, s, sum(t in body for t in qt) / len(qt) if qt else 1.0, (
            v.tier(m["regulation_type"]), v.structural(m["article"], m["chapter"]),
            1.0 - math.exp(-len(m["body"] or "") / 400.0),
            sum(t in set(v.terms(m["about"])) for t in qt) / len(qt) if qt else 0.0,
            named_by_query(m.get("enacting_body"), case["query"]))))
    docs = {meta[i]["doc_id"] for i, _ in pool}
    # Currency can only reorder a pool holding a revoked chunk whose revoker is also there.
    currency = not case["exact"] and any(
        meta[i]["revoked"] and docs & set(meta[i]["revoked_by"]) for i, _ in pool)
    return {"rows": rows, "currency": currency, "has_terms": bool(qt)}


def rank_fast(case: dict, f: dict, meta: dict, w: dict) -> list[str]:
    rows = f["rows"]
    if all(x == 0 for x in w.values()):
        return [r[0] for r in rows]
    ws = [w.get(k, 0.0) for k in FACTORS]
    floor = 0.0 if case["exact"] else w["relevance_floor"]
    demote = floor > 0 and f["has_terms"]

    def key(r):
        return (demote and r[2] < floor, -(r[1] * (1.0 + sum(a * b for a, b in zip(ws, r[3])))))
    order = [r[0] for r in sorted(rows, key=key)]
    return below_revokers(order, meta) if f["currency"] else order


def score(cases: list[dict], meta: dict, v: Vocab, w: dict,
          feats: list[dict] | None = None) -> tuple[float, float, float]:
    h5 = h10 = rr = 0.0
    for k, c in enumerate(cases):
        order = (rank_fast(c, feats[k], meta, w) if feats else rescore(c, meta, v, w))[:10]
        t = set(c["targets"])
        h5 += any(i in t for i in order[:5])
        h10 += any(i in t for i in order)
        rr += next((1 / (k + 1) for k, i in enumerate(order) if i in t), 0.0)
    n = len(cases)
    return h5 / n, h10 / n, rr / n


def fit(path: str) -> None:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    cases, meta, v = data["cases"], data["meta"], Vocab(data["vocabulary"])
    if "enacting_body" not in next(iter(meta.values())):
        meta = load_meta(psycopg.connect(os.environ["DATABASE_URL"]).cursor(), sorted(meta))

    agree = sum(rescore(c, meta, v, SHIPPED)[:10] == c["shipped"][:10] for c in cases)
    print(f"mirror: python top-10 == engine top-10 at the shipped weights on "
          f"{agree}/{len(cases)} queries")
    if agree < 0.9 * len(cases):
        raise SystemExit("the mirror does not reproduce the engine; fix it before fitting")

    train = [c for c in cases if not c["holdout"]]
    hold = [c for c in cases if c["holdout"]]
    ft = [features(c, meta, v) for c in train]
    fh = [features(c, meta, v) for c in hold]
    fast = sum(rank_fast(c, f, meta, SHIPPED)[:10] == rescore(c, meta, v, SHIPPED)[:10]
               for c, f in zip(train, ft))
    print(f"train {len(train)} · holdout {len(hold)} · fast path == mirror on {fast}/{len(train)}")
    for name, w in (("off", OFF), ("shipped", SHIPPED)):
        a, b = score(train, meta, v, w, ft), score(hold, meta, v, w, fh)
        print(f"  {name:8} train R@5 {a[0]:.3f} R@10 {a[1]:.3f} MRR {a[2]:.3f} · "
              f"holdout R@5 {b[0]:.3f} R@10 {b[1]:.3f} MRR {b[2]:.3f}")

    grid = []
    steps = [0.0, 0.25, 0.5, 0.75, 1.0]
    for fl, au, st, to, co, re_ in itertools.product(
            [0.0, 0.1, 0.2, 0.3, 0.4], steps, steps, [0.0, 0.25, 0.5], [0.0, 0.25],
            [0.0, 0.25, 0.5]):
        # ! 1 + Σw ≤ 2.00, the startup guard of contract::algorithms (SCORING.md §3).
        if au + st + to + co + re_ > 1.0 + 1e-9:
            continue
        w = {"relevance_floor": fl, "authority": au, "structural": st,
             "temporal": 0.0, "completeness": co, "topical": to, "region": re_}
        grid.append((score(train, meta, v, w, ft), w))
    # ! Ranked on MRR, ✗ Recall@5 alone: EVAL.md §3 and SCORING.md §3 record that a
    # Recall@5 maximum can be the worse ranker (positions 6-10 found nothing).
    grid.sort(key=lambda g: (-g[0][2], -g[0][1], -g[0][0]))
    print(f"grid {len(grid)} configurations, top 8 on train MRR:")
    for (r5, r10, mrr), w in grid[:8]:
        print(f"  R@5 {r5:.3f} R@10 {r10:.3f} MRR {mrr:.3f}  {w}")
    best_plain = next(w for _, w in grid if w["region"] == 0)
    for name, w in (("best", grid[0][1]), ("best without region", best_plain)):
        a, b = score(train, meta, v, w, ft), score(hold, meta, v, w, fh)
        print(f"{name}: train R@5 {a[0]:.3f} R@10 {a[1]:.3f} MRR {a[2]:.3f} · "
              f"holdout R@5 {b[0]:.3f} R@10 {b[1]:.3f} MRR {b[2]:.3f}  {w}")
    print(json.dumps({"best": grid[0][1], "best_without_region": best_plain}))


# ---- fusion + factors together: `fit_pool.py fuse pools.json` --------------------------

def arm_ranks(case: dict) -> list[tuple[int | None, int | None]]:
    """Per pool member, its rank in the dense and in the sparse arm (1-based, None if absent).
    ! Exact only while the pool holds both arms whole (`FIT_DEPTH` ≥ 2 × `PER_ARM_K`).
    ! The reply back-fills BOTH scores on every result, so a score does not say which arm
    found it: the arm is its top `PER_ARM_K` by that score. A sparse-only chunk from an
    unprobed cluster can outscore the dense arm's own members on cosine, so `fuse` prints
    how often this reproduces the engine's order before it fits anything."""
    arms = case["arms"]
    per_arm = int(os.environ.get("PER_ARM_K", "60"))
    out: list[list[int | None]] = [[None, None] for _ in arms]
    for j, key in enumerate(("dense", "bm25")):
        have = sorted((-a[key], i) for i, a in enumerate(arms) if key in a)
        for r, (_, i) in enumerate(have[:per_arm], 1):
            out[i][j] = r
    return [tuple(x) for x in out]


class Packed:
    """One case as arrays · the grid evaluates 10^4 configurations in minutes."""

    def __init__(self, case: dict, meta: dict, v: Vocab) -> None:
        import numpy as np
        f = features(case, meta, v)
        keep = {i for i, *_ in f["rows"]}
        ranks = dict(zip([i for i, _ in case["pool"]], arm_ranks(case)))
        rows = f["rows"]
        self.ids = [r[0] for r in rows]
        self.exact, self.currency = case["exact"], f["currency"]
        self.has_terms = f["has_terms"]
        inf = 10**9
        self.dr = np.array([ranks[i][0] or inf for i in self.ids], dtype=float)
        self.sr = np.array([ranks[i][1] or inf for i in self.ids], dtype=float)
        self.cov = np.array([r[2] for r in rows])
        # authority, structural, completeness, topical, region, coverage (= body match share)
        self.F = np.array([list(r[3]) + [r[2]] for r in rows]).reshape(len(rows), 6)
        self.idr = np.argsort(np.argsort(np.array(self.ids, dtype=object)))
        self.engine = [r[0] for r in rows]        # the engine's fused order, factors off
        self.targets = set(case["targets"])
        assert keep == set(self.ids)

    def order(self, wd: float, ws: float, k: float, w: list[float], floor: float,
              meta: dict) -> list[str]:
        import numpy as np
        if self.exact:     # the named path pins rank 1; fusion does not decide these
            return self.engine
        rrf = wd / (k + self.dr) + ws / (k + self.sr)
        final = rrf * (1.0 + self.F @ np.asarray(w))
        demoted = (self.cov < floor) if (floor > 0 and self.has_terms) else np.zeros(len(rrf))
        idx = np.lexsort((self.idr, -rrf, -final, demoted))
        out = [self.ids[i] for i in idx]
        # ! A floor alone is ranking too (`pipeline.rs`: the composition is not inert), so
        # currency applies with every prior at 0. Missing this lifted revoked answers into
        # the top 5 and overstated k 2 by 2.6 points against the engine.
        return below_revokers(out, meta) if (self.currency and (any(w) or floor > 0)) else out


def fuse(path: str) -> None:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    cases, meta, v = data["cases"], data["meta"], Vocab(data["vocabulary"])
    if "enacting_body" not in next(iter(meta.values())):
        meta = load_meta(psycopg.connect(os.environ["DATABASE_URL"]).cursor(), sorted(meta))
    train = [Packed(c, meta, v) for c in cases if not c["holdout"]]
    hold = [Packed(c, meta, v) for c in cases if c["holdout"]]
    zero = [0.0] * 6
    same = sum(p.order(2.0, 1.0, 60.0, zero, 0.0, meta)[:10] == p.engine[:10]
               for p in train + hold if not p.exact)
    print(f"fusion mirror: recomputed RRF top-10 == engine's on {same}/"
          f"{sum(not p.exact for p in train + hold)} queries")

    def sc(ps, cfg):
        wd, ws, k, w, fl = cfg
        h5 = h10 = rr = 0.0
        for p in ps:
            o = p.order(wd, ws, k, w, fl, meta)[:10]
            h5 += any(i in p.targets for i in o[:5])
            h10 += any(i in p.targets for i in o)
            rr += next((1 / (j + 1) for j, i in enumerate(o) if i in p.targets), 0.0)
        return h5 / len(ps), h10 / len(ps), rr / len(ps)

    shipped = [0.5, 0.25, 0.0, 0.25, 0.0, 0.0]
    for name, cfg in (("off", (2, 1, 60, zero, 0.0)), ("shipped", (2, 1, 60, shipped, 0.3))):
        a, b = sc(train, cfg), sc(hold, cfg)
        print(f"  {name:8} train R@5 {a[0]:.3f} R@10 {a[1]:.3f} MRR {a[2]:.3f} · "
              f"holdout R@5 {b[0]:.3f} R@10 {b[1]:.3f} MRR {b[2]:.3f}")
    # Stage A · fusion alone, factors off.
    fus = sorted(((sc(train, (wd, 1.0, k, zero, 0.0)), (wd, k))
                  for wd in (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
                  for k in (5.0, 10.0, 20.0, 40.0, 60.0, 100.0)), key=lambda g: -g[0][2])
    print("fusion, factors off, top 5 on train MRR:")
    for (a5, a10, m), (wd, k) in fus[:5]:
        print(f"  dense:sparse {wd}:1 k {k:<5} R@5 {a5:.3f} R@10 {a10:.3f} MRR {m:.3f}")
    # Stage B · factors over the three best fusions and the shipped one.
    grid = []
    st = (0.0, 0.25, 0.5)
    for wd, k in dict.fromkeys([f[1] for f in fus[:3]] + [(2.0, 60.0)]):
        for fl, au, sr, to, re_, cv in itertools.product(
                (0.0, 0.2, 0.3), st, (0.0, 0.25), (0.0, 0.25), st, (0.0, 0.25, 0.5, 0.75, 1.0)):
            # ! 1 + Σw ≤ 2.00 (contract::algorithms).
            if au + sr + to + re_ + cv > 1.0 + 1e-9:
                continue
            cfg = (wd, 1.0, k, [au, sr, 0.0, to, re_, cv], fl)
            grid.append((sc(train, cfg), cfg))
    grid.sort(key=lambda g: (-g[0][2], -g[0][1], -g[0][0]))
    print(f"grid {len(grid)}, top 8 on train MRR "
          "(w = authority, structural, completeness, topical, region, coverage):")
    for (a5, a10, m), (wd, _, k, w, fl) in grid[:8]:
        print(f"  {wd}:1 k {k:<5} floor {fl} w {w}  R@5 {a5:.3f} R@10 {a10:.3f} MRR {m:.3f}")
    best = grid[0][1]
    b = sc(hold, best)
    print(f"best on holdout (once): R@5 {b[0]:.3f} R@10 {b[1]:.3f} MRR {b[2]:.3f}  {best}")


if __name__ == "__main__":
    {"harvest": harvest, "fit": fit, "fuse": fuse}[sys.argv[1]](sys.argv[2])
