"""Fit the multi-factor weights of SCORING.md offline, against the SHIPPED pool.

Why offline: fitting needs thousands of rankings and the engine answers one
query a second, so a 3,125-configuration grid through the server is days. This
rebuilds the pool the engine builds -- RRF over the sparse and text arms, the
same `DEFAULT_K`, the same per-arm depth -- and then reranks it in memory,
which costs one database pass and no GPU at all. `DENSE_WEIGHT` is 0.0, so the
dense arm contributes nothing to the fused order and no embedder is needed to
reproduce it. If dense ever earns a weight, this shortcut dies with it.

! --pool text fits against the TEXT ARM ALONE, which is what this harness did
until the first in-situ run. It is kept because EVAL.md quotes it, and because
the gap between the two is the point: the floor that wins by +12.5 points on
the text arm LOSES ground once BM25 is in the pool, having been fitted to do a
job sparse retrieval already does.

! Still a fitting harness, not the gate. `e2e.py` remains the only thing that
scores what ships (EVAL.md section 1).

Model (SCORING.md section 3): relevance is the base AND the gate. Factors are a
bounded multiplicative prior on top, never a replacement -- authority without
relevance ranks the most prestigious document in the corpus first for every
query, and it looks correct.

    final = relevance * (1 + sum(w_i * factor_i))

Usage:
    python dev_tools/eval/fit_factors.py            # baseline + grid search
    python dev_tools/eval/fit_factors.py --explain  # per-case movement
"""

import argparse
import collections
import itertools
import json
import math
import os
import pathlib
import re
import sys

import psycopg

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "pre_embed"))
from sparse import Bm25Vectorizer, tokenize  # noqa: E402

DSN = os.environ.get(
    "VERA_DSN", "host=127.0.0.1 port=5432 dbname=vera2 user=vera password=vera"
)
QUERIES = pathlib.Path(__file__).with_name("queries.json")

POOL = 60          # CANDIDATE_POOL; see SCORING.md section 7
PER_ARM = 20       # PER_ARM_K in pipeline.rs -- how deep each arm is read
TOP_K = 5          # scored at Recall@5, like every other number in EVAL.md
RRF_K = 60.0       # engine::fusion::DEFAULT_K

# The hierarchy of Indonesian regulation. A lookup, not a model -- and these
# are the only ten types the corpus contains, verified with
#   SELECT DISTINCT regulation_type FROM chunks;
# ! Must equal `tier()` in crates/engine/src/factors.rs. It did not: 9c50c87
# corrected the Rust table (a Perbup does not outrank the Perda it implements)
# and left this one inverted, so the weights were fitted against one hierarchy
# and applied against another. A lookup duplicated in two languages needs a
# test that they agree -- `hierarchy_matches_the_engine` below is it.
HIERARCHY = {
    "UNDANG-UNDANG": 8,
    "PERATURAN PEMERINTAH": 7,
    "PERATURAN PRESIDEN": 6,
    "INSTRUKSI PRESIDEN": 5,
    "PERATURAN DAERAH PROVINSI": 4,
    "PERATURAN GUBERNUR": 3,
    "PERATURAN DAERAH KABUPATEN": 3,
    "PERATURAN DAERAH KOTA": 3,
    "PERATURAN BUPATI": 2,
    "PERATURAN WALIKOTA": 2,
}
MAX_TIER = 10.0

SQL_TEXT_ARM = """
WITH q AS (
  SELECT to_tsquery('indonesian',
           array_to_string(tsvector_to_array(to_tsvector('indonesian', %s)), ' | ')
         ) AS tq
)
SELECT id, regulation_type, regulation_number, year, article, chapter,
       about, length(body) AS blen, body,
       (1.0 / (1.0 + (tsv <=> q.tq)))::real AS score
FROM chunks, q
WHERE indexable AND tsv @@ q.tq
ORDER BY tsv <=> q.tq
LIMIT %s
"""

SQL_SPARSE_ARM = """
SELECT id, -(sparse <#> %s::text::sparsevec) AS score
FROM chunks
WHERE indexable AND sparse IS NOT NULL
ORDER BY sparse <#> %s::text::sparsevec
LIMIT %s
"""

# ! Mirrors `store::search::OVERFETCH` and `settle`. `ORDER BY <distance>
# LIMIT k` returns an arbitrary member of any tie group straddling the limit,
# so both the engine and this harness read deeper and break ties on id. If they
# used different rules the fit would be measured on a pool the engine never
# builds -- which is the whole reason this harness exists.
OVERFETCH = 3


def settle(rows, k):
    return sorted(rows, key=lambda r: (-r[1], r[0]))[:k]

SQL_META = """
SELECT id, regulation_type, regulation_number, year, article, chapter,
       about, length(body) AS blen, body
FROM chunks WHERE id = ANY(%s)
"""

# Function words carry no topical signal and would make every `about` overlap
# look the same. Indonesian, because the queries are.
STOP = {
    "yang", "dan", "atau", "untuk", "dengan", "pada", "dari", "dalam", "oleh",
    "apakah", "bagaimana", "berapa", "siapa", "adalah", "itu", "ini", "ke",
    "di", "tidak", "dapat", "harus", "wajib", "jika", "akan", "sebagai",
}

FIELDS = ("id", "regulation_type", "regulation_number", "year",
          "article", "chapter", "about", "blen", "body")

# ! Only used to decide the FLOOR EXEMPTION, never to add candidates: the
# engine's identifier hits go to `exact_matches`, a separate channel that
# bypasses fusion entirely (pipeline.rs `exact_arm`). Reproducing it here as a
# fourth arm would fit against a pool the engine never scores.
#
# Ported from `identifier::extract`, token walk and all, rather than
# approximated with a regex -- an exemption that fires on a different set of
# queries than the engine's fits a rule the engine does not apply.


def _is_year(t):
    return len(t) == 4 and t.isdigit() and t[0] in "12"


def names_a_regulation(query):
    tokens = [t for t in re.split(r"[^0-9a-z]+", query.lower()) if t]
    i = 0
    while i < len(tokens):
        t = tokens[i]
        nxt = tokens[i + 1] if i + 1 < len(tokens) else None
        # "<number> tahun <year>", the number up to three tokens back.
        if t == "tahun" and nxt and _is_year(nxt):
            if any(c.isdigit() and not _is_year(c)
                   for c in reversed(tokens[max(0, i - 3):i])):
                return True
        # "<number> <year>", which is how "28/2007" survives tokenisation.
        if t.isdigit() and not _is_year(t) and nxt and _is_year(nxt):
            return True
        i += 1
    return False


FACTORS = ("authority", "structural", "temporal", "completeness", "topical")


def terms(text):
    return {t for t in re.findall(r"\w+", (text or "").lower())
            if len(t) > 3 and t not in STOP}


def norm(s):
    return (s or "").strip().upper()


# --- the factors -----------------------------------------------------------

def f_authority(row):
    """How binding is this instrument? SCORING.md section 2."""
    tier = HIERARCHY.get(norm(row["regulation_type"]))
    return tier / MAX_TIER if tier else 0.0


def f_structural(row):
    """An operative clause, or an annex?

    ! 80,121 of 355,621 indexable chunks are LAMPIRAN -- 22.5% of the corpus
    is annex material, and it is rarely the answer to a question about
    obligations.
    """
    article = norm(row["article"])
    chapter = norm(row["chapter"])
    if "LAMPIRAN" in article or "LAMPIRAN" in chapter:
        return 0.0
    if "PENJELASAN" in chapter:   # elucidation: explains a clause, is not one
        return 0.3
    if article.startswith("PASAL"):
        return 1.0
    return 0.5


def f_temporal(row, newest, oldest):
    """Linear recency across the corpus span.

    Newer is not automatically better in law -- a 1999 statute still governs
    unless it was repealed. Whether this earns a weight at all is the question
    the grid search answers.
    """
    y = row["year"]
    if not y or newest == oldest:
        return 0.5
    return (y - oldest) / (newest - oldest)


def f_completeness(row):
    """A whole provision, or a fragment?

    Saturating, not linear: past a few hundred characters more text is not more
    complete, and a linear term would simply rank the longest chunk first.
    """
    return 1.0 - math.exp(-(row["blen"] or 0) / 400.0)


def f_topical(row, qterms):
    """Is the INSTRUMENT about what was asked, independent of this chunk?"""
    if not qterms:
        return 0.0
    return len(qterms & terms(row["about"])) / len(qterms)


def coverage(row, qterms):
    """Share of the query's content terms this chunk actually contains.

    ! The relevance FLOOR of SCORING.md section 3. Worth +12.5 points on the
    TEXT ARM, where it was fitted -- and worth nothing in the pool the engine
    actually ranks, because BM25 is already in that pool doing the same job
    with a better instrument. See `--floor-measure evidence` for the comparison
    and EVAL.md section 5 for what that means for the shipped weights.
    """
    if not qterms:
        return 1.0
    return len(qterms & terms(row["body"])) / len(qterms)


def evidence_coverage(vz):
    """`bm25::evidence`: the IDF-WEIGHTED share of the query a chunk accounts for.

    The measure `coverage` should have been all along. Unweighted overlap
    counts "pemerintah" -- which occurs in a large fraction of the corpus --
    for exactly as much as the one rare term that identifies the answer, so a
    chunk can clear the floor on function-adjacent words alone. This weights
    each term by how much it narrows the corpus.

    ! Different scale, so its floors are NOT comparable to the unweighted
    ones; both are swept from scratch rather than carried across.
    """
    mx = max(vz.idf.values())

    def measure(row, qterms, _cache={}):
        want = _cache.get(id(qterms))
        if want is None:
            want = {t: vz.idf.get(t, mx) for t in set(qterms)}
            _cache[id(qterms)] = want
        total = sum(want.values())
        if total <= 0.0:
            return 1.0
        have = set(tokenize(row["body"] or ""))
        return sum(w for t, w in want.items() if t in have) / total

    return measure


# Set by main(). (tokenise-the-query, score-one-row) -- the floor's measure is
# selectable and does NOT share `terms()` with `f_topical`: one asks "is this
# chunk about the subject", the other "does this chunk contain the question".
COVER = (terms, coverage)


def score_pool(rows, qterms, qfloor, weights, floor=0.0, exempt=False):
    # ! Applied BEFORE scoring, and this is what makes the gate real. Without
    # it the prior (bounded at 1 + sum(weights) = 2.0x) exceeds the entire
    # spread of RRF relevance across a 60-candidate pool (1.98x), so metadata
    # alone can lift pool rank 59 to rank 1. Measured, not feared.
    #
    # ! Skipped when the query names a regulation, mirroring pipeline.rs: the
    # content terms of "PP 26 tahun 2009" are {tahun, 2009}, which no clause
    # body contains, so the floor drops the whole pool. An identifier IS the
    # relevance signal.
    if not exempt:
        kept = [r for r in rows if COVER[1](r, qfloor) >= floor]
        # ! Never empty on account of the floor. "Nothing matches" is the
        # domain gate's decision, not this one.
        rows = kept or rows[:1]
    years = [r["year"] for r in rows if r["year"]]
    newest, oldest = (max(years), min(years)) if years else (0, 0)
    out = []
    for row in rows:
        # ! The pool's own score, ✗ 1/(k + position in the survivors). The
        # engine keeps what RRF produced and filtering does not renumber it;
        # a harness that renumbers is fitting a ranking nothing serves. This
        # was measured to agree at the old shipped config (EVAL.md) -- it does
        # not agree everywhere, and agreeing by luck is not a reason to keep it.
        relevance = row["rel"]
        f = {
            "authority": f_authority(row),
            "structural": f_structural(row),
            "temporal": f_temporal(row, newest, oldest),
            "completeness": f_completeness(row),
            "topical": f_topical(row, qterms),
        }
        prior = sum(weights[k] * f[k] for k in FACTORS)
        out.append((relevance * (1.0 + prior), row))
    out.sort(key=lambda t: -t[0])
    return out


# --- scoring against the labels -------------------------------------------

def text_pool(conn, _vz, query):
    """The text arm alone, deepest-first. What this harness fitted against
    until the first in-situ run showed the result did not transfer."""
    raw = conn.execute(SQL_TEXT_ARM, (query, POOL * OVERFETCH)).fetchall()
    order = settle([(r[0], r[9]) for r in raw], POOL)
    meta = {r[0]: dict(zip(FIELDS, r)) for r in raw}
    rows = []
    for rank, (cid, _) in enumerate(order):
        row = meta[cid]
        row["rel"] = 1.0 / (RRF_K + rank)
        rows.append(row)
    return rows


def fused_pool(conn, vz, query):
    """The pool the engine builds: RRF over the sparse and text arms.

    ! Dense is absent because `DENSE_WEIGHT` is 0.0 and an arm at weight zero
    contributes nothing to the fused order (`arm_weights` zeroes the weight, it
    does not skip retrieval). That is what lets this run without a GPU, and it
    stops being true the moment dense earns a weight.
    """
    lit = vz.to_sparsevec(vz.query(query))
    text_ids = [c for c, _ in settle(
        [(r[0], r[9]) for r in
         conn.execute(SQL_TEXT_ARM, (query, PER_ARM * OVERFETCH)).fetchall()],
        PER_ARM)]
    sparse_ids = [c for c, _ in settle(
        conn.execute(SQL_SPARSE_ARM, (lit, lit, PER_ARM * OVERFETCH)).fetchall(),
        PER_ARM)]

    # engine::fusion: over RANKS, never over scores, 1-based like DEFAULT_K.
    fused = collections.defaultdict(float)
    for ids in (sparse_ids, text_ids):
        for i, cid in enumerate(ids, 1):
            fused[cid] += 1.0 / (RRF_K + i)
    order = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))[:POOL]

    meta = {r[0]: dict(zip(FIELDS, r))
            for r in conn.execute(SQL_META, ([c for c, _ in order],)).fetchall()}
    rows = []
    for cid, score in order:
        row = meta[cid]
        row["rel"] = score
        rows.append(row)
    return rows


POOLS = {"fused": fused_pool, "text": text_pool}


def load_cases(conn, vz, build):
    cases = json.loads(QUERIES.read_text(encoding="utf-8"))["queries"]
    loaded = []
    for case in cases:
        # out_of_domain is labelled with an empty answer on purpose: correct
        # behaviour is a refusal, which reranking does not affect.
        if case["type"] == "out_of_domain" or not case.get("answer_articles"):
            continue
        rows = build(conn, vz, case["query"])
        want = {(norm(lab["regulation_type"]), norm(lab["regulation_number"]),
                 lab["year"], norm(lab["article"]))
                for lab in case["answer_articles"]}
        loaded.append((case, rows, want, terms(case["query"]),
                       COVER[0](case["query"]),
                       names_a_regulation(case["query"])))
    return loaded


def key_of(row):
    return (norm(row["regulation_type"]), norm(row["regulation_number"]),
            row["year"], norm(row["article"]))


def recall_at_k(loaded, weights, k=TOP_K, floor=0.0):
    hits = 0
    per_type = collections.defaultdict(lambda: [0, 0])
    for case, rows, want, qterms, qfloor, exempt in loaded:
        got = {key_of(r) for _, r in
               score_pool(rows, qterms, qfloor, weights, floor, exempt)[:k]}
        hit = bool(got & want)
        hits += hit
        per_type[case["type"]][0] += hit
        per_type[case["type"]][1] += 1
    return hits / len(loaded), per_type


# ! Two scales, swept separately. IDF-weighted evidence concentrates on rare
# terms, so the same numeric floor admits a different set of chunks under each
# measure and carrying a threshold across would apply a value nothing measured.
FLOORS = (0.0, 0.2, 0.3, 0.4, 0.5)
FLOORS_EVIDENCE = (0.0, 0.1, 0.2, 0.3, 0.4)


# ! The prior is bounded by 1 + sum(w). If that bound exceeds the pool's own
# relevance spread, metadata alone can lift the last candidate to first, which
# is invariant 9's failure mode wearing a good Recall@5. The cap is the largest
# sum(w) allowed to ship; the sweep below shows what enforcing it costs.
WEIGHT_CAP = 1.0


def pool_spread(loaded):
    """best/worst relevance within each case's pool."""
    return sorted(max(r["rel"] for r in rows) / min(r["rel"] for r in rows)
                  for _, rows, _, _, _, _ in loaded)


def leave_one_out(loaded, configs, hits, subset=None):
    """The only number worth quoting.

    ! Taking the best of N configurations on 40 cases overfits, and by about
    ten points here. Each fold refits on the other 39 and is scored on the one
    held out, so the configuration never sees the case it is judged on.
    """
    n = len(loaded)
    pick = list(range(len(configs))) if subset is None else list(subset)
    correct = 0
    chosen = collections.Counter()
    for i in range(n):
        best = max(pick, key=lambda j: sum(hits[j]) - hits[j][i])
        correct += hits[best][i]
        floor, w = configs[best]
        chosen[(floor, tuple(round(w[k], 2) for k in FACTORS))] += 1
    return correct / n, chosen


def rank_of(case, weights, floor=0.0):
    _, rows, want, qterms, qfloor, exempt = case
    for i, (_, row) in enumerate(
            score_pool(rows, qterms, qfloor, weights, floor, exempt)):
        if key_of(row) in want:
            return i + 1
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--explain", action="store_true",
                    help="show every case the fitted configuration moved")
    ap.add_argument("--pool", choices=sorted(POOLS), default="fused",
                    help="fused = what the engine ranks; text = the single arm "
                         "this harness used to fit against")
    ap.add_argument("--floor-measure", choices=("terms", "evidence"),
                    default="terms",
                    help="what the relevance floor counts: unweighted content "
                         "terms, or bm25::evidence's IDF-weighted share")
    args = ap.parse_args()

    zero = dict.fromkeys(FACTORS, 0.0)

    vz = None
    if args.pool == "fused":
        vocab = os.environ.get("BM25_VOCAB")
        # ! No default. Sparse vectors are indexed by vocabulary POSITION, so
        # the wrong vocabulary compares unrelated dimensions and returns
        # plausible nonsense rather than failing.
        if not vocab:
            raise SystemExit(
                "set BM25_VOCAB to the vocabulary this corpus was built with "
                "· --pool text needs no vocabulary"
            )
        vz = Bm25Vectorizer.load(pathlib.Path(vocab))

    global COVER
    if args.floor_measure == "evidence":
        if vz is None:
            raise SystemExit("--floor-measure evidence needs BM25_VOCAB")
        COVER = (tokenize, evidence_coverage(vz))

    with psycopg.connect(DSN) as conn:
        conn.execute("SET statement_timeout = '120s'")
        loaded = load_cases(conn, vz, POOLS[args.pool])
    n = len(loaded)

    base, base_types = recall_at_k(loaded, zero)
    exempt_n = sum(1 for c in loaded if c[5])
    print(f"cases: {n}   pool: {args.pool} ({POOL})   "
          f"scored at Recall@{TOP_K}   floor-exempt: {exempt_n}")
    print(f"floor measure: {args.floor_measure}")
    print()
    print(f"baseline · {args.pool} pool, no floor, no factors     {base:6.1%}")

    # The floor alone, before any factor earns anything. Measured first because
    # it turns out to be worth more than all of them together.
    print()
    floors = FLOORS_EVIDENCE if args.floor_measure == "evidence" else FLOORS
    print("relevance floor alone (no factors):")
    for f in floors:
        r, _ = recall_at_k(loaded, zero, floor=f)
        print(f"  floor {f:<5} {r:6.1%}   {r - base:+5.1%}")

    best_floor_solo = max(floors, key=lambda f: recall_at_k(loaded, zero, floor=f)[0])
    floor_base, _ = recall_at_k(loaded, zero, floor=best_floor_solo)
    print()
    print(f"per factor alone, at floor {best_floor_solo}:")
    grid = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
    for fac in FACTORS:
        best_r, best_w = max(
            (recall_at_k(loaded, {**zero, fac: w}, floor=best_floor_solo)[0], w)
            for w in grid
        )
        d = best_r - floor_base
        sign = "=" if abs(d) < 1e-9 else ("+" if d > 0 else "-")
        print(f"  {fac:<14} best {best_r:6.1%}  at w={best_w:<4}  {sign}{abs(d):5.1%}")

    # ! Floor and weights are fitted JOINTLY. Fitting them separately credits a
    # factor for work the floor was doing -- which is exactly what happened the
    # first time round: `completeness` earned 0.25 as a crude relevance proxy
    # (longer chunks contain more query terms) and drops to 0.0 once a real
    # floor exists. `temporal` is held at 0.0, having been measured to add
    # nothing at any weight under any floor.
    coarse = [0.0, 0.25, 0.5, 1.0, 1.5]
    configs = [
        (f, {**zero, "authority": a, "structural": s, "completeness": c, "topical": t})
        for f in floors
        for a, s, c, t in itertools.product(coarse, repeat=4)
    ]

    hits = [
        [
            1 if {key_of(r) for _, r in
                  score_pool(rows, qt, qf, w, f, ex)[:TOP_K]} & want else 0
            for _, rows, want, qt, qf, ex in loaded
        ]
        for f, w in configs
    ]

    best_i = max(range(len(configs)), key=lambda i: sum(hits[i]))
    best_floor, best_w = configs[best_i]
    loo, chosen = leave_one_out(loaded, configs, hits)

    print()
    print(f"joint fit over {len(configs)} configurations:")
    print(f"  best in-sample   {sum(hits[best_i]) / n:6.1%}")
    print(f"  LEAVE-ONE-OUT    {loo:6.1%}   <- the only number worth quoting")
    print(f"  floor {best_floor} · "
          + "  ".join(f"{k}={v}" for k, v in best_w.items() if v))

    # --- invariant 9, as arithmetic -------------------------------------
    spreads = pool_spread(loaded)
    med = spreads[len(spreads) // 2]
    print()
    print(f"pool relevance spread (best/worst): min {spreads[0]:.2f}x   "
          f"median {med:.2f}x   max {spreads[-1]:.2f}x")
    print("  a prior bounded ABOVE the spread can reorder the pool on metadata")
    print("  alone — invariant 9. What the constraint costs:")
    print(f"  {'cap':<6}{'bound':>7}{'LOO':>8}{'in-sample':>11}"
          f"{'inside':>9}   best configuration")
    for cap in (0.5, WEIGHT_CAP, 1.5, 2.0, 2.5):
        idx = [i for i, (_, w) in enumerate(configs)
               if sum(w.values()) <= cap + 1e-9]
        if not idx:
            continue
        top = max(sum(hits[i]) for i in idx)
        tied = [i for i in idx if sum(hits[i]) == top]
        f, w = configs[tied[0]]
        named = " ".join(f"{k}={v}" for k, v in w.items() if v) or "no factors"
        inside = sum(1 for sp in spreads if 1 + cap <= sp)
        mark = "  <- shipped" if abs(cap - WEIGHT_CAP) < 1e-9 else ""
        print(f"  {cap:<6}{1 + cap:>6.2f}x{leave_one_out(loaded, configs, hits, idx)[0]:>8.1%}"
              f"{top / n:>11.1%}{inside:>6}/{n}   floor {f} · {named}{mark}")
        if abs(cap - WEIGHT_CAP) < 1e-9 and len(tied) > 1:
            print(f"  {'':6}{'':7}{'':8}{'':11}{'':9}   "
                  f"! {len(tied)} configurations tie here; the fit cannot "
                  f"separate them")

    beat = sum(1 for h in hits if sum(h) / n > base)
    print()
    print(f"  {beat} of {len(configs)} configurations beat the baseline "
          f"({beat / len(configs):.0%})")
    print("  chosen across the folds:")
    for (f, w), count in chosen.most_common(3):
        named = "  ".join(f"{k}={v}" for k, v in zip(FACTORS, w) if v)
        print(f"    floor {f} · {named or 'no factors'}   x{count}")

    _, fitted_types = recall_at_k(loaded, best_w, floor=best_floor)
    print()
    print(f"  {'shape':<16}{'base':>7}{'fitted':>9}")
    for shape in sorted(fitted_types,
                        key=lambda s: base_types[s][0] - fitted_types[s][0]):
        b, ft = base_types[shape], fitted_types[shape]
        flag = "" if b[0] == ft[0] else ("  <-- gained" if ft[0] > b[0] else "  <-- LOST")
        print(f"  {shape:<16}{b[0]}/{b[1]:>5}{ft[0]}/{ft[1]:>7}{flag}")

    if args.explain:
        print()
        print("moved cases (rank in pool, baseline -> fitted):")
        for entry in loaded:
            case = entry[0]
            a = rank_of(entry, zero, 0.0)
            b = rank_of(entry, best_w, best_floor)
            if a != b:
                print(f"  {case['id']}  {case['type']:<14} "
                      f"{a if a else '-':>4} -> {b if b else '-':>4}   {case['source']}")


if __name__ == "__main__":
    main()
