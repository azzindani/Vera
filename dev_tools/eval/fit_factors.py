"""Fit the multi-factor weights of SCORING.md offline, against the text arm.

Why offline: fitting needs hundreds of rankings, and the engine needs an
embedder for every one of them. The text arm is the best single arm (40.9%),
runs straight against Postgres, and needs no GPU -- so the weights can be
fitted here and only the WINNER paid for in an e2e run.

! This is a FITTING harness, not the gate. It reranks one arm, so its absolute
numbers are not the engine's. `e2e.py` remains the only thing that scores what
ships (EVAL.md section 1). What transfers is the SHAPE of the result: which
factors earn their weight and which do not.

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

import psycopg

DSN = os.environ.get(
    "VERA_DSN", "host=127.0.0.1 port=5432 dbname=vera2 user=vera password=vera"
)
QUERIES = pathlib.Path(__file__).with_name("queries.json")

POOL = 60          # CANDIDATE_POOL; see SCORING.md section 7
TOP_K = 5          # scored at Recall@5, like every other number in EVAL.md
RRF_K = 60.0

# The hierarchy of Indonesian regulation. A lookup, not a model -- and these
# are the only ten types the corpus contains, verified with
#   SELECT DISTINCT regulation_type FROM chunks;
HIERARCHY = {
    "UNDANG-UNDANG": 8,
    "PERATURAN PEMERINTAH": 6,
    "PERATURAN PRESIDEN": 5,
    "INSTRUKSI PRESIDEN": 5,
    "PERATURAN GUBERNUR": 4,
    "PERATURAN BUPATI": 3,
    "PERATURAN WALIKOTA": 3,
    "PERATURAN DAERAH PROVINSI": 3,
    "PERATURAN DAERAH KABUPATEN": 2,
    "PERATURAN DAERAH KOTA": 2,
}
MAX_TIER = 10.0

SQL_POOL = """
WITH q AS (
  SELECT to_tsquery('indonesian',
           array_to_string(tsvector_to_array(to_tsvector('indonesian', %s)), ' | ')
         ) AS tq
)
SELECT id, regulation_type, regulation_number, year, article, chapter,
       about, length(body) AS blen, body
FROM chunks, q
WHERE indexable AND tsv @@ q.tq
ORDER BY tsv <=> q.tq
LIMIT %s
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

    ! The relevance FLOOR of SCORING.md section 3, and the single most valuable
    dial measured here: it is worth more on its own (+12.5 points) than every
    factor weight combined. Unweighted term overlap, not IDF-weighted -- the
    engine has `bm25::evidence`, which is strictly better, but the floor below
    was fitted against THIS measure and the two live on different scales.
    Refitting against evidence is the obvious next experiment.
    """
    if not qterms:
        return 1.0
    return len(qterms & terms(row["body"])) / len(qterms)


def score_pool(rows, qterms, weights, floor=0.0):
    # ! Applied BEFORE scoring, and this is what makes the gate real. Without
    # it the prior (bounded at 1 + sum(weights) = 2.0x) exceeds the entire
    # spread of RRF relevance across a 60-candidate pool (1.98x), so metadata
    # alone can lift pool rank 59 to rank 1. Measured, not feared.
    kept = [r for r in rows if coverage(r, qterms) >= floor]
    # ! Never empty on account of the floor. "Nothing matches" is the domain
    # gate's decision, not this one.
    rows = kept or rows[:1]
    years = [r["year"] for r in rows if r["year"]]
    newest, oldest = (max(years), min(years)) if years else (0, 0)
    out = []
    for rank, row in enumerate(rows):
        relevance = 1.0 / (RRF_K + rank)      # the arm's own verdict
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

def load_cases(conn):
    cases = json.loads(QUERIES.read_text(encoding="utf-8"))["queries"]
    loaded = []
    for case in cases:
        # out_of_domain is labelled with an empty answer on purpose: correct
        # behaviour is a refusal, which reranking does not affect.
        if case["type"] == "out_of_domain" or not case.get("answer_articles"):
            continue
        rows = [dict(zip(FIELDS, r))
                for r in conn.execute(SQL_POOL, (case["query"], POOL)).fetchall()]
        want = {(norm(lab["regulation_type"]), norm(lab["regulation_number"]),
                 lab["year"], norm(lab["article"]))
                for lab in case["answer_articles"]}
        loaded.append((case, rows, want, terms(case["query"])))
    return loaded


def key_of(row):
    return (norm(row["regulation_type"]), norm(row["regulation_number"]),
            row["year"], norm(row["article"]))


def recall_at_k(loaded, weights, k=TOP_K, floor=0.0):
    hits = 0
    per_type = collections.defaultdict(lambda: [0, 0])
    for case, rows, want, qterms in loaded:
        got = {key_of(r) for _, r in score_pool(rows, qterms, weights, floor)[:k]}
        hit = bool(got & want)
        hits += hit
        per_type[case["type"]][0] += hit
        per_type[case["type"]][1] += 1
    return hits / len(loaded), per_type


FLOORS = (0.0, 0.2, 0.3, 0.4, 0.5)


def leave_one_out(loaded, configs, hits):
    """The only number worth quoting.

    ! Taking the best of N configurations on 40 cases overfits, and by about
    ten points here. Each fold refits on the other 39 and is scored on the one
    held out, so the configuration never sees the case it is judged on.
    """
    n = len(loaded)
    correct = 0
    chosen = collections.Counter()
    for i in range(n):
        best = max(range(len(configs)), key=lambda j: sum(hits[j]) - hits[j][i])
        correct += hits[best][i]
        floor, w = configs[best]
        chosen[(floor, tuple(round(w[k], 2) for k in FACTORS))] += 1
    return correct / n, chosen


def rank_of(rows, want, qterms, weights, floor=0.0):
    for i, (_, row) in enumerate(score_pool(rows, qterms, weights, floor)):
        if key_of(row) in want:
            return i + 1
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--explain", action="store_true",
                    help="show every case the fitted configuration moved")
    args = ap.parse_args()

    zero = dict.fromkeys(FACTORS, 0.0)

    with psycopg.connect(DSN) as conn:
        conn.execute("SET statement_timeout = '120s'")
        loaded = load_cases(conn)
    n = len(loaded)

    base, base_types = recall_at_k(loaded, zero)
    print(f"cases: {n}   pool: {POOL}   scored at Recall@{TOP_K}")
    print()
    print(f"baseline · text arm, no floor, no factors      {base:6.1%}")

    # The floor alone, before any factor earns anything. Measured first because
    # it turns out to be worth more than all of them together.
    print()
    print("relevance floor alone (no factors):")
    for f in FLOORS:
        r, _ = recall_at_k(loaded, zero, floor=f)
        print(f"  floor {f:<5} {r:6.1%}   {r - base:+5.1%}")

    best_floor_solo = max(FLOORS, key=lambda f: recall_at_k(loaded, zero, floor=f)[0])
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
        for f in FLOORS
        for a, s, c, t in itertools.product(coarse, repeat=4)
    ]

    hits = [
        [
            1 if {key_of(r) for _, r in score_pool(rows, qt, w, f)[:TOP_K]} & want else 0
            for _, rows, want, qt in loaded
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
        for case, rows, want, qterms in loaded:
            a = rank_of(rows, want, qterms, zero)
            b = rank_of(rows, want, qterms, best_w, best_floor)
            if a != b:
                print(f"  {case['id']}  {case['type']:<14} "
                      f"{a if a else '-':>4} -> {b if b else '-':>4}   {case['source']}")


if __name__ == "__main__":
    main()
