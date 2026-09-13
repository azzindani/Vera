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
       about, length(body) AS blen
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
          "article", "chapter", "about", "blen")

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


def score_pool(rows, qterms, weights):
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


def recall_at_k(loaded, weights, k=TOP_K):
    hits = 0
    per_type = collections.defaultdict(lambda: [0, 0])
    for case, rows, want, qterms in loaded:
        got = {key_of(r) for _, r in score_pool(rows, qterms, weights)[:k]}
        hit = bool(got & want)
        hits += hit
        per_type[case["type"]][0] += hit
        per_type[case["type"]][1] += 1
    return hits / len(loaded), per_type


def rank_of(rows, want, qterms, weights):
    for i, (_, row) in enumerate(score_pool(rows, qterms, weights)):
        if key_of(row) in want:
            return i + 1
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--explain", action="store_true",
                    help="show every case the fitted weights moved")
    args = ap.parse_args()

    zero = dict.fromkeys(FACTORS, 0.0)

    with psycopg.connect(DSN) as conn:
        conn.execute("SET statement_timeout = '120s'")
        loaded = load_cases(conn)

    base, base_types = recall_at_k(loaded, zero)
    print(f"cases: {len(loaded)}   pool: {POOL}   scored at Recall@{TOP_K}\n")
    print(f"baseline (text arm, no factors)      {base:6.1%}")

    # One factor at a time, so a factor that does nothing is visible as doing
    # nothing rather than hidden inside a combination that happens to work.
    print("\nper factor, alone:")
    grid = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
    solo = {}
    for f in FACTORS:
        best_r, best_w = max((recall_at_k(loaded, {**zero, f: w})[0], w)
                             for w in grid)
        solo[f] = (best_r, best_w)
        d = best_r - base
        sign = "=" if abs(d) < 1e-9 else ("+" if d > 0 else "-")
        print(f"  {f:<14} best {best_r:6.1%}  at w={best_w:<4}  {sign}{abs(d):5.1%}")

    live = [f for f in FACTORS if solo[f][0] > base]
    print(f"\njoint search over: {live if live else 'nothing earned a weight'}")
    if not live:
        return

    coarse = [0.0, 0.25, 0.5, 1.0, 1.5]
    best_w, best_r = zero, base
    for combo in itertools.product(coarse, repeat=len(live)):
        w = {**zero, **dict(zip(live, combo))}
        r, _ = recall_at_k(loaded, w)
        if r > best_r:
            best_r, best_w = r, w
    print(f"  best {best_r:6.1%}   ({best_r - base:+.1%} over baseline)")
    print("  weights: " + "  ".join(f"{k}={v}" for k, v in best_w.items() if v))

    _, fitted_types = recall_at_k(loaded, best_w)
    print(f"\n  {'shape':<16}{'base':>7}{'fitted':>9}")
    for shape in sorted(fitted_types,
                        key=lambda s: base_types[s][0] - fitted_types[s][0]):
        b, f = base_types[shape], fitted_types[shape]
        flag = "" if b[0] == f[0] else ("  <-- gained" if f[0] > b[0] else "  <-- LOST")
        print(f"  {shape:<16}{b[0]}/{b[1]:>5}{f[0]}/{f[1]:>7}{flag}")

    if args.explain:
        print("\nmoved cases (rank in pool, baseline -> fitted):")
        for case, rows, want, qterms in loaded:
            a = rank_of(rows, want, qterms, zero)
            b = rank_of(rows, want, qterms, best_w)
            if a != b:
                print(f"  {case['id']}  {case['type']:<14} "
                      f"{a if a else '-':>4} -> {b if b else '-':>4}   {case['source']}")


if __name__ == "__main__":
    main()
