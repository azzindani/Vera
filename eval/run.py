"""Score the retrieval arms against the labeled query set · EVAL.md §3.

Reports Recall@k and MRR per arm and for RRF fusion, so a dial change that
helps one arm and hurts the whole is visible rather than averaged away.

! Different question shapes need different scoring, and averaging them into a
single number hides the thing you need to see:

  exact_ref       any chunk of the named regulation counts · mechanical label
  multi_tier      any of several genuinely-correct clauses counts
  underspecified  likewise · the question really does have many right answers
  hard_negative   the right clause must OUTRANK its near-identical siblings
  out_of_domain   there is no right answer · the engine should return nothing.
                  That is an engine-level decision (CLAUDE.md invariant 13),
                  not an arm-level one, so it is measured separately, against
                  the routing score rather than against retrieved rows.

Usage:
    python eval/run.py
    python eval/run.py --dense-column dense_ctx    # compare a candidate column
    python eval/run.py --k 10
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipelines" / "pre_embed"))
from sparse import Bm25Vectorizer, tokenize  # noqa: E402

PG = os.environ.get(
    "DATABASE_URL", "host=localhost port=5432 dbname=vera user=vera password=vera"
)
TEI = os.environ.get("EMBED_ENDPOINT", "http://localhost:8080")
VOCAB = ROOT / ".test" / "runs" / "spike-01.bm25.json"

PER_ARM = 50
RRF_K = 60

# The domain gate's floors, mirroring crates/mcp/src/pipeline.rs. Kept here so
# a threshold change is re-scored against the labelled set before it ships,
# rather than after someone notices real questions coming back empty.
DOMAIN_FLOOR = 0.45          # nearest-centroid similarity
DOMAIN_LEXICAL_FLOOR = 0.40  # IDF mass of the query the evidence accounts for
GATE_SAMPLE = 5

# The domain gate's two floors, mirroring crates/mcp/src/pipeline.rs. Kept here
# so a threshold change can be re-scored against the labelled set before it
# ships, rather than after someone notices real questions coming back empty.
DOMAIN_FLOOR = 0.45          # nearest-centroid similarity
DOMAIN_LEXICAL_FLOOR = 0.40  # IDF mass of the query the evidence accounts for
GATE_SAMPLE = 5
ARMS = ["dense", "sparse", "tsv", "ident", "RRF(all)", "RRF(s+t)", "RRF(s+d)"]

# Mirrors crates/mcp/src/identifier.rs. An exact_ref query is served by the
# global identifier path (invariant 4), NOT by the vector arms — scoring it
# against dense/sparse/tsv alone reports 0% for a path that works.
IDENT_RE = [
    re.compile(r"\b(\d{1,4})\s*/\s*(\d{4})\b"),
    re.compile(r"(?i)\bnomor\s+(\d{1,4})\s+tahun\s+(\d{4})\b"),
    re.compile(r"(?i)\bno\.?\s*(\d{1,4})\s+tahun\s+(\d{4})\b"),
    re.compile(r"(?i)\b(\d{1,4})\s+tahun\s+(\d{4})\b"),
]


def parse_identifier(q):
    for rx in IDENT_RE:
        m = rx.search(q)
        if m:
            return m.group(1), int(m.group(2))
    return None

# OR, not plainto_tsquery. plainto ANDs every term, so a natural question needs
# one chunk holding all its lexemes and matches nothing. See store/search.rs.
ORQ = "array_to_string(tsvector_to_array(to_tsvector('indonesian', %s)), ' | ')::tsquery"


def embed(text):
    body = json.dumps({"inputs": text, "truncate": False}).encode()
    req = urllib.request.Request(
        f"{TEI}/embed", data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)[0]


def evidence(vz, cur, query, sparse_ids):
    """Share of the query's IDF mass the retrieved evidence accounts for.

    Mirrors QueryVectorizer::evidence. Unknown terms are charged the maximum:
    a word this corpus has never seen is the strongest signal it cannot answer.
    """
    terms = set(tokenize(query))
    if not terms:
        return 0.0
    mx = max(vz.idf.values())
    want = {t: vz.idf.get(t, mx) for t in terms}
    total = sum(want.values())
    if total <= 0:
        return 0.0
    ids = sparse_ids[:GATE_SAMPLE]
    if not ids:
        return 0.0
    cur.execute("SELECT body FROM chunks WHERE id = ANY(%s)", (ids,))
    pooled = set()
    for (body,) in cur.fetchall():
        pooled |= set(tokenize(body))
    return sum(w for t, w in want.items() if t in pooled) / total


def evidence(vz, cur, query, sparse_ids):
    """Share of the query's IDF mass the retrieved evidence accounts for.

    Mirrors QueryVectorizer::evidence. Unknown terms are charged the maximum:
    a word this corpus has never seen is the strongest possible evidence that
    it cannot answer the question.
    """
    terms = set(tokenize(query))
    if not terms:
        return 0.0
    mx = max(vz.idf.values())
    want = {t: vz.idf.get(t, mx) for t in terms}
    total = sum(want.values())
    ids = sparse_ids[:GATE_SAMPLE]
    if total <= 0 or not ids:
        return 0.0
    cur.execute("SELECT body FROM chunks WHERE id = ANY(%s)", (ids,))
    pooled = set()
    for (body,) in cur.fetchall():
        pooled |= set(tokenize(body))
    return sum(w for t, w in want.items() if t in pooled) / total


def rank_of(ranked, targets):
    for i, cid in enumerate(ranked, 1):
        if cid in targets:
            return i
    return None


def fuse(*lists):
    f: dict[str, float] = defaultdict(float)
    for ids in lists:
        for i, cid in enumerate(ids, 1):
            f[cid] += 1.0 / (RRF_K + i)
    return [c for c, _ in sorted(f.items(), key=lambda kv: (-kv[1], kv[0]))]


def resolve_targets(cur, q):
    """The set of chunk ids that count as correct for this query.

    ! Labels name ARTICLES, not chunk ids. A chunk id is an artefact of how the
    corpus was split — re-chunking changes every one of them — while an article
    is a property of the law. Any chunk of a labelled article counts, which is
    also the honest reading: if an article is split across three chunks, all
    three are the answer.
    """
    ids = set(q.get("answer_chunks", []))
    for ref in q.get("answer_articles", []):
        cur.execute(
            "SELECT id FROM chunks WHERE regulation_type=%s AND regulation_number=%s"
            " AND year=%s AND about=%s AND article=%s",
            (ref["regulation_type"], ref["regulation_number"], ref["year"],
             ref["about"], ref["article"]),
        )
        ids |= {r[0] for r in cur.fetchall()}
    reg = q.get("answer_regulation")
    if reg:
        cur.execute(
            "SELECT id FROM chunks WHERE regulation_type=%s"
            " AND regulation_number=%s AND year=%s",
            (reg["regulation_type"], reg["regulation_number"], reg["year"]),
        )
        ids |= {r[0] for r in cur.fetchall()}
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dense-column", default="dense")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--by-type", action="store_true", help="break results down by type")
    args = ap.parse_args()
    import psycopg

    spec = json.loads((Path(__file__).parent / "queries.json").read_text(encoding="utf-8"))
    allq = spec["queries"]
    vz = Bm25Vectorizer.load(VOCAB)
    pg = psycopg.connect(PG)
    cur = pg.cursor()

    # Cluster centroids, for the domain gate that invariant 13 needs.
    cur.execute("SELECT centroid::text FROM clusters")
    cents = []
    for (c,) in cur.fetchall():
        v = [float(x) for x in c.strip("[]").split(",")]
        n = math.sqrt(sum(a * a for a in v)) or 1.0
        cents.append([a / n for a in v])

    scored, ood, hard = [], [], []

    for q in allq:
        qv_raw = embed(q["query"])
        n = math.sqrt(sum(a * a for a in qv_raw)) or 1.0
        qn = [a / n for a in qv_raw]
        domain_score = max(sum(a * b for a, b in zip(qn, c)) for c in cents)

        cur.execute(
            "SELECT id FROM chunks WHERE indexable AND sparse IS NOT NULL"
            " ORDER BY sparse <#> %s::text::sparsevec LIMIT %s",
            (vz.to_sparsevec(vz.query(q["query"])), PER_ARM),
        )
        sparse = [r[0] for r in cur.fetchall()]
        lex = evidence(vz, cur, q["query"], sparse)
        exempt = q["type"] == "exact_ref"   # invariant 4
        refused = not exempt and (
            lex < DOMAIN_LEXICAL_FLOOR or domain_score < DOMAIN_FLOOR
        )

        if q["type"] == "out_of_domain":
            ood.append((q["id"], domain_score, lex, refused, q["query"]))
            continue

        targets = resolve_targets(cur, q)
        if not targets:
            print(f"! {q['id']} has no resolvable target — skipped", file=sys.stderr)
            continue

        qv = "[" + ",".join(f"{x:.6g}" for x in qv_raw) + "]"
        cur.execute(
            f"SELECT id FROM chunks WHERE indexable AND {args.dense_column} IS NOT NULL"
            f" ORDER BY {args.dense_column} <=> %s::text::halfvec LIMIT %s",
            (qv, PER_ARM),
        )
        dense = [r[0] for r in cur.fetchall()]

        cur.execute(
            f"SELECT id FROM chunks WHERE indexable AND tsv @@ {ORQ}"
            f" ORDER BY ts_rank(tsv, {ORQ}) DESC LIMIT %s",
            (q["query"], q["query"], PER_ARM),
        )
        tsv = [r[0] for r in cur.fetchall()]

        ident = []
        got = parse_identifier(q["query"])
        if got:
            cur.execute(
                "SELECT id FROM chunks WHERE regulation_number=%s AND year=%s"
                " ORDER BY chunk_no LIMIT %s", (got[0], got[1], PER_ARM))
            ident = [r[0] for r in cur.fetchall()]

        ranked = {
            "dense": dense, "sparse": sparse, "tsv": tsv, "ident": ident,
            "RRF(all)": fuse(dense, sparse, tsv),
            "RRF(s+t)": fuse(sparse, tsv),
            "RRF(s+d)": fuse(sparse, dense),
        }
        ranks = {a: rank_of(ids, targets) for a, ids in ranked.items()}
        scored.append((q, ranks, domain_score, lex, refused))

        if q["type"] == "hard_negative":
            bad = set(q["must_not_rank_first"])
            row = {}
            for a, ids in ranked.items():
                good_r, bad_r = rank_of(ids, targets), rank_of(ids, bad)
                row[a] = (good_r, bad_r)
            hard.append((q["id"], row))

    # -- headline ----------------------------------------------------------
    n = len(scored)
    print(f"scored {n} retrievable queries · {len(ood)} out-of-domain measured separately")
    print(f"dense column: {args.dense_column}\n")
    print(f"{'arm':<9} {'Recall@' + str(args.k):>9} {'MRR':>7}")
    print("-" * 27)
    for a in ARMS:
        hits = sum(1 for _, r, *_ in scored if r[a] and r[a] <= args.k)
        mrr = sum(1.0 / r[a] for _, r, *_ in scored if r[a])
        print(f"{a:<9} {hits / n:>8.1%} {mrr / n:>7.3f}")

    # -- by question shape -------------------------------------------------
    if args.by_type:
        by = defaultdict(list)
        for q, r, *_ in scored:
            by[q["type"]].append(r)
        print(f"\n{'type':<15} {'n':>3}  " + "  ".join(f"{a:>9}" for a in ARMS))
        print("-" * (20 + 11 * len(ARMS)))
        for t, rs in sorted(by.items()):
            cells = []
            for a in ARMS:
                h = sum(1 for r in rs if r[a] and r[a] <= args.k)
                cells.append(f"{h / len(rs):>8.0%} ")
            print(f"{t:<15} {len(rs):>3}  " + " ".join(cells))

    # -- hard negatives ----------------------------------------------------
    if hard:
        print(f"\nhard negatives (rank of right answer / rank of a look-alike):")
        for qid, row in hard:
            cells = " ".join(
                f"{a}={'-' if g is None else g}/{'-' if b is None else b}"
                for a, (g, b) in row.items() if a in ("sparse", "dense", "tsv")
            )
            # ! Per arm. Requiring every arm to win would let the dead dense
            # arm mark the whole case FAIL and hide that sparse got it right.
            ok = [a for a, (g, b) in row.items()
                  if g is not None and (b is None or g < b)]
            print(f"  {qid} {('OK:' + ','.join(ok)) if ok else 'FAIL (no arm)':<28} {cells}")

    # -- the domain gate ---------------------------------------------------
    # ! Two signals, because neither works alone. Centroid similarity tracks
    # LANGUAGE, not subject: it cannot reject an everyday Indonesian question
    # such as "cara memperbaiki keran air yang bocor di dapur" (0.717, above
    # the in-domain mean). Lexical evidence cannot reject English, whose
    # function words do occur in this corpus ("what is the capital of France"
    # scores 0.789). Each covers the other's blind spot.
    print("")
    print("domain gate · refuse when lexical < "
          f"{DOMAIN_LEXICAL_FLOOR} or centroid < {DOMAIN_FLOOR}"
          "  (identifier queries exempt · invariant 4)")
    gated = [t for t in scored if t[0]["type"] != "exact_ref"]
    false_rejects = [t[0]["id"] for t in gated if t[4]]
    caught = [o for o in ood if o[3]]
    ok = len(caught) == len(ood) and not false_rejects
    print(f"  real questions refused : {len(false_rejects)}/{len(gated)}  {false_rejects}")
    print(f"  out-of-domain refused  : {len(caught)}/{len(ood)}")
    print(f"  verdict                : {'PASS' if ok else 'FAIL'}")
    if gated:
        print(f"  margin · lexical  in-domain min {min(t[3] for t in gated):.3f}"
              f"  floor {DOMAIN_LEXICAL_FLOOR}")
        print(f"  margin · centroid in-domain min {min(t[2] for t in gated):.3f}"
              f"  floor {DOMAIN_FLOOR}")
    for qid, cs, lex, ref, text in sorted(ood, key=lambda x: -x[2]):
        print(f"    {'refused' if ref else 'LEAKED!':<8} lex={lex:.3f} cs={cs:.3f}"
              f"  {qid}  {text[:44]}")


    # -- per query ---------------------------------------------------------
    print(f"\nper-query rank (- = not in top {PER_ARM}):")
    print(f"{'id':<6} {'type':<15} {'dense':>6} {'sparse':>7} {'tsv':>5} {'all':>5} {'s+d':>5}")
    for q, r, *_ in scored:
        f = lambda v: str(v) if v else "-"  # noqa: E731
        print(f"{q['id']:<6} {q['type']:<15} {f(r['dense']):>6} {f(r['sparse']):>7} "
              f"{f(r['tsv']):>5} {f(r['RRF(all)']):>5} {f(r['RRF(s+d)']):>5}")
    pg.close()


if __name__ == "__main__":
    main()
