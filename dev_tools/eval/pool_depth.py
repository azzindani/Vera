"""Where does the answer actually live? -- pool depth vs. rank.

Recall@5 says whether the answer reached the top of the list. It cannot say
whether the answer was ever retrieved. Those are different failures with
different fixes: the first is ranking, the second is retrieval, and a single
number conflates them.

This runs ONLY the text arm -- the best single arm, built exactly as
`crates/store/src/search.rs` builds it -- to whatever depth is asked, and
sorts every labelled case into three buckets:

  A  a chunk of the labelled ARTICLE is in the pool         ranking headroom
  B  a chunk of the labelled REGULATION is, the article not  sibling-expansion
                                                             headroom
  C  the regulation is absent entirely                       retrieval floor

A is what better ranking could deliver without retrieving anything new.
B is what sibling expansion could convert: the engine already had the right
law in hand and returned the wrong clause of it.
C is the only bucket that needs better retrieval.

! No embedder, and therefore no GPU. The dense arm carries weight 0.0
(`EMBEDDING.md` §5), so a text-arm floor is the honest arm to measure on, and
one that any machine can reproduce.

Usage:
    python dev_tools/eval/pool_depth.py            # depths 5 10 20 60
    python dev_tools/eval/pool_depth.py 5 60       # specific depths

    VERA_DSN=... overrides the connection.
"""

import collections
import json
import os
import pathlib
import sys

import psycopg

DSN = os.environ.get(
    "VERA_DSN", "host=127.0.0.1 port=5432 dbname=vera2 user=vera password=vera"
)
QUERIES = pathlib.Path(__file__).with_name("queries.json")

# ! OR semantics, matching search.rs:205. plainto_tsquery ANDs every term and
# returns zero rows for 42 of 44 queries; measuring against it would measure a
# query the engine does not run.
SQL_TEXT = """
WITH q AS (
  SELECT to_tsquery('indonesian',
           array_to_string(tsvector_to_array(to_tsvector('indonesian', %s)), ' | ')
         ) AS tq
)
SELECT regulation_type, regulation_number, year, article
FROM chunks, q
WHERE indexable AND tsv @@ q.tq
ORDER BY tsv <=> q.tq
LIMIT %s
"""


def norm(s):
    return (s or "").strip().upper()


def bucket(conn, cases, depth):
    """Sort every labelled case into A / B / C at this pool depth."""
    tally = collections.Counter()
    by_type = collections.defaultdict(collections.Counter)
    rescuable = []

    for case in cases:
        # out_of_domain cases are labelled with an empty answer on purpose:
        # the correct behaviour is a refusal, which this script does not test.
        if case["type"] == "out_of_domain":
            continue
        labels = case.get("answer_articles") or []
        if not labels:
            continue

        rows = conn.execute(SQL_TEXT, (case["query"], depth)).fetchall()
        got_articles = {(norm(r[0]), norm(r[1]), r[2], norm(r[3])) for r in rows}
        got_regs = {(norm(r[0]), norm(r[1]), r[2]) for r in rows}

        want_articles = {
            (norm(l["regulation_type"]), norm(l["regulation_number"]), l["year"],
             norm(l["article"]))
            for l in labels
        }
        want_regs = {
            (norm(l["regulation_type"]), norm(l["regulation_number"]), l["year"])
            for l in labels
        }

        tally["cases"] += 1
        by_type[case["type"]]["cases"] += 1

        if got_articles & want_articles:
            key = "A"
        elif got_regs & want_regs:
            key = "B"
            rescuable.append((case["id"], case["type"], case["source"]))
        else:
            key = "C"

        tally[key] += 1
        by_type[case["type"]][key] += 1

    return tally, by_type, rescuable


def main():
    depths = [int(a) for a in sys.argv[1:]] or [5, 10, 20, 60]
    cases = json.loads(QUERIES.read_text(encoding="utf-8"))["queries"]

    with psycopg.connect(DSN) as conn:
        conn.execute("SET statement_timeout = '60s'")

        print(f"{'depth':>6}{'n':>5}{'A article':>11}{'B reg only':>12}"
              f"{'C absent':>10}{'ceiling':>9}")
        last = None
        for depth in depths:
            tally, by_type, rescuable = bucket(conn, cases, depth)
            n = tally["cases"]
            ceiling = (tally["A"] + tally["B"]) / n
            print(f"{depth:>6}{n:>5}{tally['A'] / n:>10.1%}{tally['B'] / n:>12.1%}"
                  f"{tally['C'] / n:>10.1%}{ceiling:>9.1%}")
            last = (depth, tally, by_type, rescuable)

        depth, tally, by_type, rescuable = last
        print(f"\nby question shape, at depth {depth}:")
        print(f"  {'shape':<16}{'n':>3}{'A':>4}{'B':>4}{'C':>4}")
        for t, c in sorted(by_type.items(), key=lambda kv: -kv[1]["B"]):
            print(f"  {t:<16}{c['cases']:>3}{c['A']:>4}{c['B']:>4}{c['C']:>4}")

        if rescuable:
            print(f"\nright regulation, wrong clause, at depth {depth}:")
            for i, t, s in rescuable:
                print(f"  {i}  {t:<14} {s}")


if __name__ == "__main__":
    main()
