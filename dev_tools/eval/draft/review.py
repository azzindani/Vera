"""Domain review of eval labels (`ROADMAP.md` Stage 0): export a sheet, import the verdicts.

    PGPASSWORD=... python -I dev_tools/eval/draft/review.py export --dsn ... --out review.csv [--n 120]
    python -I dev_tools/eval/draft/review.py import --sheet review.csv --reviewer "<name or role>"

`export` draws a stratified sample of in-domain cases not yet reviewed (every type
represented in proportion, at least 5 each, seeded) and writes a CSV that opens in
Excel: the question, the labelled regulation and article, the clause text read from the
corpus, and empty columns for the reviewer. UTF-8 with a BOM, so Excel shows Indonesian
characters correctly.

The reviewer fills `verdict` with one of:

    BENAR     the clause answers the question correctly and completely
    SEBAGIAN  it answers only part, or needs another article
    SALAH     it does not answer it
    AMBIGU    many other regulations answer it equally, so the label is unfair

`import` writes `review: {verdict, note, reviewer, date}` onto each case in
`queries.json`. Nothing is deleted: a case marked SALAH stays, visible, and `e2e.py`
can be run with `EVAL_REVIEWED=1` to score only cases a reviewer marked BENAR.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import random
from pathlib import Path

QUERIES = Path(__file__).resolve().parents[1] / "queries.json"
VERDICTS = {"BENAR", "SEBAGIAN", "SALAH", "AMBIGU"}
COLUMNS = ["id", "type", "query", "labelled_answer", "clause_text", "verdict", "note"]


def export(a) -> None:
    import psycopg

    spec = json.loads(QUERIES.read_text("utf-8"))["queries"]
    pool = [q for q in spec if q["type"] != "out_of_domain" and q.get("answer_articles")
            and "review" not in q]
    rng = random.Random(a.seed)
    by_type: dict[str, list] = {}
    for q in pool:
        by_type.setdefault(q["type"], []).append(q)
    picked = []
    for t, qs in sorted(by_type.items()):
        rng.shuffle(qs)
        picked += qs[:max(5, round(a.n * len(qs) / len(pool)))]
    rng.shuffle(picked)
    picked = picked[:max(a.n, 5 * len(by_type))]
    conn = psycopg.connect(a.dsn)
    with open(a.out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        for q in picked:
            labels, texts = [], []
            for ref in q["answer_articles"]:
                about = ref["about"].strip()
                rows = conn.execute(
                    "SELECT body FROM chunks WHERE regulation_type=%s AND regulation_number=%s AND year=%s"
                    " AND article=%s AND (starts_with(upper(about), upper(%s)) OR starts_with(upper(%s), upper(about)))"
                    " ORDER BY seq NULLS LAST, id",
                    (ref["regulation_type"], ref["regulation_number"], ref["year"], ref["article"],
                     about, about)).fetchall()
                labels.append(f"{ref['regulation_type']} {ref['regulation_number']}/{ref['year']} "
                              f"{ref['article']} ({about[:80]})")
                texts.append(f"[{ref['article']}] " + "\n".join(r[0] for r in rows)[:3000])
            w.writerow([q["id"], q["type"], q["query"], "\n".join(labels), "\n\n".join(texts), "", ""])
    print(f"wrote {len(picked)} cases to {a.out} · " + " · ".join(
        f"{t} {sum(1 for q in picked if q['type'] == t)}" for t in sorted(by_type)))


def import_(a) -> None:
    spec = json.loads(QUERIES.read_text("utf-8"))
    by_id = {q["id"]: q for q in spec["queries"]}
    done = skipped = 0
    with open(a.sheet, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            v = (row.get("verdict") or "").strip().upper()
            if not v:
                skipped += 1
                continue
            if v not in VERDICTS:
                raise SystemExit(f"{row['id']}: verdict {v!r} is not one of {sorted(VERDICTS)}")
            by_id[row["id"]]["review"] = {"verdict": v, "note": (row.get("note") or "").strip(),
                                          "reviewer": a.reviewer, "date": dt.date.today().isoformat()}
            done += 1
    QUERIES.write_text(json.dumps(spec, ensure_ascii=False, indent=2), "utf-8", newline="\n")
    reviewed = [q["review"]["verdict"] for q in spec["queries"] if "review" in q]
    print(f"imported {done} verdicts ({skipped} rows left blank) · reviewed so far {len(reviewed)}: " +
          " · ".join(f"{v} {reviewed.count(v)}" for v in sorted(VERDICTS)))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--dsn", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--n", type=int, default=120)
    e.add_argument("--seed", type=int, default=20261114)
    i = sub.add_parser("import")
    i.add_argument("--sheet", required=True)
    i.add_argument("--reviewer", required=True)
    a = ap.parse_args()
    export(a) if a.cmd == "export" else import_(a)


if __name__ == "__main__":
    main()
