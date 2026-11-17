"""Review the engine's answers to the questions it misses: is the result ALSO an answer?

    PGPASSWORD=... python -I dev_tools/eval/draft/alternatives.py export --dsn ... --out sheet.csv
    PGPASSWORD=... python -I dev_tools/eval/draft/alternatives.py import --dsn ... \
        --sheet sheet.csv --reviewer "<name or role>"

! Why (`ROADMAP.md` Stage 3). Each label names one clause, and Indonesian regulation repeats
itself: the same retribution clause sits in many regencies' PERDA, the same text in a law
and the older law it replaced. In a read of 30 misses about a quarter of the engine's first
results answered the question as well as the label. Whether one does is a lawyer's call,
✗ the engine's, so this exports a sheet and imports verdicts.

`export` runs every answerable question (not `exact_ref`) through the deployed engine
(`VERA_HTTP`), keeps the misses (no labelled chunk in the top 5) and writes one row per
miss with ONE candidate from that top 5:

    MIRIP    the result whose text is closest to the labelled clause (dense cosine at
             least `--similar`, default 0.85): most likely the same clause elsewhere
    TERATAS  otherwise the first result

Questions already reviewed for that candidate are skipped.

The reviewer fills `verdict`:

    JUGA_BENAR  the engine's clause answers the question as well as the labelled one
    SEBAGIAN    it answers part of it
    TIDAK       it does not answer it

`import` records every verdict on the case (`alternative_reviews`) and adds the clause of
each JUGA_BENAR to the case's `alternatives`, which `run.resolve_targets` counts ONLY when
`EVAL_ALTERNATIVES=1`. The labels themselves are never changed, so the score against the
original labels stays reportable beside the score with reviewed alternatives.

! A bias to report, ✗ hide: only the engine's own misses are reviewed, so alternatives can
only raise the score. Report both numbers; never the second alone.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
QUERIES = HERE / "queries.json"
VERDICTS = {"JUGA_BENAR", "SEBAGIAN", "TIDAK"}
COLUMNS = ["id", "type", "query", "labelled_answer", "labelled_text", "kind", "engine_rank",
           "engine_answer", "engine_text", "similarity", "verdict", "note", "candidate_chunk"]


def _label_text(conn, ref) -> tuple[str, str]:
    about = ref["about"].strip()
    rows = conn.execute(
        "SELECT body FROM chunks WHERE regulation_type=%s AND regulation_number=%s AND year=%s"
        " AND article=%s AND (starts_with(upper(about), upper(%s)) OR starts_with(upper(%s), upper(about)))"
        " AND chapter NOT ILIKE 'PENJELASAN%%' ORDER BY seq NULLS LAST, id",
        (ref["regulation_type"], ref["regulation_number"], ref["year"], ref["article"],
         about, about)).fetchall()
    return (f"{ref['regulation_type']} {ref['regulation_number']}/{ref['year']} {ref['article']}"
            f" ({about[:80]})", f"[{ref['article']}] " + "\n".join(r[0] for r in rows)[:3000])


def export(a) -> None:
    import psycopg

    sys.path.insert(0, str(HERE))
    from e2e import HttpServer
    from run import resolve_targets

    spec = json.loads(QUERIES.read_text("utf-8"))["queries"]
    conn = psycopg.connect(a.dsn)
    cur = conn.cursor()
    srv = HttpServer(os.environ.get("VERA_HTTP", "http://127.0.0.1:8081"))
    rows, seen = [], 0
    for q in spec:
        if q["type"] in ("out_of_domain", "exact_ref") or not q.get("answer_articles"):
            continue
        targets = resolve_targets(cur, q)
        if not targets:
            continue
        seen += 1
        top = [r["id"] for r in srv.search(q["query"]).get("results", [])][:5]
        if not top or any(i in targets for i in top):
            continue
        sims = dict(cur.execute(
            "SELECT c.id, max(1 - (c.dense <=> t.dense)) FROM chunks c, chunks t"
            " WHERE c.id = ANY(%s) AND t.id = ANY(%s) GROUP BY c.id",
            (top, list(targets))).fetchall())
        best = max(top, key=lambda i: sims.get(i, 0.0))
        kind, pick = ("MIRIP", best) if sims.get(best, 0.0) >= a.similar else ("TERATAS", top[0])
        if any(r.get("chunk") == pick for r in q.get("alternative_reviews", [])):
            continue
        reg, num, year, about, article, chapter, body = cur.execute(
            "SELECT regulation_type, regulation_number, year, about, article, chapter, body"
            " FROM chunks WHERE id = %s", (pick,)).fetchone()
        labels, texts = zip(*(_label_text(conn, ref) for ref in q["answer_articles"]))
        rows.append([q["id"], q["type"], q["query"], "\n".join(labels), "\n\n".join(texts),
                     kind, top.index(pick) + 1,
                     f"{reg} {num}/{year} {article or chapter or ''} ({(about or '')[:80]})",
                     body[:3000], f"{sims.get(pick, 0.0):.3f}", "", "", pick])
    rows.sort(key=lambda r: (r[5] != "MIRIP", r[0]))
    with open(a.out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        w.writerows(rows)
    mirip = sum(r[5] == "MIRIP" for r in rows)
    print(f"{seen} answerable questions · {len(rows)} misses to review · MIRIP {mirip}"
          f" · TERATAS {len(rows) - mirip} · wrote {a.out}")


def import_(a) -> None:
    import psycopg

    spec = json.loads(QUERIES.read_text("utf-8"))
    by_id = {q["id"]: q for q in spec["queries"]}
    conn = psycopg.connect(a.dsn)
    done = added = skipped = 0
    today = dt.date.today().isoformat()
    with open(a.sheet, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            v = (row.get("verdict") or "").strip().upper()
            if not v:
                skipped += 1
                continue
            if v not in VERDICTS:
                raise SystemExit(f"{row['id']}: verdict {v!r} is not one of {sorted(VERDICTS)}")
            q, chunk = by_id[row["id"]], row["candidate_chunk"]
            q.setdefault("alternative_reviews", []).append(
                {"chunk": chunk, "verdict": v, "note": (row.get("note") or "").strip(),
                 "reviewer": a.reviewer, "date": today})
            done += 1
            if v != "JUGA_BENAR":
                continue
            reg, num, year, about, article = conn.execute(
                "SELECT regulation_type, regulation_number, year, about, article FROM chunks"
                " WHERE id = %s", (chunk,)).fetchone()
            # ! The ARTICLE, as labels are, ✗ the chunk: any chunk of it counts, and a
            # chunk id does not survive re-chunking. A clause with no article label is kept
            # by chunk id, the only address it has.
            alt = ({"regulation_type": reg, "regulation_number": num, "year": year,
                    "about": about or "", "article": article} if article else {"chunk": chunk})
            q.setdefault("alternatives", []).append({**alt, "reviewer": a.reviewer, "date": today})
            added += 1
    QUERIES.write_text(json.dumps(spec, ensure_ascii=False, indent=2), "utf-8", newline="\n")
    print(f"imported {done} verdicts ({skipped} rows left blank) · {added} alternative answers added")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--dsn", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--similar", type=float, default=0.85)
    i = sub.add_parser("import")
    i.add_argument("--dsn", required=True)
    i.add_argument("--sheet", required=True)
    i.add_argument("--reviewer", required=True)
    a = ap.parse_args()
    export(a) if a.cmd == "export" else import_(a)


if __name__ == "__main__":
    main()
