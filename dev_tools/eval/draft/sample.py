"""Sample clauses from the deployed corpus for drafting eval cases (`ROADMAP.md` Stage 0).

    PGPASSWORD=... python -I dev_tools/eval/draft/sample.py --dsn "host=... dbname=vera user=vera" \
        --out <dir> [--seed 20261114]

Writes one `<type>.json` per question type: the clause(s) a drafter turns into a question,
with the metadata the label needs. Nothing here writes a question; that is the drafter's
job, under the rules in `EVAL.md` §2b, and every draft is then machine-checked
(`check.py`) and read against its clause by a second, independent reader.

Strata, as in the 2026-11-14 expansion: 45% national, 15% provincial, 40% regency/city;
one clause per document; documents already labelled in `queries.json` excluded. A type
is sampled where its marker appears (a sanction clause says *pidana* or *denda*), so the
drafter is not asked to find an exception in a clause that has none. Three types need
more than one clause:

* `multi_tier`: an implementing clause that cites `Pasal N Undang-Undang Nomor X Tahun Y`,
  paired with that article of that law when the corpus holds it.
* `hard_negative`: a clause and its nearest neighbour from another document about a
  different subject (cosine distance 0.02-0.12), so the question must discriminate.
* `underspecified`: a clause with at least three near-identical twins (distance < 0.03)
  in other documents; a generic question that every twin answers.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import psycopg

NATIONAL = ["UNDANG-UNDANG", "PERATURAN PEMERINTAH", "PERATURAN PEMERINTAH PENGGANTI UNDANG-UNDANG",
            "PERATURAN PRESIDEN", "KEPUTUSAN PRESIDEN", "INSTRUKSI PRESIDEN"]
PROVINCIAL = ["PERATURAN DAERAH PROVINSI", "PERATURAN GUBERNUR"]
LOCAL = ["PERATURAN DAERAH KABUPATEN", "PERATURAN BUPATI", "PERATURAN WALIKOTA",
         "PERATURAN DAERAH KOTA", "PERATURAN WALI KOTA", "QANUN KABUPATEN"]
LEVELS = [(NATIONAL, 0.45), (PROVINCIAL, 0.15), (LOCAL, 0.40)]

# type -> (marker regex on the body, clauses wanted)
TYPES = {
    "numeric": (r"(\d+\s*\([a-z ]+\)\s*(hari|bulan|tahun)|persen|%|Rp\s?\d)", 14),
    "conceptual": (r".", 30),
    "procedural": (r"(tata cara|permohonan|diajukan|prosedur|tahapan)", 34),
    "authority": (r"(berwenang|kewenangan|menetapkan|memberikan izin|ditetapkan oleh)", 34),
    "exception": (r"(kecuali|dikecualikan|tidak berlaku bagi|tidak termasuk|dibebaskan dari)", 46),
    "obligation": (r"\mwajib\M", 50),
    "sanction": (r"(pidana penjara|pidana kurungan|denda paling|sanksi administratif)", 58),
    "definitional": (r"yang dimaksud dengan", 64),
    "exact_ref": (r".", 80),
}
SPECIAL = {"multi_tier": 60, "hard_negative": 60, "underspecified": 60}

COLS = ("id, doc_id, regulation_type, regulation_number, year, about, enacting_body, "
        "article, body, cluster_id")
CITE = re.compile(r"Pasal\s+(\d+[A-Z]?)[^.;]{0,60}?Undang-Undang\s+Nomor\s+(\d+)\s+Tahun\s+(\d{4})",
                  re.IGNORECASE)


def row_dict(r) -> dict:
    keys = [c.strip() for c in COLS.split(",")]
    d = dict(zip(keys, r))
    d.pop("cluster_id", None)
    return d


def used_docs(conn, queries: Path) -> set[str]:
    """Documents already labelled, so a new case never re-tests a known document."""
    spec = json.loads(queries.read_text("utf-8"))["queries"]
    out: set[str] = set()
    for q in spec:
        for ref in q.get("answer_articles", []):
            about = ref["about"].strip()
            for (d,) in conn.execute(
                "SELECT DISTINCT doc_id FROM chunks WHERE regulation_type=%s AND regulation_number=%s"
                " AND year=%s AND (starts_with(upper(about), upper(%s)) OR starts_with(upper(%s), upper(about)))",
                (ref["regulation_type"], ref["regulation_number"], ref["year"], about, about)):
                out.add(d)
    return out


def candidates(conn, types: list[str], marker: str, limit: int, seed: int,
               lo: int = 400, hi: int = 2000, pct: float = 4) -> list[tuple]:
    return conn.execute(
        f"SELECT {COLS} FROM chunks TABLESAMPLE SYSTEM ({pct}) REPEATABLE (%s) "
        # ! Operative text only: an explanation is filed under the article it explains.
        "WHERE indexable AND article LIKE 'Pasal %%' AND chapter NOT LIKE 'PENJELASAN%%' "
        "AND about <> '' "
        "AND regulation_type = ANY(%s) AND length(body) BETWEEN %s AND %s AND body ~* %s "
        "LIMIT %s",
        (seed, types, lo, hi, marker, limit)).fetchall()


def take(rows, n: int, seen: set[str], rng: random.Random) -> list[tuple]:
    rows = list(rows)
    rng.shuffle(rows)
    out = []
    for r in rows:
        if len(out) == n:
            break
        if r[1] in seen:
            continue
        seen.add(r[1])
        out.append(r)
    return out


def neighbours(conn, row, k: int = 8) -> list[tuple]:
    """Nearest chunks in the clause's own cluster (exact, via the routed layout)."""
    return conn.execute(
        f"SELECT {', '.join('c.' + c.strip() for c in COLS.split(','))}, v.dense <=> q.dense AS dist "
        "FROM chunk_vectors v JOIN chunks c ON c.id = v.id, "
        "(SELECT dense FROM chunks WHERE id = %s) q "
        "WHERE v.cluster_id = %s AND v.id <> %s ORDER BY v.dense <=> q.dense LIMIT %s",
        (row[0], row[9], row[0], k)).fetchall()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--queries", type=Path, default=Path(__file__).parents[1] / "queries.json")
    ap.add_argument("--seed", type=int, default=20261114)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    conn = psycopg.connect(a.dsn)
    seen = used_docs(conn, a.queries)
    print(f"{len(seen)} documents already labelled, excluded")
    a.out.mkdir(parents=True, exist_ok=True)

    for t, (marker, want) in TYPES.items():
        picked = []
        for i, (types, share) in enumerate(LEVELS):
            n = round(want * share)
            picked += take(candidates(conn, types, marker, n * 12, a.seed + i), n, seen, rng)
        (a.out / f"{t}.json").write_text(
            json.dumps([{"clause": row_dict(r)} for r in picked], ensure_ascii=False, indent=1), "utf-8")
        print(f"  {t:<14} {len(picked)} clauses")

    # multi_tier: an implementing clause and the law article it cites
    pairs = []
    rows = candidates(conn, NATIONAL[1:] + PROVINCIAL + LOCAL,
                      r"Pasal\s+\d+[^.;]{0,60}Undang-Undang\s+Nomor\s+\d+\s+Tahun", 20000, a.seed + 10,
                      lo=300, hi=2500, pct=60)
    rng.shuffle(rows)
    for r in rows:
        if len(pairs) == SPECIAL["multi_tier"]:
            break
        if r[1] in seen:
            continue
        m = CITE.search(r[8])
        if not m:
            continue
        law = conn.execute(
            f"SELECT {COLS} FROM chunks WHERE regulation_type='UNDANG-UNDANG' AND regulation_number=%s"
            " AND year=%s AND article=%s AND indexable AND chapter NOT LIKE 'PENJELASAN%%'"
            " AND length(body) BETWEEN 200 AND 2500 LIMIT 1",
            (m.group(2), int(m.group(3)), f"Pasal {m.group(1)}")).fetchone()
        if not law or law[1] in seen:
            continue
        seen.update({r[1], law[1]})
        pairs.append({"clause": row_dict(r), "parent": row_dict(law)})
    (a.out / "multi_tier.json").write_text(json.dumps(pairs, ensure_ascii=False, indent=1), "utf-8")
    print(f"  multi_tier     {len(pairs)} pairs")

    # hard_negative and underspecified both start from a clause's nearest neighbours
    hard, under = [], []
    pool = []
    for i, (types, _share) in enumerate(LEVELS):
        pool += candidates(conn, types, ".", 1500, a.seed + 20 + i)
    rng.shuffle(pool)
    for r in pool:
        if len(hard) >= SPECIAL["hard_negative"] and len(under) >= SPECIAL["underspecified"]:
            break
        if r[1] in seen:
            continue
        near = neighbours(conn, r)
        others = [n for n in near if n[1] != r[1]]
        twins = [n for n in others if n[-1] < 0.03]
        if len(twins) >= 3 and len(under) < SPECIAL["underspecified"]:
            seen.add(r[1])
            under.append({"clause": row_dict(r),
                          "twins": [dict(row_dict(n[:-1]), dist=round(n[-1], 4)) for n in twins]})
            continue
        foil = next((n for n in others if 0.02 <= n[-1] <= 0.12
                     and (n[5] or "").upper() != (r[5] or "").upper()), None)
        if foil and len(hard) < SPECIAL["hard_negative"]:
            seen.update({r[1], foil[1]})
            hard.append({"clause": row_dict(r), "foil": dict(row_dict(foil[:-1]), dist=round(foil[-1], 4))})
    (a.out / "hard_negative.json").write_text(json.dumps(hard, ensure_ascii=False, indent=1), "utf-8")
    (a.out / "underspecified.json").write_text(json.dumps(under, ensure_ascii=False, indent=1), "utf-8")
    print(f"  hard_negative  {len(hard)} pairs\n  underspecified {len(under)} groups")


if __name__ == "__main__":
    main()
