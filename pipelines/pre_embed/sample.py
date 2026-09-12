"""Select the corpus subset to ingest, and record exactly what was selected.

Two principles, both learned the hard way from the August/November drift:

1. Sample whole REGULATIONS, never rows. At ~45.7 chunks per regulation a
   row-based sample shreds documents -- Pasal 9 without Pasal 8 -- which breaks
   neighbour reads and makes provenance tests meaningless.

2. Write a manifest. "Which 4,380 regulations were in that run?" has to be
   answerable later or the remaining 548K cannot be continued cleanly.

Deterministic: selection is ordered by a stable hash of the regulation key, so
the same target size always yields the same set. No RANDOM().

Usage:
    python sample.py --target-chunks 5000  --out fixture.json
    python sample.py --target-chunks 200000 --out spike.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

DB = Path(__file__).resolve().parents[2] / ".test" / "ID_REG_DB_2511" / "id_regulations.db"

# Deliberately NOT the corpus's natural distribution. 35% of the corpus is
# regency bylaws and only 5.2% is UNDANG-UNDANG, but national-tier regulations
# are what users actually query. Over-sample them so the eval measures the
# queries that matter; keep enough bylaws that near-duplicate pressure is real.
STRATA = {
    "UNDANG-UNDANG": 0.25,
    "PERATURAN PEMERINTAH": 0.20,
    "PERATURAN PRESIDEN": 0.10,
    "PERATURAN DAERAH KABUPATEN": 0.15,
    "PERATURAN BUPATI": 0.10,
    "PERATURAN DAERAH KOTA": 0.08,
    "PERATURAN WALIKOTA": 0.05,
    "PERATURAN GUBERNUR": 0.04,
    "PERATURAN DAERAH PROVINSI": 0.03,
}

REG_KEY = "regulation_type || '|' || regulation_number || '|' || year"


def stable_order(key: str) -> str:
    """Deterministic pseudo-random ordering · same input, same sample, forever."""
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def pick_regulations(con: sqlite3.Connection, target_chunks: int) -> dict:
    """Choose whole regulations per stratum until the chunk budget is met."""
    chosen: dict[str, dict] = {}
    per_stratum_chunks = {k: int(target_chunks * w) for k, w in STRATA.items()}

    for rtype, budget in per_stratum_chunks.items():
        rows = con.execute(
            f"""
            SELECT {REG_KEY} AS k, regulation_number, year, COUNT(*) AS n
            FROM regulations
            WHERE regulation_type = ?
            GROUP BY k
            HAVING n BETWEEN 3 AND 400
            """,
            (rtype,),
        ).fetchall()
        rows.sort(key=lambda r: stable_order(r[0]))

        taken = 0
        for k, number, year, n in rows:
            if taken >= budget:
                break
            chosen[k] = {"reason": f"stratum:{rtype}", "chunks": n}
            taken += n

    return chosen


def plant_edge_cases(con: sqlite3.Connection, chosen: dict) -> dict:
    """Add the cases that make the test suite worth running.

    Test data with one planted adversarial case beats ten times as much
    realistic data. Each of these guards a specific invariant.
    """
    # key -> [reasons]. One regulation often satisfies several cases at once
    # (the cukai example also contains boilerplate); a scalar would silently
    # overwrite, and the fixture would appear to be missing coverage it has.
    planted: dict[str, list[str]] = defaultdict(list)

    def add(key: str, reason: str):
        if not key:
            return
        if key not in chosen:
            n = con.execute(
                f"SELECT COUNT(*) FROM regulations WHERE {REG_KEY} = ?", (key,)
            ).fetchone()[0]
            chosen[key] = {"reason": reason, "chunks": n}
        # Record it either way · a case silently satisfied by the stratified
        # draw is still a case the fixture must be known to cover.
        if reason not in planted[key]:
            planted[key].append(reason)

    # A known-answer query used throughout development: excise sanctions.
    row = con.execute(
        f"SELECT {REG_KEY} FROM regulations WHERE about LIKE '%SANKSI ADMINISTRASI%'"
        " AND about LIKE '%CUKAI%' LIMIT 1"
    ).fetchone()
    if row:
        add(row[0], "known-answer:cukai-sanctions")

    # Hard negatives: same regulation_number, different year. These are what a
    # wrong-but-plausible answer looks like in a legal corpus.
    for key, in con.execute(
        f"""
        SELECT {REG_KEY} FROM regulations
        WHERE regulation_number IN (
            SELECT regulation_number FROM regulations
            GROUP BY regulation_number HAVING COUNT(DISTINCT year) > 3
        )
        GROUP BY {REG_KEY} LIMIT 6
        """
    ):
        add(key, "hard-negative:same-number-different-year")

    # Truncated at ingest (exactly 32767 chars) -- must surface as incomplete,
    # never be cited as a whole article.
    row = con.execute(
        f"SELECT {REG_KEY} FROM regulations WHERE LENGTH(content) = 32767 LIMIT 1"
    ).fetchone()
    if row:
        add(row[0], "edge:truncated-at-32767")

    # Partial locator: an article but no chapter. Every row in this corpus has
    # an article, so a fully empty locator does not occur -- the realistic case
    # is a half-populated one, which must still render a usable citation.
    row = con.execute(
        f"SELECT {REG_KEY} FROM regulations WHERE chapter = 'N/A'"
        " AND article IS NOT NULL AND article <> '' LIMIT 1"
    ).fetchone()
    if row:
        add(row[0], "edge:partial-locator-no-chapter")

    # A very short chunk -- snippet logic must not pad or ellipsise it.
    row = con.execute(
        f"SELECT {REG_KEY} FROM regulations WHERE LENGTH(content) < 60"
        " AND LENGTH(content) > 0 LIMIT 1"
    ).fetchone()
    if row:
        add(row[0], "edge:very-short-chunk")

    # "Cukup jelas." ("self-explanatory") -- boilerplate filling 44K+ rows of
    # explanatory notes. Semantically empty, so it must be stored and readable
    # but never indexed: 44K near-identical vectors would distort ranking.
    row = con.execute(
        f"SELECT {REG_KEY} FROM regulations WHERE TRIM(content) = 'Cukup jelas.' LIMIT 1"
    ).fetchone()
    if row:
        add(row[0], "edge:boilerplate-not-indexable")

    return dict(planted)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-chunks", type=int, default=5000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    con = sqlite3.connect(DB)
    chosen = pick_regulations(con, args.target_chunks)
    planted = plant_edge_cases(con, chosen)

    # Resolve to the actual chunk ids this run will ingest.
    #
    # ! ONE scan, filtered in Python. The composite regulation key is not an
    # indexed expression, so a per-regulation lookup is a full table scan each
    # time -- fine for 184 regulations, pathological for 4,400.
    ids: list[str] = []
    by_type: dict[str, int] = defaultdict(int)
    wanted = set(chosen)
    for gid, rtype, rnum, yr in con.execute(
        "SELECT global_id, regulation_type, regulation_number, year FROM regulations"
    ):
        if f"{rtype}|{rnum}|{yr}" in wanted:
            ids.append(gid)
    for key in chosen:
        by_type[key.split("|")[0]] += chosen[key]["chunks"]

    total_chunks = sum(c["chunks"] for c in chosen.values())
    manifest = {
        "created": datetime.now(timezone.utc).isoformat(),
        "source_db": str(DB),
        "target_chunks": args.target_chunks,
        "actual_chunks": total_chunks,
        "regulations": len(chosen),
        "by_regulation_type": dict(sorted(by_type.items(), key=lambda kv: -kv[1])),
        "planted_cases": planted,
        "selection": {k: v["reason"] for k, v in chosen.items()},
        "chunk_ids": sorted(ids, key=int),
    }

    Path(args.out).write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"regulations selected : {len(chosen):,}")
    print(f"chunks               : {total_chunks:,} (target {args.target_chunks:,})")
    cases = sorted({r for rs in planted.values() for r in rs})
    print(f"planted regulations  : {len(planted)}  covering {len(cases)} case types")
    for case in cases:
        keys = [k for k, rs in planted.items() if case in rs]
        print(f"    {case:<38} {len(keys):>2}x  e.g. {keys[0]}")
    print("\nby regulation_type:")
    for t, n in sorted(by_type.items(), key=lambda kv: -kv[1]):
        print(f"    {t:<32} {n:>7,}  {n / total_chunks:6.1%}")
    print(f"\nmanifest -> {args.out}")
    con.close()


if __name__ == "__main__":
    main()
