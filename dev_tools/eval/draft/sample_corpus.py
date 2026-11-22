"""Sample clauses from ANY served corpus for drafting a citation-labelled eval set.

    PGPASSWORD=... python -I dev_tools/eval/draft/sample_corpus.py \
        --dsn "host=... dbname=vera user=vera" --config draft/corpus_ca_laws.json --out <dir>

`sample.py` is the Indonesian sampler, with its strata and markers written in Python. This
one reads them from a config file, one per corpus, in that corpus's language:

    {"corpus": "ca_laws", "seed": 20261122, "min_chars": 200,
     "strata": [{"types": ["Act"], "share": 0.45}, {"types": ["SOR"], "share": 0.45},
                {"types": "*", "share": 0.10}],
     "types": {"definitional": {"marker": "\\\\bmeans\\\\b", "n": 14}, ...},
     "pairs": {"hard_negative": 10, "underspecified": 6}}

It writes one `<type>.json` per type: the clause a drafter turns into a question, with the
citation (`identifier`) and the instrument number (`regulation_number`) the label names
(`answer_identifiers`, `answer_documents`; `run.resolve_targets`). As in `sample.py`, one
clause per document, a type is sampled only where its marker appears, and two types take
more than one clause:

* `hard_negative`: a clause and its nearest neighbour in ANOTHER document (cosine distance
  0.02-0.12), so the question must discriminate between them.
* `underspecified`: a clause with at least three near-identical twins (distance < 0.03) in
  other documents, for a generic question every twin answers.

! Nothing here writes a question. The drafter writes them under `EVAL.md` §2b, and every
draft is machine-checked (`check.py`) before it is scored.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import psycopg

COLS = "id, doc_id, identifier, regulation_type, regulation_number, about, heading_path, body"


def row_dict(r: tuple) -> dict:
    keys = [c.strip() for c in COLS.split(",")]
    return dict(zip(keys, r, strict=True))


def candidates(conn, types, marker: str, limit: int, seed: int, min_chars: int) -> list[tuple]:
    """Servable chunks of these instrument types whose body matches `marker`, in a seeded
    random order. `types == "*"` is every type the other strata do not name."""
    where = ["length(body) >= %s", "body ~* %s", "identifier <> ''"]
    params: list = [min_chars, marker]
    if isinstance(types, list):
        where.append("regulation_type = ANY(%s)")
        params.append(types)
    elif isinstance(types, dict):  # {"not": [...]}
        where.append("NOT (regulation_type = ANY(%s))")
        params.append(types["not"])
    sql = (f"SELECT {COLS} FROM chunks WHERE {' AND '.join(where)}"
           f" ORDER BY md5(id || %s) LIMIT %s")
    with conn.cursor() as cur:
        cur.execute(sql, (*params, str(seed), limit))
        return cur.fetchall()


def take(rows, n: int, seen: set[str]) -> list[tuple]:
    out = []
    for r in rows:
        if r[1] in seen:
            continue
        seen.add(r[1])
        out.append(r)
        if len(out) == n:
            break
    return out


def neighbours(conn, row, k: int = 8) -> list[tuple]:
    """The nearest chunks in other documents, with their cosine distance."""
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT {', '.join('c.' + x.strip() for x in COLS.split(','))},"
            " c.dense <=> q.dense AS d FROM chunks c,"
            " (SELECT dense FROM chunks WHERE id = %s) q"
            " WHERE c.doc_id <> %s ORDER BY c.dense <=> q.dense LIMIT %s",
            (row[0], row[1], k),
        )
        return cur.fetchall()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    seed, min_chars = cfg["seed"], cfg.get("min_chars", 200)
    rng = random.Random(seed)
    named = [t for s in cfg["strata"] if isinstance(s["types"], list) for t in s["types"]]
    strata = [({"not": named} if s["types"] == "*" else s["types"], s["share"])
              for s in cfg["strata"]]
    args.out.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    with psycopg.connect(args.dsn) as conn:
        for name, t in cfg["types"].items():
            picked: list[tuple] = []
            for types, share in strata:
                want = max(1, round(t["n"] * share))
                rows = candidates(conn, types, t["marker"], 40 * want, seed, min_chars)
                picked += take(rows, want, seen)
            rng.shuffle(picked)
            cases = [row_dict(r) for r in picked[: t["n"]]]
            (args.out / f"{name}.json").write_text(
                json.dumps(cases, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"{name:14} {len(cases)}")

        pairs = cfg.get("pairs", {})
        pool = candidates(conn, "*", ".", 4000, seed + 1, min_chars)
        hard, under = [], []
        for r in pool:
            if len(hard) >= pairs.get("hard_negative", 0) and \
               len(under) >= pairs.get("underspecified", 0):
                break
            if r[1] in seen:
                continue
            near = neighbours(conn, r)
            twins = [n for n in near if n[-1] < 0.03]
            other = next((n for n in near if 0.02 <= n[-1] <= 0.12), None)
            if len(twins) >= 3 and len(under) < pairs.get("underspecified", 0):
                seen.add(r[1])
                under.append({"clause": row_dict(r), "twins": [row_dict(n[:-1]) for n in twins]})
            elif other and len(hard) < pairs.get("hard_negative", 0):
                seen.add(r[1])
                hard.append({"clause": row_dict(r), "neighbour": row_dict(other[:-1]),
                             "distance": round(float(other[-1]), 4)})
        for name, cases in (("hard_negative", hard), ("underspecified", under)):
            (args.out / f"{name}.json").write_text(
                json.dumps(cases, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"{name:14} {len(cases)}")


if __name__ == "__main__":
    main()
