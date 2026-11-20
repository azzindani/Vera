"""Offline: a vocabulary bridge the corpus declares, tried on the sparse arm.

    POOLS=best_pools.json PAIRS=bridge_pairs.json DATABASE_URL=... \
        python -I dev_tools/eval/bridge_sim.py

Indonesian regulations define their own short forms: "Badan Penyelenggara Jaminan Sosial
yang selanjutnya disingkat BPJS", "... yang selanjutnya disebut Perda". Mined from every
chunk that says so, a short form is kept only when one long form owns it: at least
MIN_DOCS regulations and SHARE of them agree. "Menteri" or "Daerah", whose long form
changes with every regulation, fall out by themselves.

A question that contains one form gets the other appended for BM25 (`pg_search`, as the
sparse arm runs it). MODES picks which pairs and which direction:

* `all`       every pair, both ways
* `acr`       acronyms only ("BPJS", "DPPA-SKPD": one token in capitals), both ways
* `acr>long`  an acronym in the question adds its long form
* `long>acr`  a long form in the question adds its acronym

Reports, over the questions the bridge touches: where the labelled answer ranks in
BM25's top 20 (the arm's depth) and top 200, before and after, with wins and losses;
and the same for the misses `miss_diag.py` wrote (MISSES, optional).
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

import psycopg

MIN_DOCS = int(os.environ.get("MIN_DOCS", "3"))
SHARE = float(os.environ.get("SHARE", "0.6"))
DEFINE = re.compile(
    r"(?:^|[.;:\n]|\d+\.)\s*([A-Z][^.;:\n]{2,150}?)\s*,?\s+yang\s+selanjutnya\s+"
    r"(?:disingkat|disebut)(?:\s+dengan)?\s+[\"“]?([A-Z][\w/&-]*(?:\s+[A-Z][\w/&-]*){0,3})"
)


def mine(cur) -> dict[str, dict]:
    path = Path(os.environ.get("PAIRS", "bridge_pairs.json"))
    if path.exists():
        return json.loads(path.read_text("utf-8"))
    cur.execute("SET statement_timeout = 0")
    cur.execute("SELECT doc_id, body FROM chunks"
                " WHERE body ~ 'yang selanjutnya (disingkat|disebut)'")
    owners: dict[str, Counter] = defaultdict(Counter)
    spelled: dict[str, Counter] = defaultdict(Counter)
    seen: set[tuple[str, str, str]] = set()
    for doc, body in cur.fetchall():
        for long, short in DEFINE.findall(body):
            long = " ".join(long.split()).lower()
            short = " ".join(short.split())
            if not 2 <= len(long.split()) <= 12 or len(short) < 3 or short.lower() == long:
                continue
            key = (doc, short.lower(), long)
            if key in seen:
                continue
            seen.add(key)
            owners[short.lower()][long] += 1
            spelled[short.lower()][short] += 1
    pairs = {}
    for short, longs in owners.items():
        (long, n), total = longs.most_common(1)[0], sum(longs.values())
        if n >= MIN_DOCS and n / total >= SHARE:
            form = spelled[short].most_common(1)[0][0]
            # ! An acronym is one token in capitals ("BPJS", "DPPA-SKPD"); "Pemerintah" or
            # "Dinas Kesehatan" is an alias, a word the question uses in its plain sense.
            acronym = " " not in form and form.upper() == form and any(ch.isalpha() for ch in form)
            pairs[short] = {"long": long, "acronym": acronym}
    path.write_text(json.dumps(pairs, ensure_ascii=False, indent=0), "utf-8")
    return pairs


def bridged(question: str, pairs: dict, mode: str) -> list[str]:
    q = question.lower()
    added = []
    for short, p in pairs.items():
        if mode != "all" and not p["acronym"]:
            continue
        long = p["long"]
        has_short = re.search(r"(?<![\w-])" + re.escape(short) + r"(?![\w-])", q)
        has_long = long in q
        if has_short and not has_long and mode in ("all", "acr", "acr>long"):
            added.append(long)
        elif has_long and not has_short and mode in ("all", "acr", "long>acr"):
            added.append(short)
    return added


def bm25(cur, text: str) -> list[str]:
    cur.execute("SELECT s.id FROM (SELECT id, paradedb.score(id) AS sc FROM chunks"
                " WHERE id @@@ paradedb.match('body', %s) ORDER BY paradedb.score(id) DESC"
                " LIMIT 400) s JOIN chunks c ON c.id = s.id WHERE c.indexable"
                " ORDER BY s.sc DESC LIMIT 200", (text,))
    return [r[0] for r in cur.fetchall()]


def report(label: str, rs: list[dict]) -> None:
    if not rs:
        print(f"  [{label}] touched 0")
        return
    line = f"  [{label}] touched {len(rs)}"
    for cut in (20, 200):
        def inside(x):
            return bool(x and x <= cut)
        b = sum(inside(r["before"]) for r in rs)
        a = sum(inside(r["after"]) for r in rs)
        win = sum(inside(r["after"]) and not inside(r["before"]) for r in rs)
        loss = sum(inside(r["before"]) and not inside(r["after"]) for r in rs)
        line += f"   BM25 top {cut}: {b} -> {a} (+{win} -{loss})"
    print(line)


def main() -> None:
    cur = psycopg.connect(os.environ["DATABASE_URL"]).cursor()
    pairs = mine(cur)
    print(f"pairs kept: {len(pairs)} (min {MIN_DOCS} regulations, share {SHARE}), "
          f"acronyms {sum(p['acronym'] for p in pairs.values())}")
    cases = json.loads(Path(os.environ["POOLS"]).read_text("utf-8"))
    spec = {q["id"]: q for q in json.loads((Path(__file__).parent / "queries.json")
                                           .read_text("utf-8"))["queries"]}
    misses = set()
    if os.environ.get("MISSES"):
        misses = {r["id"] for r in json.loads(Path(os.environ["MISSES"]).read_text("utf-8"))}

    cache: dict[str, list[str]] = {}

    def top(text: str) -> list[str]:
        if text not in cache:
            cache[text] = bm25(cur, text)
        return cache[text]

    for mode in os.environ.get("MODES", "all,acr,acr>long,long>acr").split(","):
        rows = []
        for c in cases:
            q = spec[c["id"]]["query"]
            added = bridged(q, pairs, mode)
            if not added:
                continue
            t = set(c["targets"])
            before, after = top(q), top(q + " " + " ".join(added))
            rank_b = next((k + 1 for k, i in enumerate(before) if i in t), None)
            rank_a = next((k + 1 for k, i in enumerate(after) if i in t), None)
            rows.append({"id": c["id"], "holdout": c["holdout"], "added": added,
                         "before": rank_b, "after": rank_a, "miss": c["id"] in misses})
        print(f"\nmode {mode}")
        report("all", rows)
        report("holdout", [r for r in rows if r["holdout"]])
        report("misses", [r for r in rows if r["miss"]])
        for r in rows[:6]:
            print(f"    {r['id']} {r['before']} -> {r['after']}  + {r['added']}")


if __name__ == "__main__":
    main()
