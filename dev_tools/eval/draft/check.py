"""Machine checks on drafted cases, then merge into `queries.json` (`ROADMAP.md` Stage 0).

    PGPASSWORD=... python -I dev_tools/eval/draft/check.py build  --dsn ... --drafts <out dir> --staged <file>
    PGPASSWORD=... python -I dev_tools/eval/draft/check.py merge  --staged <file> --verdicts <file>

`build` turns each drafted case into a labelled case:

* **Labels come from the database, never from the drafter.** The drafter names chunk ids;
  the regulation, number, year, title and article of each are read back from `chunks`.
  A typo in a drafted label cannot exist.
* **Leakage**: a query sharing 6 or more consecutive words with its clause is rejected
  (the rule given to drafters is 4; 4-5 is reported as a warning). Outside `exact_ref`, a
  query naming the regulation's number together with its year is rejected.
* **Resolution**: the article label must resolve, through `run.resolve_targets`, to the
  chunk the case was drafted from. A label the eval cannot score is not a label.
* **near_duplicates**: documents holding a clause within cosine distance 0.03 of the
  answer, other than the labelled ones. A hit on such a twin scores as a miss, so `e2e.py`
  reports Recall@5 with and without these cases.
* **Ids and split**: new in-domain cases are `s001...`, new out-of-domain `o041...`.
  `holdout` is decided once, by md5 of the id (36 of 100 for in-domain, so the set as a
  whole is about 30% holdout; 30 of 100 for out-of-domain) and never changed afterwards.

`staged` also carries each case's clause text, for the second reader. `merge` keeps only
the cases the second reader marked `answers`, and writes them into `queries.json` with
their verdict in `provenance`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import psycopg

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run import resolve_targets  # noqa: E402

QUERIES = Path(__file__).resolve().parents[1] / "queries.json"
WORD = re.compile(r"\w+")
META = ("SELECT id, doc_id, regulation_type, regulation_number, year, about, article, body, "
        "enacting_body, cluster_id FROM chunks WHERE id = ANY(%s)")
TODAY = __import__("datetime").date.today().isoformat()


def longest_run(a: str, b: str) -> int:
    x, y = WORD.findall(a.lower()), WORD.findall(b.lower())
    best, prev = 0, [0] * (len(y) + 1)
    for i in range(1, len(x) + 1):
        cur = [0] * (len(y) + 1)
        for j in range(1, len(y) + 1):
            if x[i - 1] == y[j - 1]:
                cur[j] = prev[j - 1] + 1
                best = max(best, cur[j])
        prev = cur
    return best


def held(qid: str, per100: int) -> bool:
    return int(hashlib.md5(qid.encode()).hexdigest(), 16) % 100 < per100


def near_duplicates(conn, row, exclude_docs: set[str]) -> int:
    rows = conn.execute(
        "SELECT c.doc_id FROM chunk_vectors v JOIN chunks c ON c.id = v.id, "
        "(SELECT dense FROM chunks WHERE id = %s) q "
        "WHERE v.cluster_id = %s AND v.dense <=> q.dense < 0.03",
        (row["id"], row["cluster_id"])).fetchall()
    return len({d for (d,) in rows} - exclude_docs)


def build(a) -> None:
    conn = psycopg.connect(a.dsn)
    cur = conn.cursor()
    spec = json.loads(QUERIES.read_text("utf-8"))
    known = {q["query"].strip().lower() for q in spec["queries"]}
    n_s = max(int(q["id"][1:]) for q in spec["queries"] if q["id"].startswith("s"))
    n_o = max(int(q["id"][1:]) for q in spec["queries"] if q["id"].startswith("o"))
    staged, rejected, skips = [], [], []
    for path in sorted(Path(a.drafts).glob("*.json")):
        for d in json.loads(path.read_text("utf-8")):
            if "skip" in d:
                skips.append({"file": path.name, **d})
                continue
            if d["query"].strip().lower() in known:
                rejected.append({**d, "why": "duplicate query"})
                continue
            known.add(d["query"].strip().lower())
            if d["type"] == "out_of_domain":
                n_o += 1
                qid = f"o{n_o:03d}"
                staged.append({"id": qid, "type": "out_of_domain", "subtype": d.get("subtype"),
                               "difficulty": d.get("difficulty"), "query": d["query"],
                               "answer_chunks": [], "source": d.get("source", "not answerable from this corpus"),
                               "provenance": f"written {TODAY} (stage 0)", "holdout": held(qid, 30)})
                continue
            ids = [d["chunk_id"], *d.get("extra_chunk_ids", [])]
            rows = {r[0]: dict(zip(("id", "doc_id", "regulation_type", "regulation_number", "year",
                                     "about", "article", "body", "enacting_body", "cluster_id"), r))
                    for r in cur.execute(META, (ids,)).fetchall()}
            if d["chunk_id"] not in rows:
                rejected.append({**d, "why": "chunk id not in corpus"})
                continue
            main = rows[d["chunk_id"]]
            run = longest_run(d["query"], main["body"])
            if run >= 6:
                rejected.append({**d, "why": f"copies {run} words of its clause"})
                continue
            if d["type"] != "exact_ref" and re.search(
                    rf"\b{re.escape(str(main['regulation_number']))}\b", d["query"]) and str(main["year"]) in d["query"]:
                rejected.append({**d, "why": "names the regulation's number and year"})
                continue
            missing = [i for i in ids if i not in rows]
            if missing:
                rejected.append({**d, "why": f"extra chunk ids not in corpus: {missing}"})
                continue
            # ! A chunk with no title cannot be labelled: `resolve_targets` matches on the
            # title, so its label would score nothing. Such a twin or parent is dropped.
            if not (main["about"] or "").strip():
                rejected.append({**d, "why": "answer chunk has no title, cannot be labelled"})
                continue
            ids = [i for i in ids if (rows[i]["about"] or "").strip()]
            if d["type"] == "multi_tier" and len(ids) < 2:
                rejected.append({**d, "why": "parent has no title, cannot be labelled"})
                continue
            if d["type"] == "underspecified" and len(ids) < 4:
                rejected.append({**d, "why": "fewer than 3 labellable twins"})
                continue
            articles, seen = [], set()
            for i in ids:
                r = rows[i]
                key = (r["regulation_type"], r["regulation_number"], r["year"], r["about"], r["article"])
                if key not in seen:
                    seen.add(key)
                    articles.append({"regulation_type": r["regulation_type"],
                                     "regulation_number": r["regulation_number"], "year": r["year"],
                                     "about": r["about"], "article": r["article"]})
            n_s += 1
            qid = f"s{n_s:03d}"
            case = {"id": qid, "type": d["type"], "difficulty": d.get("difficulty"), "query": d["query"],
                    "source": f"{main['regulation_type']} {main['regulation_number']}/{main['year']} "
                              f"{main['article']} — {d.get('subject', '')}".strip(),
                    "answer_articles": articles}
            if d.get("must_not_rank_first"):
                case["must_not_rank_first"] = d["must_not_rank_first"]
            if d.get("note"):
                case["note"] = d["note"]
            targets = resolve_targets(cur, case)
            if d["chunk_id"] not in targets:
                n_s -= 1
                rejected.append({**d, "why": "label does not resolve to its own chunk"})
                continue
            if d["type"] != "underspecified":
                nd = near_duplicates(conn, main, {rows[i]["doc_id"] for i in ids})
                if nd:
                    case["near_duplicates"] = nd
            case["provenance"] = (f"drafted {TODAY} from chunk {d['chunk_id']} of id_reg_markdown (stage 0); "
                                  f"machine-checked; NOT reviewed by a domain expert")
            case["holdout"] = held(qid, 36)
            staged.append({**case, "_check": {
                "longest_copied_run": run,
                "clauses": [{"id": i, "role": "answer", "regulation": f"{rows[i]['regulation_type']} "
                             f"{rows[i]['regulation_number']}/{rows[i]['year']}", "about": rows[i]["about"],
                             "region": rows[i]["enacting_body"], "article": rows[i]["article"],
                             "body": rows[i]["body"]} for i in ids]}})
            for foil in d.get("must_not_rank_first", []):
                r = cur.execute(META, ([foil],)).fetchone()
                if r:
                    staged[-1]["_check"]["clauses"].append(
                        {"id": r[0], "role": "foil", "regulation": f"{r[2]} {r[3]}/{r[4]}", "about": r[5],
                         "region": r[8], "article": r[6], "body": r[7]})
    Path(a.staged).write_text(json.dumps({"staged": staged, "rejected": rejected, "skips": skips},
                                         ensure_ascii=False, indent=1), "utf-8")
    by: dict[str, int] = {}
    for s in staged:
        by[s["type"]] = by.get(s["type"], 0) + 1
    print(f"staged {len(staged)} · rejected {len(rejected)} · skipped by drafters {len(skips)}")
    print("  " + " · ".join(f"{k} {v}" for k, v in sorted(by.items())))
    for r in rejected:
        print(f"  ✗ {r.get('chunk_id', '')} {r['why']}")
    mismatch = [s for s in skips if "metadata" in s.get("why", "").lower()]
    print(f"  drafters flagged {len(mismatch)} metadata mismatches")


def merge(a) -> None:
    """Merge staged cases. Without `--verdicts`, every case not in `--drop` is merged with
    `--note` (what second reading it had, e.g. a sample) appended to its provenance."""
    staged = json.loads(Path(a.staged).read_text("utf-8"))["staged"]
    verdicts = ({v["id"]: v for v in json.loads(Path(a.verdicts).read_text("utf-8"))}
                if a.verdicts else None)
    drop = set(a.drop.split(",")) if a.drop else set()
    spec = json.loads(QUERIES.read_text("utf-8"))
    kept = dropped = 0
    for s in staged:
        s = {k: v for k, v in s.items() if k != "_check"}
        if s["id"] in drop:
            dropped += 1
            continue
        if verdicts is not None:
            v = verdicts.get(s["id"])
            want = "out_of_domain" if s["type"] == "out_of_domain" else "answers"
            if not v or v["verdict"] != want:
                dropped += 1
                continue
            if s["type"] != "out_of_domain":
                s["provenance"] += f"; second reader: answers ({v.get('why', '')[:120]})"
        elif a.note and s["type"] != "out_of_domain":
            s["provenance"] += f"; {a.note}"
        spec["queries"].append(s)
        kept += 1
    QUERIES.write_text(json.dumps(spec, ensure_ascii=False, indent=2), "utf-8", newline="\n")
    print(f"merged {kept} · dropped by the second reader {dropped} · queries.json now {len(spec['queries'])}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--dsn", required=True)
    b.add_argument("--drafts", required=True)
    b.add_argument("--staged", required=True)
    m = sub.add_parser("merge")
    m.add_argument("--staged", required=True)
    m.add_argument("--verdicts")
    m.add_argument("--drop", default="", help="comma-separated staged ids to leave out")
    m.add_argument("--note", default="", help="appended to each in-domain provenance")
    a = ap.parse_args()
    build(a) if a.cmd == "build" else merge(a)


if __name__ == "__main__":
    main()
