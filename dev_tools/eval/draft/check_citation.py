"""Machine checks on a citation-labelled set drafted from `sample_corpus.py` output.

    PGPASSWORD=... python -I dev_tools/eval/draft/check_citation.py \
        --dsn "host=... dbname=vera user=vera" --corpus ca_laws --prefix ca \
        --samples <sample dir> --drafts <drafts.json> [<drafts.json> ...] \
        --out dev_tools/eval/queries_ca_laws.json

`check.py` labels Indonesian articles by their fields. A corpus whose profile states a
citation per chunk is labelled by that citation instead (`answer_identifiers`, the
instrument in `answer_documents`; `run.resolve_targets`). The drafter writes only the
question, keyed by the clause it was drafted from:

    {"exact_ref:3": "What does section 17 of ... require ...?",
     "out_of_domain:general:0": "How do I make sourdough bread rise better?"}

`<type>:<n>` is clause n of `<samples>/<type>.json`; every key the drafts leave out is a
clause the drafter skipped. As in `check.py`:

* **Labels come from the database, never from the drafter**: the citation and the
  instrument are read back from `chunks` by the sampled chunk id.
* **Leakage**: 6 or more consecutive words shared with the clause is rejected; 4-5 is
  reported, for the drafter to reword. Outside `exact_ref`, a query naming the
  instrument's number (`SOR/2016-152`, `C-46`) is rejected.
* **Resolution**: the label must resolve to the clause it was drafted from.
* `hard_negative`: the neighbour is `must_not_rank_first`. `underspecified`: every twin is
  an answer, and at least three must be labellable.
* **Ids and split**: `<prefix>001...` in-domain, `<prefix>o01...` out-of-domain; holdout
  by md5 of the id, 36 of 100 in-domain and 30 of 100 out-of-domain, as in `check.py`.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date
from pathlib import Path

import psycopg

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from check import held, longest_run  # noqa: E402
from run import resolve_targets  # noqa: E402

META = ("SELECT id, doc_id, identifier, regulation_number, about, body FROM chunks "
        "WHERE id = ANY(%s)")
KEYS = ("id", "doc_id", "identifier", "regulation_number", "about", "body")


def near_twins(conn, chunk_id: str, exclude_docs: set[str]) -> int:
    """Documents, other than the labelled ones, holding a clause within distance 0.03."""
    rows = conn.execute(
        "SELECT c.doc_id FROM chunks c, (SELECT dense FROM chunks WHERE id = %s) q"
        " WHERE c.dense <=> q.dense < 0.03", (chunk_id,)).fetchall()
    return len({d for (d,) in rows} - exclude_docs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--samples", type=Path, required=True)
    ap.add_argument("--drafts", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    drafts: dict[str, str] = {}
    for p in a.drafts:
        drafts |= json.loads(p.read_text("utf-8"))
    samples = {p.stem: json.loads(p.read_text("utf-8")) for p in a.samples.glob("*.json")
               if not p.stem.startswith("drafts")}
    today = date.today().isoformat()
    cases, rejected, warned = [], [], []
    n_in = n_out = 0
    with psycopg.connect(a.dsn) as conn:
        cur = conn.cursor()
        for key, query in drafts.items():
            kind, _, rest = key.partition(":")
            if kind == "out_of_domain":
                subtype = rest.split(":")[0]
                n_out += 1
                qid = f"{a.prefix}o{n_out:02d}"
                cases.append({"id": qid, "type": "out_of_domain", "subtype": subtype,
                              "difficulty": "easy" if subtype == "general" else "hard",
                              "query": query, "answer_chunks": [],
                              "source": f"not answerable from {a.corpus}",
                              "provenance": f"written {today}", "holdout": held(qid, 30)})
                continue
            item = samples[kind][int(rest)]
            if kind == "hard_negative":
                clause, others, foils = item["clause"], [], [item["neighbour"]["id"]]
            elif kind == "underspecified":
                clause, others, foils = item["clause"], [t["id"] for t in item["twins"]], []
            else:
                clause, others, foils = item, [], []
            ids = [clause["id"], *others]
            rows = {r[0]: dict(zip(KEYS, r)) for r in cur.execute(META, (ids,)).fetchall()}
            main_row = rows.get(clause["id"])
            if main_row is None:
                rejected.append((key, "chunk id not in corpus"))
                continue
            run = longest_run(query, main_row["body"])
            if run >= 6:
                rejected.append((key, f"copies {run} words of its clause"))
                continue
            if run >= 4:
                warned.append((key, f"shares {run} words with its clause"))
            if kind != "exact_ref" and re.search(
                    rf"(?<![\w/-]){re.escape(main_row['regulation_number'])}(?![\w/-])", query):
                rejected.append((key, "names the instrument's number"))
                continue
            labelled = [rows[i] for i in ids if i in rows and rows[i]["identifier"]]
            if kind == "underspecified" and len(labelled) < 4:
                rejected.append((key, "fewer than 3 labellable twins"))
                continue
            idents = list(dict.fromkeys(r["identifier"] for r in labelled))
            docs = list(dict.fromkeys(r["regulation_number"] for r in labelled))
            n_in += 1
            qid = f"{a.prefix}{n_in:03d}"
            case = {"id": qid, "type": kind, "difficulty": "medium", "query": query,
                    "source": f"{main_row['identifier']} — {main_row['about']}",
                    "answer_identifiers": idents, "answer_documents": docs}
            if foils:
                case["must_not_rank_first"] = foils
            targets = resolve_targets(cur, case)
            if clause["id"] not in targets:
                n_in -= 1
                rejected.append((key, "label does not resolve to its own chunk"))
                continue
            case["label_chunks"] = len(targets)
            if kind != "underspecified":
                nd = near_twins(conn, clause["id"], {r["doc_id"] for r in labelled})
                if nd:
                    case["near_duplicates"] = nd
            case["provenance"] = (f"drafted {today} from chunk {clause['id']} of {a.corpus}; "
                                  "machine-checked; NOT reviewed by a domain expert")
            case["holdout"] = held(qid, 36)
            cases.append(case)
    a.out.write_text(json.dumps(
        {"_README": [f"A citation-labelled set on {a.corpus} (EVAL_QUERIES), drafted from "
                     "sample_corpus.py clauses and machine-checked by check_citation.py. "
                     "Labels are citations read back from the database; NOT reviewed by a "
                     "domain expert."], "queries": cases},
        ensure_ascii=False, indent=2), "utf-8", newline="\n")
    by: dict[str, int] = {}
    for c in cases:
        by[c["type"]] = by.get(c["type"], 0) + 1
    print(f"{len(cases)} cases · rejected {len(rejected)} · warned {len(warned)}")
    print("  " + " · ".join(f"{k} {v}" for k, v in sorted(by.items())))
    for k, why in rejected:
        print(f"  ✗ {k} {why}")
    for k, why in warned:
        print(f"  ! {k} {why}")
    wide = [c for c in cases if c.get("label_chunks", 0) > 5]
    for c in wide:
        print(f"  wide label {c['id']} {c['answer_identifiers'][0]} -> {c['label_chunks']} chunks")


if __name__ == "__main__":
    main()
