"""Where do regional twins cost recall? Diagnostic only, through the deployed engine.

For every answerable case (exact_ref excluded):
  * twin_of_answer_above: a chunk within cosine 0.03 of a target, from another document,
    ranked in the top 5 while no target is (the right text, the wrong region's copy)
  * wasted: top-5 slots that are a near-identical copy of a higher-ranked result
  * region_named: the target's enacting_body appears in the query
"""
import json, os, sys, collections
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import psycopg
from run import resolve_targets
from e2e import HttpServer

TW = float(os.environ.get("TWIN", "0.03"))
cur = psycopg.connect(os.environ["DATABASE_URL"]).cursor()
srv = HttpServer(os.environ.get("VERA_HTTP", "http://127.0.0.1:8081"))
spec = json.loads((Path(__file__).parent / "queries.json").read_text("utf-8"))
out = []
for q in spec["queries"]:
    if q["type"] in ("out_of_domain", "exact_ref"):
        continue
    targets = resolve_targets(cur, q)
    if not targets:
        continue
    d = srv.search(q["query"])
    ids = [r["id"] for r in d.get("results", [])]
    if not ids:
        continue
    cur.execute("SELECT id, doc_id, enacting_body, regulation_type FROM chunks WHERE id = ANY(%s)",
                (ids + list(targets),))
    meta = {r[0]: r[1:] for r in cur.fetchall()}
    # twin pairs among returned + targets
    allids = list(dict.fromkeys(ids + list(targets)))
    cur.execute("SELECT a.id, b.id FROM chunks a, chunks b WHERE a.id = ANY(%s) AND b.id = ANY(%s)"
                " AND a.id < b.id AND a.dense <=> b.dense < %s", (allids, allids, TW))
    tw = collections.defaultdict(set)
    for a, b in cur.fetchall():
        tw[a].add(b); tw[b].add(a)
    tset = set(targets)
    if q["type"] == "underspecified":
        tset |= {i for t in targets for i in tw[t]}
    h5 = any(i in tset for i in ids[:5])
    twin_ans = [i for i in ids[:5] if i not in tset and any(t in tw[i] for t in targets)]
    wasted = sum(1 for k, i in enumerate(ids[:5]) if any(j in tw[i] for j in ids[:k]))
    tgt_rank = next((k for k, i in enumerate(ids) if i in tset), None)
    bodies = {meta[t][1] for t in targets if t in meta and meta[t][1]}
    named = any(b and len(b) > 3 and b.lower() in q["query"].lower() for b in bodies)
    out.append({"id": q["id"], "type": q["type"], "holdout": bool(q.get("holdout")), "h5": h5,
                "rank": tgt_rank, "twin_ans_top5": len(twin_ans), "wasted5": wasted,
                "region_named": named, "n": len(ids)})
Path(os.environ.get("OUT", "twins_diag.json")).write_text(json.dumps(out), "utf-8")

def summary(rows, label):
    m = [r for r in rows if not r["h5"]]
    print(f"[{label}] n={len(rows)} hit5={sum(r['h5'] for r in rows)} miss={len(m)}")
    print(f"  misses with the answer's twin in top5 (wrong copy): {sum(1 for r in m if r['twin_ans_top5'])}")
    print(f"    of which the query names the answer's region: {sum(1 for r in m if r['twin_ans_top5'] and r['region_named'])}")
    print(f"  misses with >=1 wasted top-5 slot: {sum(1 for r in m if r['wasted5'])}, "
          f"slots wasted total {sum(r['wasted5'] for r in m)}")
    print(f"  misses whose target is at rank 6-10: {sum(1 for r in m if r['rank'] is not None and r['rank'] < 10)}"
          f" · of those with wasted slots: {sum(1 for r in m if r['rank'] is not None and r['rank'] < 10 and r['wasted5'])}")
    print(f"  hits with wasted slots: {sum(1 for r in rows if r['h5'] and r['wasted5'])}")
    print(f"  region-named cases: {sum(r['region_named'] for r in rows)}, hit5 {sum(r['h5'] for r in rows if r['region_named'])}")

summary(out, "all")
summary([r for r in out if r["type"] == "underspecified"], "underspecified")
summary([r for r in out if r["type"] != "underspecified"], "other")
summary([r for r in out if r["holdout"]], "holdout")
