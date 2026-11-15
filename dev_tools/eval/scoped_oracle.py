"""Oracle: if the engine searched only the named region, would it find the answer?

Region-named cases only (target's enacting_body appears in the query). Region = the
target's enacting_body (an oracle: the engine would have to parse it from the query).
Dense: exact scan of the region's chunks. Sparse: BM25 top-k restricted to the region.
"""
import json, os, sys, time, urllib.request, collections
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import psycopg
from run import resolve_targets

conn = psycopg.connect(os.environ["DATABASE_URL"])
cur = conn.cursor()
cur.execute("SELECT to_jsonb(corpus_meta) ->> 'dense_query_instruction' FROM corpus_meta ORDER BY created_at DESC LIMIT 1")
instr = cur.fetchone()[0]
prefix = f"Instruct: {instr}\nQuery: " if instr else ""
def embed(t):
    req = urllib.request.Request(os.environ.get("EMBED_ENDPOINT", "http://127.0.0.1:8080") + "/embed", data=json.dumps({"inputs": prefix + t, "truncate": False}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)[0]
cur.execute("CREATE TEMP TABLE rb AS SELECT id, enacting_body FROM chunks WHERE indexable")
cur.execute("CREATE INDEX ON rb (enacting_body)")
spec = {q["id"]: q for q in json.loads((Path(__file__).parent / "queries.json").read_text("utf-8"))["queries"]}
diag = [r for r in json.loads(Path(os.environ["DIAG"]).read_text("utf-8")) if r["region_named"]]
res = collections.Counter(); rows = []
for r in diag:
    q = spec[r["id"]]
    targets = set(resolve_targets(cur, q))
    cur.execute("SELECT enacting_body FROM chunks WHERE id = %s", (next(iter(targets)),))
    body = cur.fetchone()[0]
    cur.execute("SELECT count(*) FROM rb WHERE enacting_body = %s", (body,))
    size = cur.fetchone()[0]
    v = "[" + ",".join(f"{x:.6f}" for x in embed(q["query"])) + "]"
    t0 = time.time()
    cur.execute("SELECT c.id FROM rb JOIN chunks c USING (id) WHERE rb.enacting_body = %s"
                " ORDER BY c.dense <=> %s::halfvec LIMIT 20", (body, v))
    dense = [x[0] for x in cur.fetchall()]
    td = time.time() - t0
    t0 = time.time()
    cur.execute("SELECT s.id FROM (SELECT id, paradedb.score(id) sc FROM chunks WHERE id @@@ paradedb.match('body', %s)"
                " ORDER BY paradedb.score(id) DESC LIMIT 20000) s JOIN rb USING (id) WHERE rb.enacting_body = %s"
                " ORDER BY s.sc DESC LIMIT 20", (q["query"], body))
    sparse = [x[0] for x in cur.fetchall()]
    ts = time.time() - t0
    # RRF of the two scoped lists
    sc = collections.defaultdict(float)
    for lst in (dense, sparse):
        for k, i in enumerate(lst):
            sc[i] += 1 / (60 + k + 1)
    fused = sorted(sc, key=lambda i: -sc[i])
    key = "hit" if r["h5"] else "miss"
    res[key] += 1
    res[f"{key} dense@5"] += any(i in targets for i in dense[:5])
    res[f"{key} dense@20"] += any(i in targets for i in dense[:20])
    res[f"{key} sparse@20"] += any(i in targets for i in sparse[:20])
    res[f"{key} fused@5"] += any(i in targets for i in fused[:5])
    res[f"{key} fused@10"] += any(i in targets for i in fused[:10])
    rows.append({"id": r["id"], "h5": r["h5"], "size": size, "td": td, "ts": ts,
                 "fused5": any(i in targets for i in fused[:5])})
Path(os.environ["OUT"]).write_text(json.dumps(rows), "utf-8")
for k in sorted(res):
    print(f"  {k}: {res[k]}")
sz = sorted(x["size"] for x in rows); td = sorted(x["td"] for x in rows)
print("region size p50/p90/max", sz[len(sz)//2], sz[int(len(sz)*.9)], sz[-1])
print("dense scan s p50/p90/max", round(td[len(td)//2], 2), round(td[int(len(td)*.9)], 2), round(td[-1], 2))
