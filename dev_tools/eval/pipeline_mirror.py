"""A read-only Python mirror of the engine's query path, with the SPARSE ARM SWITCHABLE.

    BM25_VOCAB=.test/runs/corpus.bm25.json python dev_tools/eval/pipeline_mirror.py
    VERA_HTTP=http://127.0.0.1:8081 ... pipeline_mirror.py     # also proves the mirror against the live engine

Why it exists: `pg_search` vs `sparsevec` is chosen by what the database HAS (`store/search.rs::has_bm25`), so
comparing the two arms through the engine would mean dropping an index from the live database. This runs the
engine's own SQL for every arm, the same RRF / floor / factor arithmetic in float32 where the Rust is f32, and
only ever SELECTs. It prints, for each sparse arm: Recall@5 / @10 / MRR, real questions refused, out-of-domain
refused, and the gate's lexical evidence per out-of-domain query (`EVAL.md` §4c-bis).

! The mirror is only worth anything while it AGREES with the engine. With VERA_HTTP set it compares the top-10 of
all 50 queries against the live server; on 2026-11-02 that was 50 of 50 identical, in order. If that line is not
all-identical, ignore everything below it -- the engine moved and this file did not.
! Mirrors `balanced` (relevance floor 0.3, authority 0.5, structural 0.25, topical 0.25) with the built-in
id_regulation vocabulary and the compose defaults (DENSE_WEIGHT 2.0, DOMAIN_FLOOR 0.39, DOMAIN_LEXICAL_FLOOR 0.40,
GATE_SAMPLE 5, CLUSTERS_PROBED 5, CLUSTER_BATCH 1). Change those in the engine and they must change here.
! Does NOT mirror sibling expansion or non-default algorithms.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import psycopg

ROOT = Path(__file__).resolve().parents[2]
VOCAB = Path(os.environ.get("BM25_VOCAB", ROOT / ".test" / "runs" / "corpus.bm25.json"))
HOST = os.environ.get("PGHOST", "127.0.0.1")  # ✗ localhost: see load_test.py
DB = os.environ.get("PGDATABASE", "vera2")
EMBED = os.environ.get("EMBED_HTTP", "http://127.0.0.1:8080").rstrip("/")
LIVE = os.environ.get("VERA_HTTP", "").rstrip("/")

F = np.float32
OVER, PER_ARM_K, PER_CLUSTER_K, PROBE, POOL, TOP_K, GATE_SAMPLE = 3, 20, 20, 5, 60, 10, 5
W_DENSE, W_SPARSE, W_TEXT, RRF_K = F(2.0), F(1.0), F(1.0), F(60.0)
DOMAIN_FLOOR, LEX_FLOOR, REL_FLOOR = F(0.39), F(0.40), F(0.3)
W_AUTH, W_STRUCT, W_TOPICAL = F(0.5), F(0.25), F(0.25)

ART = json.loads(VOCAB.read_text(encoding="utf-8"))
VOC, IDF = ART["vocab"], {k: F(v) for k, v in ART["idf"].items()}
UNK, DIM = max(IDF.values()), ART["dim"]
tok = lambda s: re.findall(r"(?u)\b\w\w+\b", s.lower())  # noqa: E731 -- bm25.rs::tokenize


def evidence(query: str, texts: list[str]) -> F:
    """bm25.rs::evidence -- share of the query's IDF mass the pooled texts account for."""
    terms = sorted(set(tok(query)))
    if not terms:
        return F(0)
    w = lambda t: IDF.get(t, UNK)  # noqa: E731
    total = F(0)
    for t in terms:
        total = F(total + w(t))
    if total <= 0:
        return F(0)
    pooled = {t for x in texts for t in tok(x)}
    got = F(0)
    for t in terms:
        if t in pooled:
            got = F(got + w(t))
    return F(got / total)


# ---- identifier.rs
TYPE_WORDS = [
    (["undang", "undang"], "UNDANG-UNDANG"), (["uu"], "UNDANG-UNDANG"),
    (["peraturan", "pemerintah"], "PERATURAN PEMERINTAH"), (["pp"], "PERATURAN PEMERINTAH"),
    (["peraturan", "presiden"], "PERATURAN PRESIDEN"), (["perpres"], "PERATURAN PRESIDEN"),
    (["instruksi", "presiden"], "INSTRUKSI PRESIDEN"), (["inpres"], "INSTRUKSI PRESIDEN"),
    (["peraturan", "gubernur"], "PERATURAN GUBERNUR"), (["pergub"], "PERATURAN GUBERNUR"),
    (["peraturan", "bupati"], "PERATURAN BUPATI"), (["perbup"], "PERATURAN BUPATI"),
    (["peraturan", "walikota"], "PERATURAN WALIKOTA"), (["peraturan", "wali", "kota"], "PERATURAN WALIKOTA"),
    (["perwali"], "PERATURAN WALIKOTA"),
    (["peraturan", "daerah", "provinsi"], "PERATURAN DAERAH PROVINSI"),
    (["peraturan", "daerah", "kabupaten"], "PERATURAN DAERAH KABUPATEN"),
    (["peraturan", "daerah", "kota"], "PERATURAN DAERAH KOTA"),
]
is_year = lambda s: len(s) == 4 and s.isascii() and s.isdigit() and s[0] in "12"  # noqa: E731
is_num = lambda s: s.isascii() and s.isdigit()  # noqa: E731


def type_before(tokens: list[str], upto: int):
    window = tokens[max(0, upto - 4):upto]
    best = None
    for words, canon in TYPE_WORDS:
        for start in range(len(window)):
            if len(window) - start >= len(words) and window[start:start + len(words)] == words:
                end = start + len(words)
                if best is None or end >= best[0]:
                    best = (end, canon)
    return best[1] if best else None


def extract_ids(query: str) -> list[dict]:
    toks = [t for t in re.split(r"[^\w]|_", query.lower()) if t]
    out: list[dict] = []

    def push(num, year, rt):
        for e in out:
            if e["number"] == num and e["year"] == year:
                e["reg_type"] = e["reg_type"] or rt
                return
        out.append({"number": num, "year": year, "reg_type": rt})

    i = 0
    while i < len(toks):
        t = toks[i]
        if t == "tahun" and i + 1 < len(toks) and is_year(toks[i + 1]):
            num = next((c for c in reversed(toks[:i][-3:]) if is_num(c) and not is_year(c)), None)
            if num is not None:
                push(num, int(toks[i + 1]), type_before(toks, i))
                i += 2
                continue
        if is_num(t) and not is_year(t) and i + 1 < len(toks) and is_year(toks[i + 1]):
            push(t, int(toks[i + 1]), type_before(toks, i))
            i += 2
            continue
        i += 1
    return out


# ---- factors.rs (built-in id_regulation vocabulary)
TIERS = [("UNDANG-UNDANG", 8), ("PERATURAN PEMERINTAH", 7), ("PERATURAN PRESIDEN", 6), ("INSTRUKSI PRESIDEN", 5),
         ("PERATURAN DAERAH PROVINSI", 4), ("PERATURAN GUBERNUR", 3), ("PERATURAN DAERAH KABUPATEN", 3),
         ("PERATURAN DAERAH KOTA", 3), ("PERATURAN BUPATI", 2), ("PERATURAN WALIKOTA", 2)]
STOP = set("yang dan atau untuk dengan pada dari dalam oleh apakah bagaimana berapa siapa adalah itu ini ke di "
           "tidak dapat harus wajib jika akan sebagai".split())
R_LAMP, R_PENJ = re.compile(r"(?i)LAMPIRAN"), re.compile(r"(?i)PENJELASAN")
R_PASAL = re.compile(r"(?i)^\s*(?:PASAL)")


def authority(rt):
    if not rt:
        return F(0)
    want = rt.strip().upper()
    c = [(k, t) for k, t in TIERS if want == k or want.startswith(k)]
    return F(F(max(c, key=lambda x: len(x[0]))[1]) / F(10.0)) if c else F(0)


def structural(article, chapter):
    a, c = article or "", chapter or ""
    if R_LAMP.search(a) or R_LAMP.search(c):
        return F(0.0)
    if R_PENJ.search(c):
        return F(0.3)
    return F(1.0) if R_PASAL.search(a) else F(0.5)


def content_terms(text: str) -> list[str]:
    out: list[str] = []
    cur: list[str] = []

    def flush():
        w = "".join(cur)
        if len(w) > 3 and w not in STOP and w not in out:
            out.append(w)
        cur.clear()

    for ch in text:
        if ch.isalnum() or ch == "_":
            cur.append(ch.lower())
        else:
            flush()
    flush()
    return out


def share(have: list[str], qterms: list[str]) -> F:
    if not qterms:
        return F(0)
    hs = set(have)
    return F(F(sum(1 for t in qterms if t in hs)) / F(len(qterms)))


def settle(rows, k):  # search.rs::settle -- score desc, id asc, cut
    return sorted(rows, key=lambda r: (-r[1], r[0]))[:k]


class Mirror:
    def __init__(self) -> None:
        self.cur = psycopg.connect(f"host={HOST} port=5432 dbname={DB} user=vera password=vera", autocommit=True).cursor()
        self.cent = [(i, np.array([float(x) for x in t.strip("[]").split(",")], dtype=np.float32))
                     for i, t in self.q("select id, centroid::text from clusters order by id")]

    def q(self, sql, params=()):
        self.cur.execute(sql, params)
        return self.cur.fetchall()

    @staticmethod
    def embed(text: str):
        req = urllib.request.Request(f"{EMBED}/embed", data=json.dumps({"inputs": [text]}).encode(),
                                     headers={"Content-Type": "application/json"})
        return np.array(json.load(urllib.request.urlopen(req, timeout=300))[0], dtype=np.float32)

    def arms(self, query: str) -> dict:
        qv = self.embed(query)
        cos = lambda c: F(np.dot(qv, c) / (np.sqrt(np.dot(qv, qv)) * np.sqrt(np.dot(c, c))))  # noqa: E731
        probed = sorted(((i, cos(c)) for i, c in self.cent), key=lambda x: (-x[1], x[0]))[:PROBE]
        lit = "[" + ",".join(f"{v:.9g}" for v in qv) + "]"
        dense: list = []
        for cid, _ in probed:  # CLUSTER_BATCH = 1
            rows = self.q("SELECT id, 1.0 - (dense <=> %s::text::halfvec) FROM chunks WHERE indexable AND dense IS NOT NULL "
                          "AND cluster_id = ANY(%s) ORDER BY dense <=> %s::text::halfvec LIMIT %s",
                          (lit, [cid], lit, PER_CLUSTER_K * OVER))
            dense += settle([(r[0], F(r[1])) for r in rows], PER_CLUSTER_K)
        dense.sort(key=lambda r: -r[1])
        rows = self.q("SELECT id, paradedb.score(id) FROM chunks WHERE indexable AND id @@@ paradedb.match('body', %s) "
                      "ORDER BY paradedb.score(id) DESC LIMIT %s", (query, PER_ARM_K * OVER))
        pg = settle([(r[0], F(r[1])) for r in rows], PER_ARM_K)
        idx = sorted({VOC[t] for t in tok(query) if t in VOC})
        sv_lit = "{" + ",".join(f"{i + 1}:1" for i in idx) + "}/" + str(DIM)
        rows = self.q("SELECT id, -(sparse <#> %s::text::sparsevec) FROM chunks WHERE indexable AND sparse IS NOT NULL "
                      "ORDER BY sparse <#> %s::text::sparsevec LIMIT %s", (sv_lit, sv_lit, PER_ARM_K * OVER))
        sv = settle([(r[0], F(r[1])) for r in rows], PER_ARM_K)
        rows = self.q("WITH q AS (SELECT array_to_string(tsvector_to_array(to_tsvector('indonesian', %s)), ' | ')::tsquery AS tq) "
                      "SELECT id, (1.0 / (1.0 + (tsv <=> q.tq)))::real FROM chunks, q WHERE indexable AND tsv @@ q.tq "
                      "ORDER BY tsv <=> q.tq LIMIT %s", (query, PER_ARM_K * OVER))
        exact: list[str] = []
        for d in extract_ids(query):
            exact += [r[0] for r in self.q(
                "SELECT id FROM chunks WHERE regulation_number = %s AND (%s::int IS NULL OR year = %s) "
                "AND (%s::text IS NULL OR regulation_type = %s) ORDER BY year DESC NULLS LAST, chunk_no, id LIMIT 5",
                (d["number"], d["year"], d["year"], d["reg_type"], d["reg_type"]))]
        return {"top_cluster": probed[0][1], "dense": dense[:PER_ARM_K], "pg": pg, "sv": sv,
                "text": settle([(r[0], F(r[1])) for r in rows], PER_ARM_K), "exact": exact}

    def bodies(self, ids):
        rows = self.q("SELECT id, body, regulation_type, article, chapter, about FROM chunks WHERE id = ANY(%s)",
                      (list(ids),)) if ids else []
        return {r[0]: r[1:] for r in rows}

    def run(self, query: str, a: dict, sparse_key: str):
        """pipeline.rs::search_with after the arms -> (refused, top-10 ids, gate lexical evidence or None)."""
        sparse, lex = a[sparse_key], None
        if not a["exact"]:
            samp = [i for i, _ in sparse[:GATE_SAMPLE]]
            bd = self.bodies(samp)
            lex = evidence(query, [bd[i][0] for i in samp if i in bd])
            if not (lex >= LEX_FLOOR and a["top_cluster"] >= DOMAIN_FLOOR):
                return True, [], lex
        score: dict[str, F] = {}
        for ranked, w in ((a["dense"], W_DENSE), (sparse, W_SPARSE), (a["text"], W_TEXT)):
            for r, (i, _) in enumerate(ranked, start=1):
                score[i] = F(score.get(i, F(0)) + F(w / F(RRF_K + F(r))))
        fused = sorted(score.items(), key=lambda x: (-x[1], x[0]))[:POOL]
        if not fused:
            return True, [], lex
        bd = self.bodies([i for i, _ in fused])
        cands = [(i, s) for i, s in fused if i in bd]
        qterms = content_terms(query)
        if not a["exact"] and qterms:
            kept = [(i, s) for i, s in cands if share(content_terms(bd[i][0]), qterms) >= REL_FLOOR]
            cands = kept or cands[:1]

        def final(c):
            i, s = c
            _, rt, art, chap, about = bd[i]
            prior = F(F(W_AUTH * authority(rt)) + F(W_STRUCT * structural(art, chap)))
            if about:
                prior = F(prior + F(W_TOPICAL * share(content_terms(about), qterms)))
            return F(s * F(F(1) + prior))

        return False, [i for i, _ in sorted(cands, key=lambda c: -final(c))[:TOP_K]], lex


def targets(m: Mirror, q: dict) -> set[str]:
    ids = {x for x in q.get("answer_chunks", [])}
    for r in q.get("answer_articles", []):
        ids |= {x[0] for x in m.q("SELECT id FROM chunks WHERE regulation_type=%s AND regulation_number=%s AND year=%s "
                                  "AND about=%s AND article=%s",
                                  (r["regulation_type"], r["regulation_number"], r["year"], r["about"], r["article"]))}
    g = q.get("answer_regulation")
    if g:
        ids |= {x[0] for x in m.q("SELECT id FROM chunks WHERE regulation_type=%s AND regulation_number=%s AND year=%s",
                                  (g["regulation_type"], g["regulation_number"], g["year"]))}
    return ids


def live_top10(queries: list[dict]) -> dict[str, list[str]]:
    n = [0]

    def call(method, params):
        n[0] += 1
        body = json.dumps({"jsonrpc": "2.0", "id": n[0], "method": method, "params": params}).encode()
        req = urllib.request.Request(f"{LIVE}/mcp", data=body, headers={"Content-Type": "application/json"})
        return json.load(urllib.request.urlopen(req, timeout=300))

    call("initialize", {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "mirror", "version": "0"}})
    out = {}
    for q in queries:
        d = json.loads(call("tools/call", {"name": "search_knowledge", "arguments": {"query": q["query"]}})["result"]["content"][0]["text"])
        out[q["id"]] = [r["id"] for r in d.get("results", [])]
    return out


def main() -> None:
    spec = json.loads((Path(__file__).parent / "queries.json").read_text(encoding="utf-8"))["queries"]
    m, t0 = Mirror(), time.time()
    res: dict[str, dict[str, tuple]] = {"pg": {}, "sv": {}}
    for q in spec:
        a = m.arms(q["query"])
        for key in ("pg", "sv"):
            res[key][q["id"]] = m.run(q["query"], a, key)
    print(f"{len(spec)} queries mirrored in {time.time() - t0:.0f}s on {DB}")

    if LIVE:
        live = live_top10(spec)
        same = sum(res["pg"][q["id"]][1] == live[q["id"]] for q in spec)
        refusal = sum(res["pg"][q["id"]][0] == (not live[q["id"]]) for q in spec)
        print(f"MIRROR (pg_search) vs LIVE: identical top-10 order {same}/{len(spec)}, same refusal decision {refusal}/{len(spec)}")
        for q in spec:
            if res["pg"][q["id"]][1] != live[q["id"]]:
                print("  differs:", q["id"], q["type"])

    tg = {q["id"]: targets(m, q) for q in spec}
    for label, key in (("pg_search (what the engine runs when the index exists)", "pg"), ("sparsevec (the fallback)", "sv")):
        n = h5 = h10 = real_ref = ood = ood_ref = 0
        rr: list[float] = []
        for q in spec:
            refused, ids, _ = res[key][q["id"]]
            if q["type"] == "out_of_domain":
                ood += 1
                ood_ref += not ids
            elif tg[q["id"]]:
                n += 1
                real_ref += not ids
                h5 += any(x in tg[q["id"]] for x in ids[:5])
                h10 += any(x in tg[q["id"]] for x in ids[:10])
                rr.append(next((1 / (i + 1) for i, x in enumerate(ids) if x in tg[q["id"]]), 0.0))
        print(f"{label:58s} n={n} R@5 {100 * h5 / n:.1f}%  R@10 {100 * h10 / n:.1f}%  MRR {sum(rr) / len(rr):.3f}  "
              f"real refused {real_ref}  out-of-domain refused {ood_ref}/{ood}")
    print("\ngate lexical evidence on the out-of-domain queries (floor 0.40; refused = True)")
    for q in spec:
        if q["type"] == "out_of_domain":
            a, b = res["pg"][q["id"]], res["sv"][q["id"]]
            print(f"  {q['id']}  pg_search {float(a[2]):.3f} refused={a[0]}  |  sparsevec {float(b[2]):.3f} refused={b[0]}")


if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        sys.exit(__doc__)
    main()
