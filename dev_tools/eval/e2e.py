"""Score the eval set through the real MCP server, over stdio or HTTP.

    DATABASE_URL=... BM25_VOCAB=... python dev_tools/eval/e2e.py
    TEXT_WEIGHT=0.5 python dev_tools/eval/e2e.py          # sweep a tunable
    VERA_EXE=./target/release/vera-mcp python dev_tools/eval/e2e.py
    VERA_HTTP=http://localhost:8081 python dev_tools/eval/e2e.py   # a running deployment

! The HTTP mode scores the DEPLOYED server — the container, its limits, its
config — rather than a subprocess started with this shell's environment. Under
the VPS overlay that is the only way to find out whether the profile in
docker-compose.vps.yml actually serves, as opposed to whether the arithmetic in
docs/operations/HARDWARE.md §2 adds up.

! `run.py` scores the ARMS: it reimplements fusion in Python with equal
weights. This drives the shipped binary instead — same weights, same gate,
same contract a client gets. The two disagreeing is the bug it exists to
catch, and it has caught one: run.py reported RRF(all) at 50.0% while the
server delivered 38.6%, because the engine's text weight was 0.0 and no
harness was measuring the engine.

Use run.py to ask "is this arm any good". Use this to ask "is the product
any good". Only the second one is a promise to a user.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import psycopg

sys.path.insert(0, str(Path(__file__).parent))
from run import resolve_targets  # noqa: E402


def _rewrites() -> dict[str, list[str]]:
    """EVAL_REWRITES: a file of {case id: [rewordings]} (`rewrites_holdout.json`), keyed
    here by the question text so `search_args` can find them."""
    if not os.environ.get("EVAL_REWRITES"):
        return {}
    by_id = json.loads(Path(os.environ["EVAL_REWRITES"]).read_text("utf-8"))
    spec = json.loads(_queries_file().read_text("utf-8"))
    return {q["query"]: by_id[q["id"]] for q in spec["queries"] if q["id"] in by_id}


def _queries_file() -> Path:
    # ! EVAL_SPLIT=fresh reads `queries_fresh.json`, a test set drafted after the engine's
    # levers were chosen. It is a separate file so that no other script (sims, fits,
    # sweeps, all of which read `queries.json`) can see it: never tuned on, by construction.
    # ! EVAL_QUERIES names another source's set (`queries_ca_laws.json`): a corpus served by
    # its own engine is scored on its own questions, labelled by citation.
    if os.environ.get("EVAL_QUERIES"):
        return Path(__file__).parent / os.environ["EVAL_QUERIES"]
    name = "queries_fresh.json" if os.environ.get("EVAL_SPLIT") == "fresh" else "queries.json"
    return Path(__file__).parent / name


REWRITES = _rewrites()


def search_args(query):
    """EVAL_ARGS='{"profile": "literal"}' adds per-request options (`TOOL_SURFACE.md`) to
    every search, to score one algorithm or weight vector instead of the server's default.
    EVAL_REWRITES sends `rewordings` with the questions it covers."""
    args = {"query": query, **json.loads(os.environ.get("EVAL_ARGS", "{}"))}
    if query in REWRITES:
        args["rewordings"] = REWRITES[query]
    return args


def same_regulation(row, ref) -> bool:
    """A result chunk's regulation against a label's, by the label's own prefix rule.

    ! The regulation level is scored apart from the article level (`ROADMAP.md`
    Stage 0): the right law with the wrong pasal is a ranking miss inside a document,
    the wrong law is a retrieval miss, and a legal user needs to know which one.
    """
    rtype, rnum, year, about = row
    if (rtype, rnum, year) != (ref["regulation_type"], ref["regulation_number"], ref["year"]):
        return False
    if not ref.get("about"):
        return True     # `answer_regulation` names type, number and year only
    a, b = (about or "").strip().upper(), ref["about"].strip().upper()
    return bool(a) and (a.startswith(b) or b.startswith(a))


def regulation_hits(cur, q, ids) -> list[bool]:
    """Per returned id: does it belong to a labelled regulation?"""
    # A citation-labelled set names its instruments by number (`answer_documents`:
    # `SOR/2018-108`, `LEGITEXT000005634379`), what the profile writes to regulation_number.
    if q.get("answer_documents") and ids:
        cur.execute("SELECT id, regulation_number FROM chunks WHERE id = ANY(%s)", (ids,))
        number = dict(cur.fetchall())
        return [number.get(i) in q["answer_documents"] for i in ids]
    refs = list(q.get("answer_articles", []))
    if q.get("answer_regulation"):
        refs.append(q["answer_regulation"])
    if not ids or not refs:
        return [False] * len(ids)
    cur.execute(
        "SELECT id, regulation_type, regulation_number, year, about FROM chunks"
        " WHERE id = ANY(%s)", (ids,))
    meta = {r[0]: r[1:] for r in cur.fetchall()}
    return [any(same_regulation(meta[i], ref) for ref in refs) if i in meta else False
            for i in ids]

# ! Platform-resolved, not hardcoded to one. A default with .exe in it is a
# default that only works on one developer's machine.
_DEFAULT_EXE = "target/release/vera-mcp" + (".exe" if os.name == "nt" else "")
EXE = os.environ.get("VERA_EXE", _DEFAULT_EXE)
PG = os.environ.get(
    "DATABASE_URL", "host=localhost port=5432 dbname=vera2 user=vera password=vera"
)
# Anything the engine reads from the environment passes straight through, so a
# sweep needs no rebuild — which is the point of invariant 12.
PASSTHROUGH = (
    "EMBED_ENDPOINT", "BM25_VOCAB", "CLUSTERS_PROBED", "PARENTS_PROBED", "TWO_LEVEL_FROM", "CLUSTER_BATCH",
    "MAX_CONCURRENCY",
    "DENSE_WEIGHT", "SPARSE_WEIGHT", "TEXT_WEIGHT", "RRF_K", "REGION_WEIGHT",
    "DOMAIN_FLOOR", "DOMAIN_LEXICAL_FLOOR", "CANARY_MIN_COSINE",
    "PER_CLUSTER_K", "PER_ARM_K", "TOP_K", "SNIPPET_CHARS", "GATE_SAMPLE",
    "QUEUE_WAIT_MS", "READ_CHUNK_CHARS", "MAX_PROVENANCE_IDS",
)


class HttpServer:
    """A deployment already running somewhere, spoken to over POST /mcp."""

    def __init__(self, base):
        self.base = base.rstrip("/")
        self._n = 0
        self.call("initialize", {
            "protocolVersion": "2024-11-05", "capabilities": {},
            "clientInfo": {"name": "eval-e2e", "version": "0"},
        })

    def call(self, method, params):
        import urllib.request
        self._n += 1
        body = json.dumps({
            "jsonrpc": "2.0", "id": self._n, "method": method, "params": params,
        }).encode()
        req = urllib.request.Request(
            f"{self.base}/mcp", data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.load(r)

    def search(self, query):
        r = self.call("tools/call", {
            "name": "search_knowledge", "arguments": search_args(query),
        })
        return json.loads(r["result"]["content"][0]["text"])

    def close(self):
        pass


class Server:
    """One MCP server on stdio, spoken to in JSON-RPC."""

    def __init__(self):
        # ! The engine requires EMBED_ENDPOINT and BM25_VOCAB and will exit
        # rather than guess. Checking here turns that into a usable message
        # instead of a subprocess that dies with its stderr thrown away.
        missing = [k for k in ("EMBED_ENDPOINT", "BM25_VOCAB")
                   if not os.environ.get(k)]
        if missing:
            sys.exit(f"set {' and '.join(missing)} · the engine will not guess")
        if not Path(EXE).exists():
            sys.exit(f"no engine at {EXE} · cargo build --release -p vera-mcp, "
                     f"or set VERA_EXE")

        env = dict(os.environ, DATABASE_URL=PG)
        # ! stderr is kept. The engine reports a refusal to serve there — a
        # canary failure, a missing vocabulary — and discarding it turns every
        # one of those into an unexplained hang on the first read.
        self.p = subprocess.Popen(
            [EXE], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=None, env=env, bufsize=0,
        )
        self._n = 0
        self.call("initialize", {
            "protocolVersion": "2024-11-05", "capabilities": {},
            "clientInfo": {"name": "eval-e2e", "version": "0"},
        })
        self.notify("notifications/initialized", {})

    def _send(self, msg):
        self.p.stdin.write((json.dumps(msg) + "\n").encode())
        self.p.stdin.flush()

    def notify(self, method, params):
        self._send({"jsonrpc": "2.0", "method": method, "params": params})

    def call(self, method, params):
        self._n += 1
        self._send({"jsonrpc": "2.0", "id": self._n, "method": method, "params": params})
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise SystemExit("engine closed stdout — check stderr for a refusal")
            try:
                m = json.loads(line)
            except json.JSONDecodeError:
                continue          # ! not ours: never assume stdout is only us
            if m.get("id") == self._n:
                return m

    def search(self, query):
        r = self.call("tools/call", {
            "name": "search_knowledge", "arguments": search_args(query),
        })
        return json.loads(r["result"]["content"][0]["text"])

    def close(self):
        self.p.kill()


def main():
    http = os.environ.get("VERA_HTTP")
    srv = HttpServer(http) if http else Server()
    cur = psycopg.connect(PG).cursor()
    spec = json.loads(_queries_file().read_text(encoding="utf-8"))

    n = hit5 = hit10 = hitall = refused_real = reg5 = 0
    deep = {15: 0, 20: 0, 25: 0}   # Recall@k past 10, read only when the reply is that deep
    depth = 0   # the deepest reply seen: TOP_K on the server, ✗ known here
    named = [0, 0, 0]   # exact_ref cases: n, article at rank 1, regulation at rank 1
    ood_n = ood_refused = 0
    rr = []
    misses = []
    # ! EVAL_SPLIT=train|holdout|all (default all). `holdout: true` cases are the ones
    # tuning must not see; a weight fitted on `train` is checked ONCE on `holdout`.
    split = os.environ.get("EVAL_SPLIT", "all")
    by_type: dict[str, list[int]] = {}
    per_case = []

    for q in spec["queries"]:
        held = bool(q.get("holdout"))
        # A `fresh` case found in `queries.json` (merged there by mistake) is still left
        # out of every other split, so numbers recorded before it existed stay comparable.
        if (split == "fresh") != bool(q.get("fresh")):
            continue
        if (split == "train" and held) or (split == "holdout" and not held):
            continue
        # EVAL_TYPES=exact_ref,out_of_domain scores only those types (for iterating).
        if os.environ.get("EVAL_TYPES") and q["type"] not in os.environ["EVAL_TYPES"].split(","):
            continue
        # ! EVAL_REVIEWED=1 scores only labels a domain reviewer marked BENAR
        # (`draft/review.py`); out-of-domain cases are kept, they have no label to review.
        if (os.environ.get("EVAL_REVIEWED") == "1" and q["type"] != "out_of_domain"
                and q.get("review", {}).get("verdict") != "BENAR"):
            continue
        d = srv.search(q["query"])
        ids = [r.get("id") for r in d.get("results", [])]

        if q["type"] == "out_of_domain":
            ood_n += 1
            ood_refused += 1 if not ids else 0
            if ids:
                misses.append(f"  ! {q['id']} answered an out-of-domain query")
            continue

        targets = resolve_targets(cur, q)
        # ! `underspecified`: any equally correct answer is a pass, and the labelled twins
        # are a sample of them, not all. A returned chunk within cosine distance 0.03 of a
        # labelled chunk is the same clause in another region's regulation, so it counts.
        if q["type"] == "underspecified" and targets and ids:
            cur.execute(
                "SELECT r.id FROM chunks r, chunks t WHERE r.id = ANY(%s) AND t.id = ANY(%s)"
                " AND r.dense <=> t.dense < 0.03", (ids, list(targets)))
            targets = targets | {r[0] for r in cur.fetchall()}
        if not targets:
            print(f"! {q['id']} has no resolvable target — skipped", file=sys.stderr)
            continue

        n += 1
        if not ids:
            refused_real += 1
            misses.append(f"  ! {q['id']} refused a real question")
        h5 = any(x in targets for x in ids[:5])
        hit5 += 1 if h5 else 0
        hit10 += 1 if any(x in targets for x in ids[:10]) else 0
        for k in deep:
            deep[k] += 1 if any(x in targets for x in ids[:k]) else 0
        # ! Anywhere in the reply. With TOP_K = CANDIDATE_POOL on the server this is
        # pool recall, the Stage 2 ceiling: what ranking could reach at best.
        hall = any(x in targets for x in ids)
        hitall += 1 if hall else 0
        depth = max(depth, len(ids))
        rr.append(next((1 / (i + 1) for i, x in enumerate(ids) if x in targets), 0.0))
        regs = regulation_hits(cur, q, ids)
        r5 = any(regs[:5])
        reg5 += 1 if r5 else 0
        if q["type"] == "exact_ref":
            named[0] += 1
            named[1] += 1 if ids and ids[0] in targets else 0
            named[2] += 1 if regs and regs[0] else 0
        t = by_type.setdefault(q["type"], [0, 0])
        t[0] += 1 if h5 else 0
        t[1] += 1
        per_case.append({"id": q["id"], "type": q["type"], "hit5": h5, "reg5": r5,
                         "rr": rr[-1], "pool": hall, "holdout": held,
                         "rank": next((i + 1 for i, x in enumerate(ids) if x in targets), None),
                         "near_duplicates": q.get("near_duplicates", 0)})

    srv.close()
    if os.environ.get("EVAL_CASES_OUT"):
        Path(os.environ["EVAL_CASES_OUT"]).write_text(json.dumps(per_case), encoding="utf-8")

    def w(key, default):
        return os.environ.get(key, default)

    print(f"through {http or EXE}")
    # ! In HTTP mode the weights belong to the SERVER, ✗ to this process, and
    # nothing here can see them: `HttpServer.call` sends method and params only.
    # Printing this shell's env next to a score measured by another process
    # invites reading `dense=0.0` off a run where the server used 2.0 — which is
    # exactly the misreading that the dense-arm defect survived on for weeks.
    if http:
        print("  weights  set on the server · not visible from here")
    else:
        print(f"  weights  dense={w('DENSE_WEIGHT', '0.0')} "
              f"sparse={w('SPARSE_WEIGHT', '1.0')} text={w('TEXT_WEIGHT', '1.0')}")
    print(f"  split {split}" + (f" · args {os.environ['EVAL_ARGS']}"
                                 if os.environ.get("EVAL_ARGS") else ""))
    print(f"  n={n}   Recall@5 {100 * hit5 / n:.1f}%   "
          f"Recall@10 {100 * hit10 / n:.1f}%   MRR {sum(rr) / len(rr):.3f}")
    # ! With n this size a Recall@5 difference is only worth reading past its noise:
    # the 95% half-width of a proportion near 60% is ~1.96 * sqrt(.24 / n).
    if depth > 10:
        print(f"  answer anywhere in the reply (depth {depth}) {100 * hitall / n:.1f}%")
        # ! Only the cut-offs the reply reaches: past `depth` the count is the reply's.
        cuts = [k for k in deep if k <= depth]
        if cuts:
            print("  " + "   ".join(f"Recall@{k} {100 * deep[k] / n:.1f}%" for k in cuts))
    print(f"  ±{196 * (0.24 / n) ** 0.5:.1f} points (95%, binomial) on Recall@5")
    print(f"  right regulation in top 5 {100 * reg5 / n:.1f}%   "
          f"(right law, wrong article {100 * (reg5 - hit5) / n:.1f}%)")
    if named[0]:
        print(f"  exact_ref n={named[0]}   article at rank 1 {100 * named[1] / named[0]:.1f}%   "
              f"named regulation at rank 1 {100 * named[2] / named[0]:.1f}%")
    clean = [c for c in per_case if not c["near_duplicates"]]
    if len(clean) < len(per_case):
        print(f"  without near-duplicate labels: n={len(clean)}  Recall@5 "
              f"{100 * sum(c['hit5'] for c in clean) / max(len(clean), 1):.1f}%")
    print("  by type (Recall@5): " + " · ".join(
        f"{k} {v[0]}/{v[1]}" for k, v in sorted(by_type.items(), key=lambda kv: -kv[1][1])))
    print(f"  gate     real refused {refused_real}/{n} · "
          f"out-of-domain refused {ood_refused}/{ood_n}")
    for m in misses:
        print(m)

    # ! A refusal of a real question and an answer to junk are contract
    # failures, ✗ quality numbers. They fail the run.
    return 1 if (refused_real or ood_refused < ood_n) else 0


if __name__ == "__main__":
    sys.exit(main())
