"""Score the eval set through the real MCP server, over stdio or HTTP.

    DATABASE_URL=... BM25_VOCAB=... python dev_tools/eval/e2e.py
    TEXT_WEIGHT=0.5 python dev_tools/eval/e2e.py          # sweep a tunable
    VERA_EXE=./target/release/vera-mcp python dev_tools/eval/e2e.py
    VERA_HTTP=http://localhost:8081 python dev_tools/eval/e2e.py   # a running deployment

! The HTTP mode scores the DEPLOYED server — the container, its limits, its
config — rather than a subprocess started with this shell's environment. Under
the VPS overlay that is the only way to find out whether the profile in
docker-compose.vps.yml actually serves, as opposed to whether the arithmetic in
docs/HARDWARE.md §2 adds up.

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
    "EMBED_ENDPOINT", "BM25_VOCAB", "CLUSTERS_PROBED", "CLUSTER_BATCH",
    "MAX_CONCURRENCY",
    "DENSE_WEIGHT", "SPARSE_WEIGHT", "TEXT_WEIGHT",
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
            "name": "search_knowledge", "arguments": {"query": query},
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
            "name": "search_knowledge", "arguments": {"query": query},
        })
        return json.loads(r["result"]["content"][0]["text"])

    def close(self):
        self.p.kill()


def main():
    http = os.environ.get("VERA_HTTP")
    srv = HttpServer(http) if http else Server()
    cur = psycopg.connect(PG).cursor()
    spec = json.loads(
        (Path(__file__).parent / "queries.json").read_text(encoding="utf-8")
    )

    n = hit5 = hit10 = refused_real = 0
    ood_n = ood_refused = 0
    rr = []
    misses = []

    for q in spec["queries"]:
        d = srv.search(q["query"])
        ids = [r.get("id") for r in d.get("results", [])]

        if q["type"] == "out_of_domain":
            ood_n += 1
            ood_refused += 1 if not ids else 0
            if ids:
                misses.append(f"  ! {q['id']} answered an out-of-domain query")
            continue

        targets = resolve_targets(cur, q)
        if not targets:
            print(f"! {q['id']} has no resolvable target — skipped", file=sys.stderr)
            continue

        n += 1
        if not ids:
            refused_real += 1
            misses.append(f"  ! {q['id']} refused a real question")
        hit5 += 1 if any(x in targets for x in ids[:5]) else 0
        hit10 += 1 if any(x in targets for x in ids[:10]) else 0
        rr.append(next((1 / (i + 1) for i, x in enumerate(ids) if x in targets), 0.0))

    srv.close()

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
    print(f"  n={n}   Recall@5 {100 * hit5 / n:.1f}%   "
          f"Recall@10 {100 * hit10 / n:.1f}%   MRR {sum(rr) / len(rr):.3f}")
    print(f"  gate     real refused {refused_real}/{n} · "
          f"out-of-domain refused {ood_refused}/{ood_n}")
    for m in misses:
        print(m)

    # ! A refusal of a real question and an answer to junk are contract
    # failures, ✗ quality numbers. They fail the run.
    return 1 if (refused_real or ood_refused < ood_n) else 0


if __name__ == "__main__":
    sys.exit(main())
