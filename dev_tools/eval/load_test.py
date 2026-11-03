"""Concurrency and latency under load, through the deployed HTTP server. stdlib only.

    python dev_tools/eval/load_test.py <label> <clients> <requests> [seed]
    python dev_tools/eval/load_test.py pinned-355K 4 48

Spreads <requests> `search_knowledge` calls over <clients> workers (the eval set's 50 queries cycle, shuffled by
`seed`) and prints served / refused / error counts, wall time, throughput, latency percentiles of the SERVED
requests, and peak CPU% and memory of every container sampled with `docker stats` while it ran.

! The default URL is 127.0.0.1, ✗ localhost. On Windows, Python resolving `localhost` costs ~2 s per request
(IPv6 attempt first, then fallback), which is more than the whole server-side query. The first 355K numbers taken
for `docs/HARDWARE.md` §8 read 3.0 s p50 for exactly this reason; the same stack answers in ~0.9 s. Set VERA_HTTP
to override, but keep an IP literal.

! A refusal (503 + Retry-After) is counted as `refused`, not as an error: it is the engine's backpressure doing
what invariant 6 says, and a retry is safe.
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

if len(sys.argv) < 4:
    sys.exit(__doc__)
LABEL, CLIENTS, TOTAL = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
SEED = int(sys.argv[4]) if len(sys.argv) > 4 else 7
URL = os.environ.get("VERA_HTTP", "http://127.0.0.1:8081").rstrip("/") + "/mcp"
QUERIES = Path(__file__).parent / "queries.json"

qs = [q["query"] for q in json.loads(QUERIES.read_text(encoding="utf-8"))["queries"]]
order = [qs[i % len(qs)] for i in range(TOTAL)]
random.Random(SEED).shuffle(order)

lock = threading.Lock()
cursor = [0]
results: list[tuple[str, float]] = []


def post(body: dict, timeout: int = 300) -> dict:
    req = urllib.request.Request(
        URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def worker() -> None:
    n = 0
    post({"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {
        "protocolVersion": "2024-11-05", "capabilities": {},
        "clientInfo": {"name": "load-test", "version": "0"}}})
    while True:
        with lock:
            if cursor[0] >= TOTAL:
                return
            query = order[cursor[0]]
            cursor[0] += 1
        n += 1
        t = time.perf_counter()
        try:
            r = post({"jsonrpc": "2.0", "id": n, "method": "tools/call",
                      "params": {"name": "search_knowledge", "arguments": {"query": query}}})
            status = "served" if "result" in r and not r["result"].get("isError") else "error"
        except urllib.error.HTTPError as e:
            status = "refused" if e.code == 503 else f"http{e.code}"
        except Exception as e:  # noqa: BLE001 -- anything else is a data point, not a crash
            status = "exc:" + type(e).__name__
        with lock:
            results.append((status, time.perf_counter() - t))


peak: dict[str, list[float]] = {}
stop = threading.Event()


def sampler() -> None:
    while not stop.is_set():
        out = subprocess.run(
            ["docker", "stats", "--no-stream", "--format", "{{.Name}}|{{.CPUPerc}}|{{.MemUsage}}"],
            capture_output=True, text=True).stdout
        for line in out.splitlines():
            try:
                name, cpu, mem = line.split("|")
                used = mem.split("/")[0].strip()
                mib = float(used[:-3]) * (1024 if used.endswith("GiB") else 1)
                p = peak.setdefault(name, [0.0, 0.0])
                p[0] = max(p[0], float(cpu.strip("%")))
                p[1] = max(p[1], mib)
            except (ValueError, IndexError):
                pass


def main() -> None:
    threading.Thread(target=sampler, daemon=True).start()
    t0 = time.perf_counter()
    threads = [threading.Thread(target=worker) for _ in range(CLIENTS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0
    stop.set()

    served = sorted(d for s, d in results if s == "served")
    counts: dict[str, int] = {}
    for s, _ in results:
        counts[s] = counts.get(s, 0) + 1

    def pct(p: float) -> float:
        return served[min(len(served) - 1, int(p / 100 * len(served)))] * 1000 if served else float("nan")

    print(f"[{LABEL}] clients={CLIENTS} requests={TOTAL} wall={wall:.1f}s "
          f"qps={len(served) / wall:.2f} counts={counts}")
    print(f"[{LABEL}]   served latency ms: p50={pct(50):.0f} p90={pct(90):.0f} p95={pct(95):.0f} "
          f"max={served[-1] * 1000 if served else float('nan'):.0f}")
    print(f"[{LABEL}]   peak (cpu%, MiB): "
          + ", ".join(f"{k}=({v[0]:.0f}%, {v[1]:.0f})" for k, v in sorted(peak.items())))


if __name__ == "__main__":
    main()
