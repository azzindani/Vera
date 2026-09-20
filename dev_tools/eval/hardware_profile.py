"""A cold-start to steady-state profile of the whole stack, sampled throughout.

Every memory figure in `HARDWARE.md` is a point measurement of one process at one
moment. None of them says what the stack *does* — what it costs stopped, what
starting costs, what a query costs while it runs, what is reclaimable afterwards,
and whether anything ratchets upward over a session.

This drives the stack through named phases and samples every container plus the
whole VM on a fixed interval, tagging each sample with the phase that produced it.

    DATABASE_URL=... EMBED_ENDPOINT=... BM25_VOCAB=... python hardware_profile.py
    INTERVAL=2 python dev_tools/eval/hardware_profile.py

! It STOPS and STARTS containers. Do not point it at anything serving traffic.

! `docker stats` memory is the cgroup's `memory.current`, which INCLUDES page
cache the kernel would hand back under pressure. For Postgres that is most of the
number and it is not a requirement -- `HARDWARE.md` §2 splits anon from file for
exactly this reason. Treat the db row as "what it has taken", ✗ "what it needs".

! The VM row is Docker Desktop's Linux VM, not the Windows host. On a real VPS
that row IS the box. It is the only figure here that answers "would this fit".

! Sampling is on a wall-clock interval, so a phase shorter than the interval can
produce no samples of its own. Phases are reported with their own duration so a
thin one is visible rather than silently missing.

! Run it with `python -X utf8` on Windows. The console is cp1252 by default and
this file prints the same separators the rest of dev_tools uses, so without it
the run dies on a box-drawing character before it measures anything.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INTERVAL = float(os.environ.get("INTERVAL", "3"))
HTTP = os.environ.get("VERA_HTTP", "http://localhost:8081").rstrip("/")
EMBED = os.environ.get("EMBED_HTTP", "http://localhost:8080").rstrip("/")
STACK = ("vera-db", "vera-embed-ref", "vera-mcp")
QUERIES = Path(__file__).parent / "queries.json"

_samples: list[dict] = []
_phase = "init"
_stop = threading.Event()


def docker(*args: str, check: bool = True) -> str:
    r = subprocess.run(("docker",) + args, capture_output=True, text=True)
    if check and r.returncode != 0:
        print(f"  ! docker {' '.join(args)} -> {r.stderr.strip()[:120]}", flush=True)
    return r.stdout.strip()


def vm_memory() -> dict[str, int]:
    """MemTotal / MemAvailable / Cached from the Docker VM, in MB.

    ! A privileged --pid=host container, because on Docker Desktop the Windows
    host's memory says nothing about whether the stack fits: the VM is the box
    the containers actually live in."""
    out = docker("run", "--rm", "--privileged", "--pid=host", "alpine",
                 "cat", "/proc/meminfo", check=False)
    want = {"MemTotal": 0, "MemAvailable": 0, "Cached": 0}
    for line in out.splitlines():
        k, _, rest = line.partition(":")
        if k in want:
            want[k] = int(rest.strip().split()[0]) // 1024
    return want


def sample() -> None:
    fmt = "{{.Name}}\t{{.MemUsage}}\t{{.CPUPerc}}"
    while not _stop.is_set():
        out = docker("stats", "--no-stream", "--format", fmt, check=False)
        row = {"t": time.monotonic(), "phase": _phase, "c": {}}
        for line in out.splitlines():
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            name, mem, cpu = parts[0], parts[1], parts[2]
            used = mem.split("/")[0].strip()
            row["c"][name] = {"mem": used, "cpu": cpu}
        _samples.append(row)
        _stop.wait(INTERVAL)


def mb(text: str) -> float:
    """'1.234GiB' / '567.8MiB' / '12.3kB' -> MB."""
    t = text.strip()
    for suffix, mult in (("GiB", 1024), ("MiB", 1), ("KiB", 1 / 1024),
                         ("GB", 1000), ("MB", 1), ("kB", 1 / 1000), ("B", 1e-6)):
        if t.endswith(suffix):
            try:
                return float(t[: -len(suffix)]) * mult
            except ValueError:
                return 0.0
    return 0.0


def phase(name: str) -> None:
    global _phase
    _phase = name
    print(f"\n── {name}", flush=True)


def await_health(timeout: float = 120.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        try:
            with urllib.request.urlopen(f"{HTTP}/health", timeout=5) as r:
                if r.status == 200:
                    return True
        except (urllib.error.URLError, OSError):
            time.sleep(1)
    return False


def search(query: str, timeout: float = 180) -> float:
    body = json.dumps({
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "search_knowledge", "arguments": {"query": query}},
    }).encode()
    req = urllib.request.Request(
        f"{HTTP}/mcp", data=body, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            r.read()
    except (urllib.error.URLError, OSError):
        pass
    return (time.perf_counter() - t0) * 1000


def burst(queries: list[str], n: int) -> tuple[int, int]:
    """n concurrent requests. Returns (served, refused-or-failed)."""
    ok = [0, 0]
    lock = threading.Lock()

    def one(q: str) -> None:
        body = json.dumps({
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": "search_knowledge", "arguments": {"query": q}},
        }).encode()
        req = urllib.request.Request(
            f"{HTTP}/mcp", data=body, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                r.read()
            with lock:
                ok[0] += 1
        except Exception:
            with lock:
                ok[1] += 1

    ts = [threading.Thread(target=one, args=(queries[i % len(queries)],))
          for i in range(n)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return ok[0], ok[1]


def report() -> None:
    order: list[str] = []
    for s in _samples:
        if s["phase"] not in order:
            order.append(s["phase"])
    names: list[str] = []
    for s in _samples:
        for c in s["c"]:
            if c not in names:
                names.append(c)
    names = [n for n in STACK if n in names] + [n for n in names if n not in STACK]

    print("\n\nPEAK CONTAINER MEMORY BY PHASE (MB · cgroup current, page cache included)\n")
    print(f"{'phase':<22}" + "".join(f"{n.replace('vera-', ''):>14}" for n in names)
          + f"{'total':>10}{'samples':>9}")
    for ph in order:
        rows = [s for s in _samples if s["phase"] == ph]
        peaks = []
        for n in names:
            vals = [mb(s["c"][n]["mem"]) for s in rows if n in s["c"]]
            peaks.append(max(vals) if vals else 0.0)
        print(f"{ph:<22}" + "".join(f"{p:>14.0f}" for p in peaks)
              + f"{sum(peaks):>10.0f}{len(rows):>9}")

    print("\nPEAK CPU BY PHASE (% of one core; 2 cores = 200%)\n")
    print(f"{'phase':<22}" + "".join(f"{n.replace('vera-', ''):>14}" for n in names))
    for ph in order:
        rows = [s for s in _samples if s["phase"] == ph]
        peaks = []
        for n in names:
            vals = []
            for s in rows:
                if n in s["c"]:
                    try:
                        vals.append(float(s["c"][n]["cpu"].rstrip("%")))
                    except ValueError:
                        pass
            peaks.append(max(vals) if vals else 0.0)
        print(f"{ph:<22}" + "".join(f"{p:>13.0f}%" for p in peaks))


def main() -> None:
    cases = [c["query"] for c in
             json.loads(QUERIES.read_text(encoding="utf-8"))["queries"]
             if c["type"] != "out_of_domain"]
    ood = [c["query"] for c in
           json.loads(QUERIES.read_text(encoding="utf-8"))["queries"]
           if c["type"] == "out_of_domain"]

    t = threading.Thread(target=sample, daemon=True)
    t.start()
    timeline: list[tuple[str, float]] = []

    def mark(name: str, fn) -> None:
        phase(name)
        t0 = time.monotonic()
        fn()
        dt = time.monotonic() - t0
        timeline.append((name, dt))
        print(f"   {dt:.1f}s", flush=True)

    def stop_all() -> None:
        docker("stop", *STACK, check=False)
        time.sleep(INTERVAL * 2)
        v = vm_memory()
        print(f"   VM: {v['MemTotal'] - v['MemAvailable']:,} MB used of "
              f"{v['MemTotal']:,} - {v['Cached']:,} cached", flush=True)

    def start_all() -> None:
        # ! ORDER MATTERS, and this is not the script being fussy. The engine's
        # startup canary re-embeds a stored chunk and refuses to serve if it
        # cannot reproduce the vector space (CLAUDE.md §7.3). Started alongside a
        # model server that is still loading, it does not retry -- it exits, and
        # stays down. Correct, and it means "start everything" is wrong.
        docker("start", "vera-db", "vera-embed-ref", check=False)
        # ! Poll /embed with a real request, ✗ /health and ✗ the logs. Both of
        # the cheaper signals are wrong here:
        #   - `docker logs` keeps the previous run's output, so "listening on"
        #     is already present the instant the container restarts;
        #   - /health answers 200 while the weights are still loading. It is a
        #     LIVENESS check, and the canary needs READINESS.
        # Either one starts the engine against a model server that cannot embed,
        # and the engine then refuses to serve and stays down.
        end = time.monotonic() + 900
        probe = json.dumps({"inputs": "ready", "truncate": False}).encode()
        while time.monotonic() < end:
            try:
                req = urllib.request.Request(
                    f"{EMBED}/embed", data=probe,
                    headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=10) as r:
                    if r.status == 200:
                        break
            except (urllib.error.URLError, OSError):
                time.sleep(3)
        else:
            sys.exit("embedder never became ready to embed")
        docker("start", "vera-mcp", check=False)
        if not await_health():
            sys.exit("engine never became healthy - docker logs vera-mcp")

    def idle() -> None:
        time.sleep(INTERVAL * 4)
        v = vm_memory()
        print(f"   VM: {v['MemTotal'] - v['MemAvailable']:,} MB used of "
              f"{v['MemTotal']:,} - {v['Cached']:,} cached", flush=True)

    def one_query() -> None:
        print(f"   cold query {search(cases[0]):.0f} ms", flush=True)

    def sequential() -> None:
        lat = [search(q) for q in cases]
        lat.sort()
        print(f"   {len(lat)} queries - p50 {lat[len(lat)//2]:.0f} ms - "
              f"p95 {lat[int(0.95*(len(lat)-1))]:.0f} ms - max {lat[-1]:.0f} ms",
              flush=True)

    def refusals() -> None:
        lat = [search(q) for q in ood]
        print(f"   {len(lat)} out-of-domain - p50 "
              f"{sorted(lat)[len(lat)//2]:.0f} ms", flush=True)

    def concurrency() -> None:
        served, failed = burst(cases, 12)
        print(f"   12 concurrent -> {served} served - {failed} refused/failed",
              flush=True)

    mark("0 - all stopped", stop_all)
    mark("1 - starting", start_all)
    mark("2 - idle after start", idle)
    mark("3 - first query (cold)", one_query)
    mark("4 - sequential eval", sequential)
    mark("5 - domain refusals", refusals)
    mark("6 - 12 concurrent", concurrency)
    mark("7 - idle after load", idle)

    _stop.set()
    t.join(timeout=INTERVAL * 2)

    print("\n\nPHASE DURATIONS\n")
    for name, dt in timeline:
        print(f"  {name:<24}{dt:>8.1f}s")
    report()
    print("\n! db memory is mostly reclaimable page cache, ✗ a requirement."
          "\n  Compare phase 7 against phase 2: what stays is what ratcheted.")


if __name__ == "__main__":
    main()
