"""What `CLUSTER_BATCH` costs in memory and buys in latency, measured.

`ARCHITECTURE.md` §4 used to claim two things: that peak RAM was
`MAX_CONCURRENCY x CLUSTER_BATCH x one cluster`, and that the 187 ms sequential
loading costs was refundable. Run on 2026-09-19, this script confirmed the second
(-182 ms at batch=5) and **refuted the first** -- engine RSS measured 14/12/14 MB
at 1/2/5, flat, because the store returns ranked ids rather than vectors.

Keep running it. The memory claim is now "flat", and a change that starts loading
rows into the engine would show up here as a column that finally moves.

    VERA_HTTP=http://localhost:8081 python dev_tools/eval/cluster_batch_sweep.py
    BATCHES=1,2,5 REPEATS=3 python dev_tools/eval/cluster_batch_sweep.py

! It RESTARTS the engine container, once per setting, and leaves it on the last
one. Do not point it at anything serving traffic.

! The container is cloned from whatever is running -- image, env, mounts,
network, limits -- with only `CLUSTER_BATCH` overridden. Hardcoding a `docker
run` here would measure this developer's machine and silently stop matching the
deployment the moment either changed.

! Peak RSS comes from the cgroup's own `memory.peak`, ✗ from sampling `docker
stats`. A sampler that polls every second cannot see a spike that lasts 80 ms,
which is exactly the shape of the thing being measured. Nothing resets that
high-water mark and nothing needs to: each row restarts the container, and a new
container is a new cgroup starting at zero. (`/sys/fs/cgroup` is mounted
read-only anyway, so a reset would fail.) Where the kernel does not expose
`memory.peak` at all -- cgroup v1, older hosts -- the column reports `n/a`
rather than a number from a different measurement.

! Latency here is END TO END through the deployed server, so it carries the
embed call and both global arms. The dense arm is ~31% of it (`HARDWARE.md`
§3): a 3x saving on that arm is not a 3x saving on the row.
"""

from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

QUERIES = Path(__file__).parent / "queries.json"
HTTP = os.environ.get("VERA_HTTP", "http://localhost:8081").rstrip("/")
NAME = os.environ.get("VERA_CONTAINER", "vera-mcp")
BATCHES = [int(b) for b in os.environ.get("BATCHES", "1,2,3,5").split(",")]
REPEATS = int(os.environ.get("REPEATS", "3"))
WARMUP = int(os.environ.get("WARMUP", "3"))
# ! out_of_domain queries are refused at layer 1 and never reach a cluster, so
# timing them would dilute the measurement with the one path this knob cannot
# touch.
SKIP = {"out_of_domain"}


def docker(*args: str) -> str:
    return subprocess.run(
        ("docker",) + args, capture_output=True, text=True, check=True
    ).stdout.strip()


def clone_with(batch: int) -> None:
    """Restart the engine with one variable changed and nothing else."""
    spec = json.loads(docker("inspect", NAME))[0]
    env = [e for e in spec["Config"]["Env"] if not e.startswith("CLUSTER_BATCH=")]
    host, net = spec["HostConfig"], spec["NetworkSettings"]["Networks"]
    (netname,) = tuple(net) or ("bridge",)

    argv = ["run", "-d", "--name", NAME, "--network", netname]
    # ! The alias, not just the network. The engine resolves `db` from
    # DATABASE_URL, and a container reattached without its aliases starts,
    # fails the canary and refuses to serve -- correctly, but confusingly.
    for alias in net[netname].get("Aliases") or []:
        argv += ["--network-alias", alias]
    for bind in host.get("Binds") or []:
        argv += ["-v", bind]
    for port, binds in (host.get("PortBindings") or {}).items():
        for b in binds:
            argv += ["-p", f"{b.get('HostIp') or ''}:{b.get('HostPort')}:{port}".lstrip(":")]
    if host.get("Memory"):
        argv += ["-m", str(host["Memory"])]
    if host.get("NanoCpus"):
        argv += ["--cpus", str(host["NanoCpus"] / 1e9)]
    if host.get("ReadonlyRootfs"):
        argv += ["--read-only"]
    for e in env + [f"CLUSTER_BATCH={batch}"]:
        argv += ["-e", e]
    argv.append(spec["Config"]["Image"])

    docker("rm", "-f", NAME)
    docker(*argv)


def await_health(timeout: float = 90.0) -> None:
    """! A container that is Up is not a server that is serving: startup reads
    centroids and runs the canary, and a canary failure is a REFUSAL to serve
    that would otherwise be timed as a very fast query."""
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        try:
            with urllib.request.urlopen(f"{HTTP}/health", timeout=5) as r:
                if r.status == 200:
                    return
        except (urllib.error.URLError, OSError):
            time.sleep(1)
    sys.exit(f"{NAME} never became healthy — `docker logs {NAME}`")


def peak_bytes() -> int | None:
    """cgroup v2 `memory.peak`; None where the kernel does not expose it.

    ! Scoped by the container's lifetime, which is why `clone_with` restarting
    it is load-bearing rather than tidy: it is what makes each row's high-water
    mark belong to that row."""
    r = subprocess.run(
        ("docker", "exec", NAME, "cat", "/sys/fs/cgroup/memory.peak"),
        capture_output=True, text=True,
    )
    return int(r.stdout.strip()) if r.returncode == 0 else None


def search(query: str) -> float:
    body = json.dumps({
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "search_knowledge", "arguments": {"query": query}},
    }).encode()
    req = urllib.request.Request(
        f"{HTTP}/mcp", data=body, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=120) as r:
        r.read()
    return (time.perf_counter() - t0) * 1000


def pct(xs: list[float], p: float) -> float:
    s = sorted(xs)
    return s[min(len(s) - 1, int(round((len(s) - 1) * p)))]


def main() -> None:
    cases = [c["query"] for c in
             json.loads(QUERIES.read_text(encoding="utf-8"))["queries"]
             if c["type"] not in SKIP]
    print(f"{len(cases)} queries x {REPEATS} repeats · container {NAME} · {HTTP}",
          flush=True)

    rows = []
    for batch in BATCHES:
        print(f"\n  CLUSTER_BATCH={batch} · restarting", flush=True)
        clone_with(batch)
        await_health()
        for q in cases[:WARMUP]:
            search(q)

        lat = [search(q) for _ in range(REPEATS) for q in cases]
        peak = peak_bytes()
        rows.append((batch, lat, peak))
        print(f"  p50 {pct(lat, 0.50):.0f} ms · p95 {pct(lat, 0.95):.0f} ms"
              + (f" · peak {peak / 1e6:.0f} MB" if peak else " · peak n/a"),
              flush=True)

    base = statistics.median(rows[0][1])
    print(f"\n{'CLUSTER_BATCH':>14}{'p50':>9}{'p95':>9}{'vs batch=1':>13}"
          f"{'peak RSS':>11}")
    for batch, lat, peak in rows:
        p50 = pct(lat, 0.50)
        print(f"{batch:>14}{p50:>8.0f}{pct(lat, 0.95):>8.0f} ms"
              f"{p50 - base:>+11.0f} ms"
              + (f"{peak / 1e6:>8.0f} MB" if peak else f"{'n/a':>11}"))
    print(f"\n! Engine left running at CLUSTER_BATCH={BATCHES[-1]}.")


if __name__ == "__main__":
    main()
