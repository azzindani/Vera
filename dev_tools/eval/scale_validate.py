"""Validate the scoring layer at both corpus scales · 355K and 5.1M.

    MSYS_NO_PATHCONV=1 python -X utf8 dev_tools/eval/scale_validate.py
    MSYS_NO_PATHCONV=1 python -X utf8 dev_tools/eval/scale_validate.py --only vera5m

Restarts the engine against each database in turn, replays the labelled query set
through every shipped algorithm, and asserts the properties that must hold at any
size. It reports latency and resident memory beside them, because a ranking
guarantee that only holds while the corpus is small is not a guarantee.

WHAT IS BEING VALIDATED, AND WHAT IS NOT

  validated    an assembled composition ranks IDENTICALLY to the compiled
               shorthand, at both scales and on every query
  validated    the same request twice returns the same ranking -- `EVAL.md` §4:
               a comparison is only worth making against an engine that does
  validated    the algorithms are genuinely distinct, so selection is not theatre
  validated    what assembly costs in latency, measured rather than assumed ·
               each query is warmed with a discarded call first, or whichever
               algorithm runs first pays for the cold embedding and the other
               looks faster by construction
  validated    the engine's resident set at each scale, from the cgroup

  ! NOT recall. `vera5m` is a 14x synthetic clone (`scale_probe.py`): every answer
  chunk exists fourteen times, so every arm finds it trivially and any Recall@5
  from it is meaningless. Ranking EQUIVALENCE is still valid there -- two scorers
  over the same pool must agree whatever the pool is -- and so is latency, which
  is the reason the 5M corpus exists.

  ! NOT a claim that the fitted weights are right. They were fitted before the
  2026-09-19 re-embed and need refitting (`SCORING.md` §9). This validates that
  the layer is faithful and fast, ✗ that its numbers are good.

! Restarting the engine is disruptive, and this tool does it repeatedly. It
restores the starting database on the way out, including after a failure --
but a HARD KILL skips the `finally` entirely and leaves the engine pointed at
whichever database it was last measuring. After interrupting this, check

    docker inspect -f '{{json .Config.Env}}' vera-mcp

before trusting any later number: a synthetic 14x clone answers every query
happily and looks exactly like the real corpus in the output.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[2]
QUERIES = ROOT / "dev_tools/eval/queries.json"
ENGINE = "vera-mcp"
HTTP = os.environ.get("VERA_HTTP", "http://localhost:8081")

# The algorithms this validates, and what each one is here to prove.
COMPILED = "balanced"
COMPOSED = "balanced_composed"
LITERAL = "literal"


def sh(args: list[str], timeout: int = 900) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, capture_output=True, text=True, timeout=timeout)


def call(name: str, **arguments) -> dict:
    body = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        }
    ).encode()
    req = urllib.request.Request(
        f"{HTTP.rstrip('/')}/mcp", data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        payload = json.load(r)
    if "result" not in payload:
        raise RuntimeError(json.dumps(payload.get("error", payload))[:300])
    return json.loads(payload["result"]["content"][0]["text"])


def current_database() -> str:
    """! Read from the CONTAINER, not from `.env`. The engine may have been
    started with an override, and restoring the wrong database afterwards is
    worse than not restoring at all."""
    out = sh(["docker", "inspect", "-f", "{{json .Config.Env}}", ENGINE]).stdout
    for entry in json.loads(out or "[]"):
        if entry.startswith("DATABASE_URL="):
            for part in entry.split():
                if part.startswith("dbname="):
                    return part.split("=", 1)[1]
    return ""


def container_env() -> dict[str, str]:
    out = sh(["docker", "inspect", "-f", "{{json .Config.Env}}", ENGINE]).stdout
    env = {}
    for entry in json.loads(out or "[]"):
        k, _, v = entry.partition("=")
        env[k] = v
    return env


def point_at(database: str) -> None:
    """Recreate the engine against `database` and wait until it will answer."""
    env = container_env()
    dsn = " ".join(
        f"dbname={database}" if p.startswith("dbname=") else p
        for p in env["DATABASE_URL"].split()
    )
    overrides = {
        "DATABASE_URL": dsn,
        "EMBED_ENDPOINT": env.get("EMBED_ENDPOINT", ""),
        "ALGORITHMS_PATH": env.get("ALGORITHMS_PATH", ""),
        "VOCABULARY_PATH": env.get("VOCABULARY_PATH", ""),
    }
    cmd = [
        "docker", "compose",
        "-f", "docker-compose.yml", "-f", "docker-compose.vps.yml",
        "up", "-d", "--no-deps", "--force-recreate", "engine",
    ]
    p = subprocess.run(
        cmd, cwd=ROOT, capture_output=True, text=True, timeout=600,
        env={**os.environ, **{k: v for k, v in overrides.items() if v}},
    )
    if p.returncode != 0:
        raise RuntimeError(f"compose: {(p.stdout + p.stderr)[-400:]}")

    # ! Poll /mcp, ✗ /health. The engine binds its port before it loads centroids
    # and runs the startup canary, so a socket that accepts proves only that the
    # process started. At 5.1M rows the centroid load is the slow part.
    deadline = time.monotonic() + 300
    last = ""
    while time.monotonic() < deadline:
        try:
            call("list_domains")
            return
        except (urllib.error.URLError, RuntimeError, OSError, json.JSONDecodeError) as e:
            last = str(e)[:120]
            time.sleep(2)
    raise RuntimeError(f"engine never became ready · {last}")


def engine_memory_mb() -> tuple[float, float]:
    """`(anon, total)` MB from the container's cgroup.

    ! anon separately, because `memory.current` includes page cache the kernel
    would reclaim under pressure — counting it as the engine's cost is what makes
    a flat resident set look like a growing one.
    """
    stat = sh(["docker", "exec", ENGINE, "cat", "/sys/fs/cgroup/memory.stat"]).stdout
    current = sh(["docker", "exec", ENGINE, "cat", "/sys/fs/cgroup/memory.current"]).stdout
    anon = 0
    for line in stat.splitlines():
        if line.startswith("anon "):
            anon = int(line.split()[1])
            break
    total = int(current.strip() or 0)
    return anon / 1e6, total / 1e6


def load_queries(limit: int) -> list[str]:
    raw = json.loads(QUERIES.read_text(encoding="utf-8"))
    cases = raw if isinstance(raw, list) else raw.get("queries", [])
    out = [c["query"] for c in cases if c.get("query")]
    return out[:limit]


def ids(response: dict) -> list[str]:
    return [r["id"] for r in response.get("results", [])]


def validate(database: str, queries: list[str]) -> dict:
    print(f"\n{'=' * 72}\n{database}\n{'=' * 72}", flush=True)
    point_at(database)

    rows = int(
        sh(["docker", "exec", "vera-db", "psql", "-U", "vera", "-d", database,
            "-tAc", "SELECT count(*) FROM chunks"]).stdout.strip()
        or 0
    )
    idle_anon, idle_total = engine_memory_mb()
    print(f"  {rows:,} chunks · engine idle {idle_anon:.0f} MB anon "
          f"({idle_total:.0f} MB incl. cache)", flush=True)

    listed = call("list_algorithms")
    available = {a["name"] for a in listed["algorithms"]}
    have_composed = COMPOSED in available

    latency: dict[str, list[float]] = {COMPILED: [], COMPOSED: [], LITERAL: []}
    mismatches: list[tuple[str, list[str], list[str]]] = []
    nondeterministic: list[str] = []
    distinct = 0
    empty = 0

    for i, q in enumerate(queries, 1):
        # ! One discarded call per query, BEFORE anything is timed. Without it
        # whichever algorithm runs first absorbs that query's cold embedding and
        # cold pages, and the second looks faster by construction -- the first
        # run of this harness reported assembly at -632 ms, which is not a
        # speed-up, it is the warm-up of the call that preceded it.
        call("search_knowledge", query=q, profile=COMPILED)

        t0 = time.monotonic()
        a = call("search_knowledge", query=q, profile=COMPILED)
        latency[COMPILED].append((time.monotonic() - t0) * 1000)
        ids_a = ids(a)
        if not ids_a:
            empty += 1

        # ! The property this whole exercise exists for: assembly must be
        # faithful. Checked per query, ✗ on an aggregate -- two scorers can agree
        # on average and disagree on individual queries, which is the shape of
        # error that quietly moves a fitted weight.
        if have_composed:
            t0 = time.monotonic()
            b = call("search_knowledge", query=q, profile=COMPOSED)
            latency[COMPOSED].append((time.monotonic() - t0) * 1000)
            ids_b = ids(b)
            if ids_a != ids_b:
                mismatches.append((q, ids_a, ids_b))

        # Determinism, and distinctness from factors-off.
        again = ids(call("search_knowledge", query=q, profile=COMPILED))
        if again != ids_a:
            nondeterministic.append(q)

        t0 = time.monotonic()
        lit = call("search_knowledge", query=q, profile=LITERAL)
        latency[LITERAL].append((time.monotonic() - t0) * 1000)
        if ids(lit) != ids_a:
            distinct += 1

        if i % 10 == 0:
            print(f"    {i}/{len(queries)} queries", flush=True)

    busy_anon, busy_total = engine_memory_mb()

    def pct(values: list[float], p: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        k = min(int(len(ordered) * p), len(ordered) - 1)
        return ordered[k]

    return {
        "database": database,
        "rows": rows,
        "queries": len(queries),
        "empty": empty,
        "mismatches": mismatches,
        "nondeterministic": nondeterministic,
        "distinct_from_literal": distinct,
        "composed_checked": have_composed,
        "idle_mb": idle_anon,
        "peak_mb": busy_anon,
        "peak_total_mb": busy_total,
        "p50": {k: statistics.median(v) if v else 0.0 for k, v in latency.items()},
        "p95": {k: pct(v, 0.95) for k, v in latency.items()},
    }


def report(results: list[dict]) -> bool:
    print(f"\n{'=' * 72}\nVALIDATION\n{'=' * 72}")
    ok = True
    for r in results:
        print(f"\n{r['database']} · {r['rows']:,} chunks · {r['queries']} queries")
        checks = [
            (
                "composed ranking == compiled ranking, every query",
                r["composed_checked"] and not r["mismatches"],
                "not checked · balanced_composed absent"
                if not r["composed_checked"]
                else f"{len(r['mismatches'])} mismatched",
            ),
            (
                "every ranking reproducible on a second call",
                not r["nondeterministic"],
                f"{len(r['nondeterministic'])} varied between calls",
            ),
            (
                "the algorithms are distinct, so selection is not theatre",
                r["distinct_from_literal"] > 0,
                f"{r['distinct_from_literal']}/{r['queries']} differ from `literal`",
            ),
            (
                "every query returned results",
                r["empty"] == 0,
                f"{r['empty']} returned nothing",
            ),
        ]
        for name, passed, detail in checks:
            ok &= passed
            print(f"  {'ok  ' if passed else 'FAIL'}  {name}")
            if not passed or "differ" in detail:
                print(f"          {detail}")
        for q, a, b in r["mismatches"][:3]:
            print(f"          ! {q[:60]}\n            compiled {a[:3]}\n            composed {b[:3]}")

    print(f"\n{'corpus':<10}{'rows':>12}{'p50 ms':>10}{'p95 ms':>10}"
          f"{'composed':>10}{'literal':>10}{'RSS MB':>9}")
    for r in results:
        print(
            f"{r['database']:<10}{r['rows']:>12,}"
            f"{r['p50'][COMPILED]:>10.0f}{r['p95'][COMPILED]:>10.0f}"
            f"{r['p50'][COMPOSED]:>10.0f}{r['p50'][LITERAL]:>10.0f}"
            f"{r['peak_mb']:>9.0f}"
        )

    # ! What assembly costs, stated as a number rather than "negligible".
    for r in results:
        base, comp = r["p50"][COMPILED], r["p50"][COMPOSED]
        if base and comp:
            delta = comp - base
            print(f"\n{r['database']}: assembly costs {delta:+.0f} ms at p50 "
                  f"({delta / base * 100:+.1f}%) · {r['rows']:,} chunks")

    print("\n! Recall is NOT validated here, and on a synthetic clone it cannot be:")
    print("  every answer chunk exists 14 times. Ranking equivalence and latency are")
    print("  what this measures. `e2e.py` owns recall, against the real corpus.")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="one database, e.g. vera2 or vera5m")
    ap.add_argument("--databases", default="vera2,vera5m")
    ap.add_argument("--queries", type=int, default=30)
    args = ap.parse_args()

    targets = [args.only] if args.only else args.databases.split(",")
    queries = load_queries(args.queries)
    if not queries:
        sys.exit("no queries · check dev_tools/eval/queries.json")

    started_on = current_database()
    print(f"engine currently on `{started_on}` · will be restored", flush=True)

    results = []
    try:
        for db in targets:
            results.append(validate(db.strip(), queries))
    finally:
        # ! Restored even on failure. Leaving the engine pointed at a synthetic
        # 14x clone is how a later measurement gets quoted as the real corpus.
        if started_on and current_database() != started_on:
            print(f"\nrestoring engine to `{started_on}`", flush=True)
            try:
                point_at(started_on)
            except RuntimeError as e:
                print(f"! could not restore: {e}", flush=True)

    sys.exit(0 if results and report(results) else 1)


if __name__ == "__main__":
    main()
