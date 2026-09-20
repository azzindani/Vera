"""Scenario tests for algorithm selection and the corpus's scoring vocabulary.

    VERA_HTTP=http://localhost:8081 python dev_tools/eval/algorithm_scenarios.py
    VERA_HTTP=http://localhost:8081 python dev_tools/eval/algorithm_scenarios.py --only B

! These are **behaviour** scenarios, ✗ a recall measurement. Nothing here claims a
Recall@5. `e2e.py` does that; this asks whether the selection surface does what it
says, against a running deployment and a real corpus, which is the half the unit
tests cannot reach:

  - the schema an agent reads and the names the engine accepts are the same list
  - an unknown name is refused rather than silently served the default
  - two algorithms actually rank differently on the same query
  - `experimental` is set whenever nobody measured the ranking
  - a bad registry or an unusable weight KILLS THE PROCESS instead of failing the
    one request that happens to name it
  - a corpus that declares its own vocabulary overrides the engine's fallback

Group A runs against the server already up. Groups B and C start their own engine
container with deliberately broken configuration and assert it refuses to serve --
so they need docker and the compose stack, and they never touch the running one.

Group D runs SQL through the `vera-db` container -- the stack publishes no port
for Postgres, so a host-side driver cannot reach it and the connection string never
has to leave the container. It creates a throwaway database, stamps a scoring
vocabulary into it, reads it back through the engine's own query, and drops it.
! It never writes to the corpus under test.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"
_results: list[tuple[str, str, str, str]] = []


def record(group: str, name: str, status: str, detail: str = "") -> None:
    _results.append((group, name, status, detail))
    mark = {PASS: "ok  ", FAIL: "FAIL", SKIP: "skip"}[status]
    print(f"  {mark}  {name}" + (f"\n          {detail}" if detail else ""), flush=True)


class Http:
    """The running deployment, over JSON-RPC."""

    def __init__(self, base: str):
        self.base = base.rstrip("/")
        self._n = 0

    def call(self, method: str, params: dict) -> dict:
        self._n += 1
        body = json.dumps(
            {"jsonrpc": "2.0", "id": self._n, "method": method, "params": params}
        ).encode()
        req = urllib.request.Request(
            f"{self.base}/mcp", data=body, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=180) as r:
            return json.load(r)

    def tool(self, name: str, **args) -> dict:
        r = self.call("tools/call", {"name": name, "arguments": args})
        if "result" not in r:
            return {"success": False, "error": json.dumps(r.get("error", r))}
        return json.loads(r["result"]["content"][0]["text"])

    def schema(self, tool: str) -> dict | None:
        tools = self.call("tools/list", {})["result"]["tools"]
        return next((t for t in tools if t["name"] == tool), None)


# ---------------------------------------------------------------------------
# A · the selection surface, against the running server
# ---------------------------------------------------------------------------

QUERY = "sanksi keterlambatan pembayaran pajak daerah"


def group_a(h: Http) -> None:
    print("\nA · selection surface (running deployment)")

    # A1 -- the tool exists at all.
    try:
        listed = h.tool("list_algorithms")
    except urllib.error.URLError as e:
        record("A", "list_algorithms responds", FAIL, f"{e} · is the engine rebuilt?")
        return
    if not listed.get("success"):
        record("A", "list_algorithms responds", FAIL, str(listed)[:200])
        return
    names = [a["name"] for a in listed["algorithms"]]
    record("A", "list_algorithms responds", PASS, f"{len(names)} · {', '.join(names)}")

    # A2 -- ! the schema and the engine must agree. An operator who adds a fitted
    # algorithm to the file and finds the schema never mentions it has an
    # algorithm no agent can ever select.
    schema = h.schema("search_knowledge")
    offered = schema["inputSchema"]["properties"]["profile"].get("enum", [])
    fitted = [a["name"] for a in listed["algorithms"] if not a["experimental"]]
    record(
        "A",
        "the schema offers exactly the fitted algorithms",
        PASS if sorted(offered) == sorted(fitted) else FAIL,
        f"schema {offered} · fitted {fitted}",
    )

    # A3 -- an unfitted algorithm is reachable but not advertised.
    unfitted = [a["name"] for a in listed["algorithms"] if a["experimental"]]
    if not unfitted:
        record("A", "an unfitted algorithm is reachable but not offered", SKIP,
               "no unfitted algorithm in this registry")
    else:
        name = unfitted[0]
        r = h.tool("search_knowledge", query=QUERY, profile=name)
        ok = r.get("success") and name not in offered
        record(
            "A",
            "an unfitted algorithm is reachable but not offered",
            PASS if ok else FAIL,
            f"`{name}` served={r.get('success')} offered={name in offered}",
        )

    # A4 -- ! the failure this prevents: a caller misspells a name, is served the
    # default, and reports a ranking it cannot reproduce.
    r = h.tool("search_knowledge", query=QUERY, profile="balancd")
    refused = not r.get("success")
    lists_valid = any(n in json.dumps(r) for n in names)
    record(
        "A",
        "an unknown algorithm is refused and the error lists the valid ones",
        PASS if refused and lists_valid else FAIL,
        f"refused={refused} lists_valid={lists_valid} · {str(r.get('error', ''))[:120]}",
    )

    # A5 -- every response carries what ranked it, whether or not the caller chose.
    r = h.tool("search_knowledge", query=QUERY)
    applied = r.get("applied", {})
    record(
        "A",
        "a response with no options still echoes the algorithm that ranked it",
        PASS if applied.get("profile") and not applied.get("experimental") else FAIL,
        f"profile={applied.get('profile')} experimental={applied.get('experimental')}",
    )

    # A6 -- two algorithms must actually rank differently, or selection is theatre.
    # `literal` is factors-off; `balanced` applies the metadata prior.
    if "literal" in names and "balanced" in names:
        a = h.tool("search_knowledge", query=QUERY, profile="balanced")
        b = h.tool("search_knowledge", query=QUERY, profile="literal")
        ids_a = [x["id"] for x in a.get("results", [])]
        ids_b = [x["id"] for x in b.get("results", [])]
        record(
            "A",
            "balanced and literal produce different orderings",
            PASS if ids_a and ids_b and ids_a != ids_b else FAIL,
            f"balanced {ids_a[:3]} · literal {ids_b[:3]}",
        )
    else:
        record("A", "balanced and literal produce different orderings", SKIP,
               "both algorithms not present")

    # A7 -- determinism. ! A comparison on 40 cases is only worth making against an
    # engine that returns the same answer twice (`EVAL.md` §4).
    one = h.tool("search_knowledge", query=QUERY, profile="balanced")
    two = h.tool("search_knowledge", query=QUERY, profile="balanced")
    same = [x["id"] for x in one.get("results", [])] == [
        x["id"] for x in two.get("results", [])
    ]
    record("A", "the same request twice returns the same ranking", PASS if same else FAIL)

    # A8 -- raw weights are honoured, echoed, and marked.
    weights = {
        "relevance_floor": 0.3,
        "authority": 0.0,
        "structural": 0.0,
        "temporal": 0.0,
        "completeness": 0.0,
        "topical": 0.0,
    }
    r = h.tool("search_knowledge", query=QUERY, factor_weights=weights)
    ap = r.get("applied", {})
    echoed = ap.get("factor_weights", {})
    # ! Compared with a tolerance, and it has to be. The weights are f32 in the
    # engine and JSON numbers are f64, so 0.3 comes back as 0.30000001192092896 --
    # the same value, widened. That is a property of the wire, ✗ a defect, and
    # rounding it in the engine would be inventing precision the value never had.
    same = echoed.keys() == weights.keys() and all(
        abs(echoed[k] - v) < 1e-6 for k, v in weights.items()
    )
    record(
        "A",
        "raw weights are marked experimental and echoed back",
        PASS if ap.get("experimental") and same else FAIL,
        f"experimental={ap.get('experimental')} echoed={echoed}",
    )

    # A9 -- ! the retry loop, which is the escalation design. An agent that judges a
    # ranking unfit calls again; nothing in the engine votes or escalates for it.
    desc = schema["inputSchema"]["properties"]["profile"].get("description", "")
    record(
        "A",
        "the schema tells an agent to retry rather than reinterpret",
        PASS if "call again" in desc and "list_algorithms" in desc else FAIL,
        desc[:110],
    )

    # A10 -- a clamp is reported, never silent.
    r = h.tool("search_knowledge", query=QUERY, top_k=500)
    clamped = r.get("applied", {}).get("clamped", [])
    record(
        "A",
        "asking past a ceiling is narrowed and the narrowing is reported",
        PASS if clamped else FAIL,
        str(clamped)[:120],
    )


# ---------------------------------------------------------------------------
# B / C · configurations that must kill the process
# ---------------------------------------------------------------------------

REFUSALS = [
    (
        "B1",
        "a prior bound above 2.00x refuses to start",
        # ! The unconstrained fit: best Recall@5 ever measured here, MRR below
        # factors-off. It must not be loadable by writing it into a file.
        {"ALGORITHMS_PATH": "/cfg/bad_bound.json"},
        {
            "bad_bound.json": {
                "algorithms": {
                    "balanced": {
                        "description": "d",
                        "factors": {
                            "relevance_floor": 0.3,
                            "authority": 1.0,
                            "structural": 0.25,
                            "temporal": 0.0,
                            "completeness": 1.0,
                            "topical": 0.5,
                        },
                    }
                }
            }
        },
        ["3.75", "2.00"],
    ),
    (
        "B2",
        "a registry with no default refuses to start",
        {"ALGORITHMS_PATH": "/cfg/no_default.json"},
        {
            "no_default.json": {
                "algorithms": {
                    "sanction": {
                        "description": "d",
                        "factors": {
                            "relevance_floor": 0.3,
                            "authority": 0.5,
                            "structural": 0.25,
                            "temporal": 0.0,
                            "completeness": 0.0,
                            "topical": 0.25,
                        },
                    }
                }
            }
        },
        ["balanced"],
    ),
    (
        "B3",
        "a typo in the registry refuses to start rather than being ignored",
        {"ALGORITHMS_PATH": "/cfg/typo.json"},
        {
            "typo.json": {
                "algorithms": {
                    "balanced": {
                        "description": "d",
                        "weights": {},
                        "factors": {
                            "relevance_floor": 0.3,
                            "authority": 0.5,
                            "structural": 0.25,
                            "temporal": 0.0,
                            "completeness": 0.0,
                            "topical": 0.25,
                        },
                    }
                }
            }
        },
        ["ALGORITHMS_PATH"],
    ),
    (
        "C1",
        "a weight the vocabulary cannot evaluate refuses to start",
        # ! The point of declaring a vocabulary. A weight of 0.5 on a factor whose
        # table is empty is not a small effect, it is NO effect -- and it is
        # indistinguishable from a weight measured and found not to help.
        {"VOCABULARY_PATH": "/cfg/empty_vocab.json"},
        {"empty_vocab.json": {"authority": {}, "structural": []}},
        ["authority", "structural"],
    ),
    (
        "C2",
        "an unrecognised structural field refuses to start",
        {"VOCABULARY_PATH": "/cfg/bad_field.json"},
        {
            "bad_field.json": {
                "structural": [{"label": "X", "field": "heading", "score": 0.5}]
            }
        },
        ["heading"],
    ),
]


def run_refusal(case, image: str, cfg_dir, compose_env: dict) -> tuple[str, str]:
    """Start a throwaway engine with broken config; it must exit non-zero and say why."""
    _, _, env, files, expect = case
    for name, body in files.items():
        (cfg_dir / name).write_text(json.dumps(body, indent=1), encoding="utf-8")
    # The scenario's own files live at /scenario so they cannot shadow /cfg.
    env = {k: v.replace("/cfg/", "/scenario/") for k, v in env.items()}
    expect = [e.replace("/cfg/", "/scenario/") for e in expect]

    # ! Every bind the real engine has, plus this scenario's config. Without
    # /vocab the process dies reading the BM25 vocabulary -- BEFORE the check under
    # test -- and the scenario would report a refusal it did not cause.
    args = ["docker", "run", "--rm", "--network", compose_env["network"]]
    for bind in compose_env["binds"]:
        args += ["-v", bind]
    args += ["-v", f"{cfg_dir.as_posix()}:/scenario:ro"]
    for k, v in {**compose_env["env"], **env}.items():
        args += ["-e", f"{k}={v}"]
    args.append(image)

    p = subprocess.run(args, capture_output=True, text=True, timeout=180)
    out = (p.stdout + p.stderr).strip()
    if p.returncode == 0:
        return FAIL, "the process started · it must refuse"
    missing = [e for e in expect if e not in out]
    if missing:
        return FAIL, f"exited {p.returncode} but did not mention {missing} · {out[-220:]}"
    return PASS, f"exited {p.returncode} · {out.splitlines()[-1][:150]}"


def groups_bc(only: str | None) -> None:
    import pathlib
    import tempfile

    print("\nB/C · configurations that must refuse to serve")

    probe = subprocess.run(
        ["docker", "inspect", "-f",
         "{{range $k,$v := .NetworkSettings.Networks}}{{$k}}{{end}}", "vera-mcp"],
        capture_output=True, text=True,
    )
    if probe.returncode != 0:
        record("B", "engine container available", SKIP, "vera-mcp is not running")
        return
    network = probe.stdout.strip().splitlines()[0]

    binds = json.loads(
        subprocess.run(
            ["docker", "inspect", "-f", "{{json .HostConfig.Binds}}", "vera-mcp"],
            capture_output=True, text=True,
        ).stdout
        or "[]"
    ) or []

    env_probe = subprocess.run(
        ["docker", "inspect", "-f", "{{json .Config.Env}}", "vera-mcp"],
        capture_output=True, text=True,
    )
    env = {}
    for entry in json.loads(env_probe.stdout):
        k, _, v = entry.partition("=")
        if k in ("DATABASE_URL", "EMBED_ENDPOINT", "BM25_VOCAB", "TRANSPORT",
                 "RUST_LOG", "HTTP_ADDR"):
            env[k] = v
    # ! stdio, so the process runs its startup checks and exits instead of binding
    # a port and waiting. A refusal is what is under test, not a server.
    env["TRANSPORT"] = "stdio"

    image_probe = subprocess.run(
        ["docker", "inspect", "-f", "{{.Config.Image}}", "vera-mcp"],
        capture_output=True, text=True,
    )
    image = image_probe.stdout.strip()

    with tempfile.TemporaryDirectory() as tmp:
        cfg = pathlib.Path(tmp)
        for case in REFUSALS:
            gid = case[0][0]
            if only and only != gid:
                continue
            try:
                status, detail = run_refusal(
                    case, image, cfg,
                    {"network": network, "env": env, "binds": binds},
                )
            except subprocess.TimeoutExpired:
                status, detail = FAIL, "timed out · a refusal must be immediate"
            record(gid, f"{case[0]} {case[1]}", status, detail)


# ---------------------------------------------------------------------------
# D · the corpus's own vocabulary wins
# ---------------------------------------------------------------------------


def psql(db: str, sql: str) -> tuple[int, str]:
    """One statement, through the database container.

    ! `docker exec`, ✗ a client on the host. The compose stack publishes no port
    for Postgres -- deliberately, it is reachable only on the internal network --
    so a host-side driver cannot connect at all, and requiring one would make this
    scenario untestable on the deployment it is meant to test. It also means the
    connection string never leaves the container, which invariant 14 is about.
    """
    p = subprocess.run(
        ["docker", "exec", "-i", "vera-db", "psql", "-U", "vera", "-d", db, "-tAc", sql],
        capture_output=True,
        text=True,
        timeout=120,
    )
    return p.returncode, (p.stdout + p.stderr).strip()


def group_d() -> None:
    """! Writes to a THROWAWAY database, never to the corpus under test.

    This is the one path unit tests cannot reach: `corpus_meta.scoring_vocabulary`
    is read from a live row, and both live corpora predate the column.
    """
    print("\nD · corpus_meta declares the vocabulary")

    probe = subprocess.run(
        ["docker", "inspect", "-f", "{{.State.Running}}", "vera-db"],
        capture_output=True, text=True,
    )
    if probe.stdout.strip() != "true":
        record("D", "corpus vocabulary overrides the fallback", SKIP, "vera-db is not running")
        return

    target = "vera_vocab_probe"
    vocab = {
        "authority": {"INTERNET STANDARD": 5, "PROPOSED STANDARD": 3},
        "authority_scale": 5,
        "annex": ["APPENDIX"],
        "operative": ["SECTION"],
        "annex_score": 0.2,
        "operative_score": 1.0,
        "labelled_score": 0.6,
        "terms": ["must", "shall"],
        "stopwords": ["the", "and"],
        "min_term_chars": 3,
    }
    literal = json.dumps(vocab).replace("'", "''")

    # ! The read the ENGINE performs, verbatim. `to_jsonb(corpus_meta) ->> ...`
    # rather than naming the column, which is what lets one query serve a table
    # that has it and a table that does not.
    engine_read = (
        "SELECT to_jsonb(corpus_meta) ->> 'scoring_vocabulary'"
        " FROM corpus_meta ORDER BY created_at DESC LIMIT 1"
    )

    try:
        psql("postgres", f'DROP DATABASE IF EXISTS "{target}"')
        rc, out = psql("postgres", f'CREATE DATABASE "{target}"')
        if rc != 0:
            record("D", "throwaway database", FAIL, out[:180])
            return

        rc, out = psql(
            target,
            "CREATE TABLE corpus_meta (id TEXT PRIMARY KEY, created_at TIMESTAMPTZ"
            " DEFAULT now(), dense_model TEXT, dense_dim INTEGER, dense_pooling TEXT,"
            " dense_normalize BOOLEAN, sparse_scheme TEXT, sparse_dim INTEGER,"
            " sparse_vocab_sha256 TEXT, dense_instruction TEXT,"
            " scoring_vocabulary JSONB);"
            " INSERT INTO corpus_meta (id, dense_model, dense_dim, dense_pooling,"
            " dense_normalize, sparse_scheme, sparse_dim, sparse_vocab_sha256,"
            f" scoring_vocabulary) VALUES ('probe','m',8,'last',true,'bm25',8,'x',"
            f" '{literal}'::jsonb)",
        )
        if rc != 0:
            record("D", "a declared vocabulary reads back", FAIL, out[:200])
            return

        rc, out = psql(target, engine_read)
        got = json.loads(out) if rc == 0 and out else None
        record(
            "D",
            "a declared vocabulary reads back through the engine's own query",
            PASS if got == vocab else FAIL,
            f"{len(got.get('authority', {})) if got else 0} authority labels",
        )

        # ! And the same query against a table WITHOUT the column must return NULL
        # rather than erroring. Every corpus loaded before Ravel added it is in that
        # state, including both of this machine's.
        psql(target, "ALTER TABLE corpus_meta DROP COLUMN scoring_vocabulary")
        rc, out = psql(target, engine_read)
        record(
            "D",
            "a pre-column corpus reads NULL rather than failing the query",
            PASS if rc == 0 and out == "" else FAIL,
            f"rc={rc} out={out[:120]!r}",
        )

        # ! And the live corpus really is in that state, which is why no ranking
        # number moved when this shipped.
        rc, out = psql("vera2", engine_read)
        record(
            "D",
            "the live corpus declares no vocabulary, so it takes the fallback",
            PASS if rc == 0 and out == "" else FAIL,
            f"rc={rc} out={out[:120]!r}",
        )
    finally:
        psql(
            "postgres",
            "SELECT pg_terminate_backend(pid) FROM pg_stat_activity"
            f" WHERE datname = '{target}' AND pid <> pg_backend_pid()",
        )
        psql("postgres", f'DROP DATABASE IF EXISTS "{target}"')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="run one group: A, B, C or D")
    args = ap.parse_args()

    base = os.environ.get("VERA_HTTP")
    if args.only in (None, "A"):
        if base:
            group_a(Http(base))
        else:
            print("\nA · skipped · set VERA_HTTP=http://localhost:8081")
    if args.only in (None, "B", "C"):
        groups_bc(args.only if args.only in ("B", "C") else None)
    if args.only in (None, "D"):
        group_d()

    print("\n" + "-" * 68)
    for status in (FAIL, SKIP, PASS):
        n = sum(1 for r in _results if r[2] == status)
        if n:
            print(f"{status:>5}  {n}")
    failed = [r for r in _results if r[2] == FAIL]
    if failed:
        print("\nfailed:")
        for g, name, _, detail in failed:
            print(f"  [{g}] {name}\n      {detail}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
