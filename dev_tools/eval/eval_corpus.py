"""Serve one corpus through the engine and score it: a measurement in one command.

    python dev_tools/eval/eval_corpus.py ca_laws
    python dev_tools/eval/eval_corpus.py id_reg_markdown --split fresh --label v6
    python dev_tools/eval/eval_corpus.py ca_laws --set DOMAIN_FLOOR=0.45   # sweep a knob

Everything a run depends on is in git: how the corpus is served (`corpora.json`), the
algorithms (`config/`), the overlay (`docker-compose.eval.yml`) and the question set. The
data is not: the database volume and the Ravel bundle it was prepared from are named, and
`ravel build` / `ravel db prepare` make them again.

1. The corpus's database container is started from its prepared volume if it is not
   running. It is never created over a volume another container holds.
2. The engine is recreated against it (`--no-deps`), with the bundle's vocabulary and
   the corpus's algorithms, and waited on until healthy.
3. `e2e.py` scores the question set through the running engine over HTTP; the per-case
   results are written beside the summary in `--out`.

! The connection string carries the password and is never printed (CLAUDE.md §7.14).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DB_IMAGE = "vera-db:pg16-rum-bm25"
DB_ARGS = [
    "postgres", "-c", "shared_preload_libraries=pg_search", "-c", "shared_buffers=512MB",
    "-c", "maintenance_work_mem=1GB", "-c", "max_connections=32",
]  # fmt: skip


def _docker(*args: str, check: bool = True) -> str:
    done = subprocess.run(["docker", *args], capture_output=True, text=True, check=False)
    if check and done.returncode:
        sys.exit(f"docker {args[0]}: {done.stderr.strip()[-400:]}")
    return done.stdout.strip()


def _password() -> str:
    """The prepared databases' password: `$EVAL_DB_PASSWORD`, else `.env`'s."""
    if os.environ.get("EVAL_DB_PASSWORD"):
        return os.environ["EVAL_DB_PASSWORD"]
    env = ROOT / ".env"
    if env.is_file():
        for line in env.read_text("utf-8").splitlines():
            if line.startswith("POSTGRES_PASSWORD="):
                return line.split("=", 1)[1].strip()
    sys.exit("no database password: set EVAL_DB_PASSWORD, or POSTGRES_PASSWORD in .env")


def _database(db: dict[str, object]) -> None:
    name, volume, port = str(db["container"]), str(db["volume"]), int(str(db["port"]))
    state = _docker("ps", "-a", "--filter", f"name=^{name}$", "--format", "{{.State}}")
    if state == "running":
        return
    if state:
        _docker("start", name)
    else:
        if not _docker("volume", "ls", "-q", "--filter", f"name=^{volume}$"):
            sys.exit(
                f"no volume {volume}: prepare one with `ravel db prepare <bundle> --dsn ...` "
                "into a fresh vera-db:pg16-rum-bm25 (corpora.json `_doc`)"
            )
        holders = _docker("ps", "-q", "--filter", f"volume={volume}")
        if holders:
            sys.exit(f"volume {volume} is held by container {holders}: not opened twice")
        _docker(
            "run", "-d", "--name", name, "-v", f"{volume}:/var/lib/postgresql/data",
            "-p", f"127.0.0.1:{port}:5432", DB_IMAGE, *DB_ARGS,
        )  # fmt: skip
    for _ in range(60):
        if "accepting" in _docker("exec", name, "pg_isready", check=False):
            return
        time.sleep(1)
    sys.exit(f"{name} did not accept connections in 60 s")


def _engine(env: dict[str, str]) -> None:
    # ! `eval.env`, ✗ the operator's `.env`: a knob not in corpora.json runs at its default.
    compose = [
        "compose", "--env-file", str(HERE / "eval.env"),
        "-f", str(ROOT / "docker-compose.yml"),
        "-f", str(ROOT / "docker-compose.eval.yml"),
    ]  # fmt: skip
    for service, extra in (("embed", []), ("engine", ["--force-recreate"])):
        done = subprocess.run(
            ["docker", *compose, "up", "-d", "--no-build", "--no-deps", *extra, service],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        if done.returncode:
            sys.exit(f"compose up {service}: {done.stderr.strip()[-400:]}")
    for _ in range(120):
        status = _docker("ps", "--filter", "name=^vera-mcp$", "--format", "{{.Status}}")
        if "(healthy)" in status:
            break
        time.sleep(2)
    else:
        sys.exit("the engine did not become healthy in 240 s: docker logs vera-mcp")
    logs = subprocess.run(
        ["docker", "logs", "vera-mcp"], capture_output=True, text=True, check=False
    )
    for line in (logs.stdout + logs.stderr).splitlines()[-200:]:
        if any(word in line for word in ("canary", "ready", "algorithm")):
            print("  engine:", line[-160:], file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("corpus")
    parser.add_argument("--split", default="all")
    parser.add_argument("--label", default="run")
    parser.add_argument("--out", type=Path, default=ROOT / ".test" / "eval")
    parser.add_argument("--set", action="append", default=[], help="KEY=VALUE for the engine")
    args = parser.parse_args()

    registry = json.loads((HERE / "corpora.json").read_text("utf-8"))["corpora"]
    if args.corpus not in registry:
        sys.exit(f"no corpus {args.corpus!r} in corpora.json · known: {sorted(registry)}")
    corpus = registry[args.corpus]
    ravel = Path(os.environ.get("RAVEL_ROOT", ROOT.parent / "Ravel"))
    bundle = ravel / corpus["bundle"]
    vocab = bundle / "corpus.bm25.json"
    if not vocab.is_file():
        sys.exit(f"{vocab}: no vocabulary · build the bundle with `ravel build` first")
    db = corpus["db"]
    password = _password()

    _database(db)
    # ! A domain is (database, schema) (`SCALING.md` §4a). Unified mode names a schema; the
    # connection's search path reaches it, `public` after it for the extensions' types.
    schema = db.get("schema")
    # ! `domains` serves every domain a file in config/ lists, one engine over all of
    # them; the schema is then only the harness's, to resolve this question set's labels.
    domains = corpus.get("domains")
    engine_options = f"?options=-csearch_path%3D{schema},public" if schema and not domains else ""
    harness_options = f" options='-c search_path={schema},public'" if schema else ""
    engine_env = dict(
        os.environ,
        DATABASE_URL=(
            f"postgres://{db['user']}:{password}@host.docker.internal:{db['port']}/{db['name']}"
            f"{engine_options}"
        ),
        VOCAB_DIR=str(bundle),
        BM25_VOCAB="/vocab/corpus.bm25.json",
        ALGORITHMS_PATH=f"/cfg/{corpus['algorithms']}",
        **({"DOMAINS_FILE": f"/cfg/{domains}", "BUNDLES_DIR": str(ravel / "work")} if domains else {}),
        **corpus.get("engine", {}),
        **dict(s.split("=", 1) for s in args.set),
    )
    _engine(engine_env)

    args.out.mkdir(parents=True, exist_ok=True)
    stem = args.out / f"{args.corpus}_{args.label}"
    harness_env = dict(
        os.environ,
        BM25_VOCAB=str(vocab),
        VERA_HTTP="http://127.0.0.1:8081",
        DATABASE_URL=(
            f"host=127.0.0.1 port={db['port']} dbname={db['name']} user={db['user']} "
            f"password={password}{harness_options}"
        ),
        # ! The fresh split is its own file (`e2e._queries_file`): read through `queries`,
        # `--split fresh` found no fresh case in it and scored nothing.
        EVAL_QUERIES=corpus["fresh_queries"] if args.split == "fresh" else corpus["queries"],
        EVAL_SPLIT=args.split,
        EVAL_CASES_OUT=f"{stem}.cases.json",
    )
    with open(f"{stem}.txt", "w", encoding="utf-8") as summary:
        done = subprocess.run(
            [sys.executable, "-X", "utf8", str(HERE / "e2e.py")],
            cwd=ROOT,
            env=harness_env,
            stdout=summary,
            stderr=subprocess.STDOUT,
            check=False,
        )
    print(Path(f"{stem}.txt").read_text("utf-8")[-2500:])
    sys.exit(done.returncode)


if __name__ == "__main__":
    main()
