"""Does physical row order decide the COLD cost of the routed dense arm?

    python dev_tools/eval/cluster_layout.py build <dbname>    # side table dense_ord, ordered by cluster_id
    python dev_tools/eval/cluster_layout.py test  <dbname>    # cold + warm dense arm, original vs ordered layout
    python dev_tools/eval/cluster_layout.py drop  <dbname>

Measured on the 5.1M-row replica (`HARDWARE.md` §8): the cold dense arm over 5 probed clusters took 5,477 ms in
the `chunks` heap and 91 ms in a table whose rows are stored in cluster order. Warm: 58 ms against 20 ms.

Why: `kmeans.py` assigns `cluster_id` AFTER ingest, so a cluster's ~2,000 rows sit wherever ingest put them, and
each row also drags its TOAST chunks. Probing one cluster touched ~35,000 buffers for 2,085 rows. In `dense_ord`
the vector is stored PLAIN (inline, ✗ TOASTed) and rows are written `ORDER BY cluster_id, id`, so a cluster is a
contiguous run of ~700 pages.

! Cold means cold. `test` restarts the db container and drops the Docker VM's page cache before each layout,
using a throwaway privileged container -- it writes to /proc/sys/vm/drop_caches of the Docker VM, so run it on a
dev box. The two layouts probe DISJOINT random clusters so neither benefits from the other's reads.
! `build` is a side table, ✗ a change to `chunks`; it costs ~13 GB at 5M rows. `drop` removes it. Nothing here
changes what the engine queries -- using it would be a store/ change plus a migration, not yet built.
! Query vectors are the cluster centroids: a stand-in with a realistic access pattern, not real queries.
"""

from __future__ import annotations

import os
import random
import subprocess
import sys
import time

import psycopg

HOST = os.environ.get("PGHOST", "127.0.0.1")
DB_CONTAINER = os.environ.get("DB_CONTAINER", "vera-db")
HELPER_IMAGE = os.environ.get("HELPER_IMAGE", "alpine:latest")

SQL = {
    "original": "SELECT id FROM chunks WHERE indexable AND dense IS NOT NULL AND cluster_id=%s "
                "ORDER BY dense <=> %s::halfvec LIMIT 60",
    "ordered": "SELECT id FROM dense_ord WHERE cluster_id=%s ORDER BY dense <=> %s::halfvec LIMIT 60",
}


def connect(db: str, retries: int = 60):
    for _ in range(retries):
        try:
            con = psycopg.connect(f"host={HOST} port=5432 dbname={db} user=vera password=vera", autocommit=True)
            con.execute("SELECT 1")
            return con
        except psycopg.OperationalError:
            time.sleep(2)
    sys.exit("database did not come back")


def go_cold(db: str):
    subprocess.run(["docker", "restart", DB_CONTAINER], capture_output=True, check=True)
    subprocess.run(["docker", "run", "--rm", "--privileged", "--pid=host", "--entrypoint", "sh", HELPER_IMAGE,
                    "-c", "sync; echo 3 > /proc/sys/vm/drop_caches"], capture_output=True, check=True)
    return connect(db)


def build(db: str) -> None:
    with connect(db) as con:
        con.execute("SET max_parallel_maintenance_workers=4")
        con.execute("SET work_mem='512MB'")
        con.execute("DROP TABLE IF EXISTS dense_ord")
        con.execute("CREATE TABLE dense_ord (cluster_id int NOT NULL, id text NOT NULL, dense halfvec(1024) NOT NULL)")
        # ! PLAIN before the insert: the default (EXTENDED) compresses and TOASTs a 2 KB value, which puts the
        # vector back in a second, unordered place -- the exact thing this table exists to avoid.
        con.execute("ALTER TABLE dense_ord ALTER COLUMN dense SET STORAGE PLAIN")
        con.execute("INSERT INTO dense_ord SELECT cluster_id, id, dense FROM chunks "
                    "WHERE indexable AND dense IS NOT NULL ORDER BY cluster_id, id")
        con.execute("CREATE INDEX dense_ord_cluster ON dense_ord (cluster_id)")
        con.execute("ANALYZE dense_ord")
        print("built", con.execute("SELECT pg_size_pretty(pg_total_relation_size('dense_ord'))").fetchone()[0])


def test(db: str) -> None:
    ids = random.Random(11).sample(range(2478), 60)
    plan = {"original": [ids[i * 5:(i + 1) * 5] for i in range(6)],
            "ordered": [ids[30 + i * 5:30 + (i + 1) * 5] for i in range(6)]}
    med = lambda v: sorted(v)[len(v) // 2]  # noqa: E731
    for layout in ("original", "ordered"):
        con = go_cold(db)
        cent = dict(con.execute("SELECT id, centroid::text FROM clusters").fetchall())

        def one_pass() -> list[float]:
            out = []
            for probe in plan[layout]:
                t = time.perf_counter()
                for cid in probe:
                    con.execute(SQL[layout], (cid, cent[cid])).fetchall()
                out.append((time.perf_counter() - t) * 1000)
            return out

        cold, warm = one_pass(), one_pass()
        print(f"{layout:9s} dense arm, 5 clusters sequential: COLD p50={med(cold):7.0f} ms max={max(cold):7.0f} "
              f"| WARM p50={med(warm):6.0f} ms", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in ("build", "test", "drop"):
        sys.exit(__doc__)
    cmd, name = sys.argv[1], sys.argv[2]
    if cmd == "build":
        build(name)
    elif cmd == "test":
        test(name)
    else:
        with connect(name) as c:
            c.execute("DROP TABLE IF EXISTS dense_ord")
            print("dropped")
