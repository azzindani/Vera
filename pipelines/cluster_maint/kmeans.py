"""Layer-2 clustering · spherical k-means over the dense vectors.

! Spherical, not Euclidean. The corpus vectors are L2-normalised, so cosine
similarity IS the dot product and centroids must be renormalised after each
update. Running plain Lloyd's on normalised vectors quietly optimises the wrong
objective.

Cluster COUNT is what makes routing meaningful, and it is a config value, not a
constant (CLAUDE.md §7.12). At ~2K rows per cluster, 171K rows give ~86
clusters, so probing 5 touches ~6% of the corpus -- the same pruning ratio the
100M design targets. Leave it at the production 10K/cluster and 171K rows would
yield 17 clusters, where probing 5 touches 29% and routing proves nothing.

Numpy only · no sklearn. The recipe is short enough to own outright, and owning
it means the assignment rule is inspectable rather than a library default.

Usage:
    python kmeans.py --rows-per-cluster 2000 --iters 15
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

PG = os.environ.get(
    "DATABASE_URL", "host=localhost port=5432 dbname=vera user=vera password=vera"
)
FETCH = 20_000


def load_vectors(pg):
    """Stream dense vectors out of Postgres into one float32 matrix."""
    ids: list[str] = []
    blocks: list[np.ndarray] = []
    with pg.cursor(name="vecs") as cur:
        cur.itersize = FETCH
        cur.execute(
            "SELECT id, dense::text FROM chunks"
            " WHERE indexable AND dense IS NOT NULL ORDER BY id"
        )
        buf: list[np.ndarray] = []
        for cid, txt in cur:
            ids.append(cid)
            buf.append(np.fromstring(txt[1:-1], sep=",", dtype=np.float32))
            if len(buf) >= FETCH:
                blocks.append(np.vstack(buf))
                buf = []
                print(f"\r  loaded {len(ids):,}", end="", flush=True)
        if buf:
            blocks.append(np.vstack(buf))
    X = np.vstack(blocks)
    print(f"\r  loaded {len(ids):,} vectors {X.shape}")
    # Guard: halfvec round-trips should already be unit norm.
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return ids, X / norms


def spherical_kmeans(X, k, iters, seed=0):
    rng = np.random.default_rng(seed)
    # k-means++ style seeding, cosine flavour: spread initial centroids out.
    idx = [int(rng.integers(len(X)))]
    d = 1.0 - X @ X[idx[0]]
    for _ in range(1, k):
        p = np.maximum(d, 0) ** 2
        s = p.sum()
        nxt = int(rng.choice(len(X), p=p / s)) if s > 0 else int(rng.integers(len(X)))
        idx.append(nxt)
        d = np.minimum(d, 1.0 - X @ X[nxt])
    C = X[idx].copy()

    assign = np.zeros(len(X), dtype=np.int32)
    for it in range(iters):
        t0 = time.time()
        # Chunked argmax over dot products · the full 171K x k matrix at once
        # would be fine here, but chunking keeps this honest at 748K too.
        moved = 0
        for s in range(0, len(X), 50_000):
            sim = X[s:s + 50_000] @ C.T
            a = np.argmax(sim, axis=1).astype(np.int32)
            moved += int((a != assign[s:s + 50_000]).sum())
            assign[s:s + 50_000] = a

        for j in range(k):
            members = X[assign == j]
            if len(members):
                c = members.sum(axis=0)
                n = np.linalg.norm(c)
                C[j] = c / n if n else C[j]
            else:
                # Empty cluster · reseed on the worst-served point.
                C[j] = X[int(np.random.default_rng(it * 1000 + j).integers(len(X)))]
        print(f"  iter {it + 1:>2}/{iters}  moved={moved:>8,}  "
              f"{time.time() - t0:.1f}s")
        if moved == 0:
            print("  converged")
            break
    return assign, C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows-per-cluster", type=int, default=2000)
    ap.add_argument("--iters", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    import psycopg

    with psycopg.connect(PG, autocommit=False) as pg:
        print("loading vectors...")
        ids, X = load_vectors(pg)

        k = max(2, len(ids) // args.rows_per_cluster)
        print(f"k = {k} clusters at ~{args.rows_per_cluster} rows each")
        assign, C = spherical_kmeans(X, k, args.iters, args.seed)

        sizes = np.bincount(assign, minlength=k)
        print(f"\ncluster sizes: min={sizes.min():,} median={int(np.median(sizes)):,}"
              f" max={sizes.max():,} empty={int((sizes == 0).sum())}")

        with pg.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS clusters (
                    id         INTEGER PRIMARY KEY,
                    corpus_id  TEXT NOT NULL REFERENCES corpus_meta(id),
                    centroid   halfvec(1024) NOT NULL,
                    row_count  BIGINT NOT NULL DEFAULT 0,
                    -- ! generation supports the atomic swap in
                    -- CLUSTER_MAINTENANCE.md · a live cluster is never mutated.
                    generation INTEGER NOT NULL DEFAULT 1
                )""")
            cur.execute("SELECT id FROM corpus_meta LIMIT 1")
            corpus_id = cur.fetchone()[0]
            cur.execute("DELETE FROM clusters WHERE corpus_id = %s", (corpus_id,))
            for j in range(k):
                lit = "[" + ",".join(f"{v:.6g}" for v in C[j]) + "]"
                cur.execute(
                    "INSERT INTO clusters (id, corpus_id, centroid, row_count)"
                    " VALUES (%s,%s,%s,%s)", (j, corpus_id, lit, int(sizes[j])))

            print("writing cluster_id...")
            cur.execute("CREATE TEMP TABLE a (id TEXT, cid INT) ON COMMIT DROP")
            with cur.copy("COPY a (id, cid) FROM STDIN") as cp:
                for cid, j in zip(ids, assign):
                    cp.write_row((cid, int(j)))
            cur.execute(
                "UPDATE chunks SET cluster_id = a.cid FROM a WHERE chunks.id = a.id")
            print(f"  {cur.rowcount:,} rows assigned")
        pg.commit()
    print("done")


if __name__ == "__main__":
    main()
