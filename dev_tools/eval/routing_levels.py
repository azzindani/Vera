"""What two-level routing costs in routing recall, against one level, on labelled questions.

    DATABASE_URL=... BM25_VOCAB=... EMBED_ENDPOINT=http://127.0.0.1:8080 \
        python dev_tools/eval/routing_levels.py [--parents 1,2,4,8] [--probe 5]

Routing recall is the share of labelled questions whose answer chunk sits in one of the
probed clusters: the ceiling the dense arm can reach (`EVAL.md` §4d). One level probes
the `--probe` nearest of every leaf centroid; two levels probe the nearest leaves among
the children of the S nearest parents (`docs/operations/SCALING.md` §2). The gap between the two
columns is exactly what the parent level costs, independent of fusion and ranking.

! Read-only. The engine's own arithmetic: cosine to unit-normalised centroids, ties on
id. Needs a database with `cluster_parents` (Ravel `db prepare`, or `ravel db parents`).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

import numpy as np
import psycopg

sys.path.insert(0, str(Path(__file__).parent))
from run import resolve_targets  # noqa: E402

EMBED = os.environ.get("EMBED_ENDPOINT", "http://127.0.0.1:8080").rstrip("/")


def unit(rows: list) -> tuple[np.ndarray, np.ndarray]:
    ids = np.array([r[0] for r in rows])
    m = np.array([np.fromstring(r[1].strip("[]"), sep=",", dtype=np.float32) for r in rows])
    return ids, m / np.linalg.norm(m, axis=1, keepdims=True)


def top(ids: np.ndarray, sims: np.ndarray, k: int) -> list[int]:
    order = np.lexsort((ids, -sims))[:k]
    return [int(ids[i]) for i in order]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parents", default="1,2,4,8")
    ap.add_argument("--probe", type=int, default=5)
    a = ap.parse_args()
    s_list = [int(x) for x in a.parents.split(",")]
    spec = json.loads((Path(__file__).parent / "queries.json").read_text(encoding="utf-8"))
    with psycopg.connect(os.environ["DATABASE_URL"]) as c:
        cur = c.cursor()
        cur.execute("SELECT to_jsonb(corpus_meta) ->> 'dense_query_instruction' FROM corpus_meta"
                    " ORDER BY created_at DESC LIMIT 1")
        instr = cur.fetchone()[0]
        prefix = f"Instruct: {instr}\nQuery: " if instr else ""
        cur.execute("SELECT id, centroid::text, parent_id FROM clusters ORDER BY id")
        rows = cur.fetchall()
        leaf_ids, leaves = unit([(r[0], r[1]) for r in rows])
        parent_of = np.array([r[2] for r in rows])
        cur.execute("SELECT id, centroid::text FROM cluster_parents ORDER BY id")
        par_ids, pars = unit(cur.fetchall())
        print(f"{len(leaf_ids)} leaves under {len(par_ids)} parents · probing {a.probe}")

        n = 0
        hit_flat = 0
        hit_two = {s: 0 for s in s_list}
        same_two = {s: 0 for s in s_list}
        for q in spec["queries"]:
            if q["type"] in ("out_of_domain", "exact_ref"):
                continue
            targets = resolve_targets(cur, q)
            if not targets:
                continue
            cur.execute("SELECT DISTINCT cluster_id FROM chunk_vectors WHERE id = ANY(%s)",
                        (list(targets),))
            homes = {r[0] for r in cur.fetchall()}
            if not homes:
                continue
            req = urllib.request.Request(
                f"{EMBED}/embed", data=json.dumps({"inputs": [prefix + q["query"]]}).encode(),
                headers={"Content-Type": "application/json"})
            v = np.array(json.load(urllib.request.urlopen(req, timeout=300))[0], dtype=np.float32)
            v /= np.linalg.norm(v)
            sims = leaves @ v
            flat = top(leaf_ids, sims, a.probe)
            n += 1
            hit_flat += bool(homes & set(flat))
            psims = pars @ v
            for s in s_list:
                chosen = set(top(par_ids, psims, s))
                mask = np.isin(parent_of, list(chosen))
                two = top(leaf_ids[mask], sims[mask], a.probe)
                hit_two[s] += bool(homes & set(two))
                same_two[s] += set(two) == set(flat)
        print(f"n={n} labelled questions")
        print(f"  one level              routing recall {hit_flat / n:.3f}")
        for s in s_list:
            print(f"  two levels, {s} parents  routing recall {hit_two[s] / n:.3f} · "
                  f"same {a.probe} leaves as one level on {same_two[s]}/{n}")


if __name__ == "__main__":
    main()
