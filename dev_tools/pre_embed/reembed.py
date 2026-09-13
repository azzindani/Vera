"""Re-embed chunks.dense with the model's reference implementation.

The corpus was embedded through a backend whose output does not match the
model. Measured on 39 eval queries over a 5,102-chunk pool: the stored space
ranks the right answer at median 857 and scores 2.6% Recall@5, against median 1
for the reference implementation. The dense arm has therefore carried weight
0.0 since it was measured.

! Writes to a NEW column, ✗ over the live one. `dense_v2` is populated
alongside `dense`, validated, and only then swapped. A crash at 80% leaves a
corpus that still serves, which is the same copy-on-write discipline cluster
maintenance uses — and the reason this can run against the live database.

! Resumable and idempotent: the work queue is `WHERE dense_v2 IS NULL`, so a
re-run continues rather than restarts.

    DATABASE_URL=... python dev_tools/pre_embed/reembed.py
    DATABASE_URL=... python dev_tools/pre_embed/reembed.py --swap   # after review

Swapping invalidates the layer-2 centroids (they are means of the OLD vectors)
and the fusion weights. Both must be redone before the numbers mean anything:

    python dev_tools/cluster_maint/kmeans.py
    VERA_HTTP=... python dev_tools/eval/e2e.py     # refit DENSE_WEIGHT
"""

from __future__ import annotations

import argparse
import os
import time

import psycopg
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

MODEL = os.environ.get("MODEL_DIR", "/model")
PG = os.environ["DATABASE_URL"]


def ensure_column(cur) -> None:
    cur.execute("SELECT dense_dim FROM corpus_meta ORDER BY created_at DESC LIMIT 1")
    dim = cur.fetchone()[0]
    cur.execute(
        f"ALTER TABLE chunks ADD COLUMN IF NOT EXISTS dense_v2 halfvec({dim})"
    )
    return dim


def swap(cur) -> None:
    """Promote dense_v2 to dense, atomically.

    ! One transaction. A reader either sees every old vector or every new one;
    a corpus half in each space would rank by comparing vectors that share no
    geometry, and would do it without erroring.
    """
    cur.execute("SELECT count(*) FROM chunks WHERE indexable AND dense_v2 IS NULL")
    missing = cur.fetchone()[0]
    if missing:
        raise SystemExit(
            f"{missing:,} indexable chunks still have no dense_v2 · "
            "finish the run before swapping"
        )
    cur.execute("ALTER TABLE chunks DROP COLUMN dense")
    cur.execute("ALTER TABLE chunks RENAME COLUMN dense_v2 TO dense")
    print("swapped · dense_v2 -> dense")
    print("! centroids and fusion weights are now stale · see the module note")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--swap", action="store_true",
                    help="promote dense_v2 to dense and stop")
    args = ap.parse_args()

    with psycopg.connect(PG, autocommit=False) as pg, pg.cursor() as cur:
        if args.swap:
            swap(cur)
            pg.commit()
            return
        dim = ensure_column(cur)
        pg.commit()
        cur.execute(
            "SELECT count(*) FROM chunks WHERE indexable AND dense_v2 IS NULL"
        )
        todo = cur.fetchone()[0]
        cur.execute("SELECT count(*) FROM chunks WHERE indexable")
        total = cur.fetchone()[0]

    print(f"{todo:,} of {total:,} indexable chunks to embed at {dim}d")
    if not todo:
        print("nothing to do · run with --swap to promote")
        return

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
    model = AutoModel.from_pretrained(
        MODEL, dtype=torch.float16 if dev == "cuda" else torch.float32
    ).to(dev).eval()
    print(f"reference implementation on {dev}")

    def embed(texts: list[str]) -> list[list[float]]:
        """Last-token pooling then L2, matching 1_Pooling/config.json.

        ! Left padding. With right padding the final position of a short
        sequence is a PAD token, so the vector is read from a place the model
        never wrote a sentence representation to — which produces normal-looking
        vectors and wrong rankings.
        """
        b = tok(texts, return_tensors="pt", padding=True, truncation=True,
                max_length=args.max_tokens).to(dev)
        with torch.no_grad():
            h = model(**b).last_hidden_state
        return F.normalize(h[:, -1].float(), p=2, dim=1).cpu().tolist()

    done, t0 = 0, time.perf_counter()
    with psycopg.connect(PG, autocommit=False) as pg:
        while True:
            with pg.cursor() as cur:
                # ! Ordered and re-queried each round rather than paged with an
                # OFFSET: rows leave the queue as they are filled, so an OFFSET
                # would skip work every batch.
                cur.execute(
                    """SELECT id, body FROM chunks
                       WHERE indexable AND dense_v2 IS NULL
                       ORDER BY id LIMIT %s""",
                    (args.batch,),
                )
                rows = cur.fetchall()
                if not rows:
                    break
                vecs = embed([r[1] for r in rows])
                cur.executemany(
                    "UPDATE chunks SET dense_v2 = %s::text::halfvec WHERE id = %s",
                    [(str(v), r[0]) for v, r in zip(vecs, rows)],
                )
            # ! Commit per batch. The GPU time already spent is what makes this
            # expensive to lose; holding one transaction over four hours would
            # also pin an enormous amount of WAL.
            pg.commit()

            done += len(rows)
            el = time.perf_counter() - t0
            rate = done / el
            eta = (todo - done) / rate / 60 if rate else 0
            print(f"  {done:,}/{todo:,}  {rate:.1f}/s  eta {eta:.0f}m", flush=True)

    print(f"\ndone · {done:,} chunks in {(time.perf_counter() - t0) / 60:.0f}m")
    print("review, then re-run with --swap to promote dense_v2 -> dense")


if __name__ == "__main__":
    main()
