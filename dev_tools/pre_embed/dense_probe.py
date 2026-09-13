"""Does re-embedding buy anything? Measure the dense arm before paying for it.

! This is the check whose ABSENCE cost the dense arm. The ingest pipeline's
round-trip gate compared the serving backend against itself, scored 0.999992,
and certified a vector space that ranks the right answer at median 32. A gate
that can only pass is worse than no gate: it buys confidence.

So this compares two spaces against each other on the SAME pool, the same
queries and the same labels, and reports one number for each:

    stored   the vectors in chunks.dense today
    fresh    the same texts re-embedded here, on GPU, via transformers

Dense-only Recall@5. No fusion, no routing (new vectors have no centroids yet),
so it is a flat scan over the pool — the harder test, not the easier one.

! It measures dense IN ISOLATION. A dense arm that ranks well can still add
little through fusion if it surfaces what sparse and text already found. This
answers "is the space broken", ✗ "what will Recall@5 become".

    DATABASE_URL=... python dev_tools/pre_embed/dense_probe.py --pool 5000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import psycopg
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]

MODEL = os.environ.get("MODEL_DIR", "/model")
PG = os.environ["DATABASE_URL"]
QUERIES = ROOT / "dev_tools" / "eval" / "queries.json"


def parse_vec(text: str) -> list[float]:
    return [float(x) for x in text.strip()[1:-1].split(",")]


def resolve_targets(cur, q):
    """Chunk ids that count as correct for this query.

    ! Labels name ARTICLES, ✗ chunk ids: a chunk id is an artefact of how the
    corpus was split, an article is a property of the law. Any chunk of a
    labelled article counts — which is also the honest reading, since an
    article split across three chunks has three correct answers.

    Mirrors dev_tools/eval/run.py. Duplicated rather than imported: run.py
    requires a BM25 vocabulary at import time and this probe has no use for
    one.
    """
    ids = set(q.get("answer_chunks", []))
    for ref in q.get("answer_articles", []):
        cur.execute(
            "SELECT id FROM chunks WHERE regulation_type=%s AND regulation_number=%s"
            " AND year=%s AND about=%s AND article=%s",
            (ref["regulation_type"], ref["regulation_number"], ref["year"],
             ref["about"], ref["article"]),
        )
        ids |= {r[0] for r in cur.fetchall()}
    return ids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=int, default=5000,
                    help="random distractors sampled alongside the answers")
    ap.add_argument("--hard", type=int, default=0,
                    help="hard negatives per answer: its nearest LEXICAL "
                         "neighbours, i.e. the near-duplicate clauses")
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    cases = json.loads(QUERIES.read_text(encoding="utf-8"))["queries"]
    # ! out_of_domain has no correct answer to rank — the domain gate handles
    # those and it is not under test. exact_ref names a regulation outright and
    # resolves to every chunk of it, which is trivial for BOTH spaces and only
    # adds noise; it is the identifier path's job, not the dense arm's.
    skip = {"out_of_domain", "exact_ref"}
    cases = [c for c in cases if c["type"] not in skip]

    with psycopg.connect(PG) as pg, pg.cursor() as cur:
        targets = {}
        for c in cases:
            ids = resolve_targets(cur, c)
            if ids:
                targets[c["id"]] = ids
        cases = [c for c in cases if c["id"] in targets]
        answer_ids = sorted({i for ids in targets.values() for i in ids})
        print(f"{len(cases)} scorable queries · {len(answer_ids)} labelled chunks")

        # ! Deterministic sample. A pool that changes between runs makes the
        # two spaces incomparable, which is the one thing this must not do.
        cur.execute(
            """SELECT id, body, dense::text FROM chunks
               WHERE indexable AND dense IS NOT NULL AND id <> ALL(%s)
               ORDER BY md5(id) LIMIT %s""",
            (answer_ids, args.pool),
        )
        pool = cur.fetchall()

        # ! Hard negatives, ✗ more random ones. A random sample of 1.4% of the
        # corpus strips out exactly the competitors that make this hard: the
        # same provision repeated across dozens of regencies. Meanwhile the
        # real dense arm scans the ROUTED clusters, which are by construction
        # the chunks most similar to the query. Random distractors therefore
        # measure an easier task than production, and flatter any space that
        # works at all.
        if args.hard:
            seen = {r[0] for r in pool}
            for aid in answer_ids:
                # ! Nearest neighbours in BM25 SPARSE space, ✗ a tsquery built
                # from the body. Two earlier attempts got this wrong:
                # plainto_tsquery ANDs its terms and returned 121 negatives for
                # a request of 4,000, and an OR over every term in a body
                # matches 98.7% of the corpus and costs 48s per answer — 81
                # minutes for one run. The sparse arm answers the same question
                # (what is lexically closest to this text) in 165ms, because it
                # is the arm the engine already uses for exactly that.
                cur.execute(
                    """SELECT id, body, dense::text FROM chunks
                       WHERE indexable AND dense IS NOT NULL AND sparse IS NOT NULL
                         AND id <> ALL(%s)
                       ORDER BY sparse <#> (SELECT sparse FROM chunks WHERE id = %s)
                       LIMIT %s""",
                    (answer_ids, aid, args.hard),
                )
                for r in cur.fetchall():
                    if r[0] not in seen:
                        seen.add(r[0])
                        pool.append(r)
            added = len(pool) - args.pool
            print(f"added {added} hard negatives "
                  f"({added / max(1, len(answer_ids)):.1f} per answer)")
            # ! Fail loudly on a pool that is hard in name only.
            if added < len(answer_ids):
                sys.exit("hard-negative selection returned almost nothing · "
                         "the pool would be random while claiming otherwise")

        # ! dense IS NOT NULL here too. A labelled chunk with no stored vector
        # cannot be scored in the stored space, and a pool that differs between
        # the two spaces makes the comparison meaningless — which is the one
        # thing this probe exists not to do.
        cur.execute(
            "SELECT id, body, dense::text FROM chunks"
            " WHERE id = ANY(%s) AND dense IS NOT NULL",
            (answer_ids,),
        )
        labelled = cur.fetchall()
        pool += labelled

    kept = {r[0] for r in labelled}
    dropped = len(answer_ids) - len(kept)
    targets = {q: ids & kept for q, ids in targets.items()}
    lost = [q for q, ids in targets.items() if not ids]
    targets = {q: ids for q, ids in targets.items() if ids}
    cases = [c for c in cases if c["id"] in targets]
    if dropped:
        print(f"! {dropped} labelled chunks have no stored vector · excluded")
    if lost:
        print(f"! {len(lost)} queries lost every target · excluded: {', '.join(lost)}")
    print(f"scoring {len(cases)} queries against {len(kept)} labelled chunks")

    ids = [r[0] for r in pool]
    bodies = [r[1] for r in pool]
    stored = np.array([parse_vec(r[2]) for r in pool], dtype=np.float32)
    stored /= np.linalg.norm(stored, axis=1, keepdims=True)
    print(f"pool: {len(ids)} chunks "
          f"({args.pool} random + hard negatives + labelled)")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
    model = AutoModel.from_pretrained(
        MODEL, dtype=torch.float16 if dev == "cuda" else torch.float32
    ).to(dev).eval()
    print(f"reference implementation on {dev}")

    def embed(texts: list[str]) -> np.ndarray:
        """Last-token pooling then L2, matching 1_Pooling/config.json.

        ! Left padding. With right padding the last position of a short
        sequence is a PAD token, so the pooled vector is read from a place the
        model never wrote a sentence representation to.
        """
        out = []
        for i in range(0, len(texts), args.batch):
            b = tok(texts[i:i + args.batch], return_tensors="pt", padding=True,
                    truncation=True, max_length=1024).to(dev)
            with torch.no_grad():
                h = model(**b).last_hidden_state
            v = F.normalize(h[:, -1].float(), p=2, dim=1)
            out.append(v.cpu().numpy())
            if i % (args.batch * 50) == 0:
                print(f"  {i}/{len(texts)}", flush=True)
        return np.concatenate(out)

    t0 = time.perf_counter()
    fresh = embed(bodies)
    dt = time.perf_counter() - t0
    print(f"embedded {len(bodies)} chunks in {dt:.0f}s "
          f"({len(bodies) / dt:.1f}/s)\n")

    queries = [c["query"] for c in cases]
    q_fresh = embed(queries)
    # ! The stored space's query vector must come from the endpoint that built
    # it, not from here — that is the whole point. TEI is at EMBED_ENDPOINT.
    q_stored = tei_embed(queries)

    def recall_at(space_docs, space_queries, k=5):
        hits = 0
        ranks = []
        for c, qv in zip(cases, space_queries):
            sims = space_docs @ qv
            order = np.argsort(-sims)
            want = targets[c["id"]]
            rank = next((r for r, j in enumerate(order, 1) if ids[j] in want), None)
            ranks.append(rank or len(ids))
            if rank and rank <= k:
                hits += 1
        return hits / len(cases), int(np.median(ranks))

    print(f"dense-only over {len(ids)} chunks, {len(cases)} queries\n")
    print(f"{'space':<10}{'Recall@5':>10}{'Recall@10':>11}{'median rank':>13}")
    for name, docs, qs in (("stored", stored, q_stored), ("fresh", fresh, q_fresh)):
        r5, med = recall_at(docs, qs, 5)
        r10, _ = recall_at(docs, qs, 10)
        print(f"{name:<10}{r5:>9.1%}{r10:>10.1%}{med:>13}")

    print("\n! Dense in isolation. Through fusion the gain is smaller than this,"
          "\n  because sparse and text already find some of the same documents.")


def tei_embed(texts: list[str]) -> np.ndarray:
    import urllib.request
    ep = os.environ.get("EMBED_ENDPOINT", "http://localhost:8080").rstrip("/")
    out = []
    for t in texts:
        req = urllib.request.Request(
            f"{ep}/embed",
            data=json.dumps({"inputs": t, "truncate": False}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=60) as r:
            out.append(json.loads(r.read())[0])
    v = np.array(out, dtype=np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


if __name__ == "__main__":
    main()
