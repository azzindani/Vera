#!/usr/bin/env bash
# Everything a freshly-ingested corpus needs before it can be queried or
# evaluated, in the order it needs it.
#
# ! Written down because the order is not obvious and getting it wrong is
# quiet: routing needs centroids, and the domain gate's centroid half needs
# them too, so an eval run against a corpus with no clusters does not fail —
# it reports numbers that mean nothing.
#
#   DATABASE_URL="host=localhost port=5432 dbname=vera2 user=vera password=vera" \
#     bash pipelines/finish_corpus.sh .test/runs/spike-02.bm25.json
#
set -euo pipefail

VOCAB="${1:?usage: finish_corpus.sh <path to run.bm25.json>}"
: "${DATABASE_URL:?set DATABASE_URL — refusing to guess which corpus to modify}"

cd "$(dirname "$0")/.."

echo "== corpus =="
python - <<'PY'
import os, psycopg
with psycopg.connect(os.environ["DATABASE_URL"]) as c:
    n, idx = c.execute(
        "SELECT count(*), count(*) FILTER (WHERE indexable) FROM chunks"
    ).fetchone()
    dense = c.execute("SELECT count(*) FROM chunks WHERE dense IS NOT NULL").fetchone()[0]
    meta = c.execute("SELECT id, dense_model, dense_dim FROM corpus_meta").fetchall()
    print(f"  {n:,} chunks · {idx:,} indexable · {dense:,} embedded")
    print(f"  corpus_meta: {meta}")
    if dense == 0:
        raise SystemExit("no embeddings — ingest did not finish")
PY

echo
echo "== layer-2 clusters (k-means) =="
# ! Must run AFTER embedding: centroids are computed from the dense vectors.
python pipelines/cluster_maint/kmeans.py --rows-per-cluster 2000

echo
echo "== eval · 50 labelled cases =="
# Labels are article-level, so they resolve against whatever chunk ids this
# corpus uses. Nothing here needs updating after a re-chunk.
EMBED_ENDPOINT="${EMBED_ENDPOINT:-http://localhost:8080}" \
  python eval/run.py --by-type

echo
echo "done · compare against the previous corpus before switching .mcp.json"
