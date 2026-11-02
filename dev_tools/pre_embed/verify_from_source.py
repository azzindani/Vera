"""Reproduce a deployed corpus from its source dataset, and check it byte for byte.

    python dev_tools/pre_embed/verify_from_source.py vocab  <source_dir> --sha256 <corpus_meta.sparse_vocab_sha256> [--out FILE]
    python dev_tools/pre_embed/verify_from_source.py chunks <source_dir> <chunks.parquet>

<source_dir> is a download of Azzindani/ID_REG_KG_2511: parquet shards with `global_id`, `content` and the
metadata columns (`MANIFEST.tsv` is optional; every `*.parquet` under the directory is read, in name order).

`vocab` refits BM25 the way `ingest.py --fit-on-full-corpus` does -- `chunking.split_article` over EVERY article,
then `Bm25Vectorizer.fit(max_features=20000)` -- and compares the sha256 with the one `corpus_meta` stored. A match
means the artifact is exactly the one the corpus was built with; no reconstruction is involved. Reproduced on
2026-11-02: ea8949e3...3d96d, n_docs 1,657,712 (`EMBEDDING.md` §6).

`chunks` re-chunks the source with the CURRENT chunking.py and compares to a deployed export: ids, bodies, the
`indexable` flag and the metadata columns. It is the check that re-chunking has not drifted, which would silently
invalidate every stored vector (`EMBEDDING.md` §6).

! CPU only, read-only. `vocab` holds a document-frequency table for the whole 1.66M-part corpus: ~20 minutes and a
few GB of RAM.
! The deployed `year` column is an int where the source holds a string; '0000' becomes 0. That is the one
expected metadata difference and is reported separately rather than as a mismatch.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chunking import is_indexable, split_article  # noqa: E402
from sparse import Bm25Vectorizer  # noqa: E402

META = ["regulation_type", "regulation_number", "year", "about", "chapter", "article"]


def shards(source: Path) -> list[Path]:
    found = sorted(source.glob("*.parquet"))
    if not found:
        sys.exit(f"no parquet shards under {source}")
    return found


def vocab(args) -> int:
    files, t0 = shards(Path(args.source)), time.time()
    stats = {"articles": 0, "parts": 0}

    def docs():
        for n, f in enumerate(files, 1):
            for content in pq.read_table(f, columns=["content"]).column("content").to_pylist():
                if content is None:
                    continue
                stats["articles"] += 1
                for part in split_article(content):
                    stats["parts"] += 1
                    yield part
            print(f"shard {n}/{len(files)} articles {stats['articles']:,} parts {stats['parts']:,} "
                  f"{time.time() - t0:.0f}s", flush=True)

    vz = Bm25Vectorizer.fit(docs(), max_features=20_000)
    out = Path(args.out)
    sha = vz.save(out)
    print(f"fit: dim={vz.dim} n_docs={vz.n_docs:,} avgdl={vz.avgdl:.4f} -> {out}")
    print(f"refit sha256 : {sha}")
    if args.sha256:
        print(f"stored sha256: {args.sha256} | {'MATCH' if sha == args.sha256 else 'DIFFERS'}")
        return 0 if sha == args.sha256 else 1
    return 0


def chunks(args) -> int:
    dep = pq.read_table(args.deployed).to_pydict()
    at = {i: j for j, i in enumerate(dep["id"])}
    wanted = {i.split("#")[0] for i in dep["id"]}
    seen: set[str] = set()
    eq = ne = idx_ne = meta_ne = year_zero = 0
    examples: list[tuple[str, str, str]] = []
    for f in shards(Path(args.source)):
        t = pq.read_table(f, columns=["global_id", "content"] + META).to_pydict()
        for k, gid in enumerate(t["global_id"]):
            gid = str(gid)
            if gid not in wanted:
                continue
            parts = split_article(t["content"][k] or "")
            for i, part in enumerate(parts):
                cid = gid if len(parts) == 1 else f"{gid}#{i}"
                j = at.get(cid)
                if j is None:
                    continue
                seen.add(cid)
                if dep["body"][j] == part:
                    eq += 1
                else:
                    ne += 1
                    if len(examples) < 3:
                        examples.append((cid, dep["body"][j][:100], part[:100]))
                idx_ne += bool(dep["indexable"][j]) != is_indexable(part)
                for col in META:
                    if str(dep[col][j]) != str(t[col][k]):
                        if col == "year" and str(dep[col][j]) == "0" and str(t[col][k]) == "0000":
                            year_zero += 1
                        else:
                            meta_ne += 1
                        break
    missing = set(dep["id"]) - seen
    print(f"deployed chunks {len(dep['id']):,}; regenerated {len(seen):,}; missing {len(missing):,}")
    print(f"body identical {eq:,}, different {ne:,}; indexable flag differs {idx_ne:,}; "
          f"metadata mismatches {meta_ne:,}; year '0000'->0 (expected) {year_zero:,}")
    for e in examples:
        print("  differing body:", e)
    return 0 if not (missing or ne or idx_ne or meta_ne) else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("vocab")
    v.add_argument("source")
    v.add_argument("--sha256", help="corpus_meta.sparse_vocab_sha256 to compare against")
    v.add_argument("--out", default="refit.bm25.json")
    c = sub.add_parser("chunks")
    c.add_argument("source")
    c.add_argument("deployed", help="chunks.parquet from the export")
    args = ap.parse_args()
    return vocab(args) if args.cmd == "vocab" else chunks(args)


if __name__ == "__main__":
    sys.exit(main())
