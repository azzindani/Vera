"""Would a richer embedded header move the answer up? A pre-test, train only, NOT YET RUN.

The corpus was not embedded from the bare body. Ravel's profile (`id_regulation_md@1.3`,
`pipeline.derive.embed_text`) put a header in front of every chunk:

    PERATURAN DAERAH KABUPATEN 1/2016 › PERLINDUNGAN DAN PEMBERDAYAAN PETAMBAK GARAM › Umum › Pasal 38

    <body>

What it leaves out is the region (`enacting_body`, SAMPANG here). Hundreds of regions enact
the same clause, so the only thing in a regional chunk's vector that says WHOSE clause it
is, is missing. Two headers, each scored against the STORED vectors:

  region   the 1.3 template with the region after the type, nothing else changed.
  full     Ravel `id_regulation_md@1.4`: labelled lines (Peraturan, Nomor, Tentang, Bab,
           Letak, Isi), the region and the full heading path.

✗ A second vector per regulation: the context belongs in the one embedded text.

! `stored` is the vector the engine searches, read from the database. `ravel` re-embeds the
chunk from Ravel's own template and must reproduce `stored` at cosine ~1.0 (it does: 1.0000,
plain tokenizer, no extra token). The script prints that
check; a figure from a recipe that does not reproduce the corpus is not a comparison.

Candidates per question, the way the dense arm finds them: the 10 clusters nearest the
question (the engine probes 5), their 200 nearest chunks from `chunk_vectors`, up to 300
more from the regions of the top 20 (a region header lifts every chunk of that region, so
those are its likeliest competitors), and the labelled answer. ✗ The engine's ranking: no
fusion, no prior. A chunk outside the set that the header lifts is not seen, so a gain here
is an upper bound and a loss is real.

Reads `queries.json` (train) only, never `queries_fresh.json`.

    docker run --rm --gpus all -v C:/Github/Vera:/w -v <model>:/model -v <vocab>:/vocab \
      -e BM25_VOCAB=/vocab/corpus.bm25.json -e DATABASE_URL="host=host.docker.internal ..." \
      --entrypoint python ravel-job:gpu /w/dev_tools/eval/header_sim.py
"""

from __future__ import annotations

import json
import os
import random
import re
import string
import sys
import time
from pathlib import Path

import psycopg
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from run import resolve_targets  # noqa: E402

INSTR = ("Instruct: Given a legal or regulatory question, retrieve passages that answer "
         "it\nQuery: ")
N = int(os.environ.get("N", "150"))
SEED = int(os.environ.get("SEED", "20261121"))
CUTS = (5, 10, 15, 20, 25)
# ! No extra token: Ravel's template with the plain tokenizer reproduces the stored vectors
# at cosine 1.0000 (12 sampled chunks). The extra-<|im_end|> recipe of the FIRST 1.7M run
# (Ravel `EMBED_VERIFICATION.md`) is gone from this corpus: it reads 0.80 here.

# Ravel's template, `id_regulation_md@1.3`, verbatim: the stored space.
RAVEL = (["{regulation_type} {regulation_number}/{year}", "{about}", "{chapter}", "{article}"],
         " › ", frozenset({"N/A"}))
# `enacting_body` values that name no region (Vera migration 0007's exclusions), as empty.
NO_REGION = frozenset({"N/A", "INDONESIA", "KABUPATEN", "KOTA", "PROVINSI", "DAERAH",
                       "PEMERINTAH", "NEGARA", "MA", "RI"})
# The 1.3 template with the region after the type, nothing else changed.
REGION = (["{regulation_type} {enacting_body} {regulation_number}/{year}", "{about}",
           "{chapter}", "{article}"], " › ", NO_REGION)
# Labelled lines, every placing field: the law as questions name it, its subject, the
# chapter WITH its title (`heading_path` often drops it: `BAB II › Bagian Pertama › Pasal 4`)
# and the full path (Bagian and Paragraf titles, the pasal). ✗ status or effective date:
# they change after a vector is written, and Vera scores revocation itself.
FULL = (["Peraturan: {regulation_type} {enacting_body}", "Nomor: {regulation_number} Tahun {year}",
         "Tentang: {about}", "Bab: {chapter}", "Letak: {heading_path}", "Isi:"], "\n", NO_REGION)

dev = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = os.environ.get("MODEL_DIR", "/model")
tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
model = AutoModel.from_pretrained(
    MODEL, dtype=torch.float16 if dev == "cuda" else torch.float32).to(dev).eval()


def render(r: dict, template: tuple, body: bool = True) -> str:
    """`chunk.derive.embed_text` from Ravel, for an arbitrary (context, separator, skip)."""
    context, sep, skip = template

    def value(name):
        raw = r.get(name)
        text = "" if raw is None else str(raw).strip()
        return "" if text in skip else text
    parts = []
    for template in context:
        names = [n for _, n, _, _ in string.Formatter().parse(template) if n]
        values = {n: value(n) for n in names}
        if names and not any(values.values()):
            continue
        rendered = re.sub(r"\s+", " ", template.format(**values)).strip(" /-—")
        if rendered:
            parts.append(rendered)
    head = sep.join(parts)
    if not body:
        return head or "(tanpa judul)"
    return f"{head}\n\n{r['body']}" if head else r["body"]


def embed(texts: list[str], extra: bool, batch: int = 16) -> torch.Tensor:
    """Left padding, last token, L2; `extra` appends Ravel's one extra <|im_end|>."""
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    out = [None] * len(texts)
    for s in range(0, len(order), batch):
        sel = order[s:s + batch]
        ids = tok([texts[i] for i in sel], truncation=True, max_length=1023)["input_ids"]
        if extra:
            ids = [[*x, tok.eos_token_id] for x in ids]
        b = tok.pad({"input_ids": ids}, padding=True, return_tensors="pt").to(dev)
        with torch.no_grad():
            h = model(**b).last_hidden_state
        v = F.normalize(h[:, -1].float(), p=2, dim=1).cpu()
        for k, i in enumerate(sel):
            out[i] = v[k]
    return torch.stack(out)


def vec(text: str) -> torch.Tensor:
    return torch.tensor([float(x) for x in text.strip("[]").split(",")])


COLS = ("c.id, regulation_type, enacting_body, regulation_number, year, about, chapter,"
        " article, heading_path, body, c.dense::text")


def main() -> None:
    spec = json.loads((Path(__file__).parent / "queries.json").read_text("utf-8"))
    pool = [q for q in spec["queries"] if not q.get("holdout") and not q.get("fresh")
            and q["type"] not in ("exact_ref", "out_of_domain")]
    random.Random(SEED).shuffle(pool)
    pg = psycopg.connect(os.environ["DATABASE_URL"])
    cur = pg.cursor()
    cur.execute("SELECT max(generation) FROM clusters")
    gen = cur.fetchone()[0]
    rows_out, repro, t0 = [], [], time.perf_counter()
    for q in pool:
        if len(rows_out) >= N:
            break
        targets = resolve_targets(cur, q)
        if not targets:
            continue
        qv = embed([INSTR + q["query"]], extra=False)[0]
        lit = "[" + ",".join(f"{x:.6f}" for x in qv.tolist()) + "]"
        cur.execute("SELECT id FROM clusters WHERE generation = %s"
                    " ORDER BY centroid <=> %s::halfvec LIMIT 10", (gen, lit))
        probed = [i for (i,) in cur.fetchall()]
        cur.execute("SELECT v.id, c.enacting_body FROM chunk_vectors v JOIN chunks c USING (id)"
                    " WHERE v.cluster_id = ANY(%s) ORDER BY v.dense <=> %s::halfvec LIMIT 200",
                    (probed, lit))
        near = cur.fetchall()
        regions = sorted({e for _, e in near[:20] if e})
        ids = {i for i, _ in near} | set(targets)
        if regions:
            cur.execute("SELECT id FROM chunks WHERE indexable AND enacting_body = ANY(%s)"
                        " ORDER BY dense <=> %s::halfvec LIMIT 300", (regions, lit))
            ids |= {i for (i,) in cur.fetchall()}
        cur.execute(f"SELECT {COLS} FROM chunks c WHERE c.id = ANY(%s) AND c.dense IS NOT NULL",
                    (list(ids),))
        names = [d.name for d in cur.description]
        cands = [dict(zip(names, r)) for r in cur.fetchall()]
        stored_m = torch.stack([vec(c["dense"]) for c in cands])
        stored = stored_m @ qv
        ravel_m = embed([render(c, RAVEL) for c in cands], extra=False)
        repro.extend(F.cosine_similarity(ravel_m, stored_m).tolist())
        region = embed([render(c, REGION) for c in cands], extra=False) @ qv
        full = embed([render(c, FULL) for c in cands], extra=False) @ qv

        def rank(scores: torch.Tensor) -> int:
            order = torch.argsort(scores, descending=True).tolist()
            return next((k + 1 for k, i in enumerate(order) if cands[i]["id"] in targets), 10**6)

        rows_out.append({"id": q["id"], "type": q["type"], "n": len(cands),
                         "stored": rank(stored), "region": rank(region), "full": rank(full)})
        r = rows_out[-1]
        print(f"{len(rows_out):3d} {q['id']} n={r['n']} stored {r['stored']} region {r['region']}"
              f" full {r['full']}  repro min {min(repro):.4f}"
              f"  {time.perf_counter() - t0:.0f}s", file=sys.stderr, flush=True)

    n = len(rows_out)
    rp = torch.tensor(repro)
    print(f"\nheader_sim · train · n={n} · seed {SEED} · dense arm only, candidate set")
    print(f"  Ravel template reproduces stored: cosine min {rp.min():.4f} mean {rp.mean():.4f}"
          f" over {len(rp)} chunk vectors")
    sides = ("stored", "region", "full")
    for side in sides:
        print(f"  {side:8s} " + "   ".join(
            f"@{k} {100 * sum(r[side] <= k for r in rows_out) / n:.1f}%" for k in CUTS))
    for side in sides[1:]:
        print(f"  {side:8s} vs stored " + "   ".join(
            f"@{k} +{sum(r[side] <= k < r['stored'] for r in rows_out)}"
            f" -{sum(r['stored'] <= k < r[side] for r in rows_out)}" for k in CUTS))
    Path(os.environ.get("OUT", "/tmp/header_sim.json")).write_text(json.dumps(rows_out))


if __name__ == "__main__":
    main()
