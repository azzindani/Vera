"""How fast does THIS machine embed the corpus? Same recipe as reembed.py, without the database.

    # 1. one-off, on the host: a reproducible sample of indexable chunks
    python dev_tools/pre_embed/embed_throughput.py sample <chunks.parquet> <sample.json>
    # 2. in the CUDA probe image (the sample is a plain JSON file because that image has no pyarrow)
    docker run --rm --gpus all -v <model_dir>:/model:ro -v <dir_with_sample>:/bench:ro \
        vera-probe:cu124 python /bench/embed_throughput.py run /bench/sample.json 2048 16,32,64 float16

Recipe is `reembed.py`'s: left padding, last-token pooling, L2 normalise, max 1024 tokens, one length-sorted window.
Database I/O is excluded -- the number is the model's, not the writer's. `ingest.py` measured the writer at
22 rows/s per-row against 2,268 batched (`FLUSH_ROWS`), so a real run sits below this figure.

Measured 2026-11-02, RTX 3070 Ti Laptop 8 GB, fp16, 2,048 random chunks (mean 552 chars):
    batch 16  57.8 chunks/s  11,214 tok/s  peak 2.7 GB   -> 355,621 chunks in ~1.7 h
    batch 32  51.9 chunks/s  10,072 tok/s  peak 4.2 GB   -> ~1.9 h
(batch 64 was not run). The corpus's last full embedding took ~6 h end to end (`HARDWARE.md` §5).
! fp16 here matches `reembed.py`. The SERVING embedder is float32 on CPU (`docker/embed_cpu`); the two agree to
cosine 0.99999 on the canary chunk, which is why a corpus built on the GPU is servable from the CPU.
"""

from __future__ import annotations

import json
import random
import sys
import time


def sample(parquet: str, out: str) -> None:
    import pyarrow.parquet as pq

    t = pq.read_table(parquet, columns=["id", "body", "indexable"]).to_pydict()
    rows = [(i, b) for i, b, ix in zip(t["id"], t["body"], t["indexable"]) if ix]
    random.Random(42).shuffle(rows)
    json.dump({"total": len(rows), "sample": rows[:4096]}, open(out, "w", encoding="utf-8"))
    print(f"{len(rows):,} indexable chunks; wrote 4,096 to {out}")


def run(path: str, n: int, batches: list[int], dtype: str) -> None:
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", dev, torch.cuda.get_device_name(0) if dev == "cuda" else "", "| dtype", dtype, flush=True)
    d = json.load(open(path, encoding="utf-8"))
    rows = [tuple(x) for x in d["sample"][:n]]
    print(f"indexable chunks {d['total']:,}; sample {len(rows)}; "
          f"mean chars {sum(len(b) for _, b in rows) / len(rows):.0f}", flush=True)
    tok = AutoTokenizer.from_pretrained("/model", padding_side="left")
    model = AutoModel.from_pretrained("/model", dtype=getattr(torch, dtype)).to(dev).eval()

    def embed(texts):
        b = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(dev)
        with torch.no_grad():
            h = model(**b).last_hidden_state
        F.normalize(h[:, -1].float(), p=2, dim=1).cpu()
        return int(b["attention_mask"].sum())

    embed([rows[0][1]] * 2)  # warm-up: CUDA init and kernel selection
    order = sorted(range(len(rows)), key=lambda i: len(rows[i][1]))
    for bs in batches:
        if dev == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t0, toks = time.perf_counter(), 0
        for s in range(0, len(order), bs):
            toks += embed([rows[j][1] for j in order[s:s + bs]])
        if dev == "cuda":
            torch.cuda.synchronize()
        el = time.perf_counter() - t0
        peak = torch.cuda.max_memory_allocated() / 2**20 if dev == "cuda" else 0
        rate = len(rows) / el
        print(f"batch {bs:>3}: {el:6.1f}s {rate:6.1f} chunks/s {toks / el:8.0f} tok/s peakGPU {peak:5.0f} MiB "
              f"-> 355,621 chunks ~ {355621 / rate / 3600:.2f} h", flush=True)


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "sample":
        sample(sys.argv[2], sys.argv[3])
    elif len(sys.argv) >= 6 and sys.argv[1] == "run":
        run(sys.argv[2], int(sys.argv[3]), [int(x) for x in sys.argv[4].split(",")], sys.argv[5])
    else:
        sys.exit(__doc__)
