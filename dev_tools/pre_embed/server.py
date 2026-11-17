"""A CPU embedding server speaking the same `/embed` the engine already calls.

    MODEL_DIR=/data/model DTYPE=bfloat16 python server.py

! Exists because the pinned TEI CPU image cannot serve this model: ONNX Runtime
declines `last_token` pooling, the candle backend takes over, and MKL's SGEMM
faults (`docs/HARDWARE.md` §1). The GPU image is fine; this is the 2 vCPU / 4 GB
profile's only gap.

! The pooling, the normalization and the dtype are the WHOLE contract. The
corpus declares its space and the engine's startup canary re-embeds a stored
chunk and refuses to serve below cosine 0.98 — so a mistake here cannot ship
silently, it stops the process. That guard is why quantizing is a cheap
experiment rather than a dangerous one.
"""

from __future__ import annotations

import json
import copy
import functools
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, DynamicCache

MODEL_DIR = os.environ.get("MODEL_DIR", "/data/model")
DTYPE = os.environ.get("DTYPE", "float32")
QUANTIZE = os.environ.get("QUANTIZE", "0") == "1"
PORT = int(os.environ.get("PORT", "80"))
# ! Must match the model's real context. Truncating silently would embed a
# vector that does not represent its text.
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "32768"))
THREADS = int(os.environ.get("TORCH_THREADS", "2"))
# ! A corpus embedded with an extra closing token after the tokenizer's own (Ravel's
# `append_eos` recipe: `<|im_end|>`, id 151645, after `<|endoftext|>`) lives in a different
# space from a plain tokenizer call (cosine ~0.64). Queries must be closed the same way.
# Empty = standard. The corpus canary (reference.npy) is what proves the choice.
APPEND_TOKEN_ID = int(os.environ["APPEND_TOKEN_ID"]) if os.environ.get("APPEND_TOKEN_ID") else None

torch.set_num_threads(THREADS)

DTYPES = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}

print(f"loading {MODEL_DIR} dtype={DTYPE} quantize={QUANTIZE} threads={THREADS} "
      f"append_token_id={APPEND_TOKEN_ID}",
      file=sys.stderr, flush=True)
tok = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModel.from_pretrained(MODEL_DIR, torch_dtype=DTYPES[DTYPE]).eval()

if QUANTIZE:
    # ! Linear layers only. `nn.Embedding` is 151,936 x 1024 and dynamic
    # quantization leaves it alone, so this does not halve total resident size —
    # it halves the part that dominates compute.
    model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    print("quantized · qint8 over nn.Linear", file=sys.stderr, flush=True)

# ! Three changes that leave every vector as it was (`ROADMAP.md` Stage 4). Each has a
# switch so it can be measured against its absence; none is a different space.
#
# NARROW_EMBEDDINGS: the token table (151,669 x 1024, a quarter of the weights) is only
# ever looked up, never multiplied. Stored as bfloat16 and widened per looked-up row, it
# is EXACT when the checkpoint itself is bfloat16: widening bf16 -> fp32 is lossless,
# so the rows are bit-identical to the fp32 table from_pretrained built. ~310 MB.
NARROW_EMBEDDINGS = os.environ.get("NARROW_EMBEDDINGS", "1") == "1"
# PREFIX_CACHE: every query opens with the same instruction. The model is causal, so the
# instruction's keys and values do not depend on the question; computed once, they are
# reused, and only the question's tokens run. Measured cosine against the full pass:
# 1.000000 over 40 eval questions, 17 of a median 52 tokens saved; the startup canary
# reproduces at 1.00000.
PREFIX_CACHE = os.environ.get("PREFIX_CACHE", "1") == "1"
# SERIAL: one forward pass at a time. Off, because measured it loses: 4 clients on 2
# cores serve 2.47 q/s serialised against 2.85 in parallel (ROADMAP.md Stage 4). The
# passes overlap their tokenisation, Python and memory stalls with each other's math.
SERIAL = os.environ.get("SERIAL", "0") == "1"
# A shared opening shorter than this is not worth a cache.
MIN_PREFIX = 8

if NARROW_EMBEDDINGS and DTYPE == "float32" and not QUANTIZE:
    # ! The checkpoint's own config, ✗ `model.config`: loading with a torch_dtype may
    # rewrite the latter to the dtype it was loaded AS.
    with open(os.path.join(MODEL_DIR, "config.json"), encoding="utf-8") as f:
        stored = str(json.load(f).get("torch_dtype", ""))
    if stored == "bfloat16":
        table = model.get_input_embeddings()

        class _Narrow(torch.nn.Module):
            def __init__(self, weight: torch.Tensor) -> None:
                super().__init__()
                self.weight = torch.nn.Parameter(weight.to(torch.bfloat16), requires_grad=False)

            def forward(self, ids: torch.Tensor) -> torch.Tensor:
                return F.embedding(ids, self.weight).float()

        model.set_input_embeddings(_Narrow(table.weight.detach()))
        del table
        print("token table kept as bfloat16, widened per row (lossless: checkpoint is bf16)",
              file=sys.stderr, flush=True)
    else:
        print(f"token table left float32 · checkpoint is {stored or 'unknown'}, "
              "narrowing would not be lossless", file=sys.stderr, flush=True)

_lock = threading.Lock()
_prefix: dict = {"ids": None, "cache": None, "misses": 0, "last": None}


def _token_ids(text: str) -> list[int]:
    if APPEND_TOKEN_ID is None:
        return tok(text, truncation=True, max_length=MAX_TOKENS)["input_ids"]
    ids = tok(text, truncation=True, max_length=MAX_TOKENS - 1)["input_ids"]
    return [*ids, APPEND_TOKEN_ID]


def _learn_prefix(ids: list[int]) -> None:
    """Adopt the opening two consecutive requests share, when there is none or the current
    one has stopped matching. The boundary token is left to the question's side: BPE may
    merge it with the question's first word."""
    last, _prefix["last"] = _prefix["last"], ids
    if last is None or (_prefix["ids"] is not None and _prefix["misses"] < 3):
        return
    n = 0
    while n < min(len(last), len(ids)) and last[n] == ids[n]:
        n += 1
    n -= 1
    if n < MIN_PREFIX:
        return
    cache = DynamicCache()
    with torch.inference_mode():
        model(input_ids=torch.tensor([ids[:n]]), past_key_values=cache, use_cache=True)
    _prefix.update(ids=ids[:n], cache=cache, misses=0)
    print(f"prefix cache · {n} shared opening tokens", file=sys.stderr, flush=True)


def _embed_one(text: str) -> list[float]:
    ids = _token_ids(text)
    pre = _prefix["ids"]
    if pre is not None and len(ids) > len(pre) and ids[: len(pre)] == pre:
        cache = copy.deepcopy(_prefix["cache"])
        with torch.inference_mode():
            out = model(input_ids=torch.tensor([ids[len(pre):]]), past_key_values=cache,
                        use_cache=True).last_hidden_state
        _prefix["misses"] = 0
        return F.normalize(out[0, -1].float(), p=2, dim=0).tolist()
    if pre is not None:
        _prefix["misses"] += 1
    with torch.inference_mode():
        out = model(input_ids=torch.tensor([ids])).last_hidden_state
    _learn_prefix(ids)
    return F.normalize(out[0, -1].float(), p=2, dim=0).tolist()


def embed(texts: list[str]) -> list[list[float]]:
    if SERIAL:
        with _lock:
            return _embed(texts)
    return _embed(texts)


def _embed(texts: list[str]) -> list[list[float]]:
    if len(texts) == 1:
        return [list(_single(texts[0]))]
    return _embed_batch(texts)


def _single_uncached(text: str) -> tuple[float, ...]:
    if PREFIX_CACHE:
        # ! Unlocked by default. Each request copies the cache before extending it, so
        # the shared one is never written after it is built; two requests learning at
        # once only waste a pass.
        return tuple(_embed_one(text))
    return tuple(_embed_batch([text])[0])


# QUERY_CACHE: the last N single texts and their vectors, least recently used out. Exact
# (the vector is a function of the text) and bounded (N x 1024 floats, ~8 MB at the
# default as Python floats). An agent that retries or rephrases back to an earlier
# question pays for the engine and the database, ✗ for the model. 0 turns it off.
QUERY_CACHE = int(os.environ.get("QUERY_CACHE", "1024"))
if QUERY_CACHE < 0:
    sys.exit(f"QUERY_CACHE={QUERY_CACHE} · a cache cannot hold fewer than 0 entries")
_single = functools.lru_cache(maxsize=QUERY_CACHE)(_single_uncached)


def _embed_batch(texts: list[str]) -> list[list[float]]:
    if APPEND_TOKEN_ID is None:
        batch = tok(texts, padding=True, truncation=True, max_length=MAX_TOKENS,
                    return_tensors="pt")
    else:
        ids = tok(texts, padding=False, truncation=True, max_length=MAX_TOKENS - 1,
                  return_attention_mask=False)
        ids["input_ids"] = [[*seq, APPEND_TOKEN_ID] for seq in ids["input_ids"]]
        batch = tok.pad(ids, padding=True, return_tensors="pt")
    with torch.no_grad():
        out = model(**batch).last_hidden_state
    # ! LAST-TOKEN pooling, the value `corpus_meta` declares. Mean-pooling the
    # same weights is a different vector space that returns plausible, wrong
    # rankings — which is exactly what the canary exists to catch.
    mask = batch["attention_mask"]
    idx = mask.sum(dim=1) - 1
    pooled = out[torch.arange(out.size(0)), idx]
    return F.normalize(pooled.float(), p=2, dim=1).tolist()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, code: int, payload) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._send(200, {"status": "ok"})
        elif self.path == "/info":
            self._send(200, {
                "model_id": MODEL_DIR,
                "model_dtype": DTYPE,
                "quantized": QUANTIZE,
                "model_type": {"embedding": {"pooling": "last_token"}},
            })
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/embed":
            self._send(404, {"error": "not found"})
            return
        n = int(self.headers.get("Content-Length", "0"))
        req = json.loads(self.rfile.read(n) or b"{}")
        inputs = req.get("inputs", [])
        if isinstance(inputs, str):
            inputs = [inputs]
        try:
            self._send(200, embed(inputs))
        except Exception as e:  # noqa: BLE001 — the client needs the reason
            self._send(500, {"error": str(e)})

    def log_message(self, *_args) -> None:
        """stdout is not a log channel here either."""


if __name__ == "__main__":
    print(f"listening on :{PORT}", file=sys.stderr, flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
