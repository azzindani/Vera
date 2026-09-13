"""A CPU embedding server speaking the same `/embed` the engine already calls.

    MODEL_DIR=/data/model DTYPE=bfloat16 python server.py

! IT CANNOT SERVE THE CURRENT CORPUS, and that is what it was built to find
out. It reproduces `dev_tools/pre_embed/dense_probe.py` — the model's reference
implementation, last-token pooling and L2, the same thing `1_Pooling/config.json`
specifies — and its vectors land at cosine **0.002–0.19** against the stored
ones, where the canary needs 0.98.

! That is not a bug here. It returns **byte-identical vectors to TEI
`cpu-1.9.3`**, agreeing to five decimals across five chunks. Two independent
implementations agreeing with each other and disagreeing with the corpus is what
established that the CORPUS is the outlier: it is in the space of the pinned
`86-1.7.2` image and nothing else reproduces it (`docs/EMBEDDING.md` §5,
`docs/HARDWARE.md` §1).

Kept rather than deleted for two reasons: it is the measurement that pinned the
defect down, and it is a working CPU embedder for this model on the day the
corpus is in a space a second implementation can reproduce. It is ✗ part of the
serving stack today — `docker-compose.yml` does not reference it.

! The pooling, the normalization and the dtype are the WHOLE contract. The
engine's startup canary re-embeds a stored chunk and refuses to serve below
cosine 0.98, so a mistake here cannot ship silently — it stops the process.
That guard is what made this a safe experiment, and it is what refused this
server.
"""

from __future__ import annotations

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

MODEL_DIR = os.environ.get("MODEL_DIR", "/data/model")
DTYPE = os.environ.get("DTYPE", "float32")
QUANTIZE = os.environ.get("QUANTIZE", "0") == "1"
PORT = int(os.environ.get("PORT", "80"))
# ! Must match the model's real context. Truncating silently would embed a
# vector that does not represent its text.
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "32768"))
THREADS = int(os.environ.get("TORCH_THREADS", "2"))

torch.set_num_threads(THREADS)

DTYPES = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}

print(f"loading {MODEL_DIR} dtype={DTYPE} quantize={QUANTIZE} threads={THREADS}",
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


def embed(texts: list[str]) -> list[list[float]]:
    batch = tok(texts, padding=True, truncation=True, max_length=MAX_TOKENS,
                return_tensors="pt")
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
