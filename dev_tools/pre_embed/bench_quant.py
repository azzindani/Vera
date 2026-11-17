"""Inside vera-embed: which Linear layers can be dynamic-int8 without moving the space?
argv[1] instruction; queries on stdin. Prints, per variant: p50 latency and min / median
cosine against the float32 reference, over the queries (with instruction) and over
documents (the queries' text without instruction, as a stand-in for chunks)."""
import copy
import statistics
import sys
import time

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

torch.set_num_threads(2)
tok = AutoTokenizer.from_pretrained("/data/model")
base = AutoModel.from_pretrained("/data/model", torch_dtype=torch.float32).eval()
prefix = f"Instruct: {sys.argv[1]}\nQuery: "
qs = [q.strip() for q in sys.stdin if q.strip()][:30]
texts = [prefix + q for q in qs]


def emb(model, text):
    b = tok([text], return_tensors="pt")
    with torch.inference_mode():
        out = model(**b).last_hidden_state
    return F.normalize(out[0, -1].float(), dim=0)


ref = [emb(base, t) for t in texts]

VARIANTS = {
    "mlp gate+up": ("gate_proj", "up_proj"),
    "mlp all": ("gate_proj", "up_proj", "down_proj"),
    "attn qkv": ("q_proj", "k_proj", "v_proj"),
    "all but down_proj": ("gate_proj", "up_proj", "q_proj", "k_proj", "v_proj", "o_proj"),
    "all": ("gate_proj", "up_proj", "down_proj", "q_proj", "k_proj", "v_proj", "o_proj"),
}
only = sys.argv[2].split(",") if len(sys.argv) > 2 else list(VARIANTS)
for name in only:
    keep = VARIANTS[name]
    m = copy.deepcopy(base)
    targets = {n for n, mod in m.named_modules()
               if isinstance(mod, torch.nn.Linear) and n.split(".")[-1] in keep}
    m = torch.ao.quantization.quantize_dynamic(m, targets, dtype=torch.qint8)
    emb(m, texts[0])
    ts, cs = [], []
    for t, r in zip(texts, ref):
        s = time.perf_counter()
        e = emb(m, t)
        ts.append(time.perf_counter() - s)
        cs.append(float(e @ r))
    print(f"  {name:20} p50 {statistics.median(ts) * 1000:4.0f} ms  cosine min "
          f"{min(cs):.4f} median {statistics.median(cs):.4f}", flush=True)
    del m
s = time.perf_counter()
for t in texts:
    emb(base, t)
print(f"  {'float32':20} p50 {(time.perf_counter() - s) / len(texts) * 1000:4.0f} ms (mean)")
