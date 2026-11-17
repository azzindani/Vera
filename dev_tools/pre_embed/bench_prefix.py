"""Inside vera-embed: does reusing the instruction prefix's KV cache reproduce the embedding,
and what does it save? argv[1] = instruction, then queries on stdin (one per line)."""
import copy
import statistics
import sys
import time

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, DynamicCache

torch.set_num_threads(int(sys.argv[2]) if len(sys.argv) > 2 else 2)
tok = AutoTokenizer.from_pretrained("/data/model")
model = AutoModel.from_pretrained("/data/model", torch_dtype=torch.float32).eval()
prefix = f"Instruct: {sys.argv[1]}\nQuery: "
queries = [q.strip() for q in sys.stdin if q.strip()][:40]


def full(text):
    b = tok([text], return_tensors="pt")
    with torch.inference_mode():
        out = model(**b).last_hidden_state
    return F.normalize(out[0, -1].float(), dim=0), b["input_ids"].shape[1]


ids0 = tok([prefix + queries[0]])["input_ids"][0]
ids1 = tok([prefix + queries[1]])["input_ids"][0]
lcp = 0
while lcp < min(len(ids0), len(ids1)) and ids0[lcp] == ids1[lcp]:
    lcp += 1
P = lcp - 1  # leave the boundary token to the query side
pre = torch.tensor([ids0[:P]])
with torch.inference_mode():
    cache = DynamicCache()
    model(input_ids=pre, past_key_values=cache, use_cache=True)
print("prefix tokens", P)


def cached(text):
    ids = tok([text])["input_ids"][0]
    if ids[:P] != ids0[:P]:
        return None, len(ids)
    rest = torch.tensor([ids[P:]])
    c = copy.deepcopy(cache)
    with torch.inference_mode():
        out = model(input_ids=rest, past_key_values=c, use_cache=True).last_hidden_state
    return F.normalize(out[0, -1].float(), dim=0), len(ids)


for f in (full, cached):  # warm
    f(prefix + queries[0])
tf, tc, cos, lens = [], [], [], []
for q in queries:
    t = time.perf_counter()
    a, n = full(prefix + q)
    tf.append(time.perf_counter() - t)
    t = time.perf_counter()
    b, _ = cached(prefix + q)
    tc.append(time.perf_counter() - t)
    lens.append(n)
    cos.append(float(a @ b) if b is not None else float("nan"))
print(f"tokens median {statistics.median(lens)}  full p50 {statistics.median(tf)*1000:.0f} ms"
      f"  cached p50 {statistics.median(tc)*1000:.0f} ms  min cosine {min(cos):.6f}"
      f"  misses {sum(c != c for c in cos)}")
