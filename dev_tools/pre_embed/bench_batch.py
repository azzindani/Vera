"""Does running concurrent questions as ONE pass beat running them one by one?

    docker run --rm -i --cpuset-cpus 2,3 -v <model>:/data/model:ro -v <dir>:/w:ro \
        --entrypoint python vera-embed:cpu /w/bench_batch.py "<instruction>" < questions.txt

With the instruction's KV cache reused (`server.py` PREFIX_CACHE). The questions' tokens are
RIGHT-padded and no attention mask is passed: the model is causal, so a real token never
attends to a pad after it, and each question is pooled at its own last real token. Prints
per batch size the time of one batched pass against the same questions run singly, and
the lowest cosine between the two.
"""
import copy
import statistics
import sys
import time

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, DynamicCache

torch.set_num_threads(2)
tok = AutoTokenizer.from_pretrained("/data/model")
model = AutoModel.from_pretrained("/data/model", torch_dtype=torch.float32).eval()
prefix = f"Instruct: {sys.argv[1]}\nQuery: "
qs = [q.strip() for q in sys.stdin if q.strip()][:40]
ids = [tok(prefix + q)["input_ids"] for q in qs]
n = 0
while all(len(x) > n and x[n] == ids[0][n] for x in ids):
    n += 1
P = n - 1
cache = DynamicCache()
with torch.inference_mode():
    model(input_ids=torch.tensor([ids[0][:P]]), past_key_values=cache, use_cache=True)
pad = tok.pad_token_id if tok.pad_token_id is not None else 0


def single(x):
    c = copy.deepcopy(cache)
    with torch.inference_mode():
        out = model(input_ids=torch.tensor([x[P:]]), past_key_values=c,
                    use_cache=True).last_hidden_state
    return F.normalize(out[0, -1], dim=0)


def batched(xs):
    rest = [x[P:] for x in xs]
    width = max(map(len, rest))
    t = torch.tensor([r + [pad] * (width - len(r)) for r in rest])
    c = copy.deepcopy(cache)
    c.batch_repeat_interleave(len(xs))
    with torch.inference_mode():
        out = model(input_ids=t, past_key_values=c, use_cache=True).last_hidden_state
    last = torch.tensor([len(r) - 1 for r in rest])
    return F.normalize(out[torch.arange(len(xs)), last], dim=1)


single(ids[0])
batched(ids[:2])
print(f"prefix {P} tokens")
for b in (1, 2, 4, 8):
    ts, tb, cos = [], [], []
    for i in range(0, len(ids) - b + 1, b):
        group = ids[i:i + b]
        s = time.perf_counter()
        ref = [single(x) for x in group]
        ts.append(time.perf_counter() - s)
        s = time.perf_counter()
        got = batched(group)
        tb.append(time.perf_counter() - s)
        cos += [float(r @ g) for r, g in zip(ref, got)]
    print(f"  batch {b}: singly {statistics.median(ts) * 1000:5.0f} ms  batched "
          f"{statistics.median(tb) * 1000:5.0f} ms  per question {statistics.median(tb) / b * 1000:4.0f}"
          f" ms  min cosine {min(cos):.6f}", flush=True)
