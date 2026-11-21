"""Fit the `balanced` factor weights through the live engine, ✗ a mirror of it.

Each configuration is one `e2e.py` run on EVAL_SPLIT=train with
EVAL_ARGS={"factor_weights": ...}, three at a time:

    VERA_HTTP=... DATABASE_URL=... BM25_VOCAB=... python fit_engine.py out.jsonl grid
    ... python fit_engine.py out.jsonl floor '<best weights json>'
    ... python fit_engine.py - once fresh '<weights json>'      # the winner, ONCE

Why not `fit_pool.py`: its mirror reproduces the engine only while the engine has no
ordering rule the mirror lacks. On 2026-11-21 it agreed on 169 of 604 queries (the
structure orderings and the currency lift came after it). The engine cannot disagree with
itself. The cost is one engine call per query per configuration, minutes each.
! Ranking does not depend on cores: run the stack without the VPS pin to fit, then
measure latency pinned.
"""

import itertools
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# ! VERA_HTTP, DATABASE_URL and BM25_VOCAB come from the environment, as for e2e.py.
ENV = {**os.environ, "PYTHONUTF8": "1"}
E2E = str(Path(__file__).with_name("e2e.py"))
SHIPPED = {"relevance_floor": 0.3, "authority": 0.5, "structural": 0.25,
           "temporal": 0.0, "completeness": 0.0, "topical": 0.25}


def run(w: dict, split: str = "train") -> dict:
    env = {**ENV, "EVAL_SPLIT": split, "EVAL_ARGS": json.dumps({"factor_weights": w})}
    t = time.time()
    out = subprocess.run([sys.executable, "-X", "utf8", E2E], env=env, capture_output=True,
                         text=True, encoding="utf-8").stdout
    m = re.search(r"n=(\d+)\s+Recall@5 ([\d.]+)%\s+Recall@10 ([\d.]+)%\s+MRR ([\d.]+)", out)
    deep = dict(re.findall(r"Recall@(15|20|25) ([\d.]+)%", out))
    if not m:
        raise RuntimeError(out[-2000:])
    return {"w": w, "split": split, "n": int(m[1]), "r5": float(m[2]), "r10": float(m[3]),
            "mrr": float(m[4]), **{f"r{k}": float(v) for k, v in deep.items()},
            "s": round(time.time() - t)}


def main() -> None:
    out, mode = Path(sys.argv[1]), sys.argv[2]
    if mode == "grid":
        configs = [SHIPPED]
        for a, s, t in itertools.product((0, .25, .5, .75, 1), repeat=3):
            if a + s + t <= 1 + 1e-9 and a + s + t >= .5 - 1e-9:
                w = {**SHIPPED, "authority": a, "structural": s, "topical": t}
                if w != SHIPPED:
                    configs.append(w)
    elif mode == "floor":
        best = json.loads(sys.argv[3])
        configs = [{**best, "relevance_floor": f} for f in (0.0, 0.2, 0.4)]
    else:  # one config on one split: `once <split> <w-json>`
        print(json.dumps(run(json.loads(sys.argv[4]), sys.argv[3])))
        return
    print(f"{len(configs)} configs", flush=True)
    with ThreadPoolExecutor(3) as pool, open(out, "a", encoding="utf-8") as f:
        for r in pool.map(run, configs):
            f.write(json.dumps(r) + "\n")
            f.flush()
            w = r["w"]
            print(f"a {w['authority']} s {w['structural']} t {w['topical']} fl {w['relevance_floor']}"
                  f"  R@5 {r['r5']} R@10 {r['r10']} R@25 {r.get('r25')} MRR {r['mrr']}  {r['s']}s",
                  flush=True)


if __name__ == "__main__":
    main()
