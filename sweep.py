"""Run every compression arm over a range of ratios and record all four metrics for each.

Arms:
  quant   n-bit weight quantisation -- the strong, nearly-free baseline
  svd     plain truncated SVD of the MLP projections
  asvd    the same, weighted by activation statistics collected on chess positions

Each configuration reloads the fused model, because the factorisation is destructive surgery.
That costs ~15s per config and is far cheaper than debugging state that leaked between arms.
"""
import argparse, json, time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

import compress as C
from eval_compress import evaluate_all

BASE = "Qwen/Qwen2.5-1.5B-Instruct"
OUT = Path("runs/compress")


def calib_prompts(n=32):
    return [json.loads(l)["prompt"] for l in open("data/test.jsonl")][-n:]  # tail: not the eval slice


def build(arm, level, adapter, scales_cache):
    """Return ((model, tok), tag, params) for one configuration."""
    model, tok = C.load_fused(BASE, adapter)
    dense = C.param_count(model)

    if arm == "quant":
        nn.quantize(model, group_size=64, bits=level)
        mx.eval(model.parameters())
        # Quantised weights are packed into uint32, so count bits rather than array elements.
        bytes_ = sum(v.nbytes for _, v in tree_flatten(model.parameters()))
        return (model, tok), f"quant_{level}bit", {"eff_params": None, "bytes": bytes_}

    alpha = 0.0 if arm == "svd" else 0.5
    scales = None
    if alpha:
        key = round(alpha, 2)
        if key not in scales_cache:
            scales_cache[key] = C.calibrate(model, tok, calib_prompts(), alpha=alpha)
        scales = scales_cache[key]
    mods = C.linear_modules(model)
    ranks = {p: C.rank_for_keep(*mods[p].weight.shape, level) for p in mods}
    C.apply_ranks(model, ranks, scales)
    mx.eval(model.parameters())
    now = C.param_count(model)
    return (model, tok), f"{arm}_keep{level:g}", {
        "params": now, "dense_params": dense, "shrink": round(1 - now / dense, 4),
        "rank": ranks[next(iter(ranks))]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="adapters_best")
    ap.add_argument("--n-move", type=int, default=120)
    ap.add_argument("--n-value", type=int, default=100)
    ap.add_argument("--arms", default="quant,svd,asvd")
    ap.add_argument("--keeps", default="0.8,0.6,0.5,0.4,0.3,0.2")
    ap.add_argument("--bits", default="8,4,3")
    ap.add_argument("--out", default=str(OUT / "results.jsonl"))
    a = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    done = set()
    if Path(a.out).exists():
        done = {json.loads(l)["tag"] for l in open(a.out)}

    jobs = []
    for arm in a.arms.split(","):
        levels = [int(b) for b in a.bits.split(",")] if arm == "quant" else \
                 [float(k) for k in a.keeps.split(",")]
        jobs += [(arm, lv) for lv in levels]

    scales_cache = {}
    for arm, level in jobs:
        tag = f"quant_{level}bit" if arm == "quant" else f"{arm}_keep{level:g}"
        if tag in done:
            print(f"skip {tag} (already recorded)", flush=True)
            continue
        t0 = time.time()
        try:
            loaded, tag, meta = build(arm, level, a.adapter, scales_cache)
            res = evaluate_all(loaded, None, a.n_move, a.n_value)
            row = {"tag": tag, "arm": arm, "level": level, **meta, **res,
                   "secs": round(time.time() - t0, 1)}
        except Exception as e:                      # one bad config should not kill the sweep
            row = {"tag": tag, "arm": arm, "level": level, "error": repr(e)[:300]}
        with open(a.out, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        del loaded
        mx.clear_cache()


if __name__ == "__main__":
    main()
