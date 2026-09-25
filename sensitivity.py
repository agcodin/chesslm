"""Which layers can afford to lose rank, and what a non-uniform budget buys.

A uniform rank across all 28 layers assumes every layer is equally redundant, which is false.
Profiling compresses one layer at a time, hard, and measures the damage; the resulting profile is
then used to spend a fixed parameter budget where it is cheapest, instead of spreading it evenly.

Two commands:
  profile   compress layer i alone, record the damage, for every i
  allocate  turn a profile into per-layer ranks that hit a target size
"""
import argparse, json
from pathlib import Path

import mlx.core as mx

import compress as C
from eval_compress import perplexity

BASE = "Qwen/Qwen2.5-1.5B-Instruct"
OUT = Path("runs/compress")


def calib_prompts(n=32):
    return [json.loads(l)["prompt"] for l in open("data/test.jsonl")][-n:]


def eval_rows(n):
    return [json.loads(l) for l in open("data/test.jsonl")][:n]


def profile(adapter, keep, n_ppl, arm):
    """Damage from compressing each layer alone. Perplexity is the cheap proxy here on purpose:
    it needs one forward pass per row, and the profile only has to rank layers, not score them."""
    rows = eval_rows(n_ppl)
    out_path = OUT / f"sensitivity_{arm}_keep{keep:g}.jsonl"
    done = {json.loads(l)["layer"] for l in open(out_path)} if out_path.exists() else set()

    model, tok = C.load_fused(BASE, adapter)
    from engine import Policy
    base_ppl = perplexity(Policy((model, tok), adapter=None), rows)
    print(json.dumps({"layer": "dense", "ppl": round(base_ppl, 4)}), flush=True)
    del model

    n_layers = 28
    for i in range(n_layers):
        if i in done:
            continue
        model, tok = C.load_fused(BASE, adapter)
        scales = C.calibrate(model, tok, calib_prompts(), alpha=0.5) if arm == "asvd" else None
        mods = C.linear_modules(model)
        ranks = {p: C.rank_for_keep(*mods[p].weight.shape, keep)
                 for p in mods if p.startswith(f"model.layers.{i}.")}
        C.apply_ranks(model, ranks, scales)
        mx.eval(model.parameters())
        ppl = perplexity(Policy((model, tok), adapter=None), rows)
        row = {"layer": i, "keep": keep, "arm": arm, "ppl": round(ppl, 4),
               "base_ppl": round(base_ppl, 4), "delta": round(ppl - base_ppl, 4),
               "ratio": round(ppl / base_ppl, 4)}
        with open(out_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        del model
        mx.clear_cache()


def allocate(profile_path, target_keep, beta=1.0, floor=0.15, ceil=0.95) -> dict:
    """Per-layer keep fractions from a sensitivity profile, hitting `target_keep` overall.

    Sensitive layers keep more rank. `ratio` (perplexity multiplier when that layer alone is
    compressed) is the sensitivity signal; beta controls how aggressively the profile is followed,
    and bisection finds the scale that lands on the requested average.
    """
    rows = [json.loads(l) for l in open(profile_path) if json.loads(l).get("layer") != "dense"]
    sens = {r["layer"]: max(r["ratio"] - 1.0, 1e-4) ** beta for r in rows}
    mean = sum(sens.values()) / len(sens)
    rel = {i: s / mean for i, s in sens.items()}

    def keeps_for(k):
        return {i: min(ceil, max(floor, k * r)) for i, r in rel.items()}

    lo, hi = 1e-3, 50.0
    for _ in range(60):
        mid = (lo + hi) / 2
        avg = sum(keeps_for(mid).values()) / len(rel)
        if avg < target_keep:
            lo = mid
        else:
            hi = mid
    return keeps_for((lo + hi) / 2)


def ranks_from_keeps(model, keeps: dict) -> dict:
    mods = C.linear_modules(model)
    out = {}
    for p, mod in mods.items():
        i = int(p.split(".")[2])
        out[p] = C.rank_for_keep(*mod.weight.shape, keeps[i])
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["profile", "allocate"])
    ap.add_argument("--adapter", default="adapters_best")
    ap.add_argument("--keep", type=float, default=0.3)
    ap.add_argument("--arm", default="asvd", choices=["svd", "asvd"])
    ap.add_argument("--n-ppl", type=int, default=60)
    ap.add_argument("--profile", default=None)
    ap.add_argument("--target-keep", type=float, default=0.5)
    ap.add_argument("--beta", type=float, default=1.0)
    a = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    if a.cmd == "profile":
        profile(a.adapter, a.keep, a.n_ppl, a.arm)
    else:
        path = a.profile or OUT / f"sensitivity_{a.arm}_keep{a.keep:g}.jsonl"
        keeps = allocate(path, a.target_keep, a.beta)
        print(json.dumps({str(k): round(v, 3) for k, v in sorted(keeps.items())}, indent=2))
