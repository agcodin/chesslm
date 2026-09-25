"""Heal a compressed model: a short LoRA finetune on chess positions after factorisation.

Truncating ranks throws away directions in weight space, and the surviving parameters have never
been asked to compensate. Healing gives them a few hundred steps to do so. This is the step that
separates "decomposition destroys the model" from the numbers compression vendors quote, and it is
cheap: LoRA on the attention projections plus both low-rank factors, trained on the same data the
model was fine-tuned on.

Afterwards the adapters are folded back into the factors, so the saved model keeps exactly the
low-rank architecture and parameter count it had before healing -- the recovery is free at
inference time.
"""
import argparse, json, math, random, time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten
from mlx_lm.tuner.lora import LoRALinear

import compress as C

BASE = "Qwen/Qwen2.5-1.5B-Instruct"
ATTN = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj")


def attach_lora(model, rank=8, scale=20.0, factors=True):
    """LoRA on the attention projections and, optionally, on both halves of each factorisation."""
    n = 0
    for layer in model.model.layers:
        targets = [(layer, p.split(".")) for p in ATTN]
        if factors:
            for name in C.TARGETS:
                mod = layer
                for part in name.split("."):
                    mod = getattr(mod, part)
                if isinstance(mod, C.LowRankLinear):
                    targets += [(mod, ["down"]), (mod, ["up"])]
        for root, path in targets:
            obj = root
            for p in path[:-1]:
                obj = getattr(obj, p)
            base = getattr(obj, path[-1])
            setattr(obj, path[-1], LoRALinear.from_base(base, r=rank, scale=scale))
            n += 1
    model.freeze()
    for layer in model.model.layers:
        for _, m in layer.named_modules():
            if isinstance(m, LoRALinear):
                m.unfreeze(keys=["lora_a", "lora_b"], recurse=False)
    return n


def fuse_lora(model):
    fused = [(n, m.fuse()) for n, m in model.named_modules() if hasattr(m, "fuse")]
    if fused:
        model.update_modules(tree_unflatten(fused))
    mx.eval(model.parameters())


def load_rows(paths, limit=None):
    rows = []
    for p in paths:
        with open(p) as f:
            for line in f:
                rows.append(json.loads(line))
    random.shuffle(rows)
    return rows[:limit] if limit else rows


def batches(rows, tok, batch_size, max_len=192):
    """Yield (inputs, targets, mask) with the prompt masked out of the loss."""
    for i in range(0, len(rows) - batch_size + 1, batch_size):
        chunk = rows[i:i + batch_size]
        seqs, starts = [], []
        for r in chunk:
            p = tok.encode(r["prompt"])
            c = tok.encode(r["completion"]) + [tok.eos_token_id]
            seqs.append((p + c)[:max_len])
            starts.append(min(len(p), max_len))
        L = max(len(s) for s in seqs)
        pad = tok.eos_token_id
        arr = mx.array([s + [pad] * (L - len(s)) for s in seqs])
        mask = mx.array([[1.0 if (starts[j] <= t < len(seqs[j])) else 0.0 for t in range(L)]
                         for j in range(len(seqs))])
        yield arr[:, :-1], arr[:, 1:], mask[:, 1:]


def loss_fn(model, x, y, mask):
    logits = model(x).astype(mx.float32)
    ce = nn.losses.cross_entropy(logits, y) * mask
    return ce.sum() / mx.maximum(mask.sum(), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="asvd", choices=["svd", "asvd"])
    ap.add_argument("--keep", type=float, default=0.5)
    ap.add_argument("--adapter", default="adapters_best")
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-rank", type=int, default=8)
    ap.add_argument("--data", default="data_v3/train.jsonl")
    ap.add_argument("--limit", type=int, default=20000)
    ap.add_argument("--save", default=None, help="directory to write the healed model into")
    ap.add_argument("--log", default="runs/compress/heal.log")
    a = ap.parse_args()

    random.seed(0)
    model, tok = C.load_fused(BASE, a.adapter)
    mods = C.linear_modules(model)
    ranks = {p: C.rank_for_keep(*mods[p].weight.shape, a.keep) for p in mods}
    scales = None
    if a.arm == "asvd":
        calib = [json.loads(l)["prompt"] for l in open("data/test.jsonl")][-32:]
        scales = C.calibrate(model, tok, calib, alpha=0.5)
    C.apply_ranks(model, ranks, scales)
    mx.eval(model.parameters())
    n_lora = attach_lora(model, rank=a.lora_rank)
    trainable = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
    print(f"{a.arm} keep={a.keep} rank={ranks[next(iter(ranks))]} "
          f"lora_layers={n_lora} trainable={trainable/1e6:.1f}M", flush=True)

    rows = load_rows([a.data], a.limit)
    opt = optim.AdamW(learning_rate=a.lr)
    state = [model.state, opt.state]

    @mx.compile
    def step(x, y, m):
        lv, grads = nn.value_and_grad(model, loss_fn)(model, x, y, m)
        opt.update(model, grads)
        return lv

    it, t0, run = 0, time.time(), []
    log = Path(a.log)
    log.parent.mkdir(parents=True, exist_ok=True)
    while it < a.iters:
        for x, y, m in batches(rows, tok, a.batch_size):
            lv = step(x, y, m)
            mx.eval(state)
            run.append(float(lv))
            it += 1
            if it % 25 == 0:
                msg = (f"iter {it}/{a.iters} loss {sum(run[-25:])/len(run[-25:]):.4f} "
                       f"({(time.time()-t0)/it:.2f}s/it)")
                print(msg, flush=True)
                with open(log, "a") as f:
                    f.write(f"{a.arm} keep={a.keep} {msg}\n")
            if it >= a.iters:
                break
            if it % 100 == 0:
                mx.clear_cache()

    fuse_lora(model)
    out = a.save or f"models/healed_{a.arm}_keep{a.keep:g}"
    C.save_compressed(model, tok, out, BASE, ranks,
                      {"arm": a.arm, "keep": a.keep, "healed_iters": a.iters,
                       "final_loss": round(sum(run[-25:]) / len(run[-25:]), 4)})
    print(f"saved {out}  params {C.param_count(model)/1e6:.1f}M", flush=True)


if __name__ == "__main__":
    main()
