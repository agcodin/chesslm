"""Low-rank compression of the MLP projections, with activation-aware factorisation.

The three MLP projections hold ~75% of Qwen2.5-1.5B's parameters, so they are where a rank
budget buys the most. Each Linear W (d_out, d_in) becomes B @ A with A (r, d_in), B (d_out, r),
which costs r * (d_in + d_out) parameters instead of d_in * d_out.

Plain SVD minimises ||W - BA||, which is the wrong objective: what matters is the error in the
layer's *output*, x W^T. Scaling the columns of W by how strongly each input channel actually
fires on real positions, factorising that, then unscaling, minimises a closer proxy. That is the
ASVD idea, and here the calibration data is chess positions rather than generic text.
"""
import json, shutil
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

TARGETS = ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")


class LowRankLinear(nn.Module):
    """W x approximated as up(down(x)). Bias, if any, is carried unchanged on the outer factor.

    The two factors are real nn.Linear submodules rather than bare arrays so that LoRA adapters
    can be attached to them during healing exactly as they would be to any other projection.
    """

    def __init__(self, A: mx.array, B: mx.array, bias: mx.array | None = None):
        super().__init__()
        r, d_in = A.shape
        d_out = B.shape[0]
        self.down = nn.Linear(d_in, r, bias=False)
        self.up = nn.Linear(r, d_out, bias=bias is not None)
        self.down.weight = A
        self.up.weight = B
        if bias is not None:
            self.up.bias = bias

    @property
    def rank(self) -> int:
        return self.down.weight.shape[0]

    def __call__(self, x: mx.array) -> mx.array:
        return self.up(self.down(x))


def breakeven_rank(d_out: int, d_in: int) -> int:
    """Above this rank the factorisation stores more parameters than the dense matrix."""
    return (d_out * d_in) // (d_out + d_in)


def rank_for_keep(d_out: int, d_in: int, keep: float) -> int:
    """Rank whose factorisation costs `keep` times the dense matrix (keep=0.25 -> 4x smaller)."""
    return max(1, int(keep * d_out * d_in / (d_out + d_in)))


def factorize(W: mx.array, rank: int, scale: mx.array | None = None) -> tuple[mx.array, mx.array]:
    """Truncated SVD of W (optionally column-scaled by `scale`), returned as (A, B) with W ~ B @ A.

    `scale` is a per-input-channel weighting: we factorise W * scale and fold scale^-1 back into
    A, so the discarded directions are the ones that matter least for real activations.
    """
    M = W.astype(mx.float32)
    if scale is not None:
        M = M * scale[None, :]
    B, A = _thin_svd(M, rank)
    if scale is not None:
        A = A / scale[None, :]
    return A.astype(W.dtype), B.astype(W.dtype)


def _thin_svd(M: mx.array, rank: int) -> tuple[mx.array, mx.array]:
    """Rank-r factorisation M ~ B @ A via the Gram matrix of whichever side is smaller.

    A direct SVD of an 8960-row projection asks for an 8960x8960 U, which is far more work than
    the truncation needs. Both projection shapes here have a 1536-wide side, so one symmetric
    eigendecomposition of that side gives the same top-r subspace. Squaring the matrix squares its
    condition number too, which is why this runs in float32 and clamps near-zero eigenvalues.
    """
    m, n = M.shape
    if n <= m:                                  # (8960, 1536): Gram over columns
        w, V = mx.linalg.eigh(M.T @ M, stream=mx.cpu)
        V = V[:, ::-1][:, :rank]                # eigh returns ascending eigenvalues
        s = mx.sqrt(mx.maximum(w[::-1][:rank], 1e-12))
        B = (M @ V) / s[None, :]
        A = s[:, None] * V.T
    else:                                       # (1536, 8960): Gram over rows
        w, U = mx.linalg.eigh(M @ M.T, stream=mx.cpu)
        B = U[:, ::-1][:, :rank]
        A = B.T @ M                             # equals S @ Vt without forming either
    return B, A


def load_fused(base_model: str, adapter: str | None):
    """Load base + LoRA adapter and fold the adapter into the dense weights, in memory.

    The mlx_lm.fuse CLI insists on a complete Hub snapshot (README, LICENSE) which the local
    cache does not have, so do the same thing it does: every module exposing fuse() is replaced
    by its merged Linear. Compression then acts on the trained model, not on stock Qwen.
    """
    from mlx_lm import load
    model, tok = load(base_model, adapter_path=adapter)
    if adapter:
        fused = [(n, m.fuse()) for n, m in model.named_modules() if hasattr(m, "fuse")]
        if fused:
            model.update_modules(tree_unflatten(fused))
        mx.eval(model.parameters())
    return model, tok


def linear_modules(model) -> dict:
    """{"model.layers.3.mlp.up_proj": module} for every projection we are willing to compress."""
    out = {}
    for i, layer in enumerate(model.model.layers):
        for name in TARGETS:
            mod = layer
            for part in name.split("."):
                mod = getattr(mod, part)
            out[f"model.layers.{i}.{name}"] = mod
    return out


def _set_module(model, path: str, new):
    parts = path.split(".")
    obj = model
    for p in parts[:-1]:
        obj = obj[int(p)] if p.isdigit() else getattr(obj, p)
    setattr(obj, parts[-1], new)


def apply_ranks(model, ranks: dict, scales: dict | None = None) -> int:
    """Replace each named projection with a rank-r factorisation. Returns parameters removed."""
    saved = 0
    for path, r in ranks.items():
        mod = linear_modules(model)[path]
        if isinstance(mod, LowRankLinear):
            raise ValueError(f"{path} is already factorised")
        W = mod.weight
        d_out, d_in = W.shape
        if r >= breakeven_rank(d_out, d_in):
            continue  # a factorisation this wide would make the model bigger
        A, B = factorize(W, r, (scales or {}).get(path))
        bias = mod.bias if "bias" in mod else None
        _set_module(model, path, LowRankLinear(A, B, bias))
        saved += W.size - (A.size + B.size)
    return saved


def param_count(model) -> int:
    return sum(v.size for _, v in tree_flatten(model.parameters()))


# ---------------------------------------------------------------- calibration


class _Recorder(nn.Module):
    """Wraps a Linear and accumulates the mean absolute value of each input channel."""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        self._sum = None
        self._n = 0

    def __call__(self, x):
        flat = x.reshape(-1, x.shape[-1]).astype(mx.float32)
        s = mx.abs(flat).sum(axis=0)
        self._sum = s if self._sum is None else self._sum + s
        self._n += flat.shape[0]
        return self.inner(x)

    def stat(self):
        return self._sum / max(self._n, 1)


def calibrate(model, tokenizer, prompts, alpha: float = 0.5) -> dict:
    """Per-channel activation scales for each target projection, from real prompts.

    alpha interpolates between plain SVD (0.0) and fully activation-weighted (1.0); 0.5 is the
    value ASVD reports as a good default and is what the sweep uses unless told otherwise.
    """
    originals = linear_modules(model)
    recorders = {}
    for path, mod in originals.items():
        rec = _Recorder(mod)
        recorders[path] = rec
        _set_module(model, path, rec)
    try:
        for text in prompts:
            model(mx.array([tokenizer.encode(text)]))
            mx.eval([r._sum for r in recorders.values()])
    finally:
        for path, rec in recorders.items():
            _set_module(model, path, rec.inner)
    scales = {}
    for path, rec in recorders.items():
        s = rec.stat()
        s = mx.maximum(s, 1e-5) ** alpha
        scales[path] = (s / s.mean()).astype(mx.float32)  # mean-1 so ranks stay comparable
    return scales


# ---------------------------------------------------------------- save / load


def save_compressed(model, tokenizer, out_dir: str, base_model: str, ranks: dict, meta: dict):
    """Write the surgically modified weights plus enough metadata to rebuild the architecture."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    weights = dict(tree_flatten(model.parameters()))
    mx.save_safetensors(str(out / "weights.safetensors"), weights)
    (out / "compress_config.json").write_text(json.dumps(
        {"base_model": base_model, "ranks": ranks, **meta}, indent=2))
    tokenizer.save_pretrained(str(out))
    src = Path(model_path_for(base_model))
    for f in ("config.json",):
        if (src / f).exists():
            shutil.copy(src / f, out / f)


def model_path_for(base_model: str) -> str:
    """Local snapshot directory for a model id (already cached by earlier training runs)."""
    from mlx_lm.utils import get_model_path
    p = get_model_path(base_model)
    return str(p[0] if isinstance(p, tuple) else p)


def load_compressed(path: str, adapter: str | None = None):
    """Rebuild a compressed model: load the base architecture, factorise, load saved weights."""
    from mlx_lm import load
    cfg = json.loads((Path(path) / "compress_config.json").read_text())
    model, tok = load(cfg["base_model"], adapter_path=adapter)
    for p, r in cfg["ranks"].items():
        mod = linear_modules(model)[p]
        d_out, d_in = mod.weight.shape
        zeros_A = mx.zeros((r, d_in), dtype=mod.weight.dtype)
        zeros_B = mx.zeros((d_out, r), dtype=mod.weight.dtype)
        _set_module(model, p, LowRankLinear(zeros_A, zeros_B, mod.bias if "bias" in mod else None))
    weights = mx.load(str(Path(path) / "weights.safetensors"))
    model.update(tree_unflatten(list(weights.items())))
    mx.eval(model.parameters())
    return model, tok
