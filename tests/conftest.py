"""Make the suite runnable anywhere: no weights, no GPU, and no MLX on Linux CI.

engine.py and coach.py import mlx / mlx_lm / fastembed at module level. When those packages are
missing we register empty stand-in modules so the pure-Python logic (search, features, buckets,
the FastAPI app) can still be imported and tested. Tests that need the real MLX use `mlx_real`.
"""
import importlib.util
import os
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:
        return False


# CHESSLM_TEST_NO_MLX=1 forces the Linux-CI code path on a Mac.
HAVE_MLX = _available("mlx") and _available("mlx_lm") and not os.environ.get("CHESSLM_TEST_NO_MLX")


def _stub(name: str, **attrs):
    mod = types.ModuleType(name)
    mod.__dict__.update(attrs)
    sys.modules[name] = mod
    return mod


if not HAVE_MLX:
    _stub("mlx")
    _stub("mlx.core")
    _stub("mlx_lm", load=None, generate=None)
    _stub("mlx_lm.models")
    _stub("mlx_lm.models.cache", KVCache=object, make_prompt_cache=None, trim_prompt_cache=None)
if not _available("fastembed"):
    _stub("fastembed", TextEmbedding=object)

