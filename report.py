"""Build the results table, with confidence intervals, from the sweep's JSON lines.

top1 and legal are proportions over a finite test set, so they carry real sampling error. Quoting
them bare invites reading noise as signal, which is the main way a project like this goes wrong.
Every proportion here gets a Wilson 95% interval, and the relative-change column says plainly
whether the drop clears its own interval.
"""
import argparse, json, math
from pathlib import Path

RUNS = Path("runs/compress")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval: behaves sensibly for small n and proportions near 0, unlike normal."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def fmt_prop(p, n):
    lo, hi = wilson(round(p * n), n)
    return f"{p*100:.1f}% [{lo*100:.1f}–{hi*100:.1f}]"


def arm_of(r) -> str:
    """Rows written straight from eval_compress.py have no arm field; recover it from the tag."""
    if "arm" in r:
        return r["arm"]
    return r["tag"].split("_")[0]


def load(path):
    rs = [json.loads(l) for l in open(path) if "error" not in json.loads(l)]
    for r in rs:
        r["arm"] = arm_of(r)
    return rs


def size_label(r):
    if r["arm"] == "quant":
        return f"{r['bytes']/1e9:.2f} GB"
    if "params" in r:
        return f"{r['params']/1e6:.0f}M ({r['shrink']*100:.0f}% smaller)"
    return "—"


def table(rs) -> str:
    base = next(r for r in rs if r["tag"].startswith("baseline"))
    lines = ["| config | size | perplexity | legal move rate | Stockfish top-1 | value corr |",
             "|---|---|---|---|---|---|"]
    order = {"quant": 0, "svd": 1, "asvd": 2, "healed": 3}
    rs = sorted(rs, key=lambda r: (order.get(r.get("arm"), -1), -r.get("level", 0)))
    for r in rs:
        n_m = r["n_move"]
        ppl = f"{r['ppl']:.2f}"
        if r is not base:
            ppl += f" ({r['ppl']/base['ppl']:.2f}x)"
        lines.append(f"| `{r['tag']}` | {size_label(r)} | {ppl} | "
                     f"{fmt_prop(r.get('legal', 0), n_m)} | {fmt_prop(r['top1'], n_m)} | "
                     f"{r.get('value_corr', float('nan')):.3f} |")
    return "\n".join(lines)


def separated(p_a, n_a, p_b, n_b) -> bool:
    """True when two proportions' Wilson intervals do not overlap, i.e. the gap is not just noise."""
    lo_a, hi_a = wilson(round(p_a * n_a), n_a)
    lo_b, hi_b = wilson(round(p_b * n_b), n_b)
    return hi_b < lo_a or hi_a < lo_b


def divergence_note(rs) -> str:
    """Compare what perplexity says to what the capability metrics say, at matched size.

    Reported as relative retention rather than a single ratio: when perplexity barely moves, a
    "how many times larger" figure divides by something near zero and reads as drama rather than
    evidence. The thing that matters is whether a capability drop clears its own error bars.
    """
    base = next(r for r in rs if r["tag"].startswith("baseline"))
    nb = base["n_move"]
    out = []
    for r in rs:
        if r["tag"].startswith("baseline"):
            continue
        n = r["n_move"]
        ppl_ratio = r["ppl"] / base["ppl"]
        bits = []
        for key, label in (("legal", "legality"), ("top1", "top-1")):
            if key not in r:
                continue
            retained = r[key] / base[key] if base[key] else float("nan")
            flag = "significant" if separated(base[key], nb, r[key], n) else "within noise"
            bits.append(f"{label} retains {retained*100:.0f}% ({flag})")
        vc = r.get("value_corr")
        if vc is not None and base.get("value_corr"):
            bits.append(f"value corr retains {vc/base['value_corr']*100:.0f}%")
        out.append(f"- **`{r['tag']}`**: perplexity {ppl_ratio:.2f}x; " + "; ".join(bits))
    return "\n".join(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(RUNS / "results.jsonl"))
    ap.add_argument("--out", default=str(RUNS / "table.md"))
    a = ap.parse_args()
    rs = load(a.results)
    text = table(rs) + "\n\n### Divergence\n\n" + divergence_note(rs) + "\n"
    Path(a.out).write_text(text)
    print(text)
