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


def load(path):
    return [json.loads(l) for l in open(path) if "error" not in json.loads(l)]


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


def divergence_note(rs) -> str:
    """State the headline comparison and whether it survives its own error bars."""
    base = next(r for r in rs if r["tag"].startswith("baseline"))
    out = []
    for r in rs:
        if r["arm"] != "quant":
            continue
        n = r["min_n"] if "min_n" in r else r["n_move"]
        ppl_cost = r["ppl"] / base["ppl"] - 1
        top1_cost = 1 - r["top1"] / base["top1"]
        lo_b, hi_b = wilson(round(base["top1"] * base["n_move"]), base["n_move"])
        lo_r, hi_r = wilson(round(r["top1"] * n), n)
        sep = "separated" if hi_r < lo_b else "overlapping"
        ratio = (top1_cost / ppl_cost) if ppl_cost > 1e-9 else float("inf")
        out.append(f"- **{r['level']}-bit**: perplexity worsens {ppl_cost*100:+.1f}%, "
                   f"top-1 skill falls {top1_cost*100:.1f}% relative "
                   f"({ratio:.0f}x larger) — intervals {sep} (n={n})")
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
