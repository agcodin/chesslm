"""Turn the sweep's JSON lines into the figures the writeup argues from.

Each figure is rendered twice, light and dark, so the project page can serve whichever the reader's
system asks for. Colours match docs/template.html.
"""
import json, math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from report import wilson

OUT = Path("docs/figures")
RUNS = Path("runs/compress")

THEMES = {
    "light": dict(bg="#f8f9f6", ink="#18211d", soft="#4a5751", rule="#cfd6cf"),
    "dark": dict(bg="#18201c", ink="#e3eae5", soft="#a3b1a9", rule="#2d3833"),
}
ARM_COLOR = {"quant": "#0e5c55", "svd": "#b42318", "asvd": "#a16207", "healed": "#2f7a4a"}
ARM_LABEL = {"quant": "quantisation", "svd": "plain SVD", "asvd": "activation-aware SVD",
             "healed": "activation-aware SVD + healing"}
METRICS = [("ppl", "perplexity (lower better)", True),
           ("legal", "unconstrained legal-move rate", False),
           ("top1", "top-1 agreement with Stockfish", False),
           ("value_corr", "value correlation with Stockfish", False)]


def rows(path=RUNS / "results.jsonl"):
    out = []
    for line in open(path):
        r = json.loads(line)
        if "error" in r:
            continue
        r.setdefault("arm", r["tag"].split("_")[0])
        out.append(r)
    return out


def baseline(rs):
    for r in rs:
        if r["tag"].startswith("baseline"):
            return r
    raise SystemExit("no baseline row in results.jsonl")


def shrink_of(r, base):
    """Fraction of the dense model removed, comparable across arms.

    Quantisation does not change the parameter count, so its shrink is measured in bytes against
    the fp16 model; factorisation is measured in parameters. Both answer "how much smaller".
    """
    if r["arm"] == "quant":
        return 1 - r["bytes"] / (base["n_params_bytes"] if "n_params_bytes" in base else 1543714304 * 2)
    return r.get("shrink", 0.0)


def style(ax, th):
    ax.set_facecolor(th["bg"])
    for s in ax.spines.values():
        s.set_color(th["rule"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors=th["soft"], labelsize=9)
    ax.grid(True, color=th["rule"], lw=0.6, alpha=0.6)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(th["soft"])
    ax.yaxis.label.set_color(th["soft"])
    ax.title.set_color(th["ink"])


def save(fig, name, th_name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}_{th_name}.png", dpi=170,
                facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def fig_decay(rs, base, th_name, th):
    """Four panels: how each metric responds as the model gets smaller."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), facecolor=th["bg"])
    for ax, (key, label, log) in zip(axes.ravel(), METRICS):
        style(ax, th)
        for arm in ("quant", "svd", "asvd", "healed"):
            pts = sorted([(shrink_of(r, base), r[key], r.get("n_move", 0)) for r in rs
                          if r["arm"] == arm and key in r])
            if not pts:
                continue
            xs = [p[0] * 100 for p in pts]
            ys = [p[1] for p in pts]
            # legal and top1 are proportions over a finite test set: draw the sampling error
            # rather than letting a noisy line read as a trend.
            err = None
            if key in ("legal", "top1"):
                err = [[y - wilson(round(y * n), n)[0] for _, y, n in pts],
                       [wilson(round(y * n), n)[1] - y for _, y, n in pts]]
            ax.errorbar(xs, ys, yerr=err, fmt="o-", ms=4, lw=1.6, elinewidth=1.0, capsize=2.5,
                        color=ARM_COLOR[arm], label=ARM_LABEL[arm], alpha=0.95)
        ax.axhline(base[key], ls="--", lw=1.2, color=th["soft"], alpha=0.8)
        if log:
            ax.set_yscale("log")
        ax.set_xlabel("model shrunk by (%)")
        ax.set_title(label, fontsize=10.5, loc="left")
    h, l = axes[0][0].get_legend_handles_labels()
    h = h + [Line2D([], [], ls="--", lw=1.2, color=th["soft"])]
    leg = fig.legend(h, l + ["uncompressed"], loc="lower center", ncol=4, frameon=False,
                     bbox_to_anchor=(0.5, -0.04), fontsize=9)
    for t in leg.get_texts():
        t.set_color(th["soft"])
    fig.suptitle("The same compression, judged four ways", color=th["ink"], fontsize=13, x=0.09,
                 ha="left")
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    save(fig, "decay", th_name)


def fig_divergence(rs, base, th_name, th):
    """The argument in one panel: retention on each axis, for quantisation only.

    Perplexity is the metric compression work usually reports. If it retains far more than skill
    does at the same setting, then it is hiding the cost.
    """
    # Quantisation is the control: its metrics move together, so it shows what agreement looks
    # like. The factorised models are the effect. Quantisation rows come from the 800-position
    # confirmation run where available, since that is what the claim about it rests on.
    conf = {r["tag"]: r for r in rows(RUNS / "confirm.jsonl")} if (RUNS / "confirm.jsonl").exists() else {}
    by_tag = {r["tag"]: r for r in rs}
    picks = ["quant_4bit", "quant_3bit", "asvd_keep0.9", "asvd_keep0.8", "asvd_keep0.7"]
    q = [conf.get(t, by_tag.get(t)) for t in picks]
    q = [r for r in q if r]
    fig, ax = plt.subplots(figsize=(9, 4.4), facecolor=th["bg"])
    style(ax, th)
    keys = [("ppl", "perplexity"), ("legal", "legal-move rate"),
            ("top1", "Stockfish top-1"), ("value_corr", "value correlation")]
    width = 0.2
    for j, (key, label) in enumerate(keys):
        vals = []
        for r in q:
            # retention: 1.0 means the metric is unchanged from the dense model
            vals.append(base[key] / r[key] if key == "ppl" else r[key] / base[key])
        xs = [i + (j - 1.5) * width for i in range(len(q))]
        ax.bar(xs, [v * 100 for v in vals], width, label=label,
               color=list(ARM_COLOR.values())[j], alpha=0.9)
    ax.axhline(100, ls="--", lw=1.2, color=th["soft"])
    ax.set_xticks(range(len(q)))
    labels = []
    for r in q:
        labels.append(f"{r['level']}-bit" if r["arm"] == "quant"
                      else f"low-rank\n{r['shrink']*100:.0f}% smaller")
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("% of uncompressed retained")
    ax.axvline(1.5, color=th["rule"], lw=1.2)
    ax.text(0.5, 112, "quantisation:\nall four move together", ha="center", fontsize=9,
            color=th["soft"])
    ax.text(3, 112, "rank truncation:\nvalue correlation holds while legality collapses",
            ha="center", fontsize=9, color=th["soft"])
    ax.set_ylim(0, 124)
    ax.set_title("Four metrics on the same compressed model, disagreeing", fontsize=12,
                 loc="left", color=th["ink"])
    leg = ax.legend(frameon=False, fontsize=9, ncol=4, loc="upper center",
                    bbox_to_anchor=(0.5, -0.13))
    for t in leg.get_texts():
        t.set_color(th["soft"])
    fig.tight_layout()
    save(fig, "divergence", th_name)


def fig_sensitivity(th_name, th):
    paths = sorted(RUNS.glob("sensitivity_*.jsonl"))
    if not paths:
        return
    fig, ax = plt.subplots(figsize=(9, 3.6), facecolor=th["bg"])
    style(ax, th)
    for p in paths:
        rs = [json.loads(l) for l in open(p)]
        rs = [r for r in rs if r.get("layer") != "dense"]
        rs.sort(key=lambda r: r["layer"])
        arm = rs[0]["arm"] if rs else "asvd"
        ax.bar([r["layer"] for r in rs], [r["ratio"] for r in rs],
               color=ARM_COLOR.get(arm, "#0e5c55"), alpha=0.9)
    ax.axhline(1.0, ls="--", lw=1.2, color=th["soft"])
    ax.set_xlabel("transformer layer")
    ax.set_ylabel("perplexity multiplier")
    ax.set_title("Compressing one layer alone: the cost is not evenly spread", fontsize=12,
                 loc="left", color=th["ink"])
    fig.tight_layout()
    save(fig, "sensitivity", th_name)


def fig_healing(rs, base, th_name, th):
    """Before and after healing at matched size."""
    pairs = []
    for r in rs:
        if r["arm"] != "healed":
            continue
        twin = next((x for x in rs if x["arm"] == "asvd"
                     and abs(x.get("shrink", -1) - r.get("shrink", -2)) < 1e-6), None)
        if twin:
            pairs.append((r, twin))
    if not pairs:
        return
    fig, axes = plt.subplots(1, len(METRICS), figsize=(11, 3.4), facecolor=th["bg"])
    for ax, (key, label, log) in zip(axes, METRICS):
        style(ax, th)
        for i, (healed, raw) in enumerate(pairs):
            ax.plot([0, 1], [raw[key], healed[key]], "o-", lw=1.8, ms=5,
                    color=ARM_COLOR["healed"],
                    label=f"{raw['shrink']*100:.0f}% smaller" if ax is axes[0] else None)
        ax.axhline(base[key], ls="--", lw=1.2, color=th["soft"])
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["factorised", "healed"], fontsize=9)
        if log:
            ax.set_yscale("log")
        ax.set_title(label, fontsize=9.5, loc="left")
    fig.suptitle("A short finetune recovers most of what truncation destroyed", color=th["ink"],
                 fontsize=12.5, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    save(fig, "healing", th_name)


if __name__ == "__main__":
    rs = rows()
    base = baseline(rs)
    for th_name, th in THEMES.items():
        fig_decay(rs, base, th_name, th)
        fig_divergence(rs, base, th_name, th)
        fig_sensitivity(th_name, th)
        fig_healing(rs, base, th_name, th)
    print(f"wrote figures for {len(rs)} configurations into {OUT}")
