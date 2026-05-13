#!/usr/bin/env python3
"""Generate figures from output_equivalence.json and qwen_multirun_results.json.

Generates:
  - paper/figures/output_equivalence.pdf  (heatmap showing bit-identical outputs)
  - paper/figures/multirun_variance.pdf   (throughput per-run scatter + mean/std)

Usage:
    python scripts/generate_results_figures.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(ROOT, "figures")
PAPER_FIGURES_DIR = os.path.join(ROOT, "paper", "figures")

matplotlib.rcParams.update({
    "font.size": 11,
    "font.family": "serif",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})


def fig_output_equivalence():
    """Heatmap showing pass/fail for each prompt × policy combination."""
    path = os.path.join(FIGURES_DIR, "output_equivalence.json")
    with open(path) as f:
        data = json.load(f)

    configs = [k for k in data["configs"] if k != "baseline"]
    prompts = data["prompts"]
    # Shorten prompts for display
    prompt_labels = [p[:45] + "..." if len(p) > 45 else p for p in prompts]

    # Build matrix: rows=prompts, cols=configs
    matrix = []
    for i in range(len(prompts)):
        row = []
        for config in configs:
            match = data["configs"][config]["matches"][str(i)]
            row.append(1 if match else 0)
        matrix.append(row)
    matrix = np.array(matrix)

    fig, ax = plt.subplots(figsize=(7, 3.5))

    # Custom colormap: green for match
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(["#ef5350", "#4caf50"])
    im = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=0, vmax=1)

    # Labels
    ax.set_xticks(range(len(configs)))
    config_labels = ["Aggressive\n(cap=2, LRU)", "Balanced\n(cap=4, LFU)", "Generous\n(cap=8, LFU)"]
    ax.set_xticklabels(config_labels)
    ax.set_yticks(range(len(prompts)))
    ax.set_yticklabels(prompt_labels, fontsize=8)

    # Annotate cells
    for i in range(len(prompts)):
        for j in range(len(configs)):
            symbol = "PASS" if matrix[i, j] == 1 else "FAIL"
            color = "white"
            ax.text(j, i, symbol, ha="center", va="center",
                    fontsize=11, fontweight="bold", color=color)

    ax.set_title("Output Equivalence: All Policies vs. Baseline\n"
                 f"({data['equivalence']['total_pass']}/{data['equivalence']['total_tests']} bit-identical)")
    ax.set_xlabel("Policy Configuration")
    ax.set_ylabel("Test Prompt")

    plt.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(os.path.join(PAPER_FIGURES_DIR, f"output_equivalence.{ext}"))
    plt.close()
    print(f"  ✓ output_equivalence.pdf")


def fig_multirun_variance():
    """Scatter + mean/std showing individual run measurements."""
    path = os.path.join(FIGURES_DIR, "qwen_multirun_results.json")
    with open(path) as f:
        data = json.load(f)

    policies = data["policies"]
    keys = list(policies.keys())
    names = ["Aggressive\n(cap=2)", "Balanced\n(cap=4)", "Generous\n(cap=8)"]

    fig, ax = plt.subplots(figsize=(7, 4.5))

    x = np.arange(len(names))
    width = 0.5

    # Plot individual runs as scatter
    for i, k in enumerate(keys):
        vals = policies[k]["tps_values"]
        jitter = np.random.default_rng(42).uniform(-0.08, 0.08, len(vals))
        ax.scatter([i] * len(vals) + jitter, vals,
                   color="#1976d2", alpha=0.6, s=60, zorder=3,
                   edgecolor="black", linewidth=0.5)

    # Plot means with error bars
    means = [policies[k]["tps_mean"] for k in keys]
    stds = [policies[k]["tps_std"] for k in keys]
    ax.errorbar(x, means, yerr=stds, fmt="D", color="#d32f2f",
                markersize=10, capsize=8, capthick=2, linewidth=0,
                elinewidth=2, zorder=4, label="Mean ± std")

    # Baseline
    baseline = data.get("baseline", {})
    if "tps_mean" in baseline:
        bm = baseline["tps_mean"]
        bs = baseline["tps_std"]
        ax.axhline(y=bm, color="#757575", linestyle="--", linewidth=1.5)
        ax.axhspan(bm - bs, bm + bs, alpha=0.1, color="#757575")
        ax.text(len(names) - 0.5, bm + 0.05,
                f"Baseline: {bm:.2f} ± {bs:.2f} tok/s",
                fontsize=9, color="#757575", va="bottom")

    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Throughput (tok/s)")
    ax.set_title(f"Per-Run Throughput ($n$={data['runs']}, {data['warmup']} warmup)")
    ax.legend(loc="lower right")
    ax.set_ylim(0, max(means) * 1.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(os.path.join(PAPER_FIGURES_DIR, f"multirun_variance.{ext}"))
    plt.close()
    print(f"  ✓ multirun_variance.pdf")


def main():
    os.makedirs(PAPER_FIGURES_DIR, exist_ok=True)
    print("Generating figures from JSON results:")
    fig_output_equivalence()
    fig_multirun_variance()
    print(f"\nSaved to {PAPER_FIGURES_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
