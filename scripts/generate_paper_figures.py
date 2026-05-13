#!/usr/bin/env python3
"""Generate publication-quality figures from latest benchmark results.

Reads:
  - figures/qwen_multirun_results.json (multi-run throughput data)
  - figures/output_equivalence.json (correctness verification)

Generates:
  - paper/figures/policy_sweep_qwen.pdf  (3-panel: throughput, VRAM, hit rate with error bars)
  - paper/figures/vram_comparison_qwen.pdf (VRAM bar chart)

Usage:
    python scripts/generate_paper_figures.py
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


def load_multirun():
    path = os.path.join(FIGURES_DIR, "qwen_multirun_results.json")
    with open(path) as f:
        return json.load(f)


def fig_policy_sweep(data):
    """3-panel figure: throughput (with error bars), VRAM, hit rate."""
    policies = data["policies"]
    names = ["Aggressive\n(cap=2, LRU)", "Balanced\n(cap=4, LFU)", "Generous\n(cap=8, LFU)"]
    keys = list(policies.keys())

    tps_means = [policies[k]["tps_mean"] for k in keys]
    tps_stds = [policies[k]["tps_std"] for k in keys]
    gpu_means = [policies[k]["gpu_gb_mean"] for k in keys]
    hr_means = [policies[k]["hit_rate_mean"] * 100 for k in keys]

    # Also show baseline
    baseline_tps = data.get("baseline", {}).get("tps_mean", 0.5)
    baseline_std = data.get("baseline", {}).get("tps_std", 0.2)
    baseline_gpu = data.get("baseline", {}).get("gpu_gb", 12.0)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Panel 1: Throughput with error bars
    ax = axes[0]
    x = np.arange(len(names))
    bars = ax.bar(x, tps_means, yerr=tps_stds, capsize=5,
                  color="#1976d2", edgecolor="black", linewidth=0.6, width=0.6,
                  error_kw={"linewidth": 1.5, "capthick": 1.5})
    # Baseline line
    ax.axhline(y=baseline_tps, color="#d32f2f", linestyle="--", linewidth=1.5,
               label=f"Baseline: {baseline_tps:.1f} ± {baseline_std:.1f} tok/s")
    ax.axhspan(baseline_tps - baseline_std, baseline_tps + baseline_std,
               alpha=0.1, color="#d32f2f")
    for i, (m, s) in enumerate(zip(tps_means, tps_stds)):
        ax.text(i, m + s + 0.15, f"{m:.1f}", ha="center", va="bottom",
                fontweight="bold", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Tokens/s")
    ax.set_title("Throughput ($n{=}3$, mean ± std)")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_ylim(0, max(tps_means) * 1.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Panel 2: VRAM
    ax = axes[1]
    bars = ax.bar(x, gpu_means, color="#ff9800", edgecolor="black",
                  linewidth=0.6, width=0.6)
    ax.axhline(y=baseline_gpu, color="#d32f2f", linestyle="--", linewidth=1.5,
               label=f"Baseline: {baseline_gpu:.1f} GB")
    for i, v in enumerate(gpu_means):
        ax.text(i, v + 0.15, f"{v:.1f}", ha="center", va="bottom",
                fontweight="bold", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("GPU VRAM (GB)")
    ax.set_title("Memory Usage")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_ylim(0, baseline_gpu * 1.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Panel 3: Hit rate
    ax = axes[2]
    bars = ax.bar(x, hr_means, color="#388e3c", edgecolor="black",
                  linewidth=0.6, width=0.6)
    for i, v in enumerate(hr_means):
        ax.text(i, v + 0.3, f"{v:.1f}%", ha="center", va="bottom",
                fontweight="bold", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Cache Hit Rate (%)")
    ax.set_title("Cache Effectiveness")
    ax.set_ylim(0, max(hr_means) * 1.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle("Qwen1.5-MoE-A2.7B: Policy Tradeoffs (RTX 5080, Live Inference)",
                 fontsize=13, y=1.02)
    plt.tight_layout()

    for ext in ("pdf", "png"):
        plt.savefig(os.path.join(PAPER_FIGURES_DIR, f"policy_sweep_qwen.{ext}"))
    plt.close()
    print(f"  ✓ policy_sweep_qwen.pdf")


def fig_vram_comparison(data):
    """VRAM comparison: standard vs skeleton vs skeleton+cache."""
    baseline_gpu = data.get("baseline", {}).get("gpu_gb", 12.0)
    skeleton_gb = data["skeleton_gpu_gb"]
    # Use balanced policy as the representative "with cache"
    balanced_key = [k for k in data["policies"] if "balanced" in k][0]
    cache_gpu = data["policies"][balanced_key]["gpu_gb_mean"]
    vram_limit = data["vram_gb"]

    fig, ax = plt.subplots(figsize=(7, 4.5))

    labels = ["Standard\nloading", "Skeleton\nonly", "Skeleton +\nexpert cache"]
    vals = [baseline_gpu, skeleton_gb, cache_gpu]
    colors = ["#d32f2f", "#1976d2", "#388e3c"]

    bars = ax.bar(labels, vals, color=colors, width=0.5,
                  edgecolor="black", linewidth=0.8)

    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{v:.1f} GB", ha="center", va="bottom",
                fontweight="bold", fontsize=12)

    ax.axhline(y=vram_limit, color="red", linestyle="--", linewidth=1.8, alpha=0.7,
               label=f"GPU capacity ({vram_limit:.0f} GB)")

    # Annotate savings
    savings_pct = (baseline_gpu - cache_gpu) / baseline_gpu * 100
    ax.annotate(f"−{savings_pct:.0f}% VRAM",
                xy=(2, cache_gpu), xytext=(2.35, (baseline_gpu + cache_gpu) / 2),
                fontsize=11, fontweight="bold", color="#388e3c",
                arrowprops=dict(arrowstyle="->", color="#388e3c", lw=1.5),
                ha="left", va="center")

    ax.legend(fontsize=11, loc="upper right")
    ax.set_ylabel("GPU VRAM (GB)")
    ax.set_title(f"Qwen1.5-MoE-A2.7B on {data['gpu']}", fontsize=12)
    ax.set_ylim(0, max(max(vals), vram_limit) * 1.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(os.path.join(PAPER_FIGURES_DIR, f"vram_comparison_qwen.{ext}"))
    plt.close()
    print(f"  ✓ vram_comparison_qwen.pdf")


def fig_speedup_summary(data):
    """Speedup bar chart comparing all configs vs baseline."""
    policies = data["policies"]
    keys = list(policies.keys())
    names = ["Aggressive\n(cap=2)", "Balanced\n(cap=4)", "Generous\n(cap=8)"]

    baseline_tps = data.get("baseline", {}).get("tps_mean", 0.5)
    speedups = [policies[k]["tps_mean"] / baseline_tps for k in keys]

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(names))
    colors = ["#42a5f5", "#1976d2", "#0d47a1"]

    bars = ax.bar(x, speedups, color=colors, edgecolor="black",
                  linewidth=0.6, width=0.55)

    for i, (bar, s) in enumerate(zip(bars, speedups)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                f"{s:.1f}×", ha="center", va="bottom",
                fontweight="bold", fontsize=13)

    ax.axhline(y=1.0, color="#d32f2f", linestyle="--", linewidth=1.5,
               label="Baseline (1.0×)")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Speedup over unmanaged loading")
    ax.set_title("MoE-PolicyLang Speedup (Qwen1.5-MoE, RTX 5080)")
    ax.set_ylim(0, max(speedups) * 1.3)
    ax.legend(loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(os.path.join(PAPER_FIGURES_DIR, f"speedup_qwen.{ext}"))
    plt.close()
    print(f"  ✓ speedup_qwen.pdf")


def main():
    os.makedirs(PAPER_FIGURES_DIR, exist_ok=True)

    print("Loading multi-run results...")
    data = load_multirun()
    print(f"  Model: {data['model']}")
    print(f"  GPU: {data['gpu']}")
    print(f"  Runs: {data['runs']}")
    print()

    print("Generating figures:")
    fig_policy_sweep(data)
    fig_vram_comparison(data)
    fig_speedup_summary(data)

    print(f"\nAll figures saved to {PAPER_FIGURES_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
