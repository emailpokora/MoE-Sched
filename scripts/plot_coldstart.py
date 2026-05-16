#!/usr/bin/env python3
"""Generate cold-start throughput figure from bench_coldstart.py output.

Reads figures/coldstart_throughput.json and produces
figures/coldstart_throughput.pdf
"""

import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

FIGURES_DIR = Path(__file__).parent.parent / "figures"
DATA_PATH = FIGURES_DIR / "coldstart_throughput.json"
OUT_PATH = FIGURES_DIR / "coldstart_throughput.pdf"


def main():
    if not DATA_PATH.exists():
        print(f"ERROR: {DATA_PATH} not found. Run bench_coldstart.py first.")
        sys.exit(1)

    with open(DATA_PATH) as f:
        data = json.load(f)

    tps = np.array(data["per_token_tps"])
    hit_rate = np.array(data["cumulative_hit_rate"])
    window_tps = data.get("window_tps_10tok", [])
    n_tokens = len(tps)
    token_idx = np.arange(1, n_tokens + 1)

    # Smooth with rolling window for cleaner plot
    window = 5
    tps_smooth = np.convolve(tps, np.ones(window) / window, mode="valid")
    idx_smooth = token_idx[window - 1:]

    fig, ax1 = plt.subplots(figsize=(7, 3.5))

    # Throughput
    color1 = "#2563eb"
    ax1.plot(idx_smooth, tps_smooth, color=color1, linewidth=1.5, label="Throughput")
    ax1.set_xlabel("Token index", fontsize=10)
    ax1.set_ylabel("Throughput (tok/s)", color=color1, fontsize=10)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_xlim(1, n_tokens)
    ax1.set_ylim(0, max(tps_smooth) * 1.15)

    # Mark warmup region
    warmup_end = data.get("warmup_end_token", 20)
    if warmup_end > 1:
        ax1.axvspan(1, warmup_end, alpha=0.08, color="red", label="Warmup phase")

    # Hit rate on secondary axis
    ax2 = ax1.twinx()
    color2 = "#dc2626"
    ax2.plot(token_idx, hit_rate * 100, color=color2, linewidth=1.0,
             linestyle="--", alpha=0.7, label="Cache hit rate")
    ax2.set_ylabel("Cumulative hit rate (%)", color=color2, fontsize=10)
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.set_ylim(0, 100)

    # Annotations — use effective (windowed) throughput to match paper body
    if window_tps:
        first10_eff = window_tps[0]
        last10_eff = window_tps[-1]
    else:
        first10_eff = np.mean(tps[:10])
        last10_eff = np.mean(tps[-10:])
    # Place annotations near the smoothed line
    first10_y = tps_smooth[0] if len(tps_smooth) > 0 else first10_eff
    last10_y = tps_smooth[-1] if len(tps_smooth) > 0 else last10_eff
    ax1.annotate(f"First 10: {first10_eff:.2f} tok/s (eff.)",
                 xy=(5, first10_y), xytext=(35, first10_y + 0.8),
                 fontsize=8, color="gray",
                 arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))
    ax1.annotate(f"Last 10: {last10_eff:.1f} tok/s",
                 xy=(n_tokens - 5, last10_y), xytext=(n_tokens - 60, last10_y - 0.8),
                 fontsize=8, color="gray",
                 arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))

    # Legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=8)

    ax1.set_title("Cold-Start Throughput: Qwen1.5-MoE-A2.7B on RTX 5080 Laptop",
                  fontsize=10, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
    print(f"Saved: {OUT_PATH}")
    plt.close()


if __name__ == "__main__":
    main()
