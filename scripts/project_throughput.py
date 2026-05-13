#!/usr/bin/env python3
# Copyright (c) 2026 Jesse Pokora — MIT License (see LICENSE)
"""Project throughput from trace replay results + calibrated transfer cost.

Takes eval results from run_eval.py and applies a calibrated PCIe transfer
penalty on every cache miss to project what real offloading throughput
would be under each caching policy.

This is the key bridge between MoE-Sched's policy evaluation (trace replay)
and real-world offloading impact — no live offloading needed.

Usage:
    python scripts/project_throughput.py
    python scripts/project_throughput.py --trace traces/olmoe_1b_7b_trace.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

TRACES_DIR = os.path.join(ROOT, "traces")
FIG_DIR = os.path.join(ROOT, "paper", "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def load_trace_header(trace_path):
    with open(trace_path) as f:
        return json.loads(f.readline())


def load_eval_results(eval_path):
    with open(eval_path) as f:
        return json.load(f)


def project(eval_results, baseline_tok_per_sec, transfer_us, top_k):
    """Project throughput for each policy.

    Model: each generated token passes through all layers.
    Per layer, top_k experts are selected. Each miss incurs a synchronous
    CPU→GPU transfer of `transfer_us` microseconds.

    throughput = total_tokens / (base_time + total_transfer_overhead)

    where:
        base_time = total_tokens / baseline_tok_per_sec
        total_transfer_overhead = total_misses * transfer_us
    """
    header = eval_results["header"]
    num_layers = header["num_layers"]
    num_entries = header.get("num_entries", header.get("total_entries", 0))

    # Total tokens ≈ entries / num_layers (each token generates num_layers entries)
    total_tokens = num_entries / num_layers
    base_time_s = total_tokens / baseline_tok_per_sec

    projections = {}
    for name, stats in eval_results["policies"].items():
        misses = stats["misses"]
        hits = stats["hits"]
        hit_rate = stats["hit_rate"]

        # Transfer overhead for all misses (synchronous, sequential)
        overhead_s = misses * transfer_us / 1e6
        projected_time = base_time_s + overhead_s
        projected_tps = total_tokens / projected_time if projected_time > 0 else 0

        projections[name] = {
            "hits": hits,
            "misses": misses,
            "hit_rate": hit_rate,
            "transfer_overhead_s": round(overhead_s, 4),
            "projected_time_s": round(projected_time, 4),
            "projected_tok_per_sec": round(projected_tps, 2),
            "slowdown_vs_baseline": round(projected_tps / baseline_tok_per_sec, 4),
        }

    # Also project the autotuner best
    if "autotuner" in eval_results:
        best = eval_results["autotuner"]["best"]
        best_hr = best["hit_rate"]
        total_accesses = num_entries  # each entry = one expert access
        # Approximate: hit_rate applies to all top_k selections
        best_misses = int(total_accesses * (1 - best_hr))
        overhead_s = best_misses * transfer_us / 1e6
        projected_time = base_time_s + overhead_s
        projected_tps = total_tokens / projected_time if projected_time > 0 else 0

        projections["autotuner_best"] = {
            "config": best["params"],
            "hit_rate": best_hr,
            "approx_misses": best_misses,
            "transfer_overhead_s": round(overhead_s, 4),
            "projected_time_s": round(projected_time, 4),
            "projected_tok_per_sec": round(projected_tps, 2),
            "slowdown_vs_baseline": round(projected_tps / baseline_tok_per_sec, 4),
        }

    return projections, total_tokens, base_time_s


def print_results(projections, total_tokens, base_time_s, baseline_tps, transfer_us):
    print(f"\n{'='*70}")
    print("PROJECTED THROUGHPUT (trace replay + calibrated transfer cost)")
    print(f"{'='*70}")
    print(f"  Baseline (all-GPU): {baseline_tps:.1f} tok/s")
    print(f"  Transfer cost: {transfer_us:.0f} µs/expert ({transfer_us/1000:.1f} ms)")
    print(f"  Total tokens: {total_tokens:.0f}")
    print(f"  Base time: {base_time_s:.2f}s")
    print()

    hdr = f"{'Policy':<25} {'Hit Rate':>9} {'Misses':>8} {'Overhead':>10} {'tok/s':>8} {'vs base':>8}"
    print(hdr)
    print("-" * len(hdr))
    for name, p in projections.items():
        misses = p.get("misses", p.get("approx_misses", 0))
        print(f"{name:<25} {p['hit_rate']:>8.1%} {misses:>8} "
              f"{p['transfer_overhead_s']:>9.2f}s {p['projected_tok_per_sec']:>8.1f} "
              f"{p['slowdown_vs_baseline']:>7.2f}x")


def plot_projections(projections, baseline_tps, model_name, output_path):
    """Bar chart of projected throughput per policy."""
    # Sort by throughput descending
    sorted_items = sorted(projections.items(),
                          key=lambda x: x[1]["projected_tok_per_sec"], reverse=True)
    names = [n for n, _ in sorted_items]
    tps = [p["projected_tok_per_sec"] for _, p in sorted_items]
    hrs = [p["hit_rate"] for _, p in sorted_items]

    fig, ax1 = plt.subplots(figsize=(5, 3))

    colors = []
    for n in names:
        if "autotuner" in n:
            colors.append("#E57373")
        elif "lfu" in n:
            colors.append("#4C72B0")
        elif "lru" in n:
            colors.append("#55A868")
        elif "score" in n:
            colors.append("#C4A000")
        else:
            colors.append("#8C8C8C")

    bars = ax1.bar(range(len(names)), tps, color=colors, edgecolor="white", alpha=0.9)
    ax1.axhline(y=baseline_tps, color="red", linestyle="--", linewidth=0.8, alpha=0.7,
                label=f"All-GPU baseline ({baseline_tps:.0f} tok/s)")

    for i, bar in enumerate(bars):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f"{tps[i]:.1f}\n({hrs[i]:.0%})",
                 ha="center", va="bottom", fontsize=6)

    short_names = [n.replace("_", "\n") for n in names]
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels(short_names, fontsize=6, rotation=0)
    ax1.set_ylabel("Projected tok/s", fontsize=8)
    ax1.set_title(f"Projected offloading throughput — {model_name}", fontsize=8)
    ax1.tick_params(labelsize=7)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.legend(fontsize=6, loc="upper right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    print(f"\nSaved figure: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", default=os.path.join(TRACES_DIR, "olmoe_1b_7b_trace.jsonl"))
    parser.add_argument("--baseline-tps", type=float, default=None,
                        help="Baseline tok/s (all-GPU). Auto-detected from DSL demo results.")
    parser.add_argument("--transfer-us", type=float, default=None,
                        help="Transfer time per expert in µs. Auto-detected from DSL demo.")
    args = parser.parse_args()

    # ── Auto-detect calibration from DSL demo results ──
    dsl_demo = os.path.join(TRACES_DIR, "dsl_demo_results.json")
    if os.path.exists(dsl_demo):
        demo = json.load(open(dsl_demo))
        if args.baseline_tps is None:
            args.baseline_tps = demo["baseline_tps"]
        if args.transfer_us is None:
            args.transfer_us = demo["transfer_us"]
        print(f"Calibration from DSL demo: {args.baseline_tps:.1f} tok/s baseline, "
              f"{args.transfer_us:.0f} µs/transfer")

    # Fallback defaults
    if args.baseline_tps is None:
        args.baseline_tps = 38.0
    if args.transfer_us is None:
        args.transfer_us = 596.0

    # ── Load eval results ──
    trace_name = os.path.splitext(os.path.basename(args.trace))[0]
    eval_path = os.path.join(TRACES_DIR, f"eval_results_{trace_name}.json")
    if not os.path.exists(eval_path):
        print(f"Eval results not found: {eval_path}")
        print(f"Run: python scripts/run_eval.py {args.trace}")
        sys.exit(1)

    eval_results = load_eval_results(eval_path)
    header = load_trace_header(args.trace)
    top_k = header.get("top_k", 8)

    # ── Project ──
    projections, total_tokens, base_time_s = project(
        eval_results, args.baseline_tps, args.transfer_us, top_k
    )

    print_results(projections, total_tokens, base_time_s, args.baseline_tps, args.transfer_us)

    # ── Save ──
    out_path = os.path.join(TRACES_DIR, f"throughput_projection_{trace_name}.json")
    with open(out_path, "w") as f:
        json.dump({
            "trace": args.trace,
            "baseline_tok_per_sec": args.baseline_tps,
            "transfer_us": args.transfer_us,
            "total_tokens": total_tokens,
            "base_time_s": round(base_time_s, 4),
            "model": header.get("model_name", "unknown"),
            "projections": projections,
        }, f, indent=2)
    print(f"Saved: {out_path}")

    # ── Plot ──
    fig_path = os.path.join(FIG_DIR, f"throughput_projection_{trace_name}.pdf")
    plot_projections(projections, args.baseline_tps,
                     header.get("model_name", trace_name), fig_path)


if __name__ == "__main__":
    main()
