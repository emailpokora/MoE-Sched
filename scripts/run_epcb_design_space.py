# Copyright (c) 2026 Jesse Pokora — MIT License (see LICENSE)
"""EPCB Design-Space Exploration: Shannon entropy vs alternative allocation signals.

Compares six allocation strategies for per-layer cache budgeting on DeepSeek-V2-Lite:
  1. Shannon entropy (current EPCB)
  2. Activation frequency variance
  3. Top-k mass (fraction of activations going to top-k experts)
  4. KL divergence from uniform
  5. Gini coefficient
  6. Uniform baseline (equal capacity per layer)

All strategies use the same total budget and LFU eviction, differing only in how
they distribute capacity across layers.

Usage:
    python scripts/run_epcb_design_space.py

Outputs:
    traces/epcb_design_space.json     — raw results
    paper/figures/epcb_design_space.pdf — comparison figure
"""
import json
import os
import sys
import copy
import math
from collections import defaultdict
from typing import Dict, List, Sequence, Callable

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from moe_policylang.ir import PolicyIR, CacheIR, EvictionPolicy
from moe_policylang.compiler import compile_policy
from moe_policylang.runtime.hooks import PolicyHook
from moe_policylang.runtime.per_layer import (
    PerLayerHook, PerLayerConfig, RoutingEntropyTracker,
    allocate_capacity_by_entropy,
)

TRACES_DIR = os.path.join(ROOT, "traces")
FIG_DIR = os.path.join(ROOT, "paper", "figures")
os.makedirs(FIG_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════
# Trace loading
# ═══════════════════════════════════════════════════════════════════════

def load_trace(name):
    path = os.path.join(TRACES_DIR, f"{name}.jsonl")
    lines = open(path).readlines()
    header = json.loads(lines[0])
    data = [json.loads(l) for l in lines[1:]]
    return header, data


def run_policy_on_trace(hook, data):
    """Run a hook on trace data and return stats."""
    for e in data:
        hook.on_layer(
            layer_idx=e["l"],
            selected_experts=e["e"],
            scores=e.get("s"),
        )
    return hook.stats_snapshot()


# ═══════════════════════════════════════════════════════════════════════
# Allocation signal functions
# ═══════════════════════════════════════════════════════════════════════

def compute_layer_frequencies(data, num_layers, num_experts):
    """Compute per-layer expert activation frequency distributions."""
    freq = defaultdict(lambda: defaultdict(int))
    counts = defaultdict(int)
    for e in data:
        layer = e["l"]
        for eid in e["e"]:
            freq[layer][eid] += 1
            counts[layer] += 1
    # Normalize to probabilities
    probs = {}
    for l in range(num_layers):
        total = counts[l] if counts[l] > 0 else 1
        p = np.zeros(num_experts)
        for eid, c in freq[l].items():
            p[eid] = c / total
        probs[l] = p
    return probs


def signal_shannon_entropy(probs: Dict[int, np.ndarray]) -> Dict[int, float]:
    """Shannon entropy: H = -sum(p * log2(p))."""
    result = {}
    for l, p in probs.items():
        p_pos = p[p > 0]
        result[l] = float(-np.sum(p_pos * np.log2(p_pos)))
    return result


def signal_variance(probs: Dict[int, np.ndarray]) -> Dict[int, float]:
    """Variance of activation frequencies. High variance = concentrated."""
    # Invert: we want high signal = needs more cache = high entropy = LOW variance
    # So use 1/variance (or max_var - var) to allocate more to uniform layers
    result = {}
    variances = {l: float(np.var(p)) for l, p in probs.items()}
    max_var = max(variances.values()) if variances else 1.0
    for l, v in variances.items():
        # Invert: low variance (uniform) → high allocation signal
        result[l] = max_var - v + 1e-8
    return result


def signal_topk_mass(probs: Dict[int, np.ndarray], k: int = 6) -> Dict[int, float]:
    """Top-k mass: fraction of activations going to top-k experts.

    High top-k mass = concentrated = needs less cache.
    So invert: allocation signal = 1 - top_k_mass.
    """
    result = {}
    for l, p in probs.items():
        sorted_p = np.sort(p)[::-1]
        top_mass = float(np.sum(sorted_p[:k]))
        result[l] = 1.0 - top_mass + 1e-8  # invert: diffuse gets more
    return result


def signal_kl_divergence(probs: Dict[int, np.ndarray]) -> Dict[int, float]:
    """KL divergence from uniform distribution.

    KL(p || uniform) = sum(p * log(p * N)).
    Low KL = close to uniform = high entropy = needs more cache.
    So invert: allocation signal = max_kl - kl.
    """
    result = {}
    kl_vals = {}
    for l, p in probs.items():
        n = len(p)
        uniform = 1.0 / n
        kl = 0.0
        for pi in p:
            if pi > 0:
                kl += pi * math.log2(pi / uniform)
        kl_vals[l] = kl
    max_kl = max(kl_vals.values()) if kl_vals else 1.0
    for l, kl in kl_vals.items():
        result[l] = max_kl - kl + 1e-8  # invert
    return result


def signal_gini(probs: Dict[int, np.ndarray]) -> Dict[int, float]:
    """Gini coefficient of activation frequencies.

    Gini = 0 → perfectly equal, Gini → 1 → perfectly concentrated.
    Invert: uniform layers (low Gini) get more cache.
    """
    result = {}
    gini_vals = {}
    for l, p in probs.items():
        sorted_p = np.sort(p)
        n = len(sorted_p)
        if n == 0 or np.sum(sorted_p) == 0:
            gini_vals[l] = 0.0
            continue
        cumulative = np.cumsum(sorted_p)
        gini = (n + 1 - 2 * np.sum(cumulative) / cumulative[-1]) / n
        gini_vals[l] = float(gini)
    max_gini = max(gini_vals.values()) if gini_vals else 1.0
    for l, g in gini_vals.items():
        result[l] = max_gini - g + 1e-8  # invert
    return result


def signal_uniform(probs: Dict[int, np.ndarray]) -> Dict[int, float]:
    """Uniform allocation — equal signal for all layers (baseline)."""
    return {l: 1.0 for l in probs}


SIGNAL_FUNCTIONS = {
    "Shannon entropy": signal_shannon_entropy,
    "Inv. variance": signal_variance,
    "Inv. top-k mass": signal_topk_mass,
    "Inv. KL divergence": signal_kl_divergence,
    "Inv. Gini": signal_gini,
    "Uniform": signal_uniform,
}


# ═══════════════════════════════════════════════════════════════════════
# Allocation and evaluation
# ═══════════════════════════════════════════════════════════════════════

def allocate_by_signal(
    signals: Dict[int, float],
    total_budget: int,
    min_capacity: int,
    max_capacity: int,
) -> Dict[int, int]:
    """Generic capacity allocation proportional to a per-layer signal."""
    if not signals:
        return {}
    layers = sorted(signals.keys())
    total_signal = sum(signals[l] for l in layers)
    if total_signal == 0:
        per_layer = max(min_capacity, total_budget // len(layers))
        return {l: min(per_layer, max_capacity) for l in layers}

    raw = {l: (signals[l] / total_signal) * total_budget for l in layers}
    allocated = {}
    for l in layers:
        cap = int(round(raw[l]))
        cap = max(min_capacity, min(cap, max_capacity))
        allocated[l] = cap
    return allocated


def run_with_allocation(
    base_ir: PolicyIR,
    allocation: Dict[int, int],
    num_layers: int,
    num_experts: int,
    data: list,
) -> dict:
    """Run trace with per-layer cache capacities from an allocation."""
    hooks = {}
    for l in range(num_layers):
        ir = copy.deepcopy(base_ir)
        ir.cache.capacity = allocation.get(l, base_ir.cache.capacity)
        ir.name = f"{base_ir.name}_l{l}"
        compiled = compile_policy(ir)
        hooks[l] = PolicyHook(compiled)

    total_hits = 0
    total_misses = 0
    total_evictions = 0

    for e in data:
        layer = e["l"]
        if layer not in hooks:
            continue
        hooks[layer].on_layer(
            layer_idx=layer,
            selected_experts=e["e"],
            scores=e.get("s"),
        )

    for l, hook in hooks.items():
        snap = hook.stats_snapshot()
        c = snap["cache"]
        total_hits += c["hits"]
        total_misses += c["misses"]
        total_evictions += c["evictions"]

    total = total_hits + total_misses
    return {
        "hits": total_hits,
        "misses": total_misses,
        "evictions": total_evictions,
        "hit_rate": total_hits / total if total > 0 else 0.0,
        "allocation": {str(k): v for k, v in allocation.items()},
        "total_budget_used": sum(allocation.values()),
    }


# ═══════════════════════════════════════════════════════════════════════
# Main experiment
# ═══════════════════════════════════════════════════════════════════════

def run_design_space(trace_name="deepseek_v2_lite_sample"):
    print("=" * 70)
    print("EPCB DESIGN-SPACE EXPLORATION")
    print("=" * 70)

    header, data = load_trace(trace_name)
    num_layers = header["num_layers"]
    num_experts = header["num_experts"]
    top_k = header.get("top_k", 6)
    print(f"Trace: {trace_name} ({len(data)} entries, {num_layers} layers, {num_experts} experts, top-{top_k})")

    # Compute per-layer frequency distributions
    probs = compute_layer_frequencies(data, num_layers, num_experts)

    # Base policy: LFU with decay
    base_ir = PolicyIR(
        name="epcb_test",
        cache=CacheIR(capacity=32, eviction=EvictionPolicy.LFU, lfu_decay=0.9),
    )

    # Test at multiple budget levels
    budget_multipliers = [16, 24, 32, 48]
    all_results = {}

    for budget_cap in budget_multipliers:
        total_budget = num_layers * budget_cap
        min_cap = max(2, budget_cap // 4)
        max_cap = budget_cap * 2

        print(f"\n--- Budget: {budget_cap}/layer (total={total_budget}, min={min_cap}, max={max_cap}) ---")
        print(f"{'Signal':<22} {'Hit Rate':>10} {'Hits':>10} {'Misses':>10} {'Budget Used':>12}")
        print("-" * 70)

        budget_results = {}
        for signal_name, signal_fn in SIGNAL_FUNCTIONS.items():
            signals = signal_fn(probs)
            allocation = allocate_by_signal(signals, total_budget, min_cap, max_cap)
            result = run_with_allocation(base_ir, allocation, num_layers, num_experts, data)
            budget_results[signal_name] = result
            print(f"{signal_name:<22} {result['hit_rate']:>9.1%} {result['hits']:>10,} {result['misses']:>10,} {result['total_budget_used']:>12}")

        all_results[f"cap{budget_cap}"] = budget_results

    return all_results, probs, num_layers, num_experts


def plot_design_space(all_results):
    """Generate comparison figure."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Left: hit rate comparison across budgets
    ax = axes[0]
    signal_names = list(SIGNAL_FUNCTIONS.keys())
    budgets = sorted(all_results.keys(), key=lambda x: int(x.replace("cap", "")))
    budget_labels = [x.replace("cap", "") for x in budgets]

    x = np.arange(len(budgets))
    width = 0.12
    colors = ['#2196F3', '#FF9800', '#4CAF50', '#9C27B0', '#F44336', '#607D8B']

    for i, sname in enumerate(signal_names):
        rates = [all_results[b][sname]["hit_rate"] * 100 for b in budgets]
        offset = (i - len(signal_names)/2 + 0.5) * width
        bars = ax.bar(x + offset, rates, width, label=sname,
                      color=colors[i % len(colors)], alpha=0.85, edgecolor='white', linewidth=0.5)

    ax.set_xlabel("Per-layer budget", fontsize=11)
    ax.set_ylabel("Cache hit rate (%)", fontsize=11)
    ax.set_title("EPCB: Allocation Signal Comparison", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(budget_labels)
    ax.legend(fontsize=7, ncol=2, loc='upper left')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Right: delta from uniform at cap=32
    ax2 = axes[1]
    cap32 = all_results.get("cap32", {})
    if cap32:
        uniform_rate = cap32["Uniform"]["hit_rate"] * 100
        deltas = []
        names = []
        for sname in signal_names:
            if sname == "Uniform":
                continue
            rate = cap32[sname]["hit_rate"] * 100
            deltas.append(rate - uniform_rate)
            names.append(sname)

        colors_delta = ['#2196F3', '#FF9800', '#4CAF50', '#9C27B0', '#F44336']
        bar_colors = [c if d >= 0 else '#C44E52' for d, c in zip(deltas, colors_delta)]
        bars2 = ax2.barh(range(len(names)), deltas, color=bar_colors, alpha=0.85, edgecolor='white')
        ax2.set_yticks(range(len(names)))
        ax2.set_yticklabels(names, fontsize=9)
        ax2.set_xlabel("Hit rate delta vs uniform (pp)", fontsize=11)
        ax2.set_title("Improvement over Uniform (cap=32)", fontsize=12)
        ax2.axvline(0, color='black', linewidth=0.8, linestyle='-')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)

        for bar, val in zip(bars2, deltas):
            ha = 'left' if val >= 0 else 'right'
            offset = 0.3 if val >= 0 else -0.3
            ax2.text(val + offset, bar.get_y() + bar.get_height()/2,
                     f'{val:+.1f}pp', va='center', ha=ha, fontsize=9, fontweight='bold')

    fig.tight_layout()
    fig_path = os.path.join(FIG_DIR, "epcb_design_space.pdf")
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to {fig_path}")
    plt.close(fig)


def main():
    all_results, probs, num_layers, num_experts = run_design_space()

    # Save results
    output_path = os.path.join(TRACES_DIR, "epcb_design_space.json")
    # Convert numpy types for JSON serialization
    serializable = {}
    for budget_key, budget_results in all_results.items():
        serializable[budget_key] = {}
        for signal_name, result in budget_results.items():
            serializable[budget_key][signal_name] = {
                k: v for k, v in result.items() if k != "allocation"
            }
    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Plot
    plot_design_space(all_results)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Best signal per budget level")
    print("=" * 70)
    for budget_key in sorted(all_results.keys(), key=lambda x: int(x.replace("cap", ""))):
        best_name = max(all_results[budget_key], key=lambda s: all_results[budget_key][s]["hit_rate"])
        best_rate = all_results[budget_key][best_name]["hit_rate"]
        uniform_rate = all_results[budget_key]["Uniform"]["hit_rate"]
        delta = (best_rate - uniform_rate) * 100
        print(f"  {budget_key}: {best_name} ({best_rate:.1%}, +{delta:.1f}pp vs uniform)")


if __name__ == "__main__":
    main()
