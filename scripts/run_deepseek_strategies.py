"""Evaluate DeepSeek-specific caching strategies.

Tests:
1. Baseline policies at various capacities
2. Top-frequency expert pinning
3. Per-layer entropy-adaptive caching
4. Combined: pinning + large capacity + LFU

Outputs results and figures to paper/figures/.
"""
import json
import os
import sys
import copy
import math
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from moe_sched.ir import (
    PolicyIR, CacheIR, PrefetchIR, ScheduleIR,
    EvictionPolicy, PrefetchStrategy, ScheduleMode,
)
from moe_sched.compiler import compile_policy
from moe_sched.runtime.hooks import PolicyHook, build_hook
from moe_sched.runtime.per_layer import (
    PerLayerHook, PerLayerConfig, RoutingEntropyTracker,
)

TRACES_DIR = os.path.join(ROOT, "traces")
FIG_DIR = os.path.join(ROOT, "paper", "figures")
os.makedirs(FIG_DIR, exist_ok=True)


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


def get_top_experts(data, n):
    """Find the N most frequently activated experts globally."""
    freq = defaultdict(int)
    for e in data:
        for eid in e["e"]:
            freq[eid] += 1
    return [eid for eid, _ in sorted(freq.items(), key=lambda x: -x[1])[:n]]


# ═══════════════════════════════════════════════════════════════════════
# Strategy 1: Baseline policies at various capacities
# ═══════════════════════════════════════════════════════════════════════
def test_baseline_capacities(header, data):
    print("\n" + "=" * 70)
    print("Strategy 1: BASELINE POLICIES × CAPACITY SWEEP")
    print("=" * 70)

    capacities = [4, 8, 16, 24, 32, 48]
    evictions = [
        ("LRU", EvictionPolicy.LRU),
        ("LFU", EvictionPolicy.LFU),
    ]

    results = {}
    print(f"\n{'Cap':>5} {'Eviction':<8} {'Hit Rate':>10} {'Hits':>10} {'Misses':>10} {'Evictions':>10}")
    print("-" * 60)

    for cap in capacities:
        for ev_name, ev_policy in evictions:
            ir = PolicyIR(
                name=f"baseline_{ev_name.lower()}_c{cap}",
                cache=CacheIR(capacity=cap, eviction=ev_policy, lfu_decay=0.9),
            )
            compiled = compile_policy(ir)
            hook = PolicyHook(compiled)
            stats = run_policy_on_trace(hook, data)
            c = stats["cache"]
            key = f"{ev_name}_c{cap}"
            results[key] = c
            print(f"{cap:>5} {ev_name:<8} {c['hit_rate']:>9.1%} {c['hits']:>10,} {c['misses']:>10,} {c['evictions']:>10,}")

    return results


# ═══════════════════════════════════════════════════════════════════════
# Strategy 2: Pin top-frequency experts
# ═══════════════════════════════════════════════════════════════════════
def test_pinning(header, data):
    print("\n" + "=" * 70)
    print("Strategy 2: TOP-FREQUENCY EXPERT PINNING")
    print("=" * 70)

    pin_counts = [0, 4, 8, 12, 16]
    capacities = [16, 24, 32]

    results = {}
    print(f"\n{'Cap':>5} {'Pinned':>8} {'Eviction':<8} {'Hit Rate':>10} {'Hits':>10} {'Misses':>10}")
    print("-" * 55)

    for cap in capacities:
        for n_pin in pin_counts:
            if n_pin >= cap:
                continue
            top = get_top_experts(data, n_pin)
            ir = PolicyIR(
                name=f"pin{n_pin}_lfu_c{cap}",
                cache=CacheIR(
                    capacity=cap,
                    eviction=EvictionPolicy.LFU,
                    lfu_decay=0.9,
                    pin_experts=top,
                ),
            )
            compiled = compile_policy(ir)
            hook = PolicyHook(compiled)
            stats = run_policy_on_trace(hook, data)
            c = stats["cache"]
            key = f"pin{n_pin}_c{cap}"
            results[key] = c
            print(f"{cap:>5} {n_pin:>8} {'LFU':<8} {c['hit_rate']:>9.1%} {c['hits']:>10,} {c['misses']:>10,}")

    return results


# ═══════════════════════════════════════════════════════════════════════
# Strategy 3: Per-layer entropy-adaptive caching
# ═══════════════════════════════════════════════════════════════════════
def test_per_layer(header, data):
    print("\n" + "=" * 70)
    print("Strategy 3: PER-LAYER ENTROPY-ADAPTIVE CACHING")
    print("=" * 70)

    num_layers = header["num_layers"]
    num_experts = header["num_experts"]

    configs = [
        ("uniform_c8", 8, False),
        ("uniform_c16", 16, False),
        ("uniform_c32", 32, False),
        ("entropy_c8", 8, True),
        ("entropy_c16", 16, True),
        ("entropy_c32", 32, True),
    ]

    results = {}
    print(f"\n{'Config':<20} {'Hit Rate':>10} {'Hits':>10} {'Misses':>10} {'Budget':>8}")
    print("-" * 60)

    for name, base_cap, use_entropy in configs:
        ir = PolicyIR(
            name=name,
            cache=CacheIR(capacity=base_cap, eviction=EvictionPolicy.LFU, lfu_decay=0.9),
        )

        if use_entropy:
            config = PerLayerConfig(
                entropy_window=200,
                min_capacity=max(2, base_cap // 4),
                max_capacity=base_cap * 2,
                rebalance_interval=500,
                total_budget=num_layers * base_cap,
            )
            hook = PerLayerHook(ir, num_layers, num_experts, config)
        else:
            compiled = compile_policy(ir)
            hook = PolicyHook(compiled)

        stats = run_policy_on_trace(hook, data)
        c = stats["cache"]
        budget = num_layers * base_cap
        results[name] = {**c, "budget": budget}
        print(f"{name:<20} {c['hit_rate']:>9.1%} {c['hits']:>10,} {c['misses']:>10,} {budget:>8}")

        # Show capacity allocation for entropy-based
        if use_entropy and "capacities" in stats:
            caps = stats["capacities"]
            ents = stats.get("entropies", {})
            print(f"  Capacity range: {min(caps.values())}–{max(caps.values())}")
            if ents:
                print(f"  Entropy range:  {min(ents.values()):.2f}–{max(ents.values()):.2f}")

    return results


# ═══════════════════════════════════════════════════════════════════════
# Strategy 4: Best combined strategy
# ═══════════════════════════════════════════════════════════════════════
def test_combined(header, data):
    print("\n" + "=" * 70)
    print("Strategy 4: COMBINED BEST STRATEGIES")
    print("=" * 70)

    num_layers = header["num_layers"]
    num_experts = header["num_experts"]
    top_8 = get_top_experts(data, 8)
    top_12 = get_top_experts(data, 12)

    strategies = [
        # (name, capacity, eviction, pin, prefetch, per_layer)
        ("LFU c32", 32, EvictionPolicy.LFU, [], False, False),
        ("LFU c32 + pin8", 32, EvictionPolicy.LFU, top_8, False, False),
        ("LFU c32 + pin8 + hist", 32, EvictionPolicy.LFU, top_8, True, False),
        ("LFU c32 + entropy", 32, EvictionPolicy.LFU, [], False, True),
        ("LFU c32 + pin8 + entropy", 32, EvictionPolicy.LFU, top_8, False, True),
        ("LFU c48 + pin12", 48, EvictionPolicy.LFU, top_12, False, False),
        ("LFU c48 + pin12 + hist", 48, EvictionPolicy.LFU, top_12, True, False),
    ]

    results = {}
    print(f"\n{'Strategy':<30} {'Hit Rate':>10} {'Hits':>10} {'Misses':>10}")
    print("-" * 65)

    for name, cap, eviction, pin, prefetch, per_layer in strategies:
        prefetch_ir = PrefetchIR(
            strategy=PrefetchStrategy.HISTORY if prefetch else PrefetchStrategy.NONE,
            budget=4,
            history_window=50,
        )
        ir = PolicyIR(
            name=name.replace(" ", "_"),
            cache=CacheIR(
                capacity=cap,
                eviction=eviction,
                lfu_decay=0.9,
                pin_experts=list(pin),
            ),
            prefetch=prefetch_ir,
        )

        if per_layer:
            config = PerLayerConfig(
                entropy_window=200,
                min_capacity=max(2, cap // 4),
                max_capacity=cap * 2,
                rebalance_interval=500,
                total_budget=num_layers * cap,
            )
            hook = PerLayerHook(ir, num_layers, num_experts, config)
        else:
            compiled = compile_policy(ir)
            hook = PolicyHook(compiled)

        stats = run_policy_on_trace(hook, data)
        c = stats["cache"]
        results[name] = c
        print(f"{name:<30} {c['hit_rate']:>9.1%} {c['hits']:>10,} {c['misses']:>10,}")

    return results


# ═══════════════════════════════════════════════════════════════════════
# Figures
# ═══════════════════════════════════════════════════════════════════════
def fig_strategy_comparison(combined_results):
    """Bar chart comparing all combined strategies."""
    names = list(combined_results.keys())
    hrs = [combined_results[n]["hit_rate"] * 100 for n in names]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    bars = ax.barh(range(len(names)), hrs, color="#4C72B0", alpha=0.85, edgecolor="white")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Hit Rate (%)", fontsize=10)
    ax.set_title("DeepSeek-V2-Lite: Caching Strategy Comparison", fontsize=12)

    for bar, hr in zip(bars, hrs):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{hr:.1f}%", va="center", fontsize=9)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "deepseek_strategies.pdf")
    fig.savefig(path, dpi=300)
    print(f"\nSaved {path}")
    plt.close()


def fig_entropy_allocation(header, data):
    """Show per-layer entropy and resulting capacity allocation."""
    num_layers = header["num_layers"]
    num_experts = header["num_experts"]

    ir = PolicyIR(
        name="entropy_viz",
        cache=CacheIR(capacity=16, eviction=EvictionPolicy.LFU, lfu_decay=0.9),
    )
    config = PerLayerConfig(
        entropy_window=200,
        min_capacity=4,
        max_capacity=48,
        rebalance_interval=200,
        total_budget=num_layers * 16,
    )
    hook = PerLayerHook(ir, num_layers, num_experts, config)
    run_policy_on_trace(hook, data)

    stats = hook.stats_snapshot()
    entropies = stats.get("entropies", {})
    capacities = stats.get("capacities", {})

    layers = sorted(entropies.keys())
    ents = [entropies[l] for l in layers]
    caps = [capacities.get(l, 0) for l in layers]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)

    ax1.bar(layers, ents, color="#4C72B0", alpha=0.85, edgecolor="white")
    ax1.set_ylabel("Shannon Entropy", fontsize=10)
    ax1.set_title("DeepSeek-V2-Lite: Per-Layer Routing Entropy", fontsize=11)
    ax1.axhline(y=math.log2(64), color="red", linestyle="--", linewidth=1,
                label=f"max (log₂64 = {math.log2(64):.2f})")
    ax1.legend(fontsize=8)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    ax2.bar(layers, caps, color="#C44E52", alpha=0.85, edgecolor="white")
    ax2.set_xlabel("Layer Index", fontsize=10)
    ax2.set_ylabel("Allocated Capacity", fontsize=10)
    ax2.set_title("Entropy-Proportional Cache Allocation", fontsize=11)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.tight_layout()
    path = os.path.join(FIG_DIR, "deepseek_entropy_allocation.pdf")
    fig.savefig(path, dpi=300)
    print(f"Saved {path}")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Loading DeepSeek trace...")
    header, data = load_trace("deepseek_v2_lite_sample")
    print(f"  {len(data)} entries, {header['num_layers']} layers, "
          f"{header['num_experts']} experts, top-{header['top_k']}")

    baseline_results = test_baseline_capacities(header, data)
    pinning_results = test_pinning(header, data)
    per_layer_results = test_per_layer(header, data)
    combined_results = test_combined(header, data)

    # Figures
    print("\n" + "=" * 70)
    print("GENERATING FIGURES")
    print("=" * 70)
    fig_strategy_comparison(combined_results)
    fig_entropy_allocation(header, data)

    # Save all results
    all_results = {
        "baseline": {k: {kk: vv for kk, vv in v.items() if not isinstance(vv, float) or not math.isnan(vv)}
                     for k, v in baseline_results.items()},
        "pinning": pinning_results,
        "per_layer": per_layer_results,
        "combined": combined_results,
    }
    out_path = os.path.join(TRACES_DIR, "deepseek_strategy_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")
