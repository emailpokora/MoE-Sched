"""EPCB sensitivity ablation: c_min, c_max, and rebalance_interval.

Sweeps the three key EPCB hyperparameters to show:
1. How sensitive hit rate is to c_min/c_max bounds
2. How rebalancing frequency affects performance
3. Whether the reported gains are robust or fragile

Uses the same bursty trace as run_static_vs_dynamic_epcb.py for consistency.
Results → figures/epcb_sensitivity.json + console table.
"""

from __future__ import annotations

import copy
import json
import math
import random
from collections import defaultdict
from itertools import product
from pathlib import Path
from typing import Dict, List

from moe_policylang.compiler import compile_policy
from moe_policylang.ir import CacheIR, EvictionPolicy, PolicyIR
from moe_policylang.runtime.hooks import PolicyHook
from moe_policylang.runtime.per_layer import (
    PerLayerConfig,
    PerLayerHook,
    RoutingEntropyTracker,
    allocate_capacity_by_entropy,
)

# ---------------------------------------------------------------------------
# Same workload parameters as the main experiment
# ---------------------------------------------------------------------------
NUM_LAYERS = 27
NUM_EXPERTS = 64
TOP_K = 6
NUM_TOKENS = 2000
BURST_LEN = 50
BUDGET_PER_LAYER = 8
TOTAL_BUDGET = NUM_LAYERS * BUDGET_PER_LAYER  # 216


def generate_bursty_trace(seed: int = 42) -> List[List[List[int]]]:
    """Same bursty trace generator as the main experiment."""
    rng = random.Random(seed)
    n_hot = max(TOP_K, 8)
    hot = list(range(n_hot))
    all_experts = list(range(NUM_EXPERTS))
    mid_layer = NUM_LAYERS // 2

    trace = []
    for t in range(NUM_TOKENS):
        phase = (t // BURST_LEN) % 2
        token_layers = []
        for l in range(NUM_LAYERS):
            if phase == 0:
                concentrated = l < mid_layer
            else:
                concentrated = l >= mid_layer

            chosen: List[int] = []
            while len(chosen) < TOP_K:
                if concentrated:
                    pool = hot if rng.random() < 0.85 else all_experts
                else:
                    pool = all_experts
                e = rng.choice(pool)
                if e not in chosen:
                    chosen.append(e)
            token_layers.append(chosen)
        trace.append(token_layers)
    return trace


def run_periodic_epcb(trace, config: PerLayerConfig) -> Dict:
    """Run periodic EPCB with given config, return hit rate + stats."""
    base_ir = PolicyIR(
        name="periodic_epcb",
        cache=CacheIR(capacity=BUDGET_PER_LAYER, eviction=EvictionPolicy.LFU, lfu_decay=0.9),
    )
    hook = PerLayerHook(base_ir, NUM_LAYERS, NUM_EXPERTS, config)

    total_hits = total_misses = 0
    for t in range(len(trace)):
        for l in range(NUM_LAYERS):
            plan = hook.on_layer(l, trace[t][l])
            total_hits += sum(1 for d in plan.dispatches if d.cache_hit)
            total_misses += sum(1 for d in plan.dispatches if not d.cache_hit)

    total = total_hits + total_misses
    return {
        "hits": total_hits,
        "misses": total_misses,
        "hit_rate": total_hits / total if total > 0 else 0.0,
        "rebalances": hook._step_count // config.rebalance_interval,
    }


def run_uniform_baseline(trace) -> Dict:
    """Uniform allocation baseline (no EPCB)."""
    base_ir = PolicyIR(
        name="uniform",
        cache=CacheIR(capacity=BUDGET_PER_LAYER, eviction=EvictionPolicy.LFU, lfu_decay=0.9),
    )
    hooks = {}
    uniform_cap = TOTAL_BUDGET // NUM_LAYERS
    for l in range(NUM_LAYERS):
        ir = copy.deepcopy(base_ir)
        ir.cache.capacity = uniform_cap
        compiled = compile_policy(ir)
        hooks[l] = PolicyHook(compiled)

    total_hits = total_misses = 0
    for t in range(len(trace)):
        for l in range(NUM_LAYERS):
            plan = hooks[l].on_layer(l, trace[t][l])
            total_hits += sum(1 for d in plan.dispatches if d.cache_hit)
            total_misses += sum(1 for d in plan.dispatches if not d.cache_hit)

    total = total_hits + total_misses
    return {
        "hits": total_hits,
        "misses": total_misses,
        "hit_rate": total_hits / total if total > 0 else 0.0,
    }


def main():
    print("=" * 70)
    print("EPCB Sensitivity Ablation")
    print(f"  {NUM_TOKENS} tokens, {NUM_LAYERS} layers, {NUM_EXPERTS} experts, top-{TOP_K}")
    print(f"  Total budget: {TOTAL_BUDGET} slots")
    print("=" * 70)

    trace = generate_bursty_trace()

    # ── Baseline ──────────────────────────────────────────────────────
    print("\n[Baseline] Uniform allocation (no EPCB)...")
    baseline = run_uniform_baseline(trace)
    print(f"  Hit rate: {baseline['hit_rate']:.1%}")

    results = {
        "params": {
            "num_layers": NUM_LAYERS,
            "num_experts": NUM_EXPERTS,
            "top_k": TOP_K,
            "num_tokens": NUM_TOKENS,
            "total_budget": TOTAL_BUDGET,
        },
        "baseline_hit_rate": baseline["hit_rate"],
        "ablations": {},
    }

    # ── Ablation 1: c_min sweep (c_max fixed, interval fixed) ─────────
    print("\n" + "-" * 70)
    print("Ablation 1: c_min sweep (c_max=108, rebalance_interval=500)")
    print("-" * 70)
    c_min_values = [1, 2, 4, 6, 8, 12, 16]
    c_min_results = []
    for c_min in c_min_values:
        config = PerLayerConfig(
            entropy_window=200,
            min_capacity=c_min,
            max_capacity=TOTAL_BUDGET // 2,  # 108
            rebalance_interval=500,
            total_budget=TOTAL_BUDGET,
        )
        r = run_periodic_epcb(trace, config)
        c_min_results.append({"c_min": c_min, **r})
        delta = r["hit_rate"] - baseline["hit_rate"]
        print(f"  c_min={c_min:>2d}  →  hit_rate={r['hit_rate']:.1%}  (Δ={delta*100:+.1f}pp)")

    results["ablations"]["c_min_sweep"] = c_min_results

    # ── Ablation 2: c_max sweep (c_min fixed, interval fixed) ─────────
    print("\n" + "-" * 70)
    print("Ablation 2: c_max sweep (c_min=2, rebalance_interval=500)")
    print("-" * 70)
    c_max_values = [16, 32, 48, 64, 80, 108, 150, 216]
    c_max_results = []
    for c_max in c_max_values:
        config = PerLayerConfig(
            entropy_window=200,
            min_capacity=2,
            max_capacity=c_max,
            rebalance_interval=500,
            total_budget=TOTAL_BUDGET,
        )
        r = run_periodic_epcb(trace, config)
        c_max_results.append({"c_max": c_max, **r})
        delta = r["hit_rate"] - baseline["hit_rate"]
        print(f"  c_max={c_max:>3d}  →  hit_rate={r['hit_rate']:.1%}  (Δ={delta*100:+.1f}pp)")

    results["ablations"]["c_max_sweep"] = c_max_results

    # ── Ablation 3: Rebalance interval sweep ──────────────────────────
    print("\n" + "-" * 70)
    print("Ablation 3: rebalance_interval sweep (c_min=2, c_max=108)")
    print("-" * 70)
    interval_values = [10, 50, 100, 200, 500, 1000, 2000, 5000]
    interval_results = []
    for interval in interval_values:
        config = PerLayerConfig(
            entropy_window=200,
            min_capacity=2,
            max_capacity=TOTAL_BUDGET // 2,
            rebalance_interval=interval,
            total_budget=TOTAL_BUDGET,
        )
        r = run_periodic_epcb(trace, config)
        interval_results.append({"rebalance_interval": interval, **r})
        delta = r["hit_rate"] - baseline["hit_rate"]
        print(f"  interval={interval:>5d}  →  hit_rate={r['hit_rate']:.1%}  "
              f"(Δ={delta*100:+.1f}pp, {r['rebalances']} rebalances)")

    results["ablations"]["interval_sweep"] = interval_results

    # ── Ablation 4: Joint c_min × c_max grid ─────────────────────────
    print("\n" + "-" * 70)
    print("Ablation 4: c_min × c_max grid (interval=500)")
    print("-" * 70)
    grid_c_min = [1, 2, 4, 8]
    grid_c_max = [32, 64, 108, 216]
    grid_results = []
    label = "c_min\\c_max"
    header = f"{label:<12s}" + "".join(f"{cm:>8d}" for cm in grid_c_max)
    print(f"  {header}")
    for c_min in grid_c_min:
        row = f"  {c_min:<12d}"
        for c_max in grid_c_max:
            if c_min >= c_max:
                row += f"{'—':>8s}"
                continue
            config = PerLayerConfig(
                entropy_window=200,
                min_capacity=c_min,
                max_capacity=c_max,
                rebalance_interval=500,
                total_budget=TOTAL_BUDGET,
            )
            r = run_periodic_epcb(trace, config)
            grid_results.append({"c_min": c_min, "c_max": c_max, **r})
            row += f"{r['hit_rate']:>7.1%} "
        print(row)

    results["ablations"]["grid_cmin_cmax"] = grid_results

    # ── Summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    # Find best and worst from c_min sweep
    best_cmin = max(c_min_results, key=lambda x: x["hit_rate"])
    worst_cmin = min(c_min_results, key=lambda x: x["hit_rate"])
    print(f"  c_min: best={best_cmin['c_min']} ({best_cmin['hit_rate']:.1%}), "
          f"worst={worst_cmin['c_min']} ({worst_cmin['hit_rate']:.1%}), "
          f"spread={(best_cmin['hit_rate'] - worst_cmin['hit_rate'])*100:.1f}pp")

    best_cmax = max(c_max_results, key=lambda x: x["hit_rate"])
    worst_cmax = min(c_max_results, key=lambda x: x["hit_rate"])
    print(f"  c_max: best={best_cmax['c_max']} ({best_cmax['hit_rate']:.1%}), "
          f"worst={worst_cmax['c_max']} ({worst_cmax['hit_rate']:.1%}), "
          f"spread={(best_cmax['hit_rate'] - worst_cmax['hit_rate'])*100:.1f}pp")

    best_int = max(interval_results, key=lambda x: x["hit_rate"])
    worst_int = min(interval_results, key=lambda x: x["hit_rate"])
    print(f"  interval: best={best_int['rebalance_interval']} ({best_int['hit_rate']:.1%}), "
          f"worst={worst_int['rebalance_interval']} ({worst_int['hit_rate']:.1%}), "
          f"spread={(best_int['hit_rate'] - worst_int['hit_rate'])*100:.1f}pp")

    # All vs baseline
    all_hits = [r["hit_rate"] for r in c_min_results + c_max_results + interval_results]
    print(f"\n  All EPCB configs beat baseline ({baseline['hit_rate']:.1%}): "
          f"{'YES' if all(h > baseline['hit_rate'] for h in all_hits) else 'NO'}")
    print(f"  EPCB range: [{min(all_hits):.1%}, {max(all_hits):.1%}]")
    print(f"  Min improvement over baseline: {(min(all_hits) - baseline['hit_rate'])*100:.1f}pp")

    # ── Save ──────────────────────────────────────────────────────────
    out_path = Path(__file__).parent.parent / "figures" / "epcb_sensitivity.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
