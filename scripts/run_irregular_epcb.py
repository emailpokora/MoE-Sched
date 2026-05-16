"""Irregular-burst EPCB experiment: adaptive vs periodic on aperiodic workload.

The regular-burst experiment (run_static_vs_dynamic_epcb.py) uses fixed
50-token phase transitions — a pattern that periodic rebalancing matches
well.  This experiment uses *irregular* phase durations (drawn from a
geometric distribution) to demonstrate when condition-triggered adaptation
genuinely outperforms fixed-interval rebalancing.

Workload structure:
  - Same layer-heterogeneous design (first-half concentrated / second-half
    diffuse, then swap).
  - Phase durations are random: mean ~60 tokens, range 10–200.
  - Long stable stretches mean periodic wastes rebalances; sudden shifts
    mean periodic may not react fast enough.
"""

from __future__ import annotations

import copy
import json
import random
import time
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
# Parameters
# ---------------------------------------------------------------------------

NUM_LAYERS = 27
NUM_EXPERTS = 64
TOP_K = 6
NUM_TOKENS = 3000
BUDGET_PER_LAYER = 8
TOTAL_BUDGET = NUM_LAYERS * BUDGET_PER_LAYER

# Phase duration distribution: geometric with p≈1/60 → mean ~60 tokens
PHASE_MEAN = 60
PHASE_MIN = 10
PHASE_MAX = 200


# ---------------------------------------------------------------------------
# Irregular-burst trace generator
# ---------------------------------------------------------------------------

def generate_irregular_trace(
    num_tokens: int = NUM_TOKENS,
    num_layers: int = NUM_LAYERS,
    num_experts: int = NUM_EXPERTS,
    top_k: int = TOP_K,
    seed: int = 42,
) -> tuple[List[List[List[int]]], List[int]]:
    """Generate a trace with *irregular*, aperiodic phase transitions.

    Returns:
        trace: trace[token][layer] = list of selected expert IDs
        boundaries: token indices where phase transitions occur
    """
    rng = random.Random(seed)
    n_hot = max(top_k, 8)
    hot = list(range(n_hot))
    all_experts = list(range(num_experts))
    mid_layer = num_layers // 2

    # Pre-compute phase boundaries using geometric distribution
    boundaries = []
    t = 0
    while t < num_tokens:
        # Duration drawn from clipped geometric
        dur = max(PHASE_MIN, min(PHASE_MAX, int(rng.expovariate(1.0 / PHASE_MEAN))))
        t += dur
        if t < num_tokens:
            boundaries.append(t)

    trace = []
    phase = 0
    bi = 0  # next boundary index
    for t in range(num_tokens):
        if bi < len(boundaries) and t >= boundaries[bi]:
            phase = 1 - phase
            bi += 1

        token_layers = []
        for l in range(num_layers):
            if phase == 0:
                concentrated = l < mid_layer
            else:
                concentrated = l >= mid_layer

            chosen: List[int] = []
            while len(chosen) < top_k:
                if concentrated:
                    pool = hot if rng.random() < 0.85 else all_experts
                else:
                    pool = all_experts
                e = rng.choice(pool)
                if e not in chosen:
                    chosen.append(e)
            token_layers.append(chosen)
        trace.append(token_layers)

    return trace, boundaries


# ---------------------------------------------------------------------------
# Strategy runners (same logic as regular experiment)
# ---------------------------------------------------------------------------

def _base_ir(name: str) -> PolicyIR:
    return PolicyIR(
        name=name,
        cache=CacheIR(capacity=BUDGET_PER_LAYER, eviction=EvictionPolicy.LFU, lfu_decay=0.9),
    )


def _count(plan):
    h = sum(1 for d in plan.dispatches if d.cache_hit)
    m = sum(1 for d in plan.dispatches if not d.cache_hit)
    return h, m


def run_static(trace, calibration_tokens: int = 200):
    tracker = RoutingEntropyTracker(NUM_LAYERS, NUM_EXPERTS, window=calibration_tokens)
    for t in range(min(calibration_tokens, len(trace))):
        for l in range(NUM_LAYERS):
            tracker.record(l, trace[t][l])
    entropies = tracker.compute_all_entropies()
    allocation = allocate_capacity_by_entropy(entropies, TOTAL_BUDGET, 2, TOTAL_BUDGET // 2)

    hooks: Dict[int, PolicyHook] = {}
    for l in range(NUM_LAYERS):
        ir = copy.deepcopy(_base_ir("static"))
        ir.cache.capacity = allocation.get(l, BUDGET_PER_LAYER)
        hooks[l] = PolicyHook(compile_policy(ir))

    total_hits = total_misses = 0
    for t in range(len(trace)):
        for l in range(NUM_LAYERS):
            h, m = _count(hooks[l].on_layer(l, trace[t][l]))
            total_hits += h; total_misses += m
        if (t + 1) % 1000 == 0:
            print(f"    token {t+1}/{len(trace)}")

    total = total_hits + total_misses
    return {"strategy": "static_epcb", "hit_rate": total_hits / total if total else 0.0,
            "hits": total_hits, "misses": total_misses}


def run_periodic(trace, rebalance_interval: int = 500):
    config = PerLayerConfig(
        entropy_window=200, min_capacity=2, max_capacity=TOTAL_BUDGET // 2,
        rebalance_interval=rebalance_interval, total_budget=TOTAL_BUDGET,
    )
    hook = PerLayerHook(_base_ir("periodic"), NUM_LAYERS, NUM_EXPERTS, config)

    total_hits = total_misses = 0
    for t in range(len(trace)):
        for l in range(NUM_LAYERS):
            h, m = _count(hook.on_layer(l, trace[t][l]))
            total_hits += h; total_misses += m
        if (t + 1) % 1000 == 0:
            print(f"    token {t+1}/{len(trace)}")

    total = total_hits + total_misses
    return {"strategy": "periodic_epcb", "hit_rate": total_hits / total if total else 0.0,
            "hits": total_hits, "misses": total_misses,
            "rebalances": hook._step_count // rebalance_interval,
            "rebalance_interval": rebalance_interval}


def run_adaptive(trace, threshold: float = 0.30, window: int = 50, cooldown: int = 200):
    from moe_policylang.adaptive import (
        AdaptAction, AdaptCondition, AdaptIR, AdaptRule, AdaptiveHook,
    )

    config = PerLayerConfig(
        entropy_window=200, min_capacity=2, max_capacity=TOTAL_BUDGET // 2,
        rebalance_interval=999999, total_budget=TOTAL_BUDGET,
    )
    per_layer_hook = PerLayerHook(_base_ir("adaptive"), NUM_LAYERS, NUM_EXPERTS, config)

    adapt_ir = AdaptIR(rules=[
        AdaptRule(
            condition=AdaptCondition(
                metric="hit_rate", op="<", threshold=threshold, window=window,
            ),
            action=AdaptAction(param="rebalance", value="entropy"),
            cooldown=cooldown,
        ),
    ])
    hook = AdaptiveHook(per_layer_hook, adapt_ir, _base_ir("adaptive"), compile_policy)

    total_hits = total_misses = 0
    for t in range(len(trace)):
        for l in range(NUM_LAYERS):
            h, m = _count(hook.on_layer(l, trace[t][l]))
            total_hits += h; total_misses += m
        if (t + 1) % 1000 == 0:
            print(f"    token {t+1}/{len(trace)}")

    total = total_hits + total_misses
    adapt_stats = hook.stats_snapshot().get("adapt", {})
    return {"strategy": "adaptive_epcb", "hit_rate": total_hits / total if total else 0.0,
            "hits": total_hits, "misses": total_misses,
            "rebalances": adapt_stats.get("adaptations", 0),
            "threshold": threshold, "window": window, "cooldown": cooldown}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 65)
    print("Irregular-Burst EPCB Experiment")
    print(f"  {NUM_TOKENS} tokens, {NUM_LAYERS} layers, {NUM_EXPERTS} experts, top-{TOP_K}")
    print(f"  Phase durations: mean={PHASE_MEAN}, range=[{PHASE_MIN}, {PHASE_MAX}]")
    print(f"  Total budget: {TOTAL_BUDGET}")
    print("=" * 65)

    print("\nGenerating irregular trace...")
    trace, boundaries = generate_irregular_trace()
    print(f"  {len(boundaries)} phase transitions at tokens: "
          f"{boundaries[:8]}{'...' if len(boundaries) > 8 else ''}")

    # Compute inter-phase gaps to show irregularity
    gaps = [boundaries[0]] + [boundaries[i] - boundaries[i-1] for i in range(1, len(boundaries))]
    print(f"  Phase durations: min={min(gaps)}, max={max(gaps)}, "
          f"mean={sum(gaps)/len(gaps):.0f}")

    results = []

    print("\n[1/3] Static EPCB...")
    t0 = time.time()
    r = run_static(trace)
    results.append(r)
    print(f"  Hit rate: {r['hit_rate']:.1%}  ({time.time()-t0:.1f}s)")

    print("\n[2/3] Periodic EPCB (interval=500)...")
    t0 = time.time()
    r = run_periodic(trace, rebalance_interval=500)
    results.append(r)
    print(f"  Hit rate: {r['hit_rate']:.1%}  ({r['rebalances']} rebalances, {time.time()-t0:.1f}s)")

    print("\n[3/3] Adaptive EPCB (threshold=0.30, window=50, cooldown=200)...")
    t0 = time.time()
    r = run_adaptive(trace, threshold=0.30, window=50, cooldown=200)
    results.append(r)
    print(f"  Hit rate: {r['hit_rate']:.1%}  ({r['rebalances']} rebalances, {time.time()-t0:.1f}s)")

    # Summary
    print("\n" + "=" * 65)
    print(f"{'Strategy':<30} {'Hit Rate':>10} {'Rebal.':>8}")
    print("-" * 65)
    for r in results:
        rebal = str(r.get("rebalances", "—"))
        print(f"  {r['strategy']:<28} {r['hit_rate']:>9.1%} {rebal:>8}")
    print("=" * 65)

    # Save
    out_path = Path(__file__).parent.parent / "traces" / "irregular_epcb.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = {
        "parameters": {
            "num_tokens": NUM_TOKENS, "num_layers": NUM_LAYERS,
            "num_experts": NUM_EXPERTS, "top_k": TOP_K,
            "budget": TOTAL_BUDGET, "phase_mean": PHASE_MEAN,
        },
        "boundaries": boundaries,
        "phase_durations": gaps,
        "results": results,
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
