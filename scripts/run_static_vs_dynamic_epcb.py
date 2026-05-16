"""Static vs Dynamic EPCB on bursty workload.

Compares three allocation strategies:
  1. Static: one-shot allocation from a calibration pass, then frozen.
  2. Periodic EPCB: rebalances every N steps based on entropy.
  3. Adaptive EPCB: rebalances only when hit_rate drops below threshold.

Demonstrates that the adapt mechanism enables EPCB to react to distribution
shifts in bursty workloads, producing higher hit rates than static allocation.
"""

from __future__ import annotations

import copy
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from moe_policylang.compiler import compile_policy
from moe_policylang.ir import CacheIR, EvictionPolicy, PolicyIR, PrefetchIR, ScheduleIR
from moe_policylang.runtime.hooks import PolicyHook
from moe_policylang.runtime.per_layer import (
    PerLayerConfig,
    PerLayerHook,
    RoutingEntropyTracker,
    allocate_capacity_by_entropy,
)

# ---------------------------------------------------------------------------
# Bursty workload generator (matches benchmark suite but with more tokens)
# ---------------------------------------------------------------------------

NUM_LAYERS = 27
NUM_EXPERTS = 64
TOP_K = 6
NUM_TOKENS = 2000
BURST_LEN = 50  # tokens per phase
# Tight budget: ~8 slots per layer on average (less than top-k=6 would need
# for perfect caching).  This makes allocation decisions matter.
BUDGET_PER_LAYER = 8
TOTAL_BUDGET = NUM_LAYERS * BUDGET_PER_LAYER


def generate_bursty_trace(
    num_tokens: int = NUM_TOKENS,
    num_layers: int = NUM_LAYERS,
    num_experts: int = NUM_EXPERTS,
    top_k: int = TOP_K,
    burst_len: int = BURST_LEN,
    seed: int = 42,
) -> List[List[List[int]]]:
    """Generate a bursty trace with *layer-heterogeneous* phase shifts.

    In phase 0:
      - First half of layers (0..13) are *concentrated*: 90% of picks from
        a small hot set of 4 experts → low entropy, small cache needed.
      - Second half (14..26) are *diffuse*: picks spread across all 64
        experts → high entropy, large cache needed.

    In phase 1 (every other burst window):
      - The profiles swap: first-half layers become diffuse, second-half
        become concentrated.

    This creates a workload where the optimal per-layer allocation changes
    at every phase boundary — exactly the scenario that adaptive EPCB is
    designed for.

    Returns: trace[token][layer] = list of selected expert IDs.
    """
    rng = random.Random(seed)
    n_hot = max(top_k, 8)  # hot set must be >= top_k to avoid deadlock
    hot = list(range(n_hot))
    all_experts = list(range(num_experts))
    mid_layer = num_layers // 2  # layer index where profiles split

    trace = []
    for t in range(num_tokens):
        phase = (t // burst_len) % 2
        token_layers = []
        for l in range(num_layers):
            # Determine if this layer is concentrated or diffuse
            if phase == 0:
                concentrated = l < mid_layer
            else:
                concentrated = l >= mid_layer

            chosen: List[int] = []
            while len(chosen) < top_k:
                if concentrated:
                    # 85% hot, 15% any — low routing entropy
                    pool = hot if rng.random() < 0.85 else all_experts
                else:
                    # Pure uniform — high routing entropy
                    pool = all_experts
                e = rng.choice(pool)
                if e not in chosen:
                    chosen.append(e)
            token_layers.append(chosen)
        trace.append(token_layers)
    return trace


# ---------------------------------------------------------------------------
# Strategy 1: Static allocation (one-shot calibration)
# ---------------------------------------------------------------------------

def run_static(trace, calibration_tokens: int = 200):
    """Static EPCB: compute allocation from first N tokens, then freeze."""
    base_ir = PolicyIR(
        name="static_epcb",
        cache=CacheIR(capacity=BUDGET_PER_LAYER, eviction=EvictionPolicy.LFU, lfu_decay=0.9),
    )

    # Calibration pass: compute entropy from first N tokens
    tracker = RoutingEntropyTracker(NUM_LAYERS, NUM_EXPERTS, window=calibration_tokens)
    for t in range(min(calibration_tokens, len(trace))):
        for l in range(NUM_LAYERS):
            tracker.record(l, trace[t][l])
    entropies = tracker.compute_all_entropies()
    allocation = allocate_capacity_by_entropy(entropies, TOTAL_BUDGET, 2, TOTAL_BUDGET // 2)

    # Build per-layer hooks with frozen allocation
    hooks: Dict[int, PolicyHook] = {}
    for l in range(NUM_LAYERS):
        ir = copy.deepcopy(base_ir)
        ir.cache.capacity = allocation.get(l, BUDGET_PER_LAYER)
        compiled = compile_policy(ir)
        hooks[l] = PolicyHook(compiled)

    # Run full trace
    total_hits = total_misses = 0
    for t in range(len(trace)):
        for l in range(NUM_LAYERS):
            plan = hooks[l].on_layer(l, trace[t][l])
            total_hits += sum(1 for d in plan.dispatches if d.cache_hit)
            total_misses += sum(1 for d in plan.dispatches if not d.cache_hit)
        if (t + 1) % 500 == 0:
            print(f"    token {t+1}/{len(trace)}")

    total = total_hits + total_misses
    return {
        "strategy": "static_epcb",
        "hits": total_hits,
        "misses": total_misses,
        "hit_rate": total_hits / total if total > 0 else 0.0,
        "budget": TOTAL_BUDGET,
        "calibration_tokens": calibration_tokens,
    }


# ---------------------------------------------------------------------------
# Strategy 2: Periodic EPCB (rebalance every N steps)
# ---------------------------------------------------------------------------

def run_periodic(trace, rebalance_interval: int = 500):
    """Periodic EPCB: rebalance capacity every N layer dispatches."""
    base_ir = PolicyIR(
        name="periodic_epcb",
        cache=CacheIR(capacity=BUDGET_PER_LAYER, eviction=EvictionPolicy.LFU, lfu_decay=0.9),
    )
    config = PerLayerConfig(
        entropy_window=200,
        min_capacity=2,
        max_capacity=TOTAL_BUDGET // 2,
        rebalance_interval=rebalance_interval,
        total_budget=TOTAL_BUDGET,
    )
    hook = PerLayerHook(base_ir, NUM_LAYERS, NUM_EXPERTS, config)

    total_hits = total_misses = 0
    for t in range(len(trace)):
        for l in range(NUM_LAYERS):
            plan = hook.on_layer(l, trace[t][l])
            total_hits += sum(1 for d in plan.dispatches if d.cache_hit)
            total_misses += sum(1 for d in plan.dispatches if not d.cache_hit)
        if (t + 1) % 500 == 0:
            print(f"    token {t+1}/{len(trace)}")

    total = total_hits + total_misses
    stats = hook.stats_snapshot()
    return {
        "strategy": "periodic_epcb",
        "hits": total_hits,
        "misses": total_misses,
        "hit_rate": total_hits / total if total > 0 else 0.0,
        "budget": TOTAL_BUDGET,
        "rebalance_interval": rebalance_interval,
        "rebalances": hook._step_count // rebalance_interval,
    }


# ---------------------------------------------------------------------------
# Strategy 3: Adaptive EPCB (rebalance triggered by hit_rate drop)
# ---------------------------------------------------------------------------

def run_adaptive(trace, threshold: float = 0.5, window: int = 100, cooldown: int = 200, label: str = "adaptive_epcb"):
    """Adaptive EPCB: rebalance only when hit_rate drops below threshold.

    This uses the adapt mechanism: when hit_rate < threshold for `window`
    consecutive evaluations, trigger a rebalance.
    """
    from moe_policylang.adaptive import (
        AdaptAction, AdaptCondition, AdaptIR, AdaptRule, AdaptiveHook,
    )

    base_ir = PolicyIR(
        name=label,
        cache=CacheIR(capacity=BUDGET_PER_LAYER, eviction=EvictionPolicy.LFU, lfu_decay=0.9),
    )
    config = PerLayerConfig(
        entropy_window=200,
        min_capacity=2,
        max_capacity=TOTAL_BUDGET // 2,
        rebalance_interval=999999,  # effectively disabled — adapt handles it
        total_budget=TOTAL_BUDGET,
    )
    per_layer_hook = PerLayerHook(base_ir, NUM_LAYERS, NUM_EXPERTS, config)

    # Wrap with AdaptiveHook that triggers rebalance on low hit rate
    adapt_ir = AdaptIR(rules=[
        AdaptRule(
            condition=AdaptCondition(
                metric="hit_rate", op="<", threshold=threshold, window=window,
            ),
            action=AdaptAction(param="rebalance", value="entropy"),
            cooldown=cooldown,
        ),
    ])
    hook = AdaptiveHook(per_layer_hook, adapt_ir, base_ir, compile_policy)

    total_hits = total_misses = 0
    for t in range(len(trace)):
        for l in range(NUM_LAYERS):
            plan = hook.on_layer(l, trace[t][l])
            total_hits += sum(1 for d in plan.dispatches if d.cache_hit)
            total_misses += sum(1 for d in plan.dispatches if not d.cache_hit)
        if (t + 1) % 500 == 0:
            print(f"    token {t+1}/{len(trace)}")

    total = total_hits + total_misses
    adapt_stats = hook.stats_snapshot().get("adapt", {})
    return {
        "strategy": label,
        "hits": total_hits,
        "misses": total_misses,
        "hit_rate": total_hits / total if total > 0 else 0.0,
        "budget": TOTAL_BUDGET,
        "adapt_threshold": threshold,
        "adapt_window": window,
        "adapt_cooldown": cooldown,
        "rebalances": adapt_stats.get("adaptations", 0),
    }


# ---------------------------------------------------------------------------
# Strategy 4: Uniform per-layer (no entropy, equal allocation)
# ---------------------------------------------------------------------------

def run_uniform(trace):
    """Uniform per-layer: equal capacity per layer, no rebalancing."""
    base_ir = PolicyIR(
        name="uniform",
        cache=CacheIR(capacity=BUDGET_PER_LAYER, eviction=EvictionPolicy.LFU, lfu_decay=0.9),
    )
    hooks: Dict[int, PolicyHook] = {}
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
        "strategy": "uniform_per_layer",
        "hits": total_hits,
        "misses": total_misses,
        "hit_rate": total_hits / total if total > 0 else 0.0,
        "budget": TOTAL_BUDGET,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Static vs Dynamic EPCB on Bursty Workload")
    print(f"  {NUM_TOKENS} tokens, {NUM_LAYERS} layers, {NUM_EXPERTS} experts, top-{TOP_K}")
    print(f"  Burst length: {BURST_LEN} tokens, Total budget: {TOTAL_BUDGET}")
    print("=" * 60)

    print("\nGenerating bursty trace...")
    trace = generate_bursty_trace()

    results = []

    print("\n[1/4] Uniform per-layer (baseline)...")
    r = run_uniform(trace)
    results.append(r)
    print(f"  Hit rate: {r['hit_rate']:.1%}")

    print("\n[2/4] Static EPCB (one-shot calibration)...")
    r = run_static(trace, calibration_tokens=200)
    results.append(r)
    print(f"  Hit rate: {r['hit_rate']:.1%}")

    print("\n[3/4] Periodic EPCB (rebalance every 500 steps)...")
    r = run_periodic(trace, rebalance_interval=500)
    results.append(r)
    print(f"  Hit rate: {r['hit_rate']:.1%} ({r['rebalances']} rebalances)")

    print("\n[4/5] Adaptive EPCB, cd=200 (rebalance on hit_rate < 0.35)...")
    r = run_adaptive(trace, threshold=0.35, window=50, cooldown=200,
                     label="adaptive_cd200")
    results.append(r)
    print(f"  Hit rate: {r['hit_rate']:.1%} ({r['rebalances']} rebalances)")

    print("\n[5/5] Adaptive EPCB, cd=1350 (rebalance on hit_rate < 0.35)...")
    r = run_adaptive(trace, threshold=0.35, window=50, cooldown=1350,
                     label="adaptive_cd1350")
    results.append(r)
    print(f"  Hit rate: {r['hit_rate']:.1%} ({r['rebalances']} rebalances)")

    # Summary
    print("\n" + "=" * 60)
    print(f"{'Strategy':<30} {'Hit Rate':>10} {'Rebal.':>8}")
    print("-" * 60)
    for r in results:
        rebal = str(r.get("rebalances", "—"))
        print(f"  {r['strategy']:<28} {r['hit_rate']:>9.1%} {rebal:>8}")
    print("=" * 60)

    # Save results
    out_path = Path(__file__).parent.parent / "traces" / "static_vs_dynamic_epcb.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
