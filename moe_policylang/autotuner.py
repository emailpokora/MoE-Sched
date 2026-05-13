"""Policy autotuner — grid search over DSL parameter space.

Sweeps DSL parameters (capacity, eviction, decay, prefetch strategy/budget,
schedule mode) and finds optimal configurations for a given workload trace.

Usage::

    from moe_sched.autotuner import autotune

    # trace_data: list of dicts with 'l' (layer), 'e' (experts), 's' (scores)
    best, top5 = autotune(trace_data, metric='hit_rate', top_k=5)
    print(best.params, best.hit_rate)
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from moe_sched.compiler import compile_policy
from moe_sched.dsl import MoESched
from moe_sched.ir import PolicyIR
from moe_sched.runtime.hooks import build_hook


@dataclass
class TuningResult:
    """Result of one parameter configuration."""

    params: Dict[str, object]
    hit_rate: float
    evictions: int
    dispatch_mean_us: float = 0.0
    misses: int = 0
    hits: int = 0


# Default parameter grid
DEFAULT_GRID: Dict[str, list] = {
    "capacity": [4, 8, 16, 32],
    "eviction": ["lru", "lfu"],
    "lfu_decay": [0.8, 0.9, 0.95],
    "prefetch_strategy": ["none", "history"],
    "prefetch_budget": [2, 4, 8],
    "schedule_mode": ["gpu_only", "hybrid"],
}


def _expand_grid(grid: Dict[str, list]) -> List[Dict[str, object]]:
    """Expand a parameter grid into all valid combinations.

    Prunes invalid combos (e.g. lfu_decay only matters for LFU eviction,
    prefetch_budget only matters when strategy != 'none').
    """
    keys = sorted(grid.keys())
    combos = []
    for values in itertools.product(*(grid[k] for k in keys)):
        params = dict(zip(keys, values))

        # Prune: lfu_decay irrelevant for non-LFU eviction
        if params.get("eviction") != "lfu" and "lfu_decay" in params:
            if params["lfu_decay"] != grid["lfu_decay"][0]:
                continue  # only test one decay value for non-LFU

        # Prune: prefetch_budget irrelevant when strategy is 'none'
        if params.get("prefetch_strategy") == "none" and "prefetch_budget" in params:
            if params["prefetch_budget"] != grid["prefetch_budget"][0]:
                continue  # only test one budget value for 'none'

        # Prune: prefetch_budget must not exceed cache capacity
        if "prefetch_budget" in params and "capacity" in params:
            if params["prefetch_budget"] > params["capacity"]:
                continue

        combos.append(params)
    return combos


def _build_and_compile(name: str, params: Dict[str, object]) -> PolicyIR:
    """Build a PolicyIR from a flat parameter dict."""
    sched = MoESched()

    @sched.policy
    def _policy(p, _params=params):
        cache_kw = {"capacity": _params["capacity"], "eviction": _params["eviction"]}
        if _params["eviction"] == "lfu":
            cache_kw["lfu_decay"] = _params.get("lfu_decay", 0.9)
        p.cache(**cache_kw)
        p.prefetch(
            strategy=_params.get("prefetch_strategy", "none"),
            budget=_params.get("prefetch_budget", 4),
        )
        p.schedule(mode=_params.get("schedule_mode", "gpu_only"))

    return sched.policies["_policy"]


def _evaluate(
    ir: PolicyIR, trace_data: Sequence[dict], measure_latency: bool = False,
) -> TuningResult:
    """Compile a policy, replay trace, return metrics."""
    compiled = compile_policy(ir)
    hook = build_hook(compiled)

    latencies: list[float] = []
    for entry in trace_data:
        if measure_latency:
            t0 = time.perf_counter()
        hook.on_layer(
            layer_idx=entry["l"],
            selected_experts=entry["e"],
            scores=entry.get("s"),
        )
        if measure_latency:
            latencies.append(time.perf_counter() - t0)

    stats = hook.stats_snapshot()
    hits = stats["cache"]["hits"]
    misses = stats["cache"]["misses"]
    total = hits + misses
    hit_rate = hits / total if total > 0 else 0.0
    evictions = stats["cache"]["evictions"]
    mean_us = (sum(latencies) / len(latencies) * 1e6) if latencies else 0.0

    # Reconstruct params from IR for the result
    params = {
        "capacity": ir.cache.capacity,
        "eviction": ir.cache.eviction.value,
        "lfu_decay": ir.cache.lfu_decay,
        "prefetch_strategy": ir.prefetch.strategy.value,
        "prefetch_budget": ir.prefetch.budget,
        "schedule_mode": ir.schedule.mode.value,
    }

    return TuningResult(
        params=params,
        hit_rate=hit_rate,
        evictions=evictions,
        dispatch_mean_us=mean_us,
        misses=misses,
        hits=hits,
    )


def autotune(
    trace_data: Sequence[dict],
    grid: Optional[Dict[str, list]] = None,
    metric: str = "hit_rate",
    maximize: bool = True,
    top_k: int = 5,
    measure_latency: bool = False,
) -> Tuple[TuningResult, List[TuningResult]]:
    """Grid search over DSL parameters to find the optimal policy.

    Args:
        trace_data: Sequence of trace entries, each a dict with keys
            ``'l'`` (layer_idx), ``'e'`` (expert list), ``'s'`` (scores, optional).
        grid: Parameter grid to search. Defaults to :data:`DEFAULT_GRID`.
        metric: Which :class:`TuningResult` field to optimize.
        maximize: If True, find the configuration that maximizes *metric*.
        top_k: Number of top results to return.
        measure_latency: If True, measure per-dispatch latency (slower).

    Returns:
        Tuple of (best_result, top_k_results) sorted by *metric*.
    """
    if grid is None:
        grid = DEFAULT_GRID

    combos = _expand_grid(grid)
    results: list[TuningResult] = []

    for i, params in enumerate(combos):
        ir = _build_and_compile(f"autotune_{i}", params)
        result = _evaluate(ir, trace_data, measure_latency=measure_latency)
        results.append(result)

    # Sort by target metric
    results.sort(key=lambda r: getattr(r, metric), reverse=maximize)
    return results[0], results[:top_k]
