"""Per-layer adaptive policy support for MoE-PolicyLang.

Instead of applying one global cache policy across all layers, this module
enables assigning different cache capacities and eviction strategies per
layer based on observed routing entropy.  Layers with concentrated routing
(low entropy) need smaller caches; diffuse-routing layers (high entropy)
need larger caches.

Usage via Python eDSL:
    sched = MoEPolicyLang()
    @sched.policy
    def per_layer_adaptive(p):
        p.cache(capacity=8, eviction='lfu')
        p.per_layer(
            entropy_window=200,
            min_capacity=4,
            max_capacity=32,
            rebalance_interval=500,
        )

Usage via .moe file:
    policy per_layer_adaptive {
        cache { capacity = 8  eviction = lfu }
        per_layer {
            entropy_window = 200
            min_capacity = 4  max_capacity = 32
            rebalance_interval = 500
        }
    }
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from moe_policylang.compiler import CompiledPolicy, compile_policy
from moe_policylang.ir import CacheIR, PolicyIR
from moe_policylang.runtime.hooks import DispatchPlan, ExpertDispatch, PolicyHook


# ---------------------------------------------------------------------------
# Routing entropy tracker
# ---------------------------------------------------------------------------

class RoutingEntropyTracker:
    """Track per-layer routing entropy from expert activation patterns.

    Entropy is computed over a sliding window of expert activations per layer.
    Higher entropy = more uniform routing = harder to cache.
    Lower entropy = concentrated routing = easier to cache.
    """

    def __init__(self, num_layers: int, num_experts: int, window: int = 200):
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.window = window
        # Per-layer: list of recent expert activation sets
        self._history: Dict[int, List[List[int]]] = defaultdict(list)
        # Per-layer: cached entropy (recomputed periodically)
        self._entropy: Dict[int, float] = {}

    def record(self, layer_idx: int, selected_experts: Sequence[int]) -> None:
        """Record one layer's expert activations."""
        buf = self._history[layer_idx]
        buf.append(list(selected_experts))
        if len(buf) > self.window:
            del buf[:len(buf) - self.window]

    def compute_entropy(self, layer_idx: int) -> float:
        """Compute Shannon entropy of expert frequency distribution for a layer.

        Returns entropy in [0, log2(num_experts)].  Higher = more uniform.
        """
        buf = self._history.get(layer_idx, [])
        if not buf:
            return math.log2(max(self.num_experts, 2))  # assume worst case

        freq: Dict[int, int] = defaultdict(int)
        total = 0
        for activation in buf:
            for eid in activation:
                freq[eid] += 1
                total += 1

        if total == 0:
            return math.log2(max(self.num_experts, 2))

        entropy = 0.0
        for count in freq.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)

        return entropy

    def compute_all_entropies(self) -> Dict[int, float]:
        """Compute entropy for all tracked layers."""
        result = {}
        for layer_idx in self._history:
            result[layer_idx] = self.compute_entropy(layer_idx)
        self._entropy = result
        return result


# ---------------------------------------------------------------------------
# Per-layer configuration
# ---------------------------------------------------------------------------

@dataclass
class PerLayerConfig:
    """Configuration for per-layer adaptive policies."""
    entropy_window: int = 200
    min_capacity: int = 2
    max_capacity: int = 64
    rebalance_interval: int = 500
    total_budget: Optional[int] = None  # total cache slots across all layers


def allocate_capacity_by_entropy(
    entropies: Dict[int, float],
    total_budget: int,
    min_capacity: int = 2,
    max_capacity: int = 64,
) -> Dict[int, int]:
    """Allocate cache capacity across layers proportional to entropy.

    Higher entropy layers get more capacity (they need it).
    Lower entropy layers get less (they have concentrated routing).

    Args:
        entropies: layer_idx → Shannon entropy
        total_budget: total cache slots to distribute
        min_capacity: floor per layer
        max_capacity: ceiling per layer

    Returns:
        layer_idx → allocated capacity
    """
    if not entropies:
        return {}

    layers = sorted(entropies.keys())
    n = len(layers)

    # Normalize entropies to proportions
    total_entropy = sum(entropies[l] for l in layers)
    if total_entropy == 0:
        # Uniform allocation
        per_layer = max(min_capacity, total_budget // n)
        return {l: min(per_layer, max_capacity) for l in layers}

    # Proportional allocation with floor/ceiling
    raw = {l: (entropies[l] / total_entropy) * total_budget for l in layers}

    # Clamp and round
    allocated = {}
    for l in layers:
        cap = int(round(raw[l]))
        cap = max(min_capacity, min(cap, max_capacity))
        allocated[l] = cap

    return allocated


# ---------------------------------------------------------------------------
# PerLayerHook — maintains separate caches per layer
# ---------------------------------------------------------------------------

class PerLayerHook:
    """Policy hook that maintains separate cache instances per layer.

    Each layer gets its own cache with capacity allocated based on observed
    routing entropy.  The caches are periodically rebalanced as more routing
    data is observed.

    This hook wraps a base PolicyIR and creates per-layer CompiledPolicies
    with adjusted capacities.
    """

    def __init__(
        self,
        base_ir: PolicyIR,
        num_layers: int,
        num_experts: int,
        config: PerLayerConfig,
    ) -> None:
        self.base_ir = base_ir
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.config = config

        # Total budget: if not specified, use num_layers * base capacity
        self.total_budget = config.total_budget or (num_layers * base_ir.cache.capacity)

        # Entropy tracker
        self.entropy_tracker = RoutingEntropyTracker(
            num_layers=num_layers,
            num_experts=num_experts,
            window=config.entropy_window,
        )

        # Per-layer hooks — initially all use the base capacity
        self._per_layer_hooks: Dict[int, PolicyHook] = {}
        self._per_layer_capacity: Dict[int, int] = {}
        self._init_uniform()

        self._step_count = 0
        self._last_rebalance = 0

    def _init_uniform(self) -> None:
        """Initialize with uniform capacity across layers."""
        import copy
        uniform_cap = max(
            self.config.min_capacity,
            self.total_budget // self.num_layers,
        )
        uniform_cap = min(uniform_cap, self.config.max_capacity)

        for l in range(self.num_layers):
            ir = copy.deepcopy(self.base_ir)
            ir.cache.capacity = uniform_cap
            ir.name = f"{self.base_ir.name}_layer{l}"
            compiled = compile_policy(ir)
            self._per_layer_hooks[l] = PolicyHook(compiled)
            self._per_layer_capacity[l] = uniform_cap

    def _rebalance(self) -> None:
        """Rebalance cache capacities based on observed routing entropy."""
        import copy

        entropies = self.entropy_tracker.compute_all_entropies()
        if not entropies:
            return

        new_caps = allocate_capacity_by_entropy(
            entropies,
            self.total_budget,
            self.config.min_capacity,
            self.config.max_capacity,
        )

        # Only rebuild hooks for layers whose capacity changed
        for l, new_cap in new_caps.items():
            if new_cap != self._per_layer_capacity.get(l):
                ir = copy.deepcopy(self.base_ir)
                ir.cache.capacity = new_cap
                ir.name = f"{self.base_ir.name}_layer{l}"
                compiled = compile_policy(ir)
                self._per_layer_hooks[l] = PolicyHook(compiled)
                self._per_layer_capacity[l] = new_cap

        self._last_rebalance = self._step_count

    def on_layer(
        self,
        layer_idx: int,
        selected_experts: Sequence[int],
        scores: Optional[Sequence[float]] = None,
        expert_size_gb: float = 1.2,
    ) -> DispatchPlan:
        """Dispatch using the per-layer hook for this layer."""
        # Track entropy
        self.entropy_tracker.record(layer_idx, selected_experts)

        # Get or create per-layer hook
        if layer_idx not in self._per_layer_hooks:
            import copy
            ir = copy.deepcopy(self.base_ir)
            ir.cache.capacity = self.config.min_capacity
            ir.name = f"{self.base_ir.name}_layer{layer_idx}"
            compiled = compile_policy(ir)
            self._per_layer_hooks[layer_idx] = PolicyHook(compiled)
            self._per_layer_capacity[layer_idx] = self.config.min_capacity

        hook = self._per_layer_hooks[layer_idx]
        plan = hook.on_layer(layer_idx, selected_experts, scores, expert_size_gb)

        self._step_count += 1

        # Periodic rebalance
        if (self._step_count - self._last_rebalance >= self.config.rebalance_interval
                and self._step_count > self.config.entropy_window):
            self._rebalance()

        return plan

    @property
    def step_count(self) -> int:
        return self._step_count

    def stats_snapshot(self) -> dict:
        """Aggregate stats across all per-layer hooks."""
        total_hits = 0
        total_misses = 0
        total_evictions = 0
        per_layer_stats = {}

        for l, hook in sorted(self._per_layer_hooks.items()):
            snap = hook.stats_snapshot()
            c = snap["cache"]
            total_hits += c["hits"]
            total_misses += c["misses"]
            total_evictions += c["evictions"]
            per_layer_stats[l] = {
                "capacity": self._per_layer_capacity.get(l, 0),
                "hits": c["hits"],
                "misses": c["misses"],
                "hit_rate": c["hit_rate"],
            }

        total = total_hits + total_misses
        return {
            "name": self.base_ir.name + "_per_layer",
            "steps": self._step_count,
            "cache": {
                "hits": total_hits,
                "misses": total_misses,
                "evictions": total_evictions,
                "hit_rate": total_hits / total if total > 0 else 0.0,
            },
            "per_layer": per_layer_stats,
            "entropies": dict(self.entropy_tracker.compute_all_entropies()),
            "capacities": dict(self._per_layer_capacity),
            "total_budget": self.total_budget,
        }

    @property
    def compiled(self):
        """Return first layer's compiled policy for compatibility."""
        first = next(iter(self._per_layer_hooks.values()), None)
        return first.compiled if first else None

    @property
    def cache(self):
        """Return first layer's cache for compatibility (stats_snapshot is preferred)."""
        first = next(iter(self._per_layer_hooks.values()), None)
        return first.cache if first else None
