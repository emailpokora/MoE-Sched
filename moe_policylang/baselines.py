"""Hand-coded reference implementations used for DSL correctness verification.

These provide independent LRU/LFU implementations that deliberately
**do not** import from ``moe_policylang.runtime`` — proving that the DSL
compiler+hook pipeline produces *identical* dispatch decisions to hand-coded
logic.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import List, Sequence


@dataclass
class RefDispatch:
    expert_id: int
    on_gpu: bool
    cache_hit: bool


@dataclass
class RefStats:
    hits: int = 0
    misses: int = 0
    evictions: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


class HandCodedLRU:
    """Straight-line LRU cache + GPU-only scheduler, no prefetch.

    Mirrors the behavior of a MoE-PolicyLang policy built with:
        cache(capacity=N, eviction=LRU)
        schedule(mode=GPU_ONLY)

    (No prefetcher, no monitor.)  The equivalence test in
    ``tests/test_correctness.py`` runs both this class and a DSL-compiled
    hook over the same deterministic workload and asserts the per-step
    ``RefDispatch`` sequence matches.
    """

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self.capacity = capacity
        self._cache: "OrderedDict[int, bool]" = OrderedDict()
        self.stats = RefStats()
        self.dispatches: List[RefDispatch] = []

    # ------------------------------------------------------------------

    def on_layer(
        self,
        layer_idx: int,
        selected_experts: Sequence[int],
    ) -> List[RefDispatch]:
        """Process one layer of expert selections.  Returns the per-expert
        dispatch list for that layer (also appended to ``self.dispatches``)."""
        layer_dispatches: List[RefDispatch] = []
        for eid in selected_experts:
            if eid in self._cache:
                # Hit: refresh LRU position
                self._cache.move_to_end(eid)
                self.stats.hits += 1
                d = RefDispatch(expert_id=eid, on_gpu=True, cache_hit=True)
            else:
                # Miss: evict LRU if needed, then insert
                self.stats.misses += 1
                while len(self._cache) >= self.capacity:
                    self._cache.popitem(last=False)
                    self.stats.evictions += 1
                self._cache[eid] = True
                # GPU_ONLY scheduler: execute on GPU with a synchronous
                # transfer, counted as a miss but still on_gpu=True.
                d = RefDispatch(expert_id=eid, on_gpu=True, cache_hit=False)

            layer_dispatches.append(d)
            self.dispatches.append(d)

        return layer_dispatches


class HandCodedLRUFallback:
    """LRU cache with CPU-fallback scheduling.

    Mirrors:
        cache(capacity=N, eviction=LRU)
        schedule(mode=CPU_FALLBACK)
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._cache: "OrderedDict[int, bool]" = OrderedDict()
        self.stats = RefStats()
        self.dispatches: List[RefDispatch] = []

    def on_layer(
        self,
        layer_idx: int,
        selected_experts: Sequence[int],
    ) -> List[RefDispatch]:
        layer_dispatches: List[RefDispatch] = []
        for eid in selected_experts:
            if eid in self._cache:
                self._cache.move_to_end(eid)
                self.stats.hits += 1
                d = RefDispatch(expert_id=eid, on_gpu=True, cache_hit=True)
            else:
                self.stats.misses += 1
                while len(self._cache) >= self.capacity:
                    self._cache.popitem(last=False)
                    self.stats.evictions += 1
                self._cache[eid] = True
                # CPU_FALLBACK: misses execute on CPU, no GPU transfer.
                d = RefDispatch(expert_id=eid, on_gpu=False, cache_hit=False)
            layer_dispatches.append(d)
            self.dispatches.append(d)
        return layer_dispatches
