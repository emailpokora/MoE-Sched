# cython: boundscheck=False, wraparound=False, cdivision=True
"""Cython-accelerated dispatch loop.

Moves the entire PolicyHook.on_layer hot path into C-level code,
eliminating per-expert Python call overhead.  This is the Phase 3
fast path — the previous Phase 2 only accelerated individual
cache/scheduler components while the outer loop stayed in Python.

The fast hook maintains the same API as PolicyHook and is a transparent
drop-in.  When FAST_PATH_AVAILABLE is True and all components are
Cython-typed, build_hook() returns a FastPolicyHook automatically.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from moe_policylang.runtime.scheduler import ExecutionDevice


# We need lightweight result structs — avoid Python dataclass overhead
# in the inner loop by accumulating into C arrays and building the
# DispatchPlan at the end.

@dataclass
class ExpertDispatchResult:
    expert_id: int
    device: object  # ExecutionDevice
    cache_hit: bool
    transferred: bool = False


@dataclass
class DispatchPlanResult:
    layer_idx: int
    dispatches: list = field(default_factory=list)
    prefetched: list = field(default_factory=list)

    @property
    def hits(self):
        return sum(1 for d in self.dispatches if d.cache_hit)

    @property
    def misses(self):
        return sum(1 for d in self.dispatches if not d.cache_hit)


cdef class FastPolicyHook:
    """Full Cython dispatch loop — drop-in replacement for PolicyHook.

    Keeps references to the compiled components and runs the entire
    on_layer protocol in Cython, minimizing Python object creation
    and method dispatch overhead.
    """
    cdef public object compiled
    cdef public object cache
    cdef public object prefetcher
    cdef public object scheduler
    cdef public object monitor
    cdef public object triggers
    cdef public int _step_count
    cdef bint _cache_takes_score
    cdef bint _has_triggers
    cdef bint _has_monitor
    cdef bint _has_prefetch_insert

    def __init__(self, compiled):
        self.compiled = compiled
        self.cache = compiled.cache
        self.prefetcher = compiled.prefetcher
        self.scheduler = compiled.scheduler
        self.monitor = compiled.monitor
        self.triggers = compiled.triggers
        self._step_count = 0

        # Pre-compute capability flags to avoid hasattr/try-except in hot loop
        self._cache_takes_score = _check_score_cache(self.cache)
        self._has_triggers = (
            self.triggers is not None and getattr(self.triggers, 'active', False)
        )
        self._has_monitor = self.monitor is not None
        self._has_prefetch_insert = hasattr(self.cache, 'prefetch_insert')

    cpdef object on_layer(
        self,
        int layer_idx,
        list selected_experts,
        list scores=None,
        double expert_size_gb=1.2,
    ):
        """Process one MoE layer forward — fully in Cython."""
        cdef int n_experts = len(selected_experts)
        cdef int i
        cdef int eid
        cdef double score
        cdef bint hit
        cdef bint transferred
        cdef object device

        dispatches = []

        for i in range(n_experts):
            eid = selected_experts[i]
            score = scores[i] if scores is not None else 1.0

            # 1. Cache access
            if self._cache_takes_score:
                hit = self.cache.access(eid, score)
            else:
                hit = self.cache.access(eid)

            # 2. Scheduler decision
            device = self.scheduler.decide(eid, hit, expert_size_gb)

            # 3. Build dispatch record (minimal Python object creation)
            transferred = (not hit) and (device == ExecutionDevice.GPU)
            dispatches.append(ExpertDispatchResult(
                expert_id=eid,
                device=device,
                cache_hit=hit,
                transferred=transferred,
            ))

            # 4. Prefetcher usage accounting
            self.prefetcher.report_usage(eid)

            # 5. Triggers
            if self._has_triggers:
                self.triggers.on_access(eid)
                self.triggers.after_access(self.cache)

            # 6. Monitor
            if self._has_monitor:
                self.monitor.record_access(hit=hit)

        # 7. Prefetch for upcoming layers
        predicted = self.prefetcher.predict(layer_idx, selected_experts)

        # Warm cache for prefetched experts
        if self._has_prefetch_insert:
            for eid in predicted:
                if not self.cache.is_cached(eid):
                    self.cache.prefetch_insert(eid)

        self._step_count += 1

        # Build result (single Python object allocation at end)
        cdef object plan = DispatchPlanResult(layer_idx=layer_idx)
        plan.dispatches = dispatches
        plan.prefetched = list(predicted)
        return plan

    @property
    def step_count(self):
        return self._step_count

    def stats_snapshot(self):
        """Aggregate stats — same interface as PolicyHook."""
        snap = {
            "name": self.compiled.name,
            "steps": self._step_count,
            "cache": {
                "hits": self.cache.stats.hits,
                "misses": self.cache.stats.misses,
                "evictions": self.cache.stats.evictions,
                "hit_rate": self.cache.stats.hit_rate,
            },
            "prefetch": {
                "issued": self.prefetcher.stats.issued,
                "useful": self.prefetcher.stats.useful,
                "accuracy": self.prefetcher.stats.accuracy,
            },
            "scheduler": {
                "gpu": self.scheduler.stats.gpu_executions,
                "cpu": self.scheduler.stats.cpu_executions,
                "transfers": self.scheduler.stats.transfers,
            },
        }
        if self._has_triggers:
            snap["triggers"] = {
                "memory_pressure": (
                    {
                        "fired": self.triggers.memory_pressure.stats.fired,
                        "evicted": self.triggers.memory_pressure.stats.evicted,
                    }
                    if self.triggers.memory_pressure is not None else None
                ),
                "ttl": (
                    {
                        "fired": self.triggers.ttl.stats.fired,
                        "evicted": self.triggers.ttl.stats.evicted,
                    }
                    if self.triggers.ttl is not None else None
                ),
            }
        return snap


cdef bint _check_score_cache(object cache):
    """Check if this cache type takes a score argument (ScoreCache)."""
    import inspect
    sig = inspect.signature(cache.access)
    params = list(sig.parameters.keys())
    return len(params) >= 2 and params[1] != 'self'
