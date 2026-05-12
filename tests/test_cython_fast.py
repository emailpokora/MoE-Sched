"""Tests for Cython fast-path implementations (Phase 2).

Verifies that Cython-accelerated cache and scheduler produce identical
results to their pure-Python counterparts.

Run: python setup_cython.py build_ext --inplace && pytest tests/test_cython_fast.py -v
"""

from __future__ import annotations

import pytest
import time


# Skip if Cython modules not built
try:
    from moe_sched.runtime._fast import FAST_PATH_AVAILABLE
    if not FAST_PATH_AVAILABLE:
        raise ImportError
    from moe_sched.runtime._fast import (
        LRUCacheFast,
        LFUCacheFast,
        GPUOnlySchedulerFast,
        CPUFallbackSchedulerFast,
        HybridSchedulerFast,
    )
except ImportError:
    pytest.skip("Cython fast path not built", allow_module_level=True)

from moe_sched.runtime.cache import LRUCache, LFUCache
from moe_sched.runtime.scheduler import (
    GPUOnlyScheduler,
    CPUFallbackScheduler,
    HybridScheduler,
    ExecutionDevice,
)

# Shared test sequence
ACCESS_SEQ = [0, 1, 2, 3, 0, 1, 4, 0, 1, 2, 3, 4, 5, 6, 7, 0, 0, 0, 3, 5]


class TestLRUCacheFast:
    """Verify LRUCacheFast matches LRUCache behavior."""

    def test_access_hit_miss_equivalence(self):
        """Same sequence of accesses produces same hit/miss pattern."""
        py = LRUCache(capacity=4)
        cy = LRUCacheFast(capacity=4)
        for eid in ACCESS_SEQ:
            assert py.access(eid) == cy.access(eid), f"Mismatch at expert {eid}"

    def test_eviction_order_matches(self):
        """Stats match between Python and Cython after same access pattern."""
        py = LRUCache(capacity=3)
        cy = LRUCacheFast(capacity=3)
        for eid in ACCESS_SEQ:
            py.access(eid)
            cy.access(eid)
        assert py.stats.hits == cy.stats.hits
        assert py.stats.misses == cy.stats.misses
        assert py.stats.evictions == cy.stats.evictions

    def test_pinned_experts_respected(self):
        """Pinned experts never evicted in fast path."""
        cy = LRUCacheFast(capacity=3, pin_experts=[0, 1])
        for eid in range(10):
            cy.access(eid)
        assert cy.is_cached(0)
        assert cy.is_cached(1)

    def test_stats_accurate(self):
        """hits, misses, evictions counts match Python version."""
        py = LRUCache(capacity=4, pin_experts=[0])
        cy = LRUCacheFast(capacity=4, pin_experts=[0])
        for eid in ACCESS_SEQ * 3:
            py.access(eid)
            cy.access(eid)
        assert py.stats.hits == cy.stats.hits
        assert py.stats.misses == cy.stats.misses
        assert py.stats.evictions == cy.stats.evictions
        assert py.stats.loads == cy.stats.loads

    def test_prefetch_insert(self):
        """prefetch_insert adds without charging miss."""
        cy = LRUCacheFast(capacity=4)
        cy.prefetch_insert(99)
        assert cy.is_cached(99)
        assert cy.stats.misses == 0

    def test_performance_improvement(self):
        """Fast path is measurably faster than Python for 100K accesses."""
        N = 100_000
        seq = [i % 8 for i in range(N)]

        cy = LRUCacheFast(capacity=4)
        t0 = time.perf_counter()
        for eid in seq:
            cy.access(eid)
        cy_time = time.perf_counter() - t0

        py = LRUCache(capacity=4)
        t0 = time.perf_counter()
        for eid in seq:
            py.access(eid)
        py_time = time.perf_counter() - t0

        speedup = py_time / cy_time
        print(f"LRU speedup: {speedup:.1f}x (py={py_time*1e6/N:.1f}µs, cy={cy_time*1e6/N:.1f}µs)")
        assert speedup > 1.0, f"Expected speedup, got {speedup:.2f}x"


class TestLFUCacheFast:
    """Verify LFUCacheFast matches LFUCache behavior."""

    def test_access_hit_miss_equivalence(self):
        py = LFUCache(capacity=4, decay=0.9)
        cy = LFUCacheFast(capacity=4, decay=0.9)
        for eid in ACCESS_SEQ:
            assert py.access(eid) == cy.access(eid), f"Mismatch at expert {eid}"

    def test_stats_match(self):
        py = LFUCache(capacity=3, decay=0.95)
        cy = LFUCacheFast(capacity=3, decay=0.95)
        for eid in ACCESS_SEQ * 5:
            py.access(eid)
            cy.access(eid)
        assert py.stats.hits == cy.stats.hits
        assert py.stats.misses == cy.stats.misses
        assert py.stats.evictions == cy.stats.evictions

    def test_pinned(self):
        cy = LFUCacheFast(capacity=3, pin_experts=[0])
        for eid in range(10):
            cy.access(eid)
        assert cy.is_cached(0)

    def test_performance(self):
        N = 100_000
        seq = [i % 8 for i in range(N)]

        cy = LFUCacheFast(capacity=4, decay=0.9)
        t0 = time.perf_counter()
        for eid in seq:
            cy.access(eid)
        cy_time = time.perf_counter() - t0

        py = LFUCache(capacity=4, decay=0.9)
        t0 = time.perf_counter()
        for eid in seq:
            py.access(eid)
        py_time = time.perf_counter() - t0

        speedup = py_time / cy_time
        print(f"LFU speedup: {speedup:.1f}x")
        assert speedup > 1.0


class TestSchedulerFast:
    """Verify fast schedulers match Python behavior."""

    def test_gpu_only_matches(self):
        py = GPUOnlyScheduler()
        cy = GPUOnlySchedulerFast()
        for cached in [True, False, True, True, False]:
            assert py.decide(0, cached) == cy.decide(0, cached)
        assert py.stats.gpu_executions == cy.stats.gpu_executions
        assert py.stats.transfers == cy.stats.transfers

    def test_cpu_fallback_matches(self):
        py = CPUFallbackScheduler()
        cy = CPUFallbackSchedulerFast()
        for cached in [True, False, True, False, False]:
            assert py.decide(0, cached) == cy.decide(0, cached)
        assert py.stats.gpu_executions == cy.stats.gpu_executions
        assert py.stats.cpu_executions == cy.stats.cpu_executions

    def test_hybrid_matches(self):
        py = HybridScheduler(cpu_threshold_ms=50.0)
        cy = HybridSchedulerFast(cpu_threshold_ms=50.0)
        for cached, size in [(True, 1.2), (False, 1.2), (False, 5.0), (True, 0.5)]:
            assert py.decide(0, cached, size) == cy.decide(0, cached, size)
        assert py.stats.gpu_executions == cy.stats.gpu_executions
        assert py.stats.cpu_executions == cy.stats.cpu_executions
        assert py.stats.transfers == cy.stats.transfers
