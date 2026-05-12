"""Tests for the compiler: PolicyIR → CompiledPolicy."""

import pytest

from moe_sched.compiler import CompiledPolicy, compile_policy
from moe_sched.ir import (
    CacheIR,
    EvictionPolicy,
    MonitorIR,
    PolicyIR,
    PrefetchIR,
    PrefetchStrategy,
    ScheduleIR,
    ScheduleMode,
)
from moe_sched.runtime.cache import (
    FreqThresholdCache,
    LFUCache,
    LRUCache,
    ScoreCache,
)
from moe_sched.runtime.monitor import Monitor
from moe_sched.runtime.prefetch import (
    AffinityPrefetcher,
    HistoryPrefetcher,
    NullPrefetcher,
)
from moe_sched.runtime.scheduler import (
    CPUFallbackScheduler,
    GPUOnlyScheduler,
    HybridScheduler,
)
from moe_sched.runtime._fast import FAST_PATH_AVAILABLE
if FAST_PATH_AVAILABLE:
    from moe_sched.runtime._fast import (
        LRUCacheFast, LFUCacheFast,
        GPUOnlySchedulerFast, CPUFallbackSchedulerFast, HybridSchedulerFast,
    )
    _LRU = (LRUCache, LRUCacheFast)
    _LFU = (LFUCache, LFUCacheFast)
    _GPU = (GPUOnlyScheduler, GPUOnlySchedulerFast)
    _CPU = (CPUFallbackScheduler, CPUFallbackSchedulerFast)
    _HYB = (HybridScheduler, HybridSchedulerFast)
else:
    _LRU = (LRUCache,)
    _LFU = (LFUCache,)
    _GPU = (GPUOnlyScheduler,)
    _CPU = (CPUFallbackScheduler,)
    _HYB = (HybridScheduler,)


# ---------------------------------------------------------------------------
# Cache dispatch
# ---------------------------------------------------------------------------

class TestCompileCachePolicy:
    def test_lru(self):
        ir = PolicyIR(name="t", cache=CacheIR(capacity=16, eviction=EvictionPolicy.LRU))
        cp = compile_policy(ir)
        assert isinstance(cp.cache, _LRU)
        assert cp.cache.capacity == 16

    def test_lfu(self):
        ir = PolicyIR(
            name="t",
            cache=CacheIR(capacity=32, eviction=EvictionPolicy.LFU, lfu_decay=0.8),
        )
        cp = compile_policy(ir)
        assert isinstance(cp.cache, _LFU)
        assert cp.cache.decay == 0.8

    def test_score(self):
        ir = PolicyIR(
            name="t",
            cache=CacheIR(capacity=24, eviction=EvictionPolicy.SCORE, score_ema_alpha=0.5),
            prefetch=PrefetchIR(strategy=PrefetchStrategy.AFFINITY),
        )
        cp = compile_policy(ir)
        assert isinstance(cp.cache, ScoreCache)
        assert cp.cache.ema_alpha == 0.5

    def test_freq_threshold(self):
        ir = PolicyIR(
            name="t",
            cache=CacheIR(
                capacity=16,
                eviction=EvictionPolicy.FREQ_THRESHOLD,
                freq_threshold=0.1,
                freq_window=50,
            ),
        )
        cp = compile_policy(ir)
        assert isinstance(cp.cache, FreqThresholdCache)
        assert cp.cache.threshold == 0.1
        assert cp.cache.window == 50

    def test_pin_experts_forwarded(self):
        ir = PolicyIR(
            name="t",
            cache=CacheIR(capacity=16, pin_experts=[0, 5]),
        )
        cp = compile_policy(ir)
        assert cp.cache.pinned == {0, 5}


# ---------------------------------------------------------------------------
# Prefetch dispatch
# ---------------------------------------------------------------------------

class TestCompilePrefetch:
    def test_none(self):
        ir = PolicyIR(name="t", cache=CacheIR(capacity=16))
        cp = compile_policy(ir)
        assert isinstance(cp.prefetcher, NullPrefetcher)

    def test_affinity(self):
        ir = PolicyIR(
            name="t",
            cache=CacheIR(capacity=16),
            prefetch=PrefetchIR(
                strategy=PrefetchStrategy.AFFINITY,
                budget=8,
                affinity_threshold=0.2,
            ),
        )
        cp = compile_policy(ir)
        assert isinstance(cp.prefetcher, AffinityPrefetcher)
        assert cp.prefetcher.budget == 8
        assert cp.prefetcher.threshold == 0.2

    def test_history(self):
        ir = PolicyIR(
            name="t",
            cache=CacheIR(capacity=16),
            prefetch=PrefetchIR(strategy=PrefetchStrategy.HISTORY, history_window=100),
        )
        cp = compile_policy(ir)
        assert isinstance(cp.prefetcher, HistoryPrefetcher)
        assert cp.prefetcher.window == 100


# ---------------------------------------------------------------------------
# Scheduler dispatch
# ---------------------------------------------------------------------------

class TestCompileScheduler:
    def test_gpu_only(self):
        ir = PolicyIR(
            name="t",
            cache=CacheIR(capacity=16),
            schedule=ScheduleIR(mode=ScheduleMode.GPU_ONLY),
        )
        cp = compile_policy(ir)
        assert isinstance(cp.scheduler, _GPU)

    def test_cpu_fallback(self):
        ir = PolicyIR(
            name="t",
            cache=CacheIR(capacity=16),
            schedule=ScheduleIR(mode=ScheduleMode.CPU_FALLBACK),
        )
        cp = compile_policy(ir)
        assert isinstance(cp.scheduler, _CPU)

    def test_hybrid(self):
        ir = PolicyIR(
            name="t",
            cache=CacheIR(capacity=16),
            schedule=ScheduleIR(mode=ScheduleMode.HYBRID, cpu_threshold_ms=40.0),
        )
        cp = compile_policy(ir)
        assert isinstance(cp.scheduler, _HYB)
        assert cp.scheduler.cpu_threshold_ms == 40.0


# ---------------------------------------------------------------------------
# Monitor
# ---------------------------------------------------------------------------

class TestCompileMonitor:
    def test_no_monitor(self):
        ir = PolicyIR(name="t", cache=CacheIR(capacity=16))
        cp = compile_policy(ir)
        assert cp.monitor is None

    def test_with_monitor(self):
        ir = PolicyIR(
            name="t",
            cache=CacheIR(capacity=16),
            monitor=MonitorIR(metrics=["hit_rate", "latency"], window=200),
        )
        cp = compile_policy(ir)
        assert isinstance(cp.monitor, Monitor)
        assert "hit_rate" in cp.monitor.tracked_metrics


# ---------------------------------------------------------------------------
# Full compilation round-trip
# ---------------------------------------------------------------------------

class TestCompileRoundTrip:
    def test_full_policy(self, full_ir):
        cp = compile_policy(full_ir)
        assert isinstance(cp, CompiledPolicy)
        assert cp.name == "full"
        assert isinstance(cp.cache, _LFU)
        assert isinstance(cp.prefetcher, AffinityPrefetcher)
        assert isinstance(cp.scheduler, _HYB)
        assert isinstance(cp.monitor, Monitor)

    def test_compiled_cache_is_functional(self, minimal_ir):
        cp = compile_policy(minimal_ir)
        assert cp.cache.access(0) is False
        assert cp.cache.access(0) is True
        assert cp.cache.stats.hits == 1

    def test_compiled_prefetcher_is_functional(self, full_ir):
        cp = compile_policy(full_ir)
        result = cp.prefetcher.predict(0, [1, 2])
        assert isinstance(result, list)

    def test_compiled_scheduler_is_functional(self, full_ir):
        cp = compile_policy(full_ir)
        from moe_sched.runtime.scheduler import ExecutionDevice
        device = cp.scheduler.decide(0, is_cached=True)
        assert device is ExecutionDevice.GPU
