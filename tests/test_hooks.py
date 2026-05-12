"""Unit tests for the PolicyHook runtime wrapper."""

import pytest

from moe_sched import (
    CacheIR,
    EvictionPolicy,
    MoESched,
    PolicyIR,
    PrefetchIR,
    PrefetchStrategy,
    ScheduleIR,
    ScheduleMode,
    build_hook,
    compile_policy,
)
from moe_sched.runtime.scheduler import ExecutionDevice


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _simple_lru_hook(capacity: int = 4):
    ir = PolicyIR(
        name="simple_lru",
        cache=CacheIR(capacity=capacity, eviction=EvictionPolicy.LRU),
    )
    return build_hook(compile_policy(ir))


def _cpu_fallback_hook(capacity: int = 4):
    ir = PolicyIR(
        name="cpu_fb",
        cache=CacheIR(capacity=capacity, eviction=EvictionPolicy.LRU),
        schedule=ScheduleIR(mode=ScheduleMode.CPU_FALLBACK),
    )
    return build_hook(compile_policy(ir))


# ---------------------------------------------------------------------------
# Dispatch plan shape
# ---------------------------------------------------------------------------

class TestDispatchPlan:
    def test_all_misses_first_pass(self):
        hook = _simple_lru_hook(capacity=4)
        plan = hook.on_layer(layer_idx=0, selected_experts=[0, 1, 2])

        assert plan.layer_idx == 0
        assert len(plan.dispatches) == 3
        assert plan.misses == 3
        assert plan.hits == 0
        # GPU-only scheduler always routes to GPU, even on miss.
        assert all(d.device == ExecutionDevice.GPU for d in plan.dispatches)
        # Misses on GPU-only => transfer.
        assert all(d.transferred for d in plan.dispatches)

    def test_second_pass_hits(self):
        hook = _simple_lru_hook(capacity=4)
        hook.on_layer(0, [0, 1, 2])
        plan = hook.on_layer(1, [0, 1])

        assert plan.hits == 2
        assert plan.misses == 0
        assert all(d.cache_hit for d in plan.dispatches)
        assert not any(d.transferred for d in plan.dispatches)

    def test_eviction_behavior(self):
        hook = _simple_lru_hook(capacity=2)
        hook.on_layer(0, [0])      # [0]
        hook.on_layer(1, [1])      # [0, 1]
        hook.on_layer(2, [2])      # evict 0 -> [1, 2]
        plan = hook.on_layer(3, [0])  # miss: 0 was evicted
        assert plan.dispatches[0].cache_hit is False


class TestCpuFallback:
    def test_miss_routes_to_cpu(self):
        hook = _cpu_fallback_hook(capacity=4)
        plan = hook.on_layer(0, [5])
        assert plan.dispatches[0].device == ExecutionDevice.CPU
        assert plan.dispatches[0].cache_hit is False
        # CPU execution never counts as a transfer.
        assert plan.dispatches[0].transferred is False

    def test_hit_routes_to_gpu(self):
        hook = _cpu_fallback_hook(capacity=4)
        hook.on_layer(0, [5])
        plan = hook.on_layer(1, [5])
        assert plan.dispatches[0].device == ExecutionDevice.GPU
        assert plan.dispatches[0].cache_hit is True


# ---------------------------------------------------------------------------
# Stats snapshot
# ---------------------------------------------------------------------------

class TestStatsSnapshot:
    def test_snapshot_tracks_hits_and_misses(self):
        hook = _simple_lru_hook(capacity=4)
        hook.on_layer(0, [0, 1])
        hook.on_layer(1, [0, 1])  # both hit

        snap = hook.stats_snapshot()
        assert snap["name"] == "simple_lru"
        assert snap["cache"]["hits"] == 2
        assert snap["cache"]["misses"] == 2
        assert snap["cache"]["hit_rate"] == pytest.approx(0.5)
        assert snap["steps"] == 2


# ---------------------------------------------------------------------------
# Prefetcher wiring
# ---------------------------------------------------------------------------

class TestPrefetchWiring:
    def test_history_prefetcher_predicts(self):
        ir = PolicyIR(
            name="lfu_hist",
            cache=CacheIR(capacity=8, eviction=EvictionPolicy.LFU),
            prefetch=PrefetchIR(
                strategy=PrefetchStrategy.HISTORY, budget=2, history_window=20
            ),
        )
        hook = build_hook(compile_policy(ir))

        # Warm up a preference for experts {0, 1}.
        for _ in range(5):
            hook.on_layer(0, [0, 1])

        plan = hook.on_layer(1, [2])
        # Prefetch should target the historically-popular experts not in the
        # current selection.
        assert set(plan.prefetched).issubset({0, 1})


# ---------------------------------------------------------------------------
# Monitor integration
# ---------------------------------------------------------------------------

class TestMonitorIntegration:
    def test_monitor_records_when_present(self):
        sched = MoESched()
        ir = (
            sched.build("mon")
            .cache(capacity=4, eviction="lru")
            .monitor(metrics=["hit_rate"], window=50, log_interval=2)
            .done()
        )
        hook = build_hook(compile_policy(ir))
        hook.on_layer(0, [0, 1])
        hook.on_layer(1, [0])

        snap = hook.monitor.snapshot()
        assert snap.access_count == 3  # 2 + 1 experts
        # hit_rate = 1 hit out of 3 accesses
        assert snap.hit_rate == pytest.approx(1 / 3)

    def test_no_monitor_is_fine(self):
        hook = _simple_lru_hook()
        assert hook.monitor is None
        hook.on_layer(0, [0])  # should not raise
