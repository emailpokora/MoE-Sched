"""Tests for eviction triggers + LookaheadPrefetcher."""

import pytest

from moe_policylang import (
    CacheIR,
    EvictionPolicy,
    PolicyIR,
    PrefetchIR,
    PrefetchStrategy,
    build_hook,
    compile_policy,
)
from moe_policylang.runtime.cache import LRUCache
from moe_policylang.runtime.prefetch import LookaheadPrefetcher
from moe_policylang.runtime.triggers import (
    MemoryPressureTrigger,
    TriggerSet,
    TTLTrigger,
)


# ---------------------------------------------------------------------------
# MemoryPressureTrigger
# ---------------------------------------------------------------------------

class TestMemoryPressureTrigger:
    def test_fires_when_over_threshold(self):
        # 16 GB budget, 1.2 GB/expert, threshold=0.8 -> fires at >=11 experts.
        # headroom=0.4 -> target <=5 experts after eviction.
        cache = LRUCache(capacity=20)
        trigger = MemoryPressureTrigger(
            budget_gb=16.0, threshold=0.8, headroom=0.4, expert_size_gb=1.2,
        )

        for eid in range(13):
            cache.access(eid)
        assert cache.size == 13

        trigger.after_access(cache)
        # After firing, usage should be <= headroom * budget_gb (with epsilon
        # for float imprecision).
        assert (cache.size * 1.2) / 16.0 <= 0.4 + 1e-9
        assert trigger.stats.fired == 1
        assert trigger.stats.evicted > 0

    def test_no_fire_below_threshold(self):
        cache = LRUCache(capacity=20)
        trigger = MemoryPressureTrigger(
            budget_gb=16.0, threshold=0.9, headroom=0.5, expert_size_gb=1.2,
        )
        for eid in range(3):
            cache.access(eid)
        trigger.after_access(cache)
        assert trigger.stats.fired == 0
        assert cache.size == 3

    def test_invalid_params_rejected(self):
        with pytest.raises(ValueError):
            MemoryPressureTrigger(budget_gb=16.0, threshold=0.5, headroom=0.9)


# ---------------------------------------------------------------------------
# TTLTrigger
# ---------------------------------------------------------------------------

class TestTTLTrigger:
    def test_evicts_stale_experts(self):
        cache = LRUCache(capacity=10)
        trigger = TTLTrigger(ttl=5)

        # Access expert 0, then access 1-9.  By the time we've done 10 accesses
        # total, expert 0 should be stale (last_seen=1, step=10, cutoff=5).
        trigger.on_access(0)
        cache.access(0)
        for eid in range(1, 10):
            trigger.on_access(eid)
            cache.access(eid)
            trigger.after_access(cache)

        assert 0 not in cache.cache, "Expert 0 should have been TTL-evicted"
        assert trigger.stats.evicted >= 1

    def test_refreshing_access_prevents_eviction(self):
        cache = LRUCache(capacity=10)
        trigger = TTLTrigger(ttl=5)

        for step in range(1, 11):
            trigger.on_access(0)  # keep expert 0 hot
            cache.access(0)
            trigger.after_access(cache)

        assert 0 in cache.cache

    def test_pinned_never_evicted(self):
        cache = LRUCache(capacity=10, pin_experts=[0])
        trigger = TTLTrigger(ttl=2)

        trigger.on_access(0)
        cache.access(0)
        for eid in range(1, 8):
            trigger.on_access(eid)
            cache.access(eid)
            trigger.after_access(cache)

        assert 0 in cache.cache

    def test_invalid_ttl_rejected(self):
        with pytest.raises(ValueError):
            TTLTrigger(ttl=0)


# ---------------------------------------------------------------------------
# TriggerSet integration via compile_policy + PolicyHook
# ---------------------------------------------------------------------------

class TestTriggerSetIntegration:
    def test_memory_trigger_via_ir(self):
        ir = PolicyIR(
            name="mp",
            cache=CacheIR(
                capacity=30,
                eviction=EvictionPolicy.LRU,
                memory_threshold=0.5,  # low threshold -> will fire
                memory_headroom=0.3,
                memory_budget_gb=16.0,
                expert_size_gb=1.2,
            ),
        )
        hook = build_hook(compile_policy(ir))

        # Touch 20 distinct experts -> 20 * 1.2 = 24 GB > 0.5 * 16 = 8 GB.
        for eid in range(20):
            hook.on_layer(0, [eid])

        assert hook.triggers.memory_pressure is not None
        assert hook.triggers.memory_pressure.stats.fired > 0

        snap = hook.stats_snapshot()
        assert "triggers" in snap
        assert snap["triggers"]["memory_pressure"]["fired"] > 0

    def test_ttl_via_ir(self):
        ir = PolicyIR(
            name="ttl",
            cache=CacheIR(
                capacity=50, eviction=EvictionPolicy.LRU, ttl=3,
            ),
        )
        hook = build_hook(compile_policy(ir))

        for eid in range(10):
            hook.on_layer(0, [eid])

        assert hook.triggers.ttl is not None
        assert hook.triggers.ttl.stats.evicted > 0

    def test_both_triggers_coexist(self):
        ir = PolicyIR(
            name="both",
            cache=CacheIR(
                capacity=50,
                eviction=EvictionPolicy.LRU,
                memory_threshold=0.8,
                memory_headroom=0.4,
                memory_budget_gb=16.0,
                expert_size_gb=1.5,
                ttl=5,
            ),
        )
        hook = build_hook(compile_policy(ir))
        assert hook.triggers.memory_pressure is not None
        assert hook.triggers.ttl is not None

        for eid in range(15):
            hook.on_layer(0, [eid])

    def test_no_triggers_by_default(self):
        ir = PolicyIR(
            name="none",
            cache=CacheIR(capacity=10, eviction=EvictionPolicy.LRU),
        )
        hook = build_hook(compile_policy(ir))
        assert not hook.triggers.active
        snap = hook.stats_snapshot()
        assert "triggers" not in snap


# ---------------------------------------------------------------------------
# LookaheadPrefetcher
# ---------------------------------------------------------------------------

class TestLookaheadPrefetcher:
    def test_predicts_per_layer_patterns(self):
        pf = LookaheadPrefetcher(lookahead=1, budget=2, history_window=10)

        # Establish pattern: layer 0 always picks {0, 1}; layer 1 always {2, 3}.
        for _ in range(5):
            pf.predict(layer_idx=0, selected_experts=[0, 1])
            pf.predict(layer_idx=1, selected_experts=[2, 3])

        # Now at layer 0, prefetcher should predict {2, 3} for layer 1.
        preds = pf.predict(layer_idx=0, selected_experts=[0, 1])
        assert set(preds) == {2, 3}

    def test_budget_caps_predictions(self):
        pf = LookaheadPrefetcher(lookahead=2, budget=3, history_window=20)
        for _ in range(5):
            pf.predict(0, [0])
            pf.predict(1, [10, 11, 12, 13])
            pf.predict(2, [20, 21, 22, 23])

        preds = pf.predict(0, [0])
        assert len(preds) <= 3

    def test_respects_lookahead_depth(self):
        pf = LookaheadPrefetcher(lookahead=2, budget=4, history_window=10)
        for _ in range(3):
            pf.predict(0, [0])
            pf.predict(1, [10])
            pf.predict(2, [20])

        preds = pf.predict(0, [0])
        # Should see both layer 1 and layer 2 targets.
        assert 10 in preds and 20 in preds

    def test_excludes_current_selection(self):
        pf = LookaheadPrefetcher(lookahead=1, budget=2, history_window=10)
        for _ in range(3):
            pf.predict(0, [1, 2])
            pf.predict(1, [1, 2, 3])

        preds = pf.predict(0, [1, 2])
        assert 1 not in preds and 2 not in preds

    def test_accuracy_accounting(self):
        pf = LookaheadPrefetcher(lookahead=1, budget=2, history_window=5)
        for _ in range(5):
            pf.predict(0, [0])
            pf.predict(1, [5, 6])

        preds = pf.predict(0, [0])
        assert pf.stats.issued > 0
        # Report that expert 5 was actually used.
        if 5 in preds:
            pf.report_usage(5)
            assert pf.stats.useful >= 1


# ---------------------------------------------------------------------------
# LookaheadPrefetcher via compiler
# ---------------------------------------------------------------------------

class TestLookaheadViaCompiler:
    def test_lookahead_strategy_compiles(self):
        ir = PolicyIR(
            name="la",
            cache=CacheIR(capacity=8, eviction=EvictionPolicy.LRU),
            prefetch=PrefetchIR(
                strategy=PrefetchStrategy.LOOKAHEAD,
                lookahead=2,
                budget=4,
                history_window=20,
            ),
        )
        compiled = compile_policy(ir)
        assert isinstance(compiled.prefetcher, LookaheadPrefetcher)
        assert compiled.prefetcher.lookahead == 2
        assert compiled.prefetcher.budget == 4
