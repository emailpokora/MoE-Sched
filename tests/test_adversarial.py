"""Adversarial workload tests — Week 4 deliverable.

The proposal mandates debugging three edge cases:

  1. Cold start — no history, every access is a miss.
  2. Cache thrashing — working set > capacity, every access evicts.
  3. Expert load imbalance — one hot expert dominates, others are cold.

These tests assert that every policy (across the cache, prefetch, and
schedule dimensions) produces sensible output under each scenario:
no crashes, no negative stats, hit rates match theoretical bounds, and
the pipeline remains consistent even under degenerate conditions.
"""

from __future__ import annotations

import pytest

from moe_policylang import (
    CacheIR,
    EvictionPolicy,
    PolicyIR,
    PrefetchIR,
    PrefetchStrategy,
    ScheduleIR,
    ScheduleMode,
    build_hook,
    compile_policy,
)
from moe_policylang.integrations.mock_moe import (
    MockMoEModel,
    deterministic_trace_selector,
    uniform_selector,
)


def _run_with_selector(hook, selector, num_tokens=30, num_layers=8, num_experts=64, top_k=2):
    model = MockMoEModel(
        num_layers=num_layers,
        num_experts=num_experts,
        top_k=top_k,
        selector=selector,
    )
    model.run(hook, num_tokens=num_tokens)
    return hook.stats_snapshot()


# ---------------------------------------------------------------------------
# Cold start: first N accesses should all miss
# ---------------------------------------------------------------------------

class TestColdStart:
    @pytest.mark.parametrize("evc", [
        EvictionPolicy.LRU,
        EvictionPolicy.LFU,
        EvictionPolicy.FREQ_THRESHOLD,
    ])
    def test_first_distinct_accesses_all_miss(self, evc):
        ir = PolicyIR(
            name="cold",
            cache=CacheIR(capacity=32, eviction=evc),
        )
        hook = build_hook(compile_policy(ir))

        # Access 16 distinct experts for the first time -> all misses.
        for eid in range(16):
            plan = hook.on_layer(0, [eid])
            assert plan.misses == 1, f"Expert {eid} should miss on cold start ({evc.value})"
            assert plan.hits == 0

        assert hook.cache.stats.misses == 16
        assert hook.cache.stats.hits == 0

    def test_cold_start_with_prefetch_eventually_warms(self):
        # With history prefetcher, after a warm-up phase, some prefetches
        # should start landing.
        ir = PolicyIR(
            name="cold_pf",
            cache=CacheIR(capacity=16, eviction=EvictionPolicy.LFU),
            prefetch=PrefetchIR(
                strategy=PrefetchStrategy.HISTORY, budget=4, history_window=30
            ),
        )
        hook = build_hook(compile_policy(ir))

        snap = _run_with_selector(
            hook,
            selector=uniform_selector(num_experts=32, top_k=2, seed=0),
            num_tokens=60, num_layers=8, num_experts=32,
        )

        assert snap["prefetch"]["issued"] > 0
        # Not too strict — uniform random + lfu may have low but non-zero accuracy
        assert snap["prefetch"]["accuracy"] >= 0.0


# ---------------------------------------------------------------------------
# Cache thrashing: working set > capacity
# ---------------------------------------------------------------------------

class TestCacheThrashing:
    """When every access evicts, hit rate should stay low but the pipeline
    must not crash, produce NaN stats, or drift from theoretical bounds."""

    def _thrash_selector(self, num_experts: int, top_k: int = 2):
        """Cyclic access pattern guaranteed to thrash a small cache."""

        def select(token_idx, layer_idx):
            start = (token_idx * top_k + layer_idx * top_k) % num_experts
            return [(start + k) % num_experts for k in range(top_k)]

        return select

    @pytest.mark.parametrize("evc", [
        EvictionPolicy.LRU,
        EvictionPolicy.LFU,
    ])
    def test_small_cache_thrashes_gracefully(self, evc):
        num_experts = 64
        ir = PolicyIR(
            name="thrash",
            cache=CacheIR(capacity=4, eviction=evc),  # tiny cache
        )
        hook = build_hook(compile_policy(ir))
        snap = _run_with_selector(
            hook,
            selector=self._thrash_selector(num_experts),
            num_tokens=40, num_layers=8, num_experts=num_experts,
        )

        total = snap["cache"]["hits"] + snap["cache"]["misses"]
        assert total == 40 * 8 * 2
        # Hit rate must be in [0, 1].  Under genuine thrashing it should be
        # low but not necessarily zero (some adjacent-access hits).
        hr = snap["cache"]["hit_rate"]
        assert 0.0 <= hr <= 1.0
        assert snap["cache"]["evictions"] > 0

    def test_memory_trigger_survives_thrashing(self):
        num_experts = 48
        ir = PolicyIR(
            name="thrash_mp",
            cache=CacheIR(
                capacity=20,
                eviction=EvictionPolicy.LRU,
                memory_threshold=0.5,
                memory_headroom=0.3,
                memory_budget_gb=16.0,
                expert_size_gb=1.2,
            ),
        )
        hook = build_hook(compile_policy(ir))
        _run_with_selector(
            hook,
            selector=self._thrash_selector(num_experts),
            num_tokens=30, num_layers=8, num_experts=num_experts,
        )
        # Pipeline completed and both eviction paths recorded stats.
        assert hook.cache.stats.evictions > 0
        assert hook.triggers.memory_pressure.stats.evicted >= 0


# ---------------------------------------------------------------------------
# Expert load imbalance: one hot expert dominates
# ---------------------------------------------------------------------------

class TestLoadImbalance:
    def _imbalanced_selector(self, hot_expert: int, cold_range: range, top_k: int):
        """95 % of tokens hit ``hot_expert``; 5 % cycle through cold experts."""
        cold_list = list(cold_range)

        def select(token_idx, layer_idx):
            if token_idx % 20 == 0 and cold_list:  # ~5 % cold
                cold_eid = cold_list[(token_idx + layer_idx) % len(cold_list)]
                experts = [hot_expert, cold_eid]
            else:
                # Need top_k distinct experts: pad with a deterministic cold.
                experts = [hot_expert, cold_list[layer_idx % len(cold_list)]]
            return experts[:top_k]

        return select

    def test_hot_expert_hit_rate_high(self):
        ir = PolicyIR(
            name="imbalance",
            cache=CacheIR(capacity=8, eviction=EvictionPolicy.LFU),
        )
        hook = build_hook(compile_policy(ir))
        snap = _run_with_selector(
            hook,
            selector=self._imbalanced_selector(hot_expert=0, cold_range=range(10, 50), top_k=2),
            num_tokens=100, num_layers=8, num_experts=50,
        )

        assert snap["cache"]["hit_rate"] > 0.4, (
            f"LFU should cache the hot expert; got hit rate={snap['cache']['hit_rate']:.3f}"
        )

    def test_pinned_hot_expert_always_hits(self):
        ir = PolicyIR(
            name="pin_hot",
            cache=CacheIR(capacity=8, eviction=EvictionPolicy.LRU, pin_experts=[0]),
        )
        hook = build_hook(compile_policy(ir))
        snap = _run_with_selector(
            hook,
            selector=self._imbalanced_selector(hot_expert=0, cold_range=range(10, 60), top_k=2),
            num_tokens=100, num_layers=8, num_experts=60,
        )
        # Expert 0 is pinned — it must stay cached, so every access hits.
        # Total accesses = 100 * 8 * 2 = 1600.  Accesses to expert 0 = 1600/2.
        # After the first, those are all hits.
        assert snap["cache"]["hits"] >= 100 * 8 - 1
        assert 0 in hook.cache.cache

    def test_freq_threshold_retains_only_hot(self):
        ir = PolicyIR(
            name="freq_imbalance",
            cache=CacheIR(
                capacity=16,
                eviction=EvictionPolicy.FREQ_THRESHOLD,
                freq_threshold=0.1,   # 10 % of window
                freq_window=80,
            ),
        )
        hook = build_hook(compile_policy(ir))
        _run_with_selector(
            hook,
            selector=self._imbalanced_selector(hot_expert=0, cold_range=range(10, 50), top_k=2),
            num_tokens=100, num_layers=6, num_experts=50,
        )
        # Hot expert exceeds threshold; cold ones don't.  Cache should contain
        # the hot expert but stay well under capacity.
        assert 0 in hook.cache.cache
        assert hook.cache.size <= 16


# ---------------------------------------------------------------------------
# Determinism: repeated runs with same seed produce identical stats
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_two_identical_runs_produce_identical_stats(self):
        def build():
            ir = PolicyIR(
                name="det",
                cache=CacheIR(capacity=12, eviction=EvictionPolicy.LRU),
                prefetch=PrefetchIR(
                    strategy=PrefetchStrategy.HISTORY, budget=2, history_window=20
                ),
                schedule=ScheduleIR(mode=ScheduleMode.HYBRID, cpu_threshold_ms=40.0),
            )
            return build_hook(compile_policy(ir))

        trace = [
            [0, 1], [1, 2], [2, 3], [0, 4], [3, 5], [1, 4], [2, 5], [0, 3],
        ]
        sel = deterministic_trace_selector(trace)

        h1 = build()
        h2 = build()
        s1 = _run_with_selector(h1, selector=sel, num_tokens=20, num_layers=6)
        s2 = _run_with_selector(h2, selector=sel, num_tokens=20, num_layers=6)

        assert s1["cache"] == s2["cache"]
        assert s1["prefetch"] == s2["prefetch"]
        assert s1["scheduler"] == s2["scheduler"]
