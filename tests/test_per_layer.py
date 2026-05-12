"""Tests for per-layer adaptive caching and entropy tracker."""

import math
import pytest
from moe_sched.ir import PolicyIR, CacheIR, EvictionPolicy
from moe_sched.compiler import compile_policy
from moe_sched.runtime.hooks import PolicyHook
from moe_sched.runtime.per_layer import (
    RoutingEntropyTracker,
    PerLayerHook,
    PerLayerConfig,
    allocate_capacity_by_entropy,
)


# ---------------------------------------------------------------------------
# RoutingEntropyTracker
# ---------------------------------------------------------------------------

class TestEntropyTracker:
    def test_uniform_entropy(self):
        """Uniform routing should yield max entropy."""
        tracker = RoutingEntropyTracker(num_layers=1, num_experts=8, window=100)
        # Activate each of 8 experts equally across many steps
        for i in range(100):
            tracker.record(0, [i % 8])
        ent = tracker.compute_entropy(0)
        # Should be close to log2(8) = 3.0
        assert ent == pytest.approx(math.log2(8), abs=0.1)

    def test_concentrated_entropy(self):
        """All routing to one expert should yield ~0 entropy."""
        tracker = RoutingEntropyTracker(num_layers=1, num_experts=8, window=100)
        for _ in range(100):
            tracker.record(0, [0])
        ent = tracker.compute_entropy(0)
        assert ent == pytest.approx(0.0, abs=0.01)

    def test_two_experts_entropy(self):
        """50/50 split between 2 experts → entropy = 1.0."""
        tracker = RoutingEntropyTracker(num_layers=1, num_experts=8, window=100)
        for i in range(100):
            tracker.record(0, [i % 2])
        ent = tracker.compute_entropy(0)
        assert ent == pytest.approx(1.0, abs=0.05)

    def test_window_eviction(self):
        """Old data should be evicted from the window."""
        tracker = RoutingEntropyTracker(num_layers=1, num_experts=8, window=10)
        # First: all expert 0
        for _ in range(20):
            tracker.record(0, [0])
        # Then: uniform
        for i in range(10):
            tracker.record(0, [i % 8])
        ent = tracker.compute_entropy(0)
        # Window should only see the last 10 entries (uniform-ish)
        assert ent > 1.0

    def test_per_layer_independence(self):
        """Layers should have independent entropy."""
        tracker = RoutingEntropyTracker(num_layers=2, num_experts=8, window=100)
        for _ in range(100):
            tracker.record(0, [0])  # layer 0: all same
            tracker.record(1, [_ % 8])  # layer 1: uniform
        assert tracker.compute_entropy(0) < 0.1
        assert tracker.compute_entropy(1) > 2.5

    def test_compute_all(self):
        tracker = RoutingEntropyTracker(num_layers=3, num_experts=8, window=50)
        for _ in range(50):
            tracker.record(0, [0])
            tracker.record(1, [_ % 4])
            tracker.record(2, [_ % 8])
        all_ent = tracker.compute_all_entropies()
        assert 0 in all_ent and 1 in all_ent and 2 in all_ent
        assert all_ent[0] < all_ent[1] < all_ent[2]


# ---------------------------------------------------------------------------
# Capacity allocation
# ---------------------------------------------------------------------------

class TestCapacityAllocation:
    def test_proportional(self):
        """Higher entropy layers should get more capacity."""
        entropies = {0: 1.0, 1: 3.0, 2: 2.0}
        caps = allocate_capacity_by_entropy(entropies, total_budget=60, min_capacity=2, max_capacity=40)
        assert caps[1] > caps[0]  # layer 1 has highest entropy
        assert caps[2] > caps[0]  # layer 2 has higher entropy than 0

    def test_respects_min(self):
        entropies = {0: 0.1, 1: 5.9}
        caps = allocate_capacity_by_entropy(entropies, total_budget=20, min_capacity=4, max_capacity=16)
        assert caps[0] >= 4

    def test_respects_max(self):
        entropies = {0: 0.1, 1: 100.0}
        caps = allocate_capacity_by_entropy(entropies, total_budget=200, min_capacity=2, max_capacity=50)
        assert caps[1] <= 50

    def test_empty(self):
        assert allocate_capacity_by_entropy({}, total_budget=100) == {}


# ---------------------------------------------------------------------------
# PerLayerHook
# ---------------------------------------------------------------------------

class TestPerLayerHook:
    def _make_hook(self, num_layers=4, num_experts=8, cap=4):
        ir = PolicyIR(
            name="test_per_layer",
            cache=CacheIR(capacity=cap, eviction=EvictionPolicy.LFU, lfu_decay=0.9),
        )
        config = PerLayerConfig(
            entropy_window=50,
            min_capacity=2,
            max_capacity=16,
            rebalance_interval=100,
            total_budget=num_layers * cap,
        )
        return PerLayerHook(ir, num_layers, num_experts, config)

    def test_basic_dispatch(self):
        hook = self._make_hook()
        plan = hook.on_layer(0, [0, 1])
        assert plan.layer_idx == 0
        assert len(plan.dispatches) == 2

    def test_separate_caches_per_layer(self):
        """Each layer should have its own cache state."""
        hook = self._make_hook(num_layers=2, cap=4)
        # Fill layer 0 cache with experts 0-3
        for _ in range(5):
            hook.on_layer(0, [0, 1])
            hook.on_layer(0, [2, 3])
        # Layer 1 should miss on experts 0-3 (its own cache is cold)
        plan = hook.on_layer(1, [0, 1])
        assert plan.dispatches[0].cache_hit is False

    def test_stats_aggregation(self):
        hook = self._make_hook()
        hook.on_layer(0, [0, 1])
        hook.on_layer(1, [2, 3])
        hook.on_layer(0, [0, 1])  # should hit
        snap = hook.stats_snapshot()
        assert snap["cache"]["hits"] > 0
        assert snap["cache"]["misses"] > 0
        assert "per_layer" in snap

    def test_rebalance_fires(self):
        """After enough steps, rebalancing should adjust capacities."""
        hook = self._make_hook(num_layers=2, num_experts=8, cap=8)
        hook.config.rebalance_interval = 10
        hook.config.entropy_window = 5

        # Layer 0: concentrated (always expert 0)
        # Layer 1: uniform (rotate through 8)
        for i in range(20):
            hook.on_layer(0, [0])
            hook.on_layer(1, [i % 8])

        # After rebalance, layer 1 should get more capacity
        caps = hook._per_layer_capacity
        assert caps[1] >= caps[0], f"Expected layer 1 >= layer 0, got {caps}"

    def test_step_count(self):
        hook = self._make_hook()
        hook.on_layer(0, [0])
        hook.on_layer(1, [1])
        hook.on_layer(0, [0])
        assert hook.step_count == 3
