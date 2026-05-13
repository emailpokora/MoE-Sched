"""Tests for the policy autotuner (grid search over DSL parameters)."""

from __future__ import annotations

import pytest

from moe_policylang.autotuner import (
    DEFAULT_GRID,
    TuningResult,
    _build_and_compile,
    _evaluate,
    _expand_grid,
    autotune,
)


# ---------------------------------------------------------------------------
# Synthetic trace fixture
# ---------------------------------------------------------------------------

def _make_trace(n_steps: int = 200, n_layers: int = 4, n_experts: int = 8):
    """Generate a synthetic trace with locality (expert reuse)."""
    import random
    random.seed(42)
    trace = []
    for t in range(n_steps):
        for layer in range(n_layers):
            # Zipf-like: lower expert IDs are more popular
            experts = random.choices(range(n_experts), weights=[1 / (i + 1) for i in range(n_experts)], k=2)
            trace.append({"l": layer, "e": experts, "s": [0.6, 0.4]})
    return trace


TRACE = _make_trace()


# ---------------------------------------------------------------------------
# Unit tests: grid expansion
# ---------------------------------------------------------------------------

class TestExpandGrid:
    def test_full_grid_size(self):
        combos = _expand_grid(DEFAULT_GRID)
        # Should be less than full cartesian product due to pruning
        full_size = 1
        for v in DEFAULT_GRID.values():
            full_size *= len(v)
        assert len(combos) < full_size
        assert len(combos) > 0

    def test_pruning_lfu_decay(self):
        """lfu_decay variants should only appear for LFU eviction."""
        combos = _expand_grid(DEFAULT_GRID)
        for c in combos:
            if c.get("eviction") != "lfu" and "lfu_decay" in c:
                assert c["lfu_decay"] == DEFAULT_GRID["lfu_decay"][0]

    def test_pruning_prefetch_budget(self):
        """prefetch_budget variants should only appear for non-none strategy."""
        combos = _expand_grid(DEFAULT_GRID)
        for c in combos:
            if c.get("prefetch_strategy") == "none" and "prefetch_budget" in c:
                assert c["prefetch_budget"] == DEFAULT_GRID["prefetch_budget"][0]

    def test_small_grid(self):
        grid = {"capacity": [4, 8], "eviction": ["lru"]}
        combos = _expand_grid(grid)
        assert len(combos) == 2
        assert combos[0]["capacity"] == 4
        assert combos[1]["capacity"] == 8

    def test_empty_grid_key(self):
        grid = {"capacity": [8], "eviction": ["lru"]}
        combos = _expand_grid(grid)
        assert len(combos) == 1


# ---------------------------------------------------------------------------
# Unit tests: build and evaluate
# ---------------------------------------------------------------------------

class TestBuildAndEvaluate:
    def test_build_lru(self):
        params = {
            "capacity": 8,
            "eviction": "lru",
            "lfu_decay": 0.9,
            "prefetch_strategy": "none",
            "prefetch_budget": 4,
            "schedule_mode": "gpu_only",
        }
        ir = _build_and_compile("test_lru", params)
        assert ir.cache.capacity == 8
        assert ir.cache.eviction.value == "lru"

    def test_build_lfu(self):
        params = {
            "capacity": 16,
            "eviction": "lfu",
            "lfu_decay": 0.8,
            "prefetch_strategy": "history",
            "prefetch_budget": 4,
            "schedule_mode": "hybrid",
        }
        ir = _build_and_compile("test_lfu", params)
        assert ir.cache.capacity == 16
        assert ir.cache.eviction.value == "lfu"
        assert ir.cache.lfu_decay == 0.8

    def test_evaluate_returns_valid_result(self):
        params = {
            "capacity": 8,
            "eviction": "lru",
            "lfu_decay": 0.9,
            "prefetch_strategy": "none",
            "prefetch_budget": 4,
            "schedule_mode": "gpu_only",
        }
        ir = _build_and_compile("eval_test", params)
        result = _evaluate(ir, TRACE)
        assert 0.0 <= result.hit_rate <= 1.0
        assert result.hits + result.misses > 0
        assert result.evictions >= 0
        assert result.dispatch_mean_us == 0.0  # latency not measured

    def test_evaluate_with_latency(self):
        params = {
            "capacity": 8,
            "eviction": "lru",
            "lfu_decay": 0.9,
            "prefetch_strategy": "none",
            "prefetch_budget": 4,
            "schedule_mode": "gpu_only",
        }
        ir = _build_and_compile("lat_test", params)
        result = _evaluate(ir, TRACE, measure_latency=True)
        assert result.dispatch_mean_us > 0.0


# ---------------------------------------------------------------------------
# Integration: autotune
# ---------------------------------------------------------------------------

class TestAutotune:
    def test_default_grid(self):
        best, top5 = autotune(TRACE, top_k=5)
        assert isinstance(best, TuningResult)
        assert len(top5) == 5
        assert top5[0] is best
        # Best should have highest hit rate
        assert all(best.hit_rate >= r.hit_rate for r in top5)

    def test_small_grid(self):
        grid = {
            "capacity": [4, 16],
            "eviction": ["lru", "lfu"],
            "lfu_decay": [0.9],
            "prefetch_strategy": ["none"],
            "prefetch_budget": [4],
            "schedule_mode": ["gpu_only"],
        }
        best, top = autotune(TRACE, grid=grid, top_k=3)
        assert len(top) == 3
        # Larger capacity should generally give better hit rate
        assert best.params["capacity"] == 16

    def test_minimize(self):
        grid = {
            "capacity": [4, 8],
            "eviction": ["lru"],
            "lfu_decay": [0.9],
            "prefetch_strategy": ["none"],
            "prefetch_budget": [4],
            "schedule_mode": ["gpu_only"],
        }
        best, _ = autotune(TRACE, grid=grid, metric="evictions", maximize=False)
        # Larger capacity → fewer evictions
        assert best.params["capacity"] == 8

    def test_top_k_capped(self):
        grid = {
            "capacity": [8],
            "eviction": ["lru"],
            "lfu_decay": [0.9],
            "prefetch_strategy": ["none"],
            "prefetch_budget": [4],
            "schedule_mode": ["gpu_only"],
        }
        best, top = autotune(TRACE, grid=grid, top_k=10)
        # Only 1 combo in grid → top has 1 entry
        assert len(top) == 1
        assert top[0] is best

    def test_results_are_sorted(self):
        best, top = autotune(TRACE, top_k=10)
        for i in range(len(top) - 1):
            assert top[i].hit_rate >= top[i + 1].hit_rate

    def test_with_latency(self):
        grid = {
            "capacity": [4, 8],
            "eviction": ["lru"],
            "lfu_decay": [0.9],
            "prefetch_strategy": ["none"],
            "prefetch_budget": [4],
            "schedule_mode": ["gpu_only"],
        }
        best, _ = autotune(TRACE, grid=grid, measure_latency=True)
        assert best.dispatch_mean_us > 0.0
