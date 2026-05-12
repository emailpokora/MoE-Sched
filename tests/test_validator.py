"""Tests for PolicyIR validation rules.

Each validation rule in validator.py should have at least one passing and one
failing test case.
"""

import pytest

from moe_sched.errors import ValidationError
from moe_sched.ir import (
    CacheIR,
    EvictionPolicy,
    PolicyIR,
    PrefetchIR,
    PrefetchStrategy,
    ScheduleIR,
    ScheduleMode,
)
from moe_sched.validator import validate_policy


def _make(
    capacity=32,
    eviction=EvictionPolicy.LRU,
    pin=None,
    lfu_decay=0.95,
    freq_threshold=0.05,
    freq_window=100,
    score_ema_alpha=0.3,
    strategy=PrefetchStrategy.NONE,
    lookahead=1,
    budget=4,
    affinity_threshold=0.3,
    history_window=50,
    mode=ScheduleMode.GPU_ONLY,
    cpu_threshold_ms=50.0,
    overlap=True,
) -> PolicyIR:
    """Helper to build a PolicyIR with overrides."""
    return PolicyIR(
        name="test",
        cache=CacheIR(
            capacity=capacity,
            eviction=eviction,
            pin_experts=pin or [],
            lfu_decay=lfu_decay,
            freq_threshold=freq_threshold,
            freq_window=freq_window,
            score_ema_alpha=score_ema_alpha,
        ),
        prefetch=PrefetchIR(
            strategy=strategy,
            lookahead=lookahead,
            budget=budget,
            affinity_threshold=affinity_threshold,
            history_window=history_window,
        ),
        schedule=ScheduleIR(
            mode=mode,
            cpu_threshold_ms=cpu_threshold_ms,
            overlap=overlap,
        ),
    )


# ---- Cache rules ----

class TestCacheCapacity:
    def test_valid_capacity(self):
        validate_policy(_make(capacity=1, budget=1))
        validate_policy(_make(capacity=256))
        validate_policy(_make(capacity=512))

    def test_capacity_zero(self):
        with pytest.raises(ValidationError, match="capacity"):
            validate_policy(_make(capacity=0))

    def test_capacity_negative(self):
        with pytest.raises(ValidationError, match="capacity"):
            validate_policy(_make(capacity=-1))

    def test_capacity_exceeds_max(self):
        with pytest.raises(ValidationError, match="capacity"):
            validate_policy(_make(capacity=513))


class TestPinExperts:
    def test_valid_pin(self):
        validate_policy(_make(capacity=8, pin=[0, 1, 2]))

    def test_pin_negative_id(self):
        with pytest.raises(ValidationError, match="expert IDs"):
            validate_policy(_make(pin=[-1]))

    def test_pin_too_large_id(self):
        with pytest.raises(ValidationError, match="expert IDs"):
            validate_policy(_make(pin=[512]))

    def test_pin_exceeds_capacity(self):
        with pytest.raises(ValidationError, match="pin more experts"):
            validate_policy(_make(capacity=2, pin=[0, 1, 2]))


class TestLfuDecay:
    def test_valid(self):
        validate_policy(_make(lfu_decay=0.5))
        validate_policy(_make(lfu_decay=0.99))

    def test_zero(self):
        with pytest.raises(ValidationError, match="lfu_decay"):
            validate_policy(_make(lfu_decay=0.0))

    def test_one(self):
        with pytest.raises(ValidationError, match="lfu_decay"):
            validate_policy(_make(lfu_decay=1.0))

    def test_negative(self):
        with pytest.raises(ValidationError, match="lfu_decay"):
            validate_policy(_make(lfu_decay=-0.5))


class TestFreqThreshold:
    def test_valid(self):
        validate_policy(_make(freq_threshold=0.0))
        validate_policy(_make(freq_threshold=1.0))

    def test_negative(self):
        with pytest.raises(ValidationError, match="freq_threshold"):
            validate_policy(_make(freq_threshold=-0.1))

    def test_above_one(self):
        with pytest.raises(ValidationError, match="freq_threshold"):
            validate_policy(_make(freq_threshold=1.1))


class TestFreqWindow:
    def test_valid(self):
        validate_policy(_make(freq_window=1))

    def test_zero(self):
        with pytest.raises(ValidationError, match="freq_window"):
            validate_policy(_make(freq_window=0))


class TestScoreEmaAlpha:
    def test_valid(self):
        validate_policy(_make(score_ema_alpha=0.01))
        validate_policy(_make(score_ema_alpha=1.0))

    def test_zero(self):
        with pytest.raises(ValidationError, match="score_ema_alpha"):
            validate_policy(_make(score_ema_alpha=0.0))


# ---- Prefetch rules ----

class TestLookahead:
    def test_valid(self):
        validate_policy(_make(lookahead=1))
        validate_policy(_make(lookahead=5))

    def test_zero(self):
        with pytest.raises(ValidationError, match="lookahead"):
            validate_policy(_make(lookahead=0))


class TestPrefetchBudget:
    def test_valid(self):
        validate_policy(_make(budget=1))

    def test_zero(self):
        with pytest.raises(ValidationError, match="budget"):
            validate_policy(_make(budget=0))

    def test_exceeds_capacity(self):
        with pytest.raises(ValidationError, match="budget should not exceed"):
            validate_policy(_make(capacity=4, budget=5))


class TestAffinityThreshold:
    def test_valid(self):
        validate_policy(_make(affinity_threshold=0.0))
        validate_policy(_make(affinity_threshold=1.0))

    def test_negative(self):
        with pytest.raises(ValidationError, match="affinity_threshold"):
            validate_policy(_make(affinity_threshold=-0.1))


class TestHistoryWindow:
    def test_valid(self):
        validate_policy(_make(history_window=1))

    def test_zero(self):
        with pytest.raises(ValidationError, match="history_window"):
            validate_policy(_make(history_window=0))


# ---- Schedule rules ----

class TestCpuThreshold:
    def test_valid(self):
        validate_policy(_make(cpu_threshold_ms=0.1))

    def test_zero(self):
        with pytest.raises(ValidationError, match="cpu_threshold"):
            validate_policy(_make(cpu_threshold_ms=0.0))

    def test_negative(self):
        with pytest.raises(ValidationError, match="cpu_threshold"):
            validate_policy(_make(cpu_threshold_ms=-10.0))


class TestHybridOverlap:
    def test_hybrid_with_overlap(self):
        validate_policy(_make(mode=ScheduleMode.HYBRID, overlap=True))

    def test_hybrid_without_overlap(self):
        with pytest.raises(ValidationError, match="HYBRID.*overlap"):
            validate_policy(_make(mode=ScheduleMode.HYBRID, overlap=False))

    def test_gpu_only_without_overlap_ok(self):
        validate_policy(_make(mode=ScheduleMode.GPU_ONLY, overlap=False))


# ---- Cross-block rules ----

class TestScoreEvictionRequiresPrefetch:
    def test_score_with_affinity(self):
        validate_policy(_make(
            eviction=EvictionPolicy.SCORE,
            strategy=PrefetchStrategy.AFFINITY,
        ))

    def test_score_with_none_prefetch(self):
        with pytest.raises(ValidationError, match="SCORE eviction"):
            validate_policy(_make(
                eviction=EvictionPolicy.SCORE,
                strategy=PrefetchStrategy.NONE,
            ))

    def test_lru_with_none_prefetch_ok(self):
        validate_policy(_make(
            eviction=EvictionPolicy.LRU,
            strategy=PrefetchStrategy.NONE,
        ))


# ---- Multiple violations ----

class TestMultipleViolations:
    def test_reports_all_violations(self):
        with pytest.raises(ValidationError) as exc_info:
            validate_policy(_make(capacity=0, budget=0, cpu_threshold_ms=-1))
        assert len(exc_info.value.violations) >= 3
