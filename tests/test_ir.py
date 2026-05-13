"""Tests for IR dataclass construction and enum behaviour."""

import pytest

from moe_policylang.ir import (
    CacheIR,
    EvictionPolicy,
    MonitorIR,
    PolicyIR,
    PrefetchIR,
    PrefetchStrategy,
    ScheduleIR,
    ScheduleMode,
)


# ---------------------------------------------------------------------------
# Enum tests
# ---------------------------------------------------------------------------

class TestEvictionPolicy:
    def test_values(self):
        assert set(e.value for e in EvictionPolicy) == {
            "lru", "lfu", "score", "frequency_threshold", "fallback",
        }

    def test_from_string(self):
        assert EvictionPolicy("lru") is EvictionPolicy.LRU
        assert EvictionPolicy("lfu") is EvictionPolicy.LFU

    def test_invalid_string(self):
        with pytest.raises(ValueError):
            EvictionPolicy("invalid_policy")


class TestPrefetchStrategy:
    def test_values(self):
        assert set(s.value for s in PrefetchStrategy) == {
            "none", "affinity", "history", "lookahead",
        }

    def test_from_string(self):
        assert PrefetchStrategy("affinity") is PrefetchStrategy.AFFINITY


class TestScheduleMode:
    def test_values(self):
        assert set(m.value for m in ScheduleMode) == {
            "gpu_only", "cpu_fallback", "hybrid",
        }


# ---------------------------------------------------------------------------
# CacheIR
# ---------------------------------------------------------------------------

class TestCacheIR:
    def test_defaults(self):
        c = CacheIR(capacity=32)
        assert c.eviction is EvictionPolicy.LRU
        assert c.pin_experts == []
        assert c.lfu_decay == 0.95
        assert c.freq_threshold == 0.05
        assert c.freq_window == 100
        assert c.score_ema_alpha == 0.3

    def test_custom_values(self):
        c = CacheIR(
            capacity=64,
            eviction=EvictionPolicy.LFU,
            pin_experts=[0, 1, 2],
            lfu_decay=0.8,
        )
        assert c.capacity == 64
        assert c.eviction is EvictionPolicy.LFU
        assert c.pin_experts == [0, 1, 2]
        assert c.lfu_decay == 0.8

    def test_pin_experts_mutable_default(self):
        """Ensure default list is not shared between instances."""
        a = CacheIR(capacity=8)
        b = CacheIR(capacity=8)
        a.pin_experts.append(99)
        assert 99 not in b.pin_experts


# ---------------------------------------------------------------------------
# PrefetchIR
# ---------------------------------------------------------------------------

class TestPrefetchIR:
    def test_defaults(self):
        p = PrefetchIR()
        assert p.strategy is PrefetchStrategy.NONE
        assert p.lookahead == 1
        assert p.budget == 4

    def test_custom(self):
        p = PrefetchIR(strategy=PrefetchStrategy.HISTORY, history_window=200)
        assert p.history_window == 200


# ---------------------------------------------------------------------------
# ScheduleIR
# ---------------------------------------------------------------------------

class TestScheduleIR:
    def test_defaults(self):
        s = ScheduleIR()
        assert s.mode is ScheduleMode.GPU_ONLY
        assert s.overlap is True
        assert s.priority_routing is False

    def test_hybrid(self):
        s = ScheduleIR(mode=ScheduleMode.HYBRID, cpu_threshold_ms=30.0)
        assert s.cpu_threshold_ms == 30.0


# ---------------------------------------------------------------------------
# MonitorIR
# ---------------------------------------------------------------------------

class TestMonitorIR:
    def test_defaults(self):
        m = MonitorIR()
        assert m.metrics == ["hit_rate"]
        assert m.window == 100
        assert m.log_interval == 50


# ---------------------------------------------------------------------------
# PolicyIR
# ---------------------------------------------------------------------------

class TestPolicyIR:
    def test_minimal(self, minimal_ir):
        assert minimal_ir.name == "minimal"
        assert minimal_ir.cache.capacity == 16
        assert minimal_ir.prefetch.strategy is PrefetchStrategy.NONE
        assert minimal_ir.schedule.mode is ScheduleMode.GPU_ONLY
        assert minimal_ir.monitor is None

    def test_full(self, full_ir):
        assert full_ir.name == "full"
        assert full_ir.cache.eviction is EvictionPolicy.LFU
        assert full_ir.prefetch.strategy is PrefetchStrategy.AFFINITY
        assert full_ir.schedule.mode is ScheduleMode.HYBRID
        assert full_ir.monitor is not None
        assert "latency" in full_ir.monitor.metrics

    def test_equality(self):
        a = PolicyIR(name="x", cache=CacheIR(capacity=16))
        b = PolicyIR(name="x", cache=CacheIR(capacity=16))
        assert a == b

    def test_inequality(self):
        a = PolicyIR(name="x", cache=CacheIR(capacity=16))
        b = PolicyIR(name="x", cache=CacheIR(capacity=32))
        assert a != b
