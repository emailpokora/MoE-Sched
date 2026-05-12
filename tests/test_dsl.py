"""Tests for the DSL frontend: decorator API, builder API, error handling."""

import pytest

from moe_sched.dsl import MoESched, PolicyBuilder, FluentPolicyBuilder
from moe_sched.errors import DSLError, ValidationError
from moe_sched.ir import (
    EvictionPolicy,
    PolicyIR,
    PrefetchStrategy,
    ScheduleMode,
)


# ---------------------------------------------------------------------------
# Decorator API
# ---------------------------------------------------------------------------

class TestDecoratorAPI:
    def test_minimal_policy(self):
        sched = MoESched()

        @sched.policy
        def my_policy(p):
            p.cache(capacity=16)

        assert isinstance(my_policy, PolicyIR)
        assert my_policy.name == "my_policy"
        assert my_policy.cache.capacity == 16

    def test_full_policy(self):
        sched = MoESched()

        @sched.policy
        def full(p):
            p.cache(capacity=32, eviction=EvictionPolicy.LFU, pin=[0, 1])
            p.prefetch(strategy=PrefetchStrategy.AFFINITY, lookahead=2)
            p.schedule(mode=ScheduleMode.HYBRID, cpu_threshold_ms=40.0)
            p.monitor(metrics=["hit_rate", "latency"], window=200)

        assert full.cache.eviction is EvictionPolicy.LFU
        assert full.cache.pin_experts == [0, 1]
        assert full.prefetch.strategy is PrefetchStrategy.AFFINITY
        assert full.schedule.mode is ScheduleMode.HYBRID
        assert full.monitor is not None
        assert full.monitor.window == 200

    def test_string_enum_values(self):
        sched = MoESched()

        @sched.policy
        def p(b):
            b.cache(capacity=16, eviction="lfu")
            b.prefetch(strategy="affinity")
            b.schedule(mode="hybrid")

        assert p.cache.eviction is EvictionPolicy.LFU
        assert p.prefetch.strategy is PrefetchStrategy.AFFINITY
        assert p.schedule.mode is ScheduleMode.HYBRID

    def test_policy_registered_in_sched(self):
        sched = MoESched()

        @sched.policy
        def pol(p):
            p.cache(capacity=8)

        assert "pol" in sched.policies
        assert sched.policies["pol"] is pol

    def test_multiple_policies(self):
        sched = MoESched()

        @sched.policy
        def a(p):
            p.cache(capacity=8)

        @sched.policy
        def b(p):
            p.cache(capacity=64)

        assert len(sched.policies) == 2
        assert sched.policies["a"].cache.capacity == 8
        assert sched.policies["b"].cache.capacity == 64

    def test_defaults_applied(self):
        sched = MoESched()

        @sched.policy
        def d(p):
            p.cache(capacity=16)

        assert d.prefetch.strategy is PrefetchStrategy.NONE
        assert d.schedule.mode is ScheduleMode.GPU_ONLY
        assert d.monitor is None


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestDSLErrors:
    def test_missing_cache(self):
        sched = MoESched()
        with pytest.raises(DSLError, match="cache"):
            @sched.policy
            def no_cache(p):
                p.prefetch(strategy="affinity")

    def test_duplicate_cache(self):
        sched = MoESched()
        with pytest.raises(DSLError, match="Duplicate cache"):
            @sched.policy
            def dup(p):
                p.cache(capacity=16)
                p.cache(capacity=32)

    def test_duplicate_prefetch(self):
        sched = MoESched()
        with pytest.raises(DSLError, match="Duplicate prefetch"):
            @sched.policy
            def dup(p):
                p.cache(capacity=16)
                p.prefetch(strategy="none")
                p.prefetch(strategy="affinity")

    def test_duplicate_schedule(self):
        sched = MoESched()
        with pytest.raises(DSLError, match="Duplicate schedule"):
            @sched.policy
            def dup(p):
                p.cache(capacity=16)
                p.schedule(mode="gpu_only")
                p.schedule(mode="hybrid")

    def test_duplicate_monitor(self):
        sched = MoESched()
        with pytest.raises(DSLError, match="Duplicate monitor"):
            @sched.policy
            def dup(p):
                p.cache(capacity=16)
                p.monitor()
                p.monitor()

    def test_invalid_eviction_string(self):
        sched = MoESched()
        with pytest.raises(ValueError):
            @sched.policy
            def bad(p):
                p.cache(capacity=16, eviction="magic")

    def test_validation_runs_at_definition_time(self):
        sched = MoESched()
        with pytest.raises(ValidationError):
            @sched.policy
            def invalid(p):
                p.cache(capacity=0)


# ---------------------------------------------------------------------------
# Fluent builder API
# ---------------------------------------------------------------------------

class TestFluentBuilder:
    def test_basic(self):
        sched = MoESched()
        ir = (
            sched.build("test")
            .cache(capacity=32, eviction=EvictionPolicy.LRU)
            .prefetch(strategy=PrefetchStrategy.AFFINITY)
            .schedule(mode=ScheduleMode.GPU_ONLY)
            .done()
        )
        assert isinstance(ir, PolicyIR)
        assert ir.name == "test"
        assert ir.cache.capacity == 32

    def test_chaining_returns_builder(self):
        sched = MoESched()
        builder = sched.build("test").cache(capacity=16)
        assert isinstance(builder, FluentPolicyBuilder)

    def test_missing_cache_in_fluent(self):
        sched = MoESched()
        with pytest.raises(DSLError, match="cache"):
            sched.build("no_cache").done()

    def test_fluent_validation(self):
        sched = MoESched()
        with pytest.raises(ValidationError):
            sched.build("bad").cache(capacity=0).done()


# ---------------------------------------------------------------------------
# PolicyBuilder isolation
# ---------------------------------------------------------------------------

class TestPolicyBuilder:
    def test_builder_is_fresh_per_policy(self):
        """Each decorator call gets a fresh builder."""
        sched = MoESched()

        @sched.policy
        def a(p):
            p.cache(capacity=8, pin=[0])

        @sched.policy
        def b(p):
            p.cache(capacity=16)

        assert a.cache.pin_experts == [0]
        assert b.cache.pin_experts == []

    def test_register_prebuilt(self):
        from moe_sched.ir import CacheIR
        sched = MoESched()
        ir = PolicyIR(name="pre", cache=CacheIR(capacity=16))
        sched.register(ir)
        assert "pre" in sched.policies
