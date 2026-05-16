"""Tests for per_layer grammar, IR, parser, compiler, build_hook, and
adapt-rebalance integration.

Covers:
  - .moe file parsing of per_layer block
  - Python eDSL per_layer() method
  - Fluent builder per_layer()
  - Compiler storing per_layer IR on CompiledPolicy
  - build_hook creating PerLayerHook when per_layer is present
  - build_hook wrapping PerLayerHook with AdaptiveHook when adapt + per_layer
  - AdaptiveHook rebalance action dispatching to PerLayerHook._rebalance
  - .moe file parsing of rebalance adapt action
"""

from __future__ import annotations

import pytest

from moe_policylang import (
    AdaptAction,
    AdaptCondition,
    AdaptIR,
    AdaptRule,
    MoEPolicyLang,
    compile_policy,
    parse_policy,
)
from moe_policylang.adaptive import AdaptiveHook
from moe_policylang.ir import (
    AllocationSignal,
    CacheIR,
    EvictionPolicy,
    PerLayerIR,
    PolicyIR,
)
from moe_policylang.runtime.hooks import build_hook
from moe_policylang.runtime.per_layer import PerLayerHook


# ---------------------------------------------------------------------------
# .moe parsing: per_layer block
# ---------------------------------------------------------------------------

class TestPerLayerParsing:
    def test_parse_per_layer_block(self):
        ir = parse_policy("""
            policy epcb {
                cache { capacity = 16  eviction = lfu  frequency_decay = 0.9 }
                per_layer {
                    allocation = entropy
                    entropy_window = 200
                    min_capacity = 4
                    max_capacity = 48
                    rebalance_interval = 500
                    total_budget = 432
                }
            }
        """)
        assert ir.per_layer is not None
        assert ir.per_layer.allocation == AllocationSignal.ENTROPY
        assert ir.per_layer.entropy_window == 200
        assert ir.per_layer.min_capacity == 4
        assert ir.per_layer.max_capacity == 48
        assert ir.per_layer.rebalance_interval == 500
        assert ir.per_layer.total_budget == 432

    def test_parse_per_layer_minimal(self):
        """per_layer with only allocation should use defaults for the rest."""
        ir = parse_policy("""
            policy epcb_min {
                cache { capacity = 8 }
                per_layer { allocation = entropy }
            }
        """)
        assert ir.per_layer is not None
        assert ir.per_layer.allocation == AllocationSignal.ENTROPY
        assert ir.per_layer.entropy_window == 200  # default
        assert ir.per_layer.min_capacity == 2  # default

    def test_parse_per_layer_uniform_allocation(self):
        ir = parse_policy("""
            policy uniform_pl {
                cache { capacity = 8 }
                per_layer { allocation = uniform }
            }
        """)
        assert ir.per_layer.allocation == AllocationSignal.UNIFORM

    def test_no_per_layer_is_none(self):
        ir = parse_policy("""
            policy plain {
                cache { capacity = 8 }
            }
        """)
        assert ir.per_layer is None


# ---------------------------------------------------------------------------
# .moe parsing: rebalance adapt action
# ---------------------------------------------------------------------------

class TestRebalanceParsing:
    def test_parse_rebalance_in_adapt(self):
        ir = parse_policy("""
            policy adaptive_epcb {
                cache { capacity = 8  eviction = lfu  frequency_decay = 0.9 }
                per_layer { allocation = entropy }
                adapt {
                    when hit_rate < 0.3 for 50 accesses
                        { rebalance entropy }
                }
            }
        """)
        assert ir.per_layer is not None
        assert ir.adapt is not None
        assert len(ir.adapt.rules) == 1
        rule = ir.adapt.rules[0]
        assert rule.action.param == "rebalance"
        assert rule.action.value == "entropy"

    def test_parse_rebalance_with_other_rules(self):
        ir = parse_policy("""
            policy multi_rule {
                cache { capacity = 8  eviction = lfu  frequency_decay = 0.9 }
                per_layer { allocation = entropy }
                adapt {
                    when hit_rate < 0.4 for 100 accesses { eviction = lru }
                    when hit_rate < 0.3 { rebalance entropy }
                }
            }
        """)
        assert len(ir.adapt.rules) == 2
        assert ir.adapt.rules[0].action.param == "eviction"
        assert ir.adapt.rules[1].action.param == "rebalance"


# ---------------------------------------------------------------------------
# Python eDSL: per_layer()
# ---------------------------------------------------------------------------

class TestPerLayerEDSL:
    def test_decorator_per_layer(self):
        sched = MoEPolicyLang()

        @sched.policy
        def epcb_test(p):
            p.cache(capacity=16, eviction="lfu", lfu_decay=0.9)
            p.per_layer(
                allocation="entropy",
                entropy_window=200,
                min_capacity=4,
                max_capacity=48,
                total_budget=432,
            )

        ir = sched.policies["epcb_test"]
        assert ir.per_layer is not None
        assert ir.per_layer.allocation == AllocationSignal.ENTROPY
        assert ir.per_layer.total_budget == 432

    def test_fluent_per_layer(self):
        sched = MoEPolicyLang()
        ir = (
            sched.build("fluent_epcb")
            .cache(capacity=16, eviction="lfu", lfu_decay=0.9)
            .per_layer(allocation="entropy", total_budget=432)
            .done()
        )
        assert ir.per_layer is not None
        assert ir.per_layer.allocation == AllocationSignal.ENTROPY

    def test_duplicate_per_layer_raises(self):
        from moe_policylang.errors import DSLError
        sched = MoEPolicyLang()
        with pytest.raises(DSLError, match="Duplicate per_layer"):
            @sched.policy
            def dup(p):
                p.cache(capacity=16)
                p.per_layer()
                p.per_layer()


# ---------------------------------------------------------------------------
# Compiler: per_layer IR stored on CompiledPolicy
# ---------------------------------------------------------------------------

class TestPerLayerCompiler:
    def test_compiled_has_per_layer(self):
        ir = PolicyIR(
            name="comp_test",
            cache=CacheIR(capacity=16, eviction=EvictionPolicy.LFU, lfu_decay=0.9),
            per_layer=PerLayerIR(
                allocation=AllocationSignal.ENTROPY,
                total_budget=432,
            ),
        )
        compiled = compile_policy(ir)
        assert hasattr(compiled, "_per_layer_ir")
        assert compiled._per_layer_ir is not None
        assert compiled._per_layer_ir.allocation == AllocationSignal.ENTROPY

    def test_compiled_without_per_layer(self):
        ir = PolicyIR(
            name="no_pl",
            cache=CacheIR(capacity=16),
        )
        compiled = compile_policy(ir)
        pl = getattr(compiled, "_per_layer_ir", None)
        assert pl is None


# ---------------------------------------------------------------------------
# build_hook: PerLayerHook creation
# ---------------------------------------------------------------------------

class TestBuildHookPerLayer:
    def test_build_hook_creates_per_layer_hook(self):
        ir = PolicyIR(
            name="bh_test",
            cache=CacheIR(capacity=8, eviction=EvictionPolicy.LFU, lfu_decay=0.9),
            per_layer=PerLayerIR(allocation=AllocationSignal.ENTROPY, total_budget=32),
        )
        compiled = compile_policy(ir)
        hook = build_hook(compiled, num_layers=4, num_experts=8)
        assert isinstance(hook, PerLayerHook)

    def test_build_hook_per_layer_dispatches(self):
        ir = PolicyIR(
            name="dispatch_test",
            cache=CacheIR(capacity=4, eviction=EvictionPolicy.LFU, lfu_decay=0.9),
            per_layer=PerLayerIR(allocation=AllocationSignal.ENTROPY, total_budget=16),
        )
        compiled = compile_policy(ir)
        hook = build_hook(compiled, num_layers=4, num_experts=8)
        plan = hook.on_layer(0, [0, 1, 2])
        assert len(plan.dispatches) == 3
        assert plan.layer_idx == 0

    def test_build_hook_per_layer_requires_num_layers(self):
        ir = PolicyIR(
            name="missing_nl",
            cache=CacheIR(capacity=8),
            per_layer=PerLayerIR(allocation=AllocationSignal.ENTROPY, total_budget=32),
        )
        compiled = compile_policy(ir)
        with pytest.raises((TypeError, ValueError)):
            build_hook(compiled)

    def test_build_hook_without_per_layer_is_standard(self):
        ir = PolicyIR(
            name="standard",
            cache=CacheIR(capacity=8),
        )
        compiled = compile_policy(ir)
        hook = build_hook(compiled)
        assert not isinstance(hook, PerLayerHook)
        assert not isinstance(hook, AdaptiveHook)


# ---------------------------------------------------------------------------
# build_hook: adapt + per_layer → AdaptiveHook wrapping PerLayerHook
# ---------------------------------------------------------------------------

class TestBuildHookAdaptPerLayer:
    def test_adapt_wraps_per_layer(self):
        ir = PolicyIR(
            name="adapt_pl",
            cache=CacheIR(capacity=8, eviction=EvictionPolicy.LFU, lfu_decay=0.9),
            per_layer=PerLayerIR(allocation=AllocationSignal.ENTROPY, total_budget=32),
            adapt=AdaptIR(rules=[
                AdaptRule(
                    condition=AdaptCondition("hit_rate", "<", 0.5, window=1),
                    action=AdaptAction(param="rebalance", value="entropy"),
                    cooldown=50,
                ),
            ]),
        )
        compiled = compile_policy(ir)
        hook = build_hook(compiled, num_layers=4, num_experts=8)
        assert isinstance(hook, AdaptiveHook)
        assert isinstance(hook.base, PerLayerHook)

    def test_adapt_per_layer_dispatches(self):
        ir = PolicyIR(
            name="adapt_pl_dispatch",
            cache=CacheIR(capacity=4, eviction=EvictionPolicy.LFU, lfu_decay=0.9),
            per_layer=PerLayerIR(allocation=AllocationSignal.ENTROPY, total_budget=16),
            adapt=AdaptIR(rules=[
                AdaptRule(
                    condition=AdaptCondition("hit_rate", "<", 0.5, window=1),
                    action=AdaptAction(param="rebalance", value="entropy"),
                    cooldown=10,
                ),
            ]),
        )
        compiled = compile_policy(ir)
        hook = build_hook(compiled, num_layers=4, num_experts=8)

        # Run enough dispatches to trigger the adapt rule
        for i in range(30):
            plan = hook.on_layer(i % 4, [i % 8])
            assert len(plan.dispatches) == 1

        stats = hook.stats_snapshot()
        assert stats["adapt"]["adaptations"] >= 1


# ---------------------------------------------------------------------------
# AdaptiveHook: rebalance action
# ---------------------------------------------------------------------------

class TestAdaptRebalance:
    def _make_adaptive_per_layer_hook(self, cooldown=10):
        base_ir = PolicyIR(
            name="rebal_test",
            cache=CacheIR(capacity=4, eviction=EvictionPolicy.LFU, lfu_decay=0.9),
        )
        from moe_policylang.runtime.per_layer import PerLayerConfig
        config = PerLayerConfig(
            entropy_window=20,
            min_capacity=2,
            max_capacity=16,
            rebalance_interval=999999,  # disabled — adapt handles it
            total_budget=16,
        )
        per_layer_hook = PerLayerHook(base_ir, num_layers=4, num_experts=8, config=config)

        adapt_ir = AdaptIR(rules=[
            AdaptRule(
                condition=AdaptCondition("hit_rate", "<", 0.99, window=1),
                action=AdaptAction(param="rebalance", value="entropy"),
                cooldown=cooldown,
            ),
        ])
        return AdaptiveHook(per_layer_hook, adapt_ir, base_ir, compile_policy)

    def test_rebalance_fires(self):
        hook = self._make_adaptive_per_layer_hook(cooldown=5)
        # Run enough dispatches for the rule to fire
        for i in range(20):
            hook.on_layer(i % 4, [i % 8])
        stats = hook.stats_snapshot()
        assert stats["adapt"]["adaptations"] >= 1

    def test_rebalance_without_per_layer_skipped(self):
        """rebalance on a non-PerLayerHook base should be skipped."""
        from moe_policylang.runtime.hooks import PolicyHook
        base_ir = PolicyIR(
            name="plain_hook",
            cache=CacheIR(capacity=4, eviction=EvictionPolicy.LFU, lfu_decay=0.9),
        )
        compiled = compile_policy(base_ir)
        plain_hook = PolicyHook(compiled)

        adapt_ir = AdaptIR(rules=[
            AdaptRule(
                condition=AdaptCondition("hit_rate", "<", 0.99, window=1),
                action=AdaptAction(param="rebalance", value="entropy"),
                cooldown=5,
            ),
        ])
        hook = AdaptiveHook(plain_hook, adapt_ir, base_ir, compile_policy)

        for i in range(20):
            hook.on_layer(0, [i % 8])

        stats = hook.stats_snapshot()
        # Should have skipped, not fired
        assert stats["adapt"]["adaptations"] == 0
        assert stats["adapt"]["skipped"] >= 1


# ---------------------------------------------------------------------------
# PerLayerIR dataclass
# ---------------------------------------------------------------------------

class TestPerLayerIR:
    def test_defaults(self):
        pl = PerLayerIR()
        assert pl.allocation == AllocationSignal.ENTROPY
        assert pl.entropy_window == 200
        assert pl.min_capacity == 2
        assert pl.max_capacity == 64
        assert pl.rebalance_interval == 500
        assert pl.total_budget is None

    def test_custom_values(self):
        pl = PerLayerIR(
            allocation=AllocationSignal.UNIFORM,
            entropy_window=100,
            min_capacity=4,
            max_capacity=48,
            rebalance_interval=1000,
            total_budget=432,
        )
        assert pl.allocation == AllocationSignal.UNIFORM
        assert pl.total_budget == 432


# ---------------------------------------------------------------------------
# Round-trip: .moe → parse → compile → build_hook → dispatch
# ---------------------------------------------------------------------------

class TestPerLayerRoundTrip:
    def test_moe_file_to_dispatch(self):
        ir = parse_policy("""
            policy roundtrip {
                cache { capacity = 4  eviction = lfu  frequency_decay = 0.9 }
                per_layer {
                    allocation = entropy
                    entropy_window = 20
                    min_capacity = 2
                    max_capacity = 16
                    rebalance_interval = 10
                    total_budget = 16
                }
            }
        """)
        compiled = compile_policy(ir)
        hook = build_hook(compiled, num_layers=4, num_experts=8)
        assert isinstance(hook, PerLayerHook)

        # Run dispatches through rebalance cycle
        for i in range(30):
            plan = hook.on_layer(i % 4, [i % 8])
            assert plan.layer_idx == i % 4

        stats = hook.stats_snapshot()
        assert stats["steps"] == 30
        assert stats["cache"]["hits"] + stats["cache"]["misses"] == 30

    def test_moe_file_adapt_rebalance_round_trip(self):
        ir = parse_policy("""
            policy adapt_roundtrip {
                cache { capacity = 4  eviction = lfu  frequency_decay = 0.9 }
                per_layer {
                    allocation = entropy
                    entropy_window = 20
                    min_capacity = 2
                    max_capacity = 16
                    total_budget = 16
                }
                adapt {
                    when hit_rate < 0.99 { rebalance entropy }
                }
            }
        """)
        compiled = compile_policy(ir)
        hook = build_hook(compiled, num_layers=4, num_experts=8)
        assert isinstance(hook, AdaptiveHook)
        assert isinstance(hook.base, PerLayerHook)

        for i in range(30):
            hook.on_layer(i % 4, [i % 8])

        stats = hook.stats_snapshot()
        assert stats["adapt"]["adaptations"] >= 1
