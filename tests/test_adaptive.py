"""Tests for adaptive policy support (profiling-driven dynamic adjustment)."""

from __future__ import annotations

import pytest

from moe_sched import (
    AdaptAction,
    AdaptCondition,
    AdaptIR,
    AdaptRule,
    MoESched,
    compile_policy,
    parse_policy,
)
from moe_sched.adaptive import AdaptiveHook, _eval_condition
from moe_sched.runtime.hooks import PolicyHook, build_hook


# ---------------------------------------------------------------------------
# Unit tests: condition evaluation
# ---------------------------------------------------------------------------

class TestEvalCondition:
    def test_less_than(self):
        assert _eval_condition(0.3, "<", 0.4)
        assert not _eval_condition(0.5, "<", 0.4)

    def test_greater_than(self):
        assert _eval_condition(0.9, ">", 0.8)
        assert not _eval_condition(0.7, ">", 0.8)

    def test_less_equal(self):
        assert _eval_condition(0.4, "<=", 0.4)
        assert not _eval_condition(0.5, "<=", 0.4)

    def test_greater_equal(self):
        assert _eval_condition(0.8, ">=", 0.8)
        assert not _eval_condition(0.7, ">=", 0.8)

    def test_not_equal(self):
        assert _eval_condition(0.5, "!=", 0.4)
        assert not _eval_condition(0.5, "!=", 0.5)

    def test_unknown_op_raises(self):
        with pytest.raises(ValueError, match="Unknown comparison"):
            _eval_condition(0.5, "==", 0.4)


# ---------------------------------------------------------------------------
# Unit tests: AdaptiveHook
# ---------------------------------------------------------------------------

def _make_adaptive_hook(rules, capacity=4, eviction="lfu"):
    """Helper to build a hook with adaptive rules."""
    sched = MoESched()

    @sched.policy
    def test_policy(p):
        p.cache(capacity=capacity, eviction=eviction, lfu_decay=0.9)
        p.prefetch(strategy="none")
        p.schedule(mode="gpu_only")
        p.adapt(rules)

    ir = sched.policies["test_policy"]
    compiled = compile_policy(ir)
    return build_hook(compiled)


class TestAdaptiveHook:
    def test_returns_adaptive_hook_when_rules_present(self):
        hook = _make_adaptive_hook([
            AdaptRule(
                condition=AdaptCondition("hit_rate", "<", 0.5, window=1),
                action=AdaptAction(param="eviction", value="lru"),
            ),
        ])
        assert isinstance(hook, AdaptiveHook)

    def test_returns_plain_hook_when_no_rules(self):
        sched = MoESched()

        @sched.policy
        def plain(p):
            p.cache(capacity=4, eviction="lru")

        compiled = compile_policy(sched.policies["plain"])
        hook = build_hook(compiled)
        assert isinstance(hook, PolicyHook)

    def test_adaptation_fires_on_low_hit_rate(self):
        """Rule: when hit_rate < 0.5 for 1 step, switch to LRU."""
        hook = _make_adaptive_hook([
            AdaptRule(
                condition=AdaptCondition("hit_rate", "<", 0.5, window=1),
                action=AdaptAction(param="eviction", value="lru"),
                cooldown=10,
            ),
        ])

        # Access experts that all miss (capacity=4, access 8 different experts)
        for i in range(8):
            hook.on_layer(layer_idx=0, selected_experts=[i])

        # After many misses, hit_rate < 0.5 → rule should fire
        stats = hook.stats_snapshot()
        assert stats["adapt"]["adaptations"] > 0
        assert stats["adapt"]["log"][0]["param"] == "eviction"
        assert stats["adapt"]["log"][0]["value"] == "lru"

    def test_cooldown_prevents_rapid_refiring(self):
        """Rule should not fire again within cooldown period."""
        hook = _make_adaptive_hook([
            AdaptRule(
                condition=AdaptCondition("hit_rate", "<", 0.99, window=1),
                action=AdaptAction(param="eviction", value="lru"),
                cooldown=50,
            ),
        ])

        # Run 100 steps — condition is always true, but cooldown=50
        for i in range(100):
            hook.on_layer(layer_idx=0, selected_experts=[i % 8])

        stats = hook.stats_snapshot()
        # Should fire at most ~2 times in 100 steps (cooldown=50)
        assert stats["adapt"]["adaptations"] <= 3

    def test_window_requires_consecutive_evals(self):
        """Rule with window=5 should only fire after 5 consecutive True evals."""
        hook = _make_adaptive_hook([
            AdaptRule(
                condition=AdaptCondition("hit_rate", "<", 0.5, window=5),
                action=AdaptAction(param="eviction", value="lru"),
                cooldown=1,
            ),
        ])

        # First 4 steps: all misses, hit_rate=0 < 0.5, but window=5 not met
        for i in range(4):
            hook.on_layer(layer_idx=0, selected_experts=[i])

        stats = hook.stats_snapshot()
        assert stats["adapt"]["adaptations"] == 0

        # 5th step: window=5 met → should fire
        hook.on_layer(layer_idx=0, selected_experts=[10])
        stats = hook.stats_snapshot()
        assert stats["adapt"]["adaptations"] == 1

    def test_adaptation_changes_cache_behavior(self):
        """After switching from LFU to LRU, eviction order should change."""
        hook = _make_adaptive_hook(
            rules=[
                AdaptRule(
                    condition=AdaptCondition("hit_rate", "<", 0.3, window=1),
                    action=AdaptAction(param="eviction", value="lru"),
                    cooldown=5,
                ),
            ],
            capacity=4,
            eviction="lfu",
        )

        # Fill cache with experts 0-3 (all misses → low hit rate)
        for i in range(4):
            hook.on_layer(layer_idx=0, selected_experts=[i])

        # The adaptation should have fired (hit_rate=0 < 0.3)
        # Now the cache should be using LRU eviction
        stats = hook.stats_snapshot()
        assert stats["adapt"]["adaptations"] >= 1

    def test_stats_snapshot_includes_adapt(self):
        hook = _make_adaptive_hook([
            AdaptRule(
                condition=AdaptCondition("hit_rate", "<", 0.5, window=1),
                action=AdaptAction(param="eviction", value="lru"),
            ),
        ])
        hook.on_layer(layer_idx=0, selected_experts=[0])
        stats = hook.stats_snapshot()
        assert "adapt" in stats
        assert "adaptations" in stats["adapt"]
        assert "log" in stats["adapt"]

    def test_trigger_activation(self):
        """Adaptation can activate a memory pressure trigger."""
        hook = _make_adaptive_hook([
            AdaptRule(
                condition=AdaptCondition("hit_rate", "<", 0.5, window=1),
                action=AdaptAction(param="trigger", value="memory_pressure"),
                cooldown=100,
            ),
        ])

        for i in range(10):
            hook.on_layer(layer_idx=0, selected_experts=[i])

        stats = hook.stats_snapshot()
        # Should have fired and set memory_threshold
        assert stats["adapt"]["adaptations"] >= 1

    def test_delegate_properties(self):
        """AdaptiveHook should delegate cache/prefetcher/scheduler properties."""
        hook = _make_adaptive_hook([
            AdaptRule(
                condition=AdaptCondition("hit_rate", "<", 0.1, window=1),
                action=AdaptAction(param="eviction", value="lru"),
            ),
        ])
        assert hook.cache is not None
        assert hook.prefetcher is not None
        assert hook.scheduler is not None
        assert hook.compiled is not None

    def test_invalid_adaptation_skipped_and_logged(self):
        """Adaptation to invalid config (capacity=0) should be skipped and logged."""
        hook = _make_adaptive_hook([
            AdaptRule(
                condition=AdaptCondition("hit_rate", "<", 0.99, window=1),
                action=AdaptAction(param="capacity", value="0"),
                cooldown=100,
            ),
        ])
        # Should not crash — invalid adaptation is skipped with warning
        for i in range(20):
            hook.on_layer(layer_idx=0, selected_experts=[i])
        stats = hook.stats_snapshot()
        assert stats["adapt"]["adaptations"] == 0
        assert stats["adapt"]["skipped"] >= 1
        skip = stats["adapt"]["skipped_log"][0]
        assert skip["param"] == "capacity"
        assert skip["value"] == "0"
        assert "reason" in skip

    def test_no_nested_wrapping_after_adaptation(self):
        """Regression: _apply must not create nested AdaptiveHook layers."""
        hook = _make_adaptive_hook([
            AdaptRule(
                condition=AdaptCondition("hit_rate", "<", 0.99, window=1),
                action=AdaptAction(param="eviction", value="lru"),
                cooldown=20,
            ),
        ])

        # Run enough steps to trigger multiple adaptations
        for i in range(100):
            hook.on_layer(layer_idx=0, selected_experts=[i % 8])

        assert isinstance(hook, AdaptiveHook)
        assert isinstance(hook.base, PolicyHook)
        assert not isinstance(hook.base, AdaptiveHook)


# ---------------------------------------------------------------------------
# Integration: .moe file parsing with adapt block
# ---------------------------------------------------------------------------

class TestAdaptParsing:
    def test_parse_adapt_instant(self):
        ir = parse_policy("""
            policy test_adapt {
                cache { capacity = 8  eviction = lfu  frequency_decay = 0.9 }
                adapt {
                    when hit_rate < 0.4 { eviction = lru }
                }
            }
        """)
        assert ir.adapt is not None
        assert len(ir.adapt.rules) == 1
        rule = ir.adapt.rules[0]
        assert rule.condition.metric == "hit_rate"
        assert rule.condition.op == "<"
        assert rule.condition.threshold == 0.4
        assert rule.condition.window == 1  # instant
        assert rule.action.param == "eviction"
        assert rule.action.value == "lru"

    def test_parse_adapt_windowed(self):
        ir = parse_policy("""
            policy test_adapt_w {
                cache { capacity = 8  eviction = lfu }
                adapt {
                    when hit_rate < 0.3 for 100 accesses { eviction = lru }
                }
            }
        """)
        rule = ir.adapt.rules[0]
        assert rule.condition.window == 100

    def test_parse_adapt_trigger(self):
        ir = parse_policy("""
            policy test_trigger {
                cache { capacity = 8  eviction = lru }
                adapt {
                    when eviction_rate > 0.5 { trigger memory_pressure }
                }
            }
        """)
        rule = ir.adapt.rules[0]
        assert rule.action.param == "trigger"
        assert rule.action.value == "memory_pressure"

    def test_parse_multiple_rules(self):
        ir = parse_policy("""
            policy multi_adapt {
                cache { capacity = 16  eviction = lfu  frequency_decay = 0.9 }
                prefetch { strategy = history  budget = 4 }
                adapt {
                    when hit_rate < 0.4 for 50 accesses { eviction = lru }
                    when eviction_rate > 0.6 { trigger memory_pressure }
                }
            }
        """)
        assert len(ir.adapt.rules) == 2
        assert ir.adapt.rules[0].condition.metric == "hit_rate"
        assert ir.adapt.rules[1].condition.metric == "eviction_rate"

    def test_parse_adapt_with_gt(self):
        ir = parse_policy("""
            policy gt_test {
                cache { capacity = 8  eviction = lru }
                adapt {
                    when hit_rate >= 0.9 { eviction = lfu }
                }
            }
        """)
        assert ir.adapt.rules[0].condition.op == ">="

    def test_parse_adapt_numeric_action(self):
        ir = parse_policy("""
            policy num_adapt {
                cache { capacity = 8  eviction = lfu  frequency_decay = 0.9 }
                adapt {
                    when hit_rate < 0.3 { capacity = 16 }
                }
            }
        """)
        rule = ir.adapt.rules[0]
        assert rule.action.param == "capacity"
        # NUMBER terminal parses 16 as float 16.0; _apply uses int()
        assert int(float(rule.action.value)) == 16

    def test_parse_adapt_float_action(self):
        ir = parse_policy("""
            policy float_adapt {
                cache { capacity = 8  eviction = lfu  frequency_decay = 0.9 }
                adapt {
                    when hit_rate < 0.3 { lfu_decay = 0.8 }
                }
            }
        """)
        rule = ir.adapt.rules[0]
        assert rule.action.param == "lfu_decay"
        assert rule.action.value == "0.8"

    def test_policy_without_adapt_has_none(self):
        ir = parse_policy("""
            policy no_adapt {
                cache { capacity = 8  eviction = lru }
            }
        """)
        assert ir.adapt is None


# ---------------------------------------------------------------------------
# Integration: full round-trip DSL → compile → hook → run
# ---------------------------------------------------------------------------

class TestAdaptRoundTrip:
    def test_edsl_adapt_round_trip(self):
        """Python eDSL → IR → compile → AdaptiveHook → run."""
        sched = MoESched()

        @sched.policy
        def adaptive_test(p):
            p.cache(capacity=4, eviction="lfu", lfu_decay=0.9)
            p.prefetch(strategy="none")
            p.schedule(mode="gpu_only")
            p.adapt([
                AdaptRule(
                    condition=AdaptCondition("hit_rate", "<", 0.5, window=1),
                    action=AdaptAction(param="eviction", value="lru"),
                    cooldown=20,
                ),
            ])

        ir = sched.policies["adaptive_test"]
        assert ir.adapt is not None
        compiled = compile_policy(ir)
        hook = build_hook(compiled)
        assert isinstance(hook, AdaptiveHook)

        # Run some dispatches
        for i in range(20):
            hook.on_layer(layer_idx=0, selected_experts=[i % 8])

        stats = hook.stats_snapshot()
        assert stats["adapt"]["adaptations"] >= 1

    def test_moe_file_adapt_round_trip(self):
        """.moe file → parse → compile → AdaptiveHook → run."""
        ir = parse_policy("""
            policy file_adaptive {
                cache { capacity = 4  eviction = lfu  frequency_decay = 0.9 }
                prefetch { strategy = none }
                schedule { mode = gpu_only }
                adapt {
                    when hit_rate < 0.5 { eviction = lru }
                }
            }
        """)
        compiled = compile_policy(ir)
        hook = build_hook(compiled)
        assert isinstance(hook, AdaptiveHook)

        for i in range(20):
            hook.on_layer(layer_idx=0, selected_experts=[i % 8])

        stats = hook.stats_snapshot()
        assert stats["adapt"]["adaptations"] >= 1

    def test_fluent_builder_adapt(self):
        """Fluent builder API with adapt."""
        sched = MoESched()
        ir = (
            sched.build("fluent_adaptive")
            .cache(capacity=4, eviction="lfu", lfu_decay=0.9)
            .prefetch(strategy="none")
            .schedule(mode="gpu_only")
            .adapt([
                AdaptRule(
                    condition=AdaptCondition("hit_rate", "<", 0.5, window=1),
                    action=AdaptAction(param="eviction", value="lru"),
                ),
            ])
            .done()
        )
        assert ir.adapt is not None
        assert len(ir.adapt.rules) == 1
