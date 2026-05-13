"""Tests for the text-based MoE-PolicyLang DSL parser."""

from pathlib import Path

import pytest
from lark.exceptions import UnexpectedInput

from moe_policylang.errors import DSLError, ValidationError
from moe_policylang.ir import (
    EvictionPolicy,
    PrefetchStrategy,
    ScheduleMode,
)
from moe_policylang.parser import parse_file, parse_policies, parse_policy


EXAMPLES_DIR = Path(__file__).parent.parent / "examples"


# ---------------------------------------------------------------------------
# Basic parsing
# ---------------------------------------------------------------------------

class TestMinimalPolicy:
    def test_minimal_cache_only(self):
        src = """
            policy tiny {
                cache { capacity = 4 }
            }
        """
        ir = parse_policy(src)
        assert ir.name == "tiny"
        assert ir.cache.capacity == 4
        assert ir.cache.eviction == EvictionPolicy.LRU  # default
        assert ir.prefetch.strategy == PrefetchStrategy.NONE
        assert ir.schedule.mode == ScheduleMode.GPU_ONLY
        assert ir.monitor is None

    def test_empty_pin_list(self):
        src = """
            policy p {
                cache { capacity = 8  pin = [] }
            }
        """
        ir = parse_policy(src)
        assert ir.cache.pin_experts == []

    def test_comments_ignored(self):
        src = """
            # this is a top-level comment
            policy p {
                # inside the policy
                cache { capacity = 8 }  # trailing comment
            }
        """
        ir = parse_policy(src)
        assert ir.cache.capacity == 8


# ---------------------------------------------------------------------------
# Full policy with all blocks
# ---------------------------------------------------------------------------

class TestFullPolicy:
    def test_all_blocks_populated(self):
        src = """
            policy full {
                cache {
                    capacity = 32
                    eviction = lfu
                    pin = [0, 1, 2]
                    frequency_decay = 0.9
                }
                prefetch {
                    strategy = history
                    budget = 4
                    history_window = 100
                }
                schedule {
                    mode = hybrid
                    offload_threshold_ms = 40.0
                    overlap = true
                    priority_routing = false
                }
                monitor {
                    metrics = [hit_rate, latency]
                    window = 200
                    log_interval = 50
                }
            }
        """
        ir = parse_policy(src)
        assert ir.cache.capacity == 32
        assert ir.cache.eviction == EvictionPolicy.LFU
        assert ir.cache.pin_experts == [0, 1, 2]
        assert ir.cache.lfu_decay == pytest.approx(0.9)
        assert ir.prefetch.strategy == PrefetchStrategy.HISTORY
        assert ir.prefetch.budget == 4
        assert ir.prefetch.history_window == 100
        assert ir.schedule.mode == ScheduleMode.HYBRID
        assert ir.schedule.cpu_threshold_ms == pytest.approx(40.0)
        assert ir.schedule.overlap is True
        assert ir.schedule.priority_routing is False
        assert ir.monitor is not None
        assert ir.monitor.metrics == ["hit_rate", "latency"]
        assert ir.monitor.window == 200


# ---------------------------------------------------------------------------
# Multiple policies
# ---------------------------------------------------------------------------

class TestMultiplePolicies:
    def test_two_policies(self):
        src = """
            policy a {
                cache { capacity = 4 }
            }
            policy b {
                cache { capacity = 8  eviction = lfu }
            }
        """
        policies = parse_policies(src)
        assert len(policies) == 2
        assert policies[0].name == "a"
        assert policies[1].name == "b"
        assert policies[1].cache.eviction == EvictionPolicy.LFU

    def test_parse_policy_rejects_multi(self):
        src = """
            policy a { cache { capacity = 4 } }
            policy b { cache { capacity = 8 } }
        """
        with pytest.raises(DSLError, match="Expected exactly 1 policy"):
            parse_policy(src)

    def test_parse_policy_rejects_zero(self):
        with pytest.raises(UnexpectedInput):
            parse_policy("")


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrors:
    def test_missing_cache_block(self):
        src = """
            policy bad {
                schedule { mode = gpu_only }
            }
        """
        with pytest.raises(DSLError, match="cache block"):
            parse_policy(src)

    def test_duplicate_block(self):
        src = """
            policy bad {
                cache { capacity = 4 }
                cache { capacity = 8 }
            }
        """
        with pytest.raises(DSLError, match="Duplicate 'cache'"):
            parse_policy(src)

    def test_syntax_error(self):
        src = "policy p { cache { capacity }"
        with pytest.raises(UnexpectedInput):
            parse_policy(src)

    def test_unknown_eviction_raises(self):
        src = """
            policy p {
                cache { capacity = 4  eviction = bogus }
            }
        """
        with pytest.raises(DSLError, match="Unknown eviction 'bogus'"):
            parse_policy(src)

    def test_unknown_prefetch_strategy_raises(self):
        src = """
            policy p {
                cache { capacity = 4 }
                prefetch { strategy = bogus }
            }
        """
        with pytest.raises(DSLError, match="Unknown prefetch strategy 'bogus'"):
            parse_policy(src)

    def test_unknown_schedule_mode_raises(self):
        src = """
            policy p {
                cache { capacity = 4 }
                schedule { mode = bogus }
            }
        """
        with pytest.raises(DSLError, match="Unknown schedule mode 'bogus'"):
            parse_policy(src)

    def test_validation_error_propagates(self):
        # capacity=0 violates the >= 1 rule in the validator
        src = """
            policy bad { cache { capacity = 0 } }
        """
        with pytest.raises(ValidationError):
            parse_policy(src)


# ---------------------------------------------------------------------------
# Example files (the 3 policies from the proposal)
# ---------------------------------------------------------------------------

class TestExampleFiles:
    def test_lru_example(self):
        [ir] = parse_file(EXAMPLES_DIR / "lru_policy.moe")
        assert ir.name == "lru_baseline"
        assert ir.cache.eviction == EvictionPolicy.LRU
        assert ir.cache.capacity == 16

    def test_lfu_example(self):
        [ir] = parse_file(EXAMPLES_DIR / "lfu_policy.moe")
        assert ir.name == "lfu_with_prefetch"
        assert ir.cache.eviction == EvictionPolicy.LFU
        assert ir.prefetch.strategy == PrefetchStrategy.HISTORY

    def test_affinity_example(self):
        [ir] = parse_file(EXAMPLES_DIR / "affinity_policy.moe")
        assert ir.name == "affinity_hybrid"
        assert ir.cache.eviction == EvictionPolicy.SCORE
        assert ir.prefetch.strategy == PrefetchStrategy.AFFINITY
        assert ir.schedule.mode == ScheduleMode.HYBRID
        assert ir.cache.pin_experts == [0, 1]


# ---------------------------------------------------------------------------
# Equivalence with the Python eDSL
# ---------------------------------------------------------------------------

class TestEDSLEquivalence:
    """Parsed policies should produce identical IR to the Python eDSL."""

    def test_lru_matches_python_dsl(self):
        from moe_policylang import MoEPolicyLang

        sched = MoEPolicyLang()
        ir_py = (
            sched.build("lru_baseline")
            .cache(capacity=16, eviction="lru")
            .schedule(mode="gpu_only")
            .monitor(metrics=["hit_rate", "latency"], window=200, log_interval=50)
            .done()
        )
        [ir_parsed] = parse_file(EXAMPLES_DIR / "lru_policy.moe")

        assert ir_py.name == ir_parsed.name
        assert ir_py.cache == ir_parsed.cache
        assert ir_py.schedule == ir_parsed.schedule
        assert ir_py.monitor == ir_parsed.monitor


# ---------------------------------------------------------------------------
# Conditional expressions
# ---------------------------------------------------------------------------

class TestConditionalExpressions:
    """Test ``when ... else`` inline conditional parameters."""

    def test_eviction_conditional(self):
        src = """
            policy cond {
                cache { capacity = 16  eviction = lru : lfu when hit_rate > 0.5 }
            }
        """
        ir = parse_policy(src)
        # Default (first) value is used for the IR
        assert ir.cache.eviction == EvictionPolicy.LRU
        # Conditional desugars to an implicit adapt rule
        assert ir.adapt is not None
        assert len(ir.adapt.rules) == 1
        rule = ir.adapt.rules[0]
        assert rule.condition.metric == "hit_rate"
        assert rule.condition.op == ">"
        assert rule.condition.threshold == 0.5
        assert rule.action.param == "eviction"
        assert rule.action.value == "lfu"

    def test_capacity_conditional(self):
        src = """
            policy cond {
                cache { capacity = 16 : 32 when hit_rate > 0.8 }
            }
        """
        ir = parse_policy(src)
        assert ir.cache.capacity == 16  # default value
        assert ir.adapt is not None
        rule = ir.adapt.rules[0]
        assert rule.action.param == "capacity"
        assert rule.action.value == "32"

    def test_decay_conditional(self):
        src = """
            policy cond {
                cache {
                    capacity = 16
                    eviction = lfu
                    frequency_decay = 0.95 : 0.85 when hit_rate > 0.7
                }
            }
        """
        ir = parse_policy(src)
        assert ir.cache.lfu_decay == 0.95
        assert ir.adapt is not None
        rule = ir.adapt.rules[0]
        assert rule.action.param == "lfu_decay"
        assert rule.action.value == "0.85"

    def test_prefetch_strategy_conditional(self):
        src = """
            policy cond {
                cache { capacity = 16 }
                prefetch { strategy = history : lookahead when hit_rate > 0.6  budget = 4 }
            }
        """
        ir = parse_policy(src)
        assert ir.prefetch.strategy == PrefetchStrategy.HISTORY
        assert ir.adapt is not None
        rule = ir.adapt.rules[0]
        assert rule.action.param == "prefetch_strategy"
        assert rule.action.value == "lookahead"

    def test_prefetch_budget_conditional(self):
        src = """
            policy cond {
                cache { capacity = 16 }
                prefetch { strategy = history  budget = 4 : 8 when hit_rate > 0.8 }
            }
        """
        ir = parse_policy(src)
        assert ir.prefetch.budget == 4
        assert ir.adapt.rules[0].action.param == "prefetch_budget"

    def test_multiple_conditionals_merge(self):
        src = """
            policy cond {
                cache {
                    capacity = 16
                    eviction = lru : lfu when hit_rate > 0.5
                    frequency_decay = 0.95 : 0.85 when hit_rate > 0.7
                }
            }
        """
        ir = parse_policy(src)
        assert len(ir.adapt.rules) == 2

    def test_conditional_with_explicit_adapt(self):
        src = """
            policy cond {
                cache { capacity = 16  eviction = lru : lfu when hit_rate > 0.5 }
                adapt {
                    when eviction_rate > 0.3 { trigger memory_pressure }
                }
            }
        """
        ir = parse_policy(src)
        # Implicit rule + explicit rule
        assert len(ir.adapt.rules) == 2
        # Implicit comes first
        assert ir.adapt.rules[0].action.param == "eviction"
        assert ir.adapt.rules[1].action.param == "trigger"

    def test_conditional_example_file(self):
        policies = parse_file(EXAMPLES_DIR / "conditional_policy.moe")
        assert len(policies) == 1
        ir = policies[0]
        assert ir.name == "conditional_demo"
        assert ir.adapt is not None
        assert len(ir.adapt.rules) == 4


# ---------------------------------------------------------------------------
# Compilation integration
# ---------------------------------------------------------------------------

class TestCompilerIntegration:
    def test_parsed_policy_compiles(self):
        from moe_policylang import compile_policy

        [ir] = parse_file(EXAMPLES_DIR / "lfu_policy.moe")
        compiled = compile_policy(ir)
        assert compiled.name == "lfu_with_prefetch"
        assert compiled.cache is not None
        assert compiled.prefetcher is not None
        assert compiled.scheduler is not None
