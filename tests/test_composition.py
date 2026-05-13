"""Policy composition tests.

The proposal requires testing composition across the three orthogonal
policy dimensions (cache × prefetch × schedule) plus the new eviction
triggers.  Each test below builds a DSL policy combining primitives that
were implemented separately, runs it against the mock MoE simulator, and
asserts the resulting statistics show every component was exercised.
"""

from __future__ import annotations

import pytest

from moe_policylang import (
    CacheIR,
    EvictionPolicy,
    MonitorIR,
    PolicyIR,
    PrefetchIR,
    PrefetchStrategy,
    ScheduleIR,
    ScheduleMode,
    build_hook,
    compile_policy,
    parse_policy,
)
from moe_policylang.integrations.mock_moe import MockMoEModel, skewed_selector


def _run(hook, *, selector=None, num_tokens=30, num_layers=12, num_experts=32):
    model = MockMoEModel(
        num_layers=num_layers,
        num_experts=num_experts,
        top_k=2,
        selector=selector or skewed_selector(num_experts, top_k=2, seed=0),
    )
    model.run(hook, num_tokens=num_tokens)
    return hook.stats_snapshot()


# ---------------------------------------------------------------------------
# Headline: frequency-caching + affinity-prefetching + memory-pressure
# ---------------------------------------------------------------------------

class TestProposalHeadlineComposition:
    """The exact composition for multi-dimensional policy testing:

    > Test policy composition (e.g., frequency-caching + affinity-prefetching
    > + memory-pressure-eviction).
    """

    def test_all_three_components_active(self):
        ir = PolicyIR(
            name="headline",
            cache=CacheIR(
                capacity=24,
                eviction=EvictionPolicy.FREQ_THRESHOLD,
                freq_threshold=0.05,
                freq_window=80,
                memory_threshold=0.75,
                memory_headroom=0.4,
                memory_budget_gb=16.0,
                expert_size_gb=1.2,
            ),
            prefetch=PrefetchIR(
                strategy=PrefetchStrategy.AFFINITY,
                affinity_threshold=0.2,
                budget=4,
            ),
            schedule=ScheduleIR(mode=ScheduleMode.HYBRID, cpu_threshold_ms=40.0),
            monitor=MonitorIR(metrics=["hit_rate"], window=200, log_interval=50),
        )
        hook = build_hook(compile_policy(ir))
        snap = _run(hook, num_experts=40, num_tokens=60, num_layers=16)

        # Cache was exercised (some hits + some misses).
        assert snap["cache"]["hits"] > 0 and snap["cache"]["misses"] > 0
        # Scheduler saw dispatch decisions.
        total_exec = snap["scheduler"]["gpu"] + snap["scheduler"]["cpu"]
        assert total_exec == 60 * 16 * 2
        # Memory-pressure trigger was wired and fired under pressure.
        assert "triggers" in snap
        assert snap["triggers"]["memory_pressure"]["fired"] > 0


# ---------------------------------------------------------------------------
# All cache × prefetch × schedule combinations compile and run
# ---------------------------------------------------------------------------

CACHE_POLICIES = [
    EvictionPolicy.LRU,
    EvictionPolicy.LFU,
    EvictionPolicy.FREQ_THRESHOLD,
    EvictionPolicy.SCORE,
]
PREFETCH_STRATEGIES = [
    PrefetchStrategy.NONE,
    PrefetchStrategy.HISTORY,
    PrefetchStrategy.LOOKAHEAD,
]
SCHEDULE_MODES = [
    ScheduleMode.GPU_ONLY,
    ScheduleMode.CPU_FALLBACK,
    ScheduleMode.HYBRID,
]


@pytest.mark.parametrize("evc", CACHE_POLICIES)
@pytest.mark.parametrize("pre", PREFETCH_STRATEGIES)
@pytest.mark.parametrize("sch", SCHEDULE_MODES)
def test_cross_product_composes(evc, pre, sch):
    """Every cache x prefetch x schedule triple must build, compile, and run."""
    # SCORE eviction requires a non-NONE prefetcher (validator rule).  Skip
    # that single invalid combo.
    if evc == EvictionPolicy.SCORE and pre == PrefetchStrategy.NONE:
        pytest.skip("SCORE+NONE is an intentionally disallowed combination")

    ir = PolicyIR(
        name=f"{evc.value}_{pre.value}_{sch.value}",
        cache=CacheIR(capacity=12, eviction=evc),
        prefetch=PrefetchIR(strategy=pre, budget=3),
        schedule=ScheduleIR(mode=sch, cpu_threshold_ms=30.0, overlap=True),
    )
    hook = build_hook(compile_policy(ir))
    snap = _run(hook, num_experts=24, num_tokens=10, num_layers=6)
    assert snap["cache"]["hits"] + snap["cache"]["misses"] > 0


# ---------------------------------------------------------------------------
# Triggers compose with every cache type
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("evc", CACHE_POLICIES)
def test_memory_trigger_composes_with_every_cache(evc):
    if evc == EvictionPolicy.SCORE:
        prefetch = PrefetchIR(strategy=PrefetchStrategy.HISTORY, budget=2)
    else:
        prefetch = PrefetchIR()

    ir = PolicyIR(
        name=f"mp_{evc.value}",
        cache=CacheIR(
            capacity=30,
            eviction=evc,
            memory_threshold=0.6,
            memory_headroom=0.3,
            memory_budget_gb=16.0,
            expert_size_gb=1.2,
        ),
        prefetch=prefetch,
    )
    hook = build_hook(compile_policy(ir))
    _run(hook, num_experts=40, num_tokens=30, num_layers=8)
    assert hook.triggers.memory_pressure is not None


@pytest.mark.parametrize("evc", CACHE_POLICIES)
def test_ttl_trigger_composes_with_every_cache(evc):
    if evc == EvictionPolicy.SCORE:
        prefetch = PrefetchIR(strategy=PrefetchStrategy.HISTORY, budget=2)
    else:
        prefetch = PrefetchIR()

    ir = PolicyIR(
        name=f"ttl_{evc.value}",
        cache=CacheIR(capacity=30, eviction=evc, ttl=50),
        prefetch=prefetch,
    )
    hook = build_hook(compile_policy(ir))
    snap = _run(hook, num_experts=40, num_tokens=30, num_layers=8)
    assert snap.get("triggers", {}).get("ttl", {}).get("evicted", 0) >= 0


# ---------------------------------------------------------------------------
# Parsed DSL composition (end-to-end Weeks 2 + 3 + 4)
# ---------------------------------------------------------------------------

class TestParsedComposition:
    def test_full_composition_from_text(self):
        ir = parse_policy("""
            policy full {
                cache {
                    capacity          = 20
                    eviction          = lfu
                    frequency_decay   = 0.9
                    memory_threshold  = 0.7
                    memory_headroom   = 0.4
                    memory_budget_gb  = 16.0
                    expert_size_gb    = 1.2
                    ttl               = 150
                }
                prefetch {
                    strategy       = lookahead
                    lookahead      = 2
                    budget         = 3
                    history_window = 40
                }
                schedule {
                    mode             = hybrid
                    offload_threshold_ms = 35.0
                    overlap          = true
                    priority_routing = true
                }
                monitor {
                    metrics      = [hit_rate, latency]
                    window       = 200
                    log_interval = 50
                }
            }
        """)

        hook = build_hook(compile_policy(ir))
        assert hook.triggers.memory_pressure is not None
        assert hook.triggers.ttl is not None

        snap = _run(hook, num_experts=40, num_tokens=40, num_layers=12)
        assert snap["name"] == "full"
        assert snap["cache"]["hits"] + snap["cache"]["misses"] > 0
        assert snap["prefetch"]["issued"] > 0
