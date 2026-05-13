"""Week 3 correctness deliverable:
   DSL-generated policies produce identical dispatch to hand-coded baselines.

The proposal requires: "Verify correctness: DSL-generated LRU policy produces
identical outputs to hand-coded LRU baseline."

We run both the DSL-compiled hook and a self-contained reference
implementation over the *same* deterministic expert-selection trace and
assert their per-step dispatch sequences agree on:
  * which expert was accessed
  * whether it was a cache hit or miss
  * whether it executes on GPU or CPU

The reference implementations in ``moe_policylang.baselines`` are
intentionally built with plain dicts / OrderedDicts and share no code with
``moe_policylang.runtime.cache``, so agreement between the two is meaningful
evidence that the DSL→IR→compile→hook pipeline preserves semantics.
"""

from __future__ import annotations

import pytest

from moe_policylang import (
    CacheIR,
    EvictionPolicy,
    PolicyIR,
    ScheduleIR,
    ScheduleMode,
    build_hook,
    compile_policy,
    parse_policy,
)
from moe_policylang.baselines import HandCodedLRU, HandCodedLRUFallback
from moe_policylang.integrations.mock_moe import (
    MockMoEModel,
    deterministic_trace_selector,
    skewed_selector,
    uniform_selector,
)
from moe_policylang.runtime.scheduler import ExecutionDevice


# ---------------------------------------------------------------------------
# Workload helpers
# ---------------------------------------------------------------------------

def _collect_dsl_dispatch(hook, traces):
    """Run the DSL hook over a flat list of (layer_idx, experts) pairs."""
    records = []
    for layer_idx, experts in traces:
        plan = hook.on_layer(layer_idx=layer_idx, selected_experts=experts)
        for d in plan.dispatches:
            records.append((
                d.expert_id,
                d.cache_hit,
                d.device == ExecutionDevice.GPU,
            ))
    return records


def _collect_ref_dispatch(ref, traces):
    """Run the hand-coded reference over the same trace."""
    records = []
    for layer_idx, experts in traces:
        layer = ref.on_layer(layer_idx=layer_idx, selected_experts=experts)
        for d in layer:
            records.append((d.expert_id, d.cache_hit, d.on_gpu))
    return records


def _trace_from_selector(selector, num_tokens, num_layers):
    trace = []
    for t in range(num_tokens):
        for layer in range(num_layers):
            trace.append((layer, selector(t, layer)))
    return trace


# ---------------------------------------------------------------------------
# DSL LRU == hand-coded LRU
# ---------------------------------------------------------------------------

LRU_CAPACITIES = [2, 4, 8, 16]


class TestDSLLRUvsHandCoded:
    """Primary Week 3 correctness test."""

    @pytest.mark.parametrize("capacity", LRU_CAPACITIES)
    def test_uniform_workload(self, capacity):
        trace = _trace_from_selector(
            uniform_selector(num_experts=32, top_k=2, seed=42),
            num_tokens=50,
            num_layers=8,
        )

        ir = PolicyIR(
            name="dsl_lru",
            cache=CacheIR(capacity=capacity, eviction=EvictionPolicy.LRU),
            schedule=ScheduleIR(mode=ScheduleMode.GPU_ONLY),
        )
        dsl = build_hook(compile_policy(ir))
        ref = HandCodedLRU(capacity=capacity)

        dsl_records = _collect_dsl_dispatch(dsl, trace)
        ref_records = _collect_ref_dispatch(ref, trace)

        assert dsl_records == ref_records, (
            f"DSL and hand-coded LRU diverged at capacity={capacity}"
        )
        assert dsl.cache.stats.hits == ref.stats.hits
        assert dsl.cache.stats.misses == ref.stats.misses
        assert dsl.cache.stats.evictions == ref.stats.evictions

    @pytest.mark.parametrize("capacity", LRU_CAPACITIES)
    def test_skewed_workload(self, capacity):
        trace = _trace_from_selector(
            skewed_selector(num_experts=32, top_k=2, seed=7),
            num_tokens=100,
            num_layers=4,
        )

        ir = PolicyIR(
            name="dsl_lru",
            cache=CacheIR(capacity=capacity, eviction=EvictionPolicy.LRU),
            schedule=ScheduleIR(mode=ScheduleMode.GPU_ONLY),
        )
        dsl = build_hook(compile_policy(ir))
        ref = HandCodedLRU(capacity=capacity)

        assert _collect_dsl_dispatch(dsl, trace) == _collect_ref_dispatch(ref, trace)

    def test_deterministic_trace(self):
        """Small, hand-picked trace to exercise specific eviction patterns."""
        trace_events = [
            (0, [0, 1]),   # miss, miss
            (1, [2, 3]),   # miss, miss (evict 0, 1 at cap=2; cap=4 just fills)
            (2, [0, 1]),   # cap=2: miss, miss; cap=4: hit, hit
            (3, [4]),      # cap=2: miss (evict 2); cap=4: miss (evict 2)
            (4, [0, 1]),   # hits in both
        ]

        for capacity in (2, 4, 8):
            ir = PolicyIR(
                name="det",
                cache=CacheIR(capacity=capacity, eviction=EvictionPolicy.LRU),
                schedule=ScheduleIR(mode=ScheduleMode.GPU_ONLY),
            )
            dsl = build_hook(compile_policy(ir))
            ref = HandCodedLRU(capacity=capacity)
            assert _collect_dsl_dispatch(dsl, trace_events) == _collect_ref_dispatch(
                ref, trace_events
            ), f"Mismatch at capacity={capacity}"


# ---------------------------------------------------------------------------
# DSL LRU + CPU fallback == hand-coded LRU-fallback
# ---------------------------------------------------------------------------

class TestDSLLRUCpuFallbackVsHandCoded:
    @pytest.mark.parametrize("capacity", LRU_CAPACITIES)
    def test_matches_reference(self, capacity):
        trace = _trace_from_selector(
            uniform_selector(num_experts=16, top_k=2, seed=3),
            num_tokens=30,
            num_layers=6,
        )

        ir = PolicyIR(
            name="dsl_lru_cpu",
            cache=CacheIR(capacity=capacity, eviction=EvictionPolicy.LRU),
            schedule=ScheduleIR(mode=ScheduleMode.CPU_FALLBACK),
        )
        dsl = build_hook(compile_policy(ir))
        ref = HandCodedLRUFallback(capacity=capacity)

        dsl_records = _collect_dsl_dispatch(dsl, trace)
        ref_records = _collect_ref_dispatch(ref, trace)
        assert dsl_records == ref_records


# ---------------------------------------------------------------------------
# Parsed-DSL policy produces identical hook to programmatic policy
# ---------------------------------------------------------------------------

class TestTextDSLEquivalence:
    """Parsing a .moe file should yield the same behavior as building the IR
    in Python.  This chains Week 2 (parser) and Week 3 (hook) together."""

    def test_parsed_policy_matches_programmatic(self):
        parsed = parse_policy("""
            policy p {
                cache { capacity = 4  eviction = lru }
                schedule { mode = gpu_only }
            }
        """)
        programmatic = PolicyIR(
            name="p",
            cache=CacheIR(capacity=4, eviction=EvictionPolicy.LRU),
            schedule=ScheduleIR(mode=ScheduleMode.GPU_ONLY),
        )
        trace = _trace_from_selector(
            uniform_selector(num_experts=16, top_k=2, seed=11),
            num_tokens=20,
            num_layers=4,
        )

        hook_parsed = build_hook(compile_policy(parsed))
        hook_prog = build_hook(compile_policy(programmatic))

        assert _collect_dsl_dispatch(hook_parsed, trace) == _collect_dsl_dispatch(
            hook_prog, trace
        )


# ---------------------------------------------------------------------------
# Sanity: end-to-end mock-MoE run produces non-trivial stats
# ---------------------------------------------------------------------------

class TestEndToEndMock:
    def test_mock_model_drives_hook(self):
        ir = PolicyIR(
            name="smoke",
            cache=CacheIR(capacity=8, eviction=EvictionPolicy.LRU),
            schedule=ScheduleIR(mode=ScheduleMode.GPU_ONLY),
        )
        hook = build_hook(compile_policy(ir))
        model = MockMoEModel(
            num_layers=12, num_experts=32, top_k=2,
            selector=skewed_selector(num_experts=32, top_k=2, seed=1),
        )
        plans = model.run(hook, num_tokens=20)

        assert len(plans) == 12 * 20
        snap = hook.stats_snapshot()
        assert snap["cache"]["hits"] + snap["cache"]["misses"] == 12 * 20 * 2
        # Skewed workload + capacity 8 should yield a decent hit rate.
        assert snap["cache"]["hit_rate"] > 0.3
