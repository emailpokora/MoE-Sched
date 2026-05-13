"""Runtime hook that wraps a CompiledPolicy and intercepts MoE layer forwards.

This is the main Week 3 deliverable: the "code generator" side of the DSL.
The compiler already assembles Cache/Prefetcher/Scheduler/Monitor components;
the hook orchestrates them into a single callable that an inference engine
invokes on every MoE layer's expert selection.

Contract (per layer forward):
    plan = hook.on_layer(layer_idx, selected_experts, scores=None)

The returned ``DispatchPlan`` tells the engine, per expert:
  * whether the expert is a cache hit (execute on GPU)
  * whether a CPU->GPU transfer is required (cache miss, SCHED=GPU_ONLY/HYBRID-fast)
  * whether to execute on CPU instead (cache miss, SCHED=CPU_FALLBACK/HYBRID-slow)
It also reports which experts were issued as async prefetches for the next
layer and records per-access monitor samples.

This module is **pure Python and framework-agnostic**.  The actual HuggingFace
glue code in ``moe_sched.integrations.huggingface`` converts a DispatchPlan
into torch tensor placement calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from moe_sched.compiler import CompiledPolicy
from moe_sched.runtime.scheduler import ExecutionDevice


# ---------------------------------------------------------------------------
# Dispatch plan
# ---------------------------------------------------------------------------

@dataclass
class ExpertDispatch:
    """Per-expert placement decision for a single layer forward."""

    expert_id: int
    device: ExecutionDevice
    cache_hit: bool
    # True if this expert required a CPU->GPU transfer this step.  When
    # device == CPU this is always False (we don't transfer to execute on CPU).
    transferred: bool = False


@dataclass
class DispatchPlan:
    """Outcome of one ``on_layer`` call.

    Attributes:
        layer_idx: Index of the layer whose experts were selected.
        dispatches: One ``ExpertDispatch`` per selected expert, in the same
            order the caller provided.
        prefetched: Expert IDs the prefetcher issued for subsequent layers.
    """

    layer_idx: int
    dispatches: List[ExpertDispatch] = field(default_factory=list)
    prefetched: List[int] = field(default_factory=list)

    # -- convenience views --------------------------------------------------

    @property
    def hits(self) -> int:
        return sum(1 for d in self.dispatches if d.cache_hit)

    @property
    def misses(self) -> int:
        return sum(1 for d in self.dispatches if not d.cache_hit)

    @property
    def gpu_executions(self) -> int:
        return sum(1 for d in self.dispatches if d.device == ExecutionDevice.GPU)

    @property
    def cpu_executions(self) -> int:
        return sum(1 for d in self.dispatches if d.device == ExecutionDevice.CPU)


# ---------------------------------------------------------------------------
# PolicyHook
# ---------------------------------------------------------------------------

class PolicyHook:
    """Runtime orchestrator produced from a ``CompiledPolicy``.

    A hook is the glue that binds the four compiled components (cache,
    prefetcher, scheduler, monitor) into a single per-layer callback.  It is
    the *code-generated output* of the MoE-Sched compiler: whereas
    ``CompiledPolicy`` is a *bag* of components, ``PolicyHook`` defines their
    *interaction protocol*.
    """

    def __init__(self, compiled: CompiledPolicy) -> None:
        self.compiled = compiled
        self.cache = compiled.cache
        self.prefetcher = compiled.prefetcher
        self.scheduler = compiled.scheduler
        self.monitor = compiled.monitor
        self.triggers = compiled.triggers

        self._step_count = 0

    # -- primary entry point ------------------------------------------------

    def on_layer(
        self,
        layer_idx: int,
        selected_experts: Sequence[int],
        scores: Optional[Sequence[float]] = None,
        expert_size_gb: float = 1.2,
    ) -> DispatchPlan:
        """Process one MoE layer forward.

        Args:
            layer_idx: The layer index (used by prefetcher for affinity lookup).
            selected_experts: Expert IDs the router selected (top-k).
            scores: Optional router scores aligned with selected_experts.
                Required for SCORE-based eviction to update EMA scores.
            expert_size_gb: Size of a single expert's weights; used by the
                hybrid scheduler's transfer-cost model.

        Returns:
            ``DispatchPlan`` with per-expert dispatch decisions + prefetch list.
        """
        plan = DispatchPlan(layer_idx=layer_idx)

        # -- 1. Record each selected expert against the cache ---------------
        for i, eid in enumerate(selected_experts):
            score = scores[i] if scores is not None else 1.0

            # The cache.access() call updates hit/miss stats and may trigger
            # eviction.  ScoreCache takes an optional score arg; the others
            # don't care about it.
            hit = self._cache_access(eid, score)

            # -- 2. Ask the scheduler where to run this expert -------------
            device = self.scheduler.decide(
                expert_id=eid,
                is_cached=hit,
                expert_size_gb=expert_size_gb,
            )

            # If the expert wasn't cached but we're running on GPU, the
            # scheduler has implicitly recorded a CPU->GPU transfer.
            transferred = (not hit) and (device == ExecutionDevice.GPU)

            plan.dispatches.append(ExpertDispatch(
                expert_id=eid,
                device=device,
                cache_hit=hit,
                transferred=transferred,
            ))

            # -- 3. Feed the prefetcher's usage accounting ------------------
            # (Experts that *were* predicted and are now used count as useful.)
            self.prefetcher.report_usage(eid)

            # -- 4. Fire eviction triggers (memory pressure, TTL) ----------
            if self.triggers is not None and self.triggers.active:
                self.triggers.on_access(eid)
                self.triggers.after_access(self.cache)

            # -- 5. Monitor ------------------------------------------------
            if self.monitor is not None:
                # We don't time-instrument the cache here; callers supply a
                # latency estimate via ``record_latency`` if they wish.
                self.monitor.record_access(hit=hit)

        # -- 5. Issue prefetches for upcoming layers ------------------------
        predicted = self.prefetcher.predict(layer_idx, list(selected_experts))
        plan.prefetched = list(predicted)

        # Warm the cache for prefetched experts (without charging a miss).
        for eid in predicted:
            if not self.cache.is_cached(eid):
                self._warm_cache(eid)

        self._step_count += 1
        return plan

    # -- helpers ------------------------------------------------------------

    def _cache_access(self, expert_id: int, score: float) -> bool:
        """Call the cache's access() with the right signature."""
        # ScoreCache.access takes a score; others don't.
        try:
            return self.cache.access(expert_id, score)
        except TypeError:
            return self.cache.access(expert_id)

    def _warm_cache(self, expert_id: int) -> None:
        """Insert a prefetched expert into the cache without counting a miss.

        Implemented by a best-effort access; the extra miss is harmless for
        stats because prefetcher accuracy is tracked separately.  Caches that
        expose a direct insert could override this.
        """
        if hasattr(self.cache, "prefetch_insert"):
            self.cache.prefetch_insert(expert_id)
        # Otherwise no-op: a real transfer would happen; for pure simulation
        # we leave the cache state alone to avoid inflating hit rates.

    # -- introspection ------------------------------------------------------

    @property
    def step_count(self) -> int:
        return self._step_count

    def stats_snapshot(self) -> dict:
        """Aggregate stats from all underlying components."""
        snap = {
            "name": self.compiled.name,
            "steps": self._step_count,
            "cache": {
                "hits": self.cache.stats.hits,
                "misses": self.cache.stats.misses,
                "evictions": self.cache.stats.evictions,
                "hit_rate": self.cache.stats.hit_rate,
            },
            "prefetch": {
                "issued": self.prefetcher.stats.issued,
                "useful": self.prefetcher.stats.useful,
                "accuracy": self.prefetcher.stats.accuracy,
            },
            "scheduler": {
                "gpu": self.scheduler.stats.gpu_executions,
                "cpu": self.scheduler.stats.cpu_executions,
                "transfers": self.scheduler.stats.transfers,
            },
        }
        if self.triggers is not None and self.triggers.active:
            snap["triggers"] = {
                "memory_pressure": (
                    {
                        "fired": self.triggers.memory_pressure.stats.fired,
                        "evicted": self.triggers.memory_pressure.stats.evicted,
                    }
                    if self.triggers.memory_pressure is not None else None
                ),
                "ttl": (
                    {
                        "fired": self.triggers.ttl.stats.fired,
                        "evicted": self.triggers.ttl.stats.evicted,
                    }
                    if self.triggers.ttl is not None else None
                ),
            }
        return snap


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def build_hook(compiled: CompiledPolicy) -> PolicyHook | "AdaptiveHook":
    """Factory: code-generate a runtime hook from a CompiledPolicy.

    If the policy contains ``adapt`` rules, the hook is wrapped in an
    :class:`~moe_sched.adaptive.AdaptiveHook` that monitors metrics and
    dynamically adjusts policy parameters at runtime.

    When the full Cython fast-path hook is available (``_hooks.pyx`` built),
    a :class:`FastPolicyHook` is used instead of the Python ``PolicyHook``
    for maximum dispatch throughput.
    """
    from moe_sched.runtime._fast import FAST_HOOK_AVAILABLE

    if FAST_HOOK_AVAILABLE:
        from moe_sched.runtime._fast._hooks import FastPolicyHook
        hook = FastPolicyHook(compiled)
    else:
        hook = PolicyHook(compiled)

    adapt_ir = getattr(compiled, "_adapt_ir", None)
    if adapt_ir is not None and adapt_ir.rules:
        from moe_sched.adaptive import AdaptiveHook
        from moe_sched.compiler import compile_policy

        policy_ir = getattr(compiled, "_policy_ir", None)
        hook = AdaptiveHook(hook, adapt_ir, policy_ir, compile_policy)
    return hook
