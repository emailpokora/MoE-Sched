"""Compiler: translates PolicyIR into runtime components."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from moe_sched.ir import EvictionPolicy, PrefetchStrategy, ScheduleMode
from moe_sched.runtime.cache import (
    FreqThresholdCache,
    LFUCache,
    LRUCache,
    ScoreCache,
)
from moe_sched.runtime.monitor import Monitor
from moe_sched.runtime.prefetch import (
    AffinityPrefetcher,
    HistoryPrefetcher,
    LookaheadPrefetcher,
    NullPrefetcher,
)
from moe_sched.runtime.triggers import (
    MemoryPressureTrigger,
    TriggerSet,
    TTLTrigger,
)
from moe_sched.runtime.scheduler import (
    CPUFallbackScheduler,
    GPUOnlyScheduler,
    HybridScheduler,
)
from moe_sched.runtime._fast import FAST_PATH_AVAILABLE

if TYPE_CHECKING:
    from moe_sched.ir import PolicyIR


@dataclass
class CompiledPolicy:
    """Assembled runtime components produced by the compiler."""

    name: str
    cache: LRUCache | LFUCache | ScoreCache | FreqThresholdCache
    prefetcher: NullPrefetcher | AffinityPrefetcher | HistoryPrefetcher | LookaheadPrefetcher
    scheduler: GPUOnlyScheduler | CPUFallbackScheduler | HybridScheduler
    monitor: Monitor | None
    triggers: TriggerSet = field(default_factory=TriggerSet)


def _build_single_cache(eviction: "EvictionPolicy", capacity: int, cache_ir: "CacheIR"):
    """Build a single cache instance for a given eviction policy and capacity."""
    if eviction == EvictionPolicy.LRU:
        if FAST_PATH_AVAILABLE:
            from moe_sched.runtime._fast import LRUCacheFast
            return LRUCacheFast(capacity, pin_experts=cache_ir.pin_experts)
        return LRUCache(capacity, pin_experts=cache_ir.pin_experts)
    elif eviction == EvictionPolicy.LFU:
        if FAST_PATH_AVAILABLE:
            from moe_sched.runtime._fast import LFUCacheFast
            return LFUCacheFast(capacity, pin_experts=cache_ir.pin_experts, decay=cache_ir.lfu_decay)
        return LFUCache(capacity, pin_experts=cache_ir.pin_experts, decay=cache_ir.lfu_decay)
    elif eviction == EvictionPolicy.SCORE:
        return ScoreCache(capacity, pin_experts=cache_ir.pin_experts, ema_alpha=cache_ir.score_ema_alpha)
    elif eviction == EvictionPolicy.FREQ_THRESHOLD:
        return FreqThresholdCache(capacity, threshold=cache_ir.freq_threshold, window=cache_ir.freq_window, pin_experts=cache_ir.pin_experts)
    raise ValueError(f"Cannot use {eviction} as a component in fallback composition")


def compile_policy(ir: "PolicyIR") -> CompiledPolicy:
    """Compile a validated PolicyIR into a runnable CompiledPolicy."""

    # -- Cache --
    cache_ir = ir.cache
    if cache_ir.eviction == EvictionPolicy.LRU:
        if FAST_PATH_AVAILABLE:
            from moe_sched.runtime._fast import LRUCacheFast
            cache = LRUCacheFast(cache_ir.capacity, pin_experts=cache_ir.pin_experts)
        else:
            cache = LRUCache(cache_ir.capacity, pin_experts=cache_ir.pin_experts)
    elif cache_ir.eviction == EvictionPolicy.LFU:
        if FAST_PATH_AVAILABLE:
            from moe_sched.runtime._fast import LFUCacheFast
            cache = LFUCacheFast(
                cache_ir.capacity,
                pin_experts=cache_ir.pin_experts,
                decay=cache_ir.lfu_decay,
            )
        else:
            cache = LFUCache(
                cache_ir.capacity,
                pin_experts=cache_ir.pin_experts,
                decay=cache_ir.lfu_decay,
            )
    elif cache_ir.eviction == EvictionPolicy.SCORE:
        cache = ScoreCache(
            cache_ir.capacity,
            pin_experts=cache_ir.pin_experts,
            ema_alpha=cache_ir.score_ema_alpha,
        )
    elif cache_ir.eviction == EvictionPolicy.FREQ_THRESHOLD:
        cache = FreqThresholdCache(
            cache_ir.capacity,
            threshold=cache_ir.freq_threshold,
            window=cache_ir.freq_window,
            pin_experts=cache_ir.pin_experts,
        )
    elif cache_ir.eviction == EvictionPolicy.FALLBACK:
        from moe_sched.runtime.cache import FallbackCache
        primary_eviction = getattr(cache_ir, "_primary_eviction", EvictionPolicy.LFU)
        fallback_eviction = cache_ir.fallback_eviction or EvictionPolicy.LRU
        # Use explicit sizes if provided, otherwise auto-split 2/3 + 1/3
        primary_cap = getattr(cache_ir, "_primary_cap", None)
        secondary_cap = getattr(cache_ir, "_secondary_cap", None)
        if primary_cap is None or secondary_cap is None:
            primary_cap = max(1, (cache_ir.capacity * 2) // 3)
            secondary_cap = max(1, cache_ir.capacity - primary_cap)
        primary = _build_single_cache(primary_eviction, primary_cap, cache_ir)
        secondary = _build_single_cache(fallback_eviction, secondary_cap, cache_ir)
        cache = FallbackCache(primary, secondary)
    else:
        raise ValueError(f"Unknown eviction policy: {cache_ir.eviction}")

    # -- Prefetcher --
    pf_ir = ir.prefetch
    if pf_ir.strategy == PrefetchStrategy.NONE:
        prefetcher = NullPrefetcher()
    elif pf_ir.strategy == PrefetchStrategy.AFFINITY:
        prefetcher = AffinityPrefetcher(
            threshold=pf_ir.affinity_threshold,
            budget=pf_ir.budget,
        )
    elif pf_ir.strategy == PrefetchStrategy.HISTORY:
        prefetcher = HistoryPrefetcher(
            window=pf_ir.history_window,
            budget=pf_ir.budget,
        )
    elif pf_ir.strategy == PrefetchStrategy.LOOKAHEAD:
        prefetcher = LookaheadPrefetcher(
            lookahead=pf_ir.lookahead,
            budget=pf_ir.budget,
            history_window=pf_ir.history_window,
        )
    else:
        raise ValueError(f"Unknown prefetch strategy: {pf_ir.strategy}")

    # -- Scheduler --
    sched_ir = ir.schedule
    if sched_ir.mode == ScheduleMode.GPU_ONLY:
        if FAST_PATH_AVAILABLE:
            from moe_sched.runtime._fast import GPUOnlySchedulerFast
            scheduler = GPUOnlySchedulerFast()
        else:
            scheduler = GPUOnlyScheduler()
    elif sched_ir.mode == ScheduleMode.CPU_FALLBACK:
        if FAST_PATH_AVAILABLE:
            from moe_sched.runtime._fast import CPUFallbackSchedulerFast
            scheduler = CPUFallbackSchedulerFast()
        else:
            scheduler = CPUFallbackScheduler()
    elif sched_ir.mode == ScheduleMode.HYBRID:
        if FAST_PATH_AVAILABLE:
            from moe_sched.runtime._fast import HybridSchedulerFast
            scheduler = HybridSchedulerFast(cpu_threshold_ms=sched_ir.cpu_threshold_ms)
        else:
            scheduler = HybridScheduler(cpu_threshold_ms=sched_ir.cpu_threshold_ms)
    else:
        raise ValueError(f"Unknown schedule mode: {sched_ir.mode}")

    # -- Monitor --
    monitor = None
    if ir.monitor is not None:
        monitor = Monitor(
            metrics=ir.monitor.metrics,
            window=ir.monitor.window,
            log_interval=ir.monitor.log_interval,
        )

    # -- Eviction triggers --
    triggers = TriggerSet()
    if cache_ir.memory_threshold is not None:
        triggers.memory_pressure = MemoryPressureTrigger(
            budget_gb=cache_ir.memory_budget_gb,
            threshold=cache_ir.memory_threshold,
            headroom=cache_ir.memory_headroom,
            expert_size_gb=cache_ir.expert_size_gb,
        )
    if cache_ir.ttl is not None:
        triggers.ttl = TTLTrigger(ttl=cache_ir.ttl)

    compiled = CompiledPolicy(
        name=ir.name,
        cache=cache,
        prefetcher=prefetcher,
        scheduler=scheduler,
        monitor=monitor,
        triggers=triggers,
    )
    compiled._adapt_ir = getattr(ir, "adapt", None)
    compiled._policy_ir = ir
    return compiled
