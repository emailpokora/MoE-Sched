"""Intermediate Representation for MoE-PolicyLang policies."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class EvictionPolicy(Enum):
    LRU = "lru"
    LFU = "lfu"
    SCORE = "score"
    FREQ_THRESHOLD = "frequency_threshold"
    FALLBACK = "fallback"


class PrefetchStrategy(Enum):
    NONE = "none"
    AFFINITY = "affinity"
    HISTORY = "history"
    LOOKAHEAD = "lookahead"


class ScheduleMode(Enum):
    GPU_ONLY = "gpu_only"
    CPU_FALLBACK = "cpu_fallback"
    HYBRID = "hybrid"


class AllocationSignal(Enum):
    ENTROPY = "entropy"
    UNIFORM = "uniform"


# ---------------------------------------------------------------------------
# IR dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CacheIR:
    capacity: int
    eviction: EvictionPolicy = EvictionPolicy.LRU
    pin_experts: List[int] = field(default_factory=list)
    lfu_decay: float = 0.95
    freq_threshold: float = 0.05
    freq_window: int = 100
    score_ema_alpha: float = 0.3
    # -- Eviction triggers ----------------------------------------
    # Memory pressure: evict down to ``memory_headroom`` when estimated GPU
    # usage (cache_size * expert_size_gb / gpu_budget_gb) >= threshold.
    memory_threshold: Optional[float] = None   # in [0, 1], e.g. 0.9
    memory_headroom: float = 0.7               # target after eviction
    memory_budget_gb: float = 16.0             # simulated GPU budget
    expert_size_gb: float = 1.2                # per-expert estimate
    # TTL: evict any non-pinned expert unused for this many accesses.
    ttl: Optional[int] = None                  # in accesses; None = disabled
    # Composition: secondary eviction strategy for fallback layering.
    fallback_eviction: Optional["EvictionPolicy"] = None


@dataclass
class PrefetchIR:
    strategy: PrefetchStrategy = PrefetchStrategy.NONE
    lookahead: int = 1
    budget: int = 4
    affinity_threshold: float = 0.3
    history_window: int = 50


@dataclass
class ScheduleIR:
    mode: ScheduleMode = ScheduleMode.GPU_ONLY
    cpu_threshold_ms: float = 50.0
    overlap: bool = True
    priority_routing: bool = False


@dataclass
class MonitorIR:
    metrics: List[str] = field(default_factory=lambda: ["hit_rate"])
    window: int = 100
    log_interval: int = 50


@dataclass
class PerLayerIR:
    allocation: AllocationSignal = AllocationSignal.ENTROPY
    entropy_window: int = 200
    min_capacity: int = 2
    max_capacity: int = 64
    rebalance_interval: int = 500
    total_budget: Optional[int] = None


@dataclass
class PolicyIR:
    name: str
    cache: CacheIR
    prefetch: PrefetchIR = field(default_factory=PrefetchIR)
    schedule: ScheduleIR = field(default_factory=ScheduleIR)
    monitor: Optional[MonitorIR] = None
    adapt: Optional["AdaptIR"] = None
    per_layer: Optional[PerLayerIR] = None
