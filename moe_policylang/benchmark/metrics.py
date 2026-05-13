"""Metrics computation for MoE-Sched benchmark results.

Collects per-policy, per-workload measurements:
  * Throughput  — simulated tokens per second (wall-clock time for on_layer calls).
  * Latency     — mean / p50 / p99 per-token dispatch time.
  * Cache hit rate.
  * Memory      — peak simulated GPU memory (# cached experts × expert_size).
  * Dispatch overhead — fraction of per-token time spent in policy dispatch.
  * Prefetch accuracy.
  * Trigger stats (if applicable).
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class MetricsSummary:
    """Aggregated metrics for one (policy, workload) run."""

    policy_name: str
    workload_name: str

    # Throughput
    total_tokens: int = 0
    wall_time_s: float = 0.0
    tokens_per_second: float = 0.0

    # Latency (per-token dispatch time)
    latency_mean_us: float = 0.0
    latency_p50_us: float = 0.0
    latency_p99_us: float = 0.0
    latency_max_us: float = 0.0

    # Cache
    cache_hits: int = 0
    cache_misses: int = 0
    cache_evictions: int = 0
    hit_rate: float = 0.0

    # Memory
    peak_cached_experts: int = 0
    expert_size_gb: float = 1.2
    peak_gpu_memory_gb: float = 0.0

    # Prefetch
    prefetch_issued: int = 0
    prefetch_useful: int = 0
    prefetch_accuracy: float = 0.0

    # Scheduler
    gpu_executions: int = 0
    cpu_executions: int = 0
    transfers: int = 0

    # Dispatch overhead
    dispatch_overhead_pct: float = 0.0

    # Triggers (optional)
    trigger_memory_fired: int = 0
    trigger_memory_evicted: int = 0
    trigger_ttl_fired: int = 0
    trigger_ttl_evicted: int = 0


def compute_metrics(
    policy_name: str,
    workload_name: str,
    total_tokens: int,
    wall_time_s: float,
    per_token_latencies_us: List[float],
    hook_snapshot: dict,
    peak_cached: int,
    expert_size_gb: float = 1.2,
    simulated_inference_us: float = 500.0,
) -> MetricsSummary:
    """Compute a ``MetricsSummary`` from raw benchmark data.

    Args:
        per_token_latencies_us: Wall-clock microseconds for each token's
            full set of ``on_layer`` calls.
        hook_snapshot: Output of ``PolicyHook.stats_snapshot()``.
        peak_cached: Maximum number of experts cached at any point.
        simulated_inference_us: Hypothetical per-token inference time used
            to compute dispatch overhead percentage.
    """
    m = MetricsSummary(policy_name=policy_name, workload_name=workload_name)

    m.total_tokens = total_tokens
    m.wall_time_s = wall_time_s
    m.tokens_per_second = total_tokens / wall_time_s if wall_time_s > 0 else 0.0

    # Latency
    if per_token_latencies_us:
        m.latency_mean_us = statistics.mean(per_token_latencies_us)
        sorted_lat = sorted(per_token_latencies_us)
        m.latency_p50_us = sorted_lat[len(sorted_lat) // 2]
        p99_idx = min(int(len(sorted_lat) * 0.99), len(sorted_lat) - 1)
        m.latency_p99_us = sorted_lat[p99_idx]
        m.latency_max_us = sorted_lat[-1]
    else:
        m.latency_mean_us = 0.0

    # Cache
    cache = hook_snapshot.get("cache", {})
    m.cache_hits = cache.get("hits", 0)
    m.cache_misses = cache.get("misses", 0)
    m.cache_evictions = cache.get("evictions", 0)
    m.hit_rate = cache.get("hit_rate", 0.0)

    # Memory
    m.peak_cached_experts = peak_cached
    m.expert_size_gb = expert_size_gb
    m.peak_gpu_memory_gb = peak_cached * expert_size_gb

    # Prefetch
    pf = hook_snapshot.get("prefetch", {})
    m.prefetch_issued = pf.get("issued", 0)
    m.prefetch_useful = pf.get("useful", 0)
    m.prefetch_accuracy = pf.get("accuracy", 0.0)

    # Scheduler
    sched = hook_snapshot.get("scheduler", {})
    m.gpu_executions = sched.get("gpu", 0)
    m.cpu_executions = sched.get("cpu", 0)
    m.transfers = sched.get("transfers", 0)

    # Dispatch overhead
    if simulated_inference_us > 0 and m.latency_mean_us > 0:
        m.dispatch_overhead_pct = (m.latency_mean_us / simulated_inference_us) * 100.0

    # Triggers
    triggers = hook_snapshot.get("triggers", {})
    mp = triggers.get("memory_pressure") or {}
    m.trigger_memory_fired = mp.get("fired", 0)
    m.trigger_memory_evicted = mp.get("evicted", 0)
    ttl = triggers.get("ttl") or {}
    m.trigger_ttl_fired = ttl.get("fired", 0)
    m.trigger_ttl_evicted = ttl.get("evicted", 0)

    return m
