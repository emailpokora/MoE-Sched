"""Benchmark harness: drives policies through workloads and collects metrics.

The harness supports both DSL-compiled policies and hand-coded baselines.
It measures wall-clock dispatch time per token and tracks peak cache occupancy.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, List

from moe_sched.benchmark.metrics import MetricsSummary, compute_metrics
from moe_sched.benchmark.workloads import Workload
from moe_sched.compiler import CompiledPolicy
from moe_sched.runtime.hooks import PolicyHook, build_hook


@dataclass
class BenchmarkResult:
    """Result of running one policy across one workload."""

    metrics: MetricsSummary
    hook_snapshot: dict
    per_token_latencies_us: List[float] = field(default_factory=list)


class BenchmarkHarness:
    """Orchestrates benchmark runs for DSL policies and baselines.

    Usage:
        harness = BenchmarkHarness()
        result = harness.run_policy(compiled_policy, workload)
        result_baseline = harness.run_baseline(baseline_fn, workload)
    """

    def __init__(
        self,
        expert_size_gb: float = 1.2,
        simulated_inference_us: float = 500.0,
        warmup_tokens: int = 5,
    ):
        self.expert_size_gb = expert_size_gb
        self.simulated_inference_us = simulated_inference_us
        self.warmup_tokens = warmup_tokens

    def run_policy(
        self,
        compiled: CompiledPolicy,
        workload: Workload,
    ) -> BenchmarkResult:
        """Run a DSL-compiled policy through a workload and collect metrics."""
        hook = build_hook(compiled)
        selector = workload.make_selector()

        per_token_us: List[float] = []
        peak_cached = 0

        for t in range(workload.num_tokens):
            t0 = time.perf_counter()
            for layer in range(workload.num_layers):
                experts = selector(t, layer)
                hook.on_layer(
                    layer_idx=layer,
                    selected_experts=experts,
                    expert_size_gb=self.expert_size_gb,
                )
                # Track peak
                current_size = hook.cache.size
                if current_size > peak_cached:
                    peak_cached = current_size
            elapsed_us = (time.perf_counter() - t0) * 1e6
            if t >= self.warmup_tokens:
                per_token_us.append(elapsed_us)

        snapshot = hook.stats_snapshot()
        wall_s = sum(per_token_us) / 1e6
        measured_tokens = workload.num_tokens - self.warmup_tokens

        metrics = compute_metrics(
            policy_name=compiled.name,
            workload_name=workload.name,
            total_tokens=measured_tokens,
            wall_time_s=wall_s,
            per_token_latencies_us=per_token_us,
            hook_snapshot=snapshot,
            peak_cached=peak_cached,
            expert_size_gb=self.expert_size_gb,
            simulated_inference_us=self.simulated_inference_us,
        )

        return BenchmarkResult(
            metrics=metrics,
            hook_snapshot=snapshot,
            per_token_latencies_us=per_token_us,
        )

    def run_baseline(
        self,
        baseline_factory: Callable[[int], object],
        workload: Workload,
        capacity: int,
        baseline_name: str = "baseline",
    ) -> BenchmarkResult:
        """Run a hand-coded baseline through a workload.

        The baseline object must expose:
          * ``on_layer(layer_idx, selected_experts) -> list``
          * ``stats`` with ``hits``, ``misses``, ``evictions``, ``hit_rate``
        """
        baseline = baseline_factory(capacity)
        selector = workload.make_selector()

        per_token_us: List[float] = []
        peak_cached = capacity  # baselines fill up to capacity

        for t in range(workload.num_tokens):
            t0 = time.perf_counter()
            for layer in range(workload.num_layers):
                experts = selector(t, layer)
                baseline.on_layer(layer_idx=layer, selected_experts=experts)
            elapsed_us = (time.perf_counter() - t0) * 1e6
            if t >= self.warmup_tokens:
                per_token_us.append(elapsed_us)

        # Build a pseudo-snapshot matching PolicyHook format
        stats = baseline.stats
        # Compute scheduler stats from dispatch decisions
        gpu_count = sum(1 for d in baseline.dispatches if d.on_gpu)
        cpu_count = sum(1 for d in baseline.dispatches if not d.on_gpu)
        transfer_count = sum(1 for d in baseline.dispatches if d.on_gpu and not d.cache_hit)
        snapshot = {
            "name": baseline_name,
            "steps": workload.num_tokens * workload.num_layers,
            "cache": {
                "hits": stats.hits,
                "misses": stats.misses,
                "evictions": stats.evictions,
                "hit_rate": stats.hit_rate,
            },
            "prefetch": {"issued": 0, "useful": 0, "accuracy": 0.0},
            "scheduler": {"gpu": gpu_count, "cpu": cpu_count, "transfers": transfer_count},
        }

        wall_s = sum(per_token_us) / 1e6
        measured_tokens = workload.num_tokens - self.warmup_tokens

        metrics = compute_metrics(
            policy_name=baseline_name,
            workload_name=workload.name,
            total_tokens=measured_tokens,
            wall_time_s=wall_s,
            per_token_latencies_us=per_token_us,
            hook_snapshot=snapshot,
            peak_cached=peak_cached,
            expert_size_gb=self.expert_size_gb,
            simulated_inference_us=self.simulated_inference_us,
        )

        return BenchmarkResult(
            metrics=metrics,
            hook_snapshot=snapshot,
            per_token_latencies_us=per_token_us,
        )
