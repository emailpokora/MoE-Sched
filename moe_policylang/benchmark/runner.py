"""Benchmark runner: executes all policies × workloads and collects results.

Can be run as:
    python -m moe_sched.benchmark.runner [--output results.json]
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import List

from moe_sched.benchmark.harness import BenchmarkHarness, BenchmarkResult
from moe_sched.benchmark.metrics import MetricsSummary
from moe_sched.benchmark.policies import BASELINES, get_dsl_policies
from moe_sched.benchmark.workloads import ALL_WORKLOADS, Workload


def run_all(
    workloads: list[Workload] | None = None,
    capacity: int = 16,
    expert_size_gb: float = 1.2,
) -> List[MetricsSummary]:
    """Run every policy and baseline across every workload.

    Returns a flat list of MetricsSummary for tabulation.
    """
    workloads = workloads or ALL_WORKLOADS
    harness = BenchmarkHarness(expert_size_gb=expert_size_gb)
    results: List[MetricsSummary] = []

    # --- DSL policies ---
    policy_names = list(get_dsl_policies().keys())
    for pname in policy_names:
        for wl in workloads:
            # Policies are stateful — build fresh for each workload
            fresh = get_dsl_policies()[pname]
            print(f"  [{pname}] × [{wl.name}] ...", end=" ", flush=True)
            t0 = time.perf_counter()
            result = harness.run_policy(fresh, wl)
            elapsed = time.perf_counter() - t0
            print(f"done ({elapsed:.2f}s)")
            results.append(result.metrics)

    # --- Baselines ---
    for bname, (factory, desc) in BASELINES.items():
        for wl in workloads:
            print(f"  [{bname}] × [{wl.name}] ...", end=" ", flush=True)
            t0 = time.perf_counter()
            result = harness.run_baseline(factory, wl, capacity=capacity, baseline_name=bname)
            elapsed = time.perf_counter() - t0
            print(f"done ({elapsed:.2f}s)")
            results.append(result.metrics)

    return results


# ---------------------------------------------------------------------------
# Tabulation
# ---------------------------------------------------------------------------

def format_table(results: List[MetricsSummary]) -> str:
    """Format results as a text table."""
    lines = []
    header = (
        f"{'Policy':<25} {'Workload':<15} {'Tok/s':>10} "
        f"{'Hit%':>7} {'Lat_mean':>10} {'Lat_p99':>10} "
        f"{'PF_acc%':>8} {'Peak_GB':>8} {'OH%':>6}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    for m in results:
        lines.append(
            f"{m.policy_name:<25} {m.workload_name:<15} "
            f"{m.tokens_per_second:>10.1f} "
            f"{m.hit_rate * 100:>6.1f}% "
            f"{m.latency_mean_us:>9.1f}µs "
            f"{m.latency_p99_us:>9.1f}µs "
            f"{m.prefetch_accuracy * 100:>7.1f}% "
            f"{m.peak_gpu_memory_gb:>7.1f}G "
            f"{m.dispatch_overhead_pct:>5.1f}%"
        )
    return "\n".join(lines)


def results_to_dict(results: List[MetricsSummary]) -> list[dict]:
    """Convert results list to JSON-serialisable dicts."""
    return [asdict(m) for m in results]


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="MoE-Sched benchmark runner")
    ap.add_argument("--output", "-o", type=Path, default=None, help="Output JSON file")
    ap.add_argument("--capacity", type=int, default=16, help="Cache capacity for all policies")
    args = ap.parse_args()

    print("=" * 70)
    print("MoE-Sched Benchmark Suite — Week 5 Evaluation")
    print("=" * 70)

    results = run_all(capacity=args.capacity)

    print()
    print(format_table(results))
    print()

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results_to_dict(results), f, indent=2)
        print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
