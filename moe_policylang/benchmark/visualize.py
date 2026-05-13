"""Results visualization — generates tables and optional matplotlib charts.

Charts are only generated if matplotlib is available; the module degrades
gracefully to text-only output otherwise.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from moe_policylang.benchmark.metrics import MetricsSummary

try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ---------------------------------------------------------------------------
# Text tables
# ---------------------------------------------------------------------------

def hit_rate_table(results: List[MetricsSummary]) -> str:
    """Policy × Workload hit-rate matrix."""
    policies = sorted({r.policy_name for r in results})
    workloads = sorted({r.workload_name for r in results})

    lookup = {(r.policy_name, r.workload_name): r for r in results}

    col_w = 14
    header = f"{'Policy':<25}" + "".join(f"{w:>{col_w}}" for w in workloads)
    lines = [header, "-" * len(header)]
    for p in policies:
        row = f"{p:<25}"
        for w in workloads:
            m = lookup.get((p, w))
            val = f"{m.hit_rate * 100:.1f}%" if m else "N/A"
            row += f"{val:>{col_w}}"
        lines.append(row)
    return "\n".join(lines)


def throughput_table(results: List[MetricsSummary]) -> str:
    """Policy × Workload throughput (tok/s) matrix."""
    policies = sorted({r.policy_name for r in results})
    workloads = sorted({r.workload_name for r in results})

    lookup = {(r.policy_name, r.workload_name): r for r in results}

    col_w = 14
    header = f"{'Policy':<25}" + "".join(f"{w:>{col_w}}" for w in workloads)
    lines = [header, "-" * len(header)]
    for p in policies:
        row = f"{p:<25}"
        for w in workloads:
            m = lookup.get((p, w))
            val = f"{m.tokens_per_second:.0f}" if m else "N/A"
            row += f"{val:>{col_w}}"
        lines.append(row)
    return "\n".join(lines)


def latency_table(results: List[MetricsSummary]) -> str:
    """Policy × Workload latency (mean µs) matrix."""
    policies = sorted({r.policy_name for r in results})
    workloads = sorted({r.workload_name for r in results})

    lookup = {(r.policy_name, r.workload_name): r for r in results}

    col_w = 14
    header = f"{'Policy':<25}" + "".join(f"{w:>{col_w}}" for w in workloads)
    lines = [header, "-" * len(header)]
    for p in policies:
        row = f"{p:<25}"
        for w in workloads:
            m = lookup.get((p, w))
            val = f"{m.latency_mean_us:.0f}µs" if m else "N/A"
            row += f"{val:>{col_w}}"
        lines.append(row)
    return "\n".join(lines)


def overhead_table(results: List[MetricsSummary]) -> str:
    """Policy × Workload dispatch overhead (%) matrix."""
    policies = sorted({r.policy_name for r in results})
    workloads = sorted({r.workload_name for r in results})

    lookup = {(r.policy_name, r.workload_name): r for r in results}

    col_w = 14
    header = f"{'Policy':<25}" + "".join(f"{w:>{col_w}}" for w in workloads)
    lines = [header, "-" * len(header)]
    for p in policies:
        row = f"{p:<25}"
        for w in workloads:
            m = lookup.get((p, w))
            val = f"{m.dispatch_overhead_pct:.1f}%" if m else "N/A"
            row += f"{val:>{col_w}}"
        lines.append(row)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Matplotlib charts (optional)
# ---------------------------------------------------------------------------

def plot_hit_rates(results: List[MetricsSummary], output_dir: Path) -> Optional[Path]:
    """Grouped bar chart: hit rate per policy, grouped by workload."""
    if not HAS_MPL:
        return None
    return _grouped_bar(
        results,
        metric_fn=lambda m: m.hit_rate * 100,
        ylabel="Cache Hit Rate (%)",
        title="Cache Hit Rate by Policy and Workload",
        filename="hit_rate.png",
        output_dir=output_dir,
    )


def plot_throughput(results: List[MetricsSummary], output_dir: Path) -> Optional[Path]:
    if not HAS_MPL:
        return None
    return _grouped_bar(
        results,
        metric_fn=lambda m: m.tokens_per_second,
        ylabel="Tokens / second",
        title="Dispatch Throughput by Policy and Workload",
        filename="throughput.png",
        output_dir=output_dir,
    )


def plot_latency(results: List[MetricsSummary], output_dir: Path) -> Optional[Path]:
    if not HAS_MPL:
        return None
    return _grouped_bar(
        results,
        metric_fn=lambda m: m.latency_mean_us,
        ylabel="Mean Dispatch Latency (µs)",
        title="Per-Token Dispatch Latency by Policy and Workload",
        filename="latency.png",
        output_dir=output_dir,
    )


def plot_overhead(results: List[MetricsSummary], output_dir: Path) -> Optional[Path]:
    if not HAS_MPL:
        return None
    return _grouped_bar(
        results,
        metric_fn=lambda m: m.dispatch_overhead_pct,
        ylabel="Dispatch Overhead (%)",
        title="Policy Dispatch Overhead (target < 5%)",
        filename="overhead.png",
        output_dir=output_dir,
    )


def generate_all_charts(results: List[MetricsSummary], output_dir: Path) -> List[Path]:
    """Generate all available charts, returning paths of files written."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for fn in (plot_hit_rates, plot_throughput, plot_latency, plot_overhead):
        p = fn(results, output_dir)
        if p is not None:
            paths.append(p)
    return paths


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _grouped_bar(
    results: List[MetricsSummary],
    metric_fn,
    ylabel: str,
    title: str,
    filename: str,
    output_dir: Path,
) -> Path:
    import numpy as np

    policies = sorted({r.policy_name for r in results})
    workloads = sorted({r.workload_name for r in results})
    lookup = {(r.policy_name, r.workload_name): r for r in results}

    x = np.arange(len(workloads))
    width = 0.8 / len(policies)

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, p in enumerate(policies):
        vals = [metric_fn(lookup[(p, w)]) if (p, w) in lookup else 0 for w in workloads]
        ax.bar(x + i * width, vals, width, label=p)

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x + width * len(policies) / 2 - width / 2)
    ax.set_xticklabels(workloads, rotation=15, ha="right")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1))
    fig.tight_layout()

    out = output_dir / filename
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
