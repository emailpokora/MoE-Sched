"""Statistical analysis with confidence intervals for paper.

Produces:
1. Dispatch latency: 30 runs per policy × trace → mean ± 95% CI
2. Bootstrap hit rate CIs: per-token hit rates → 10K bootstrap → 95% CI
3. Pairwise significance: Wilcoxon signed-rank on per-token hit rates
4. Per-layer hit rate distributions: box plots
5. Saves all results to traces/stats_results.json + paper/figures/
"""
import json
import os
import sys
import time
import glob
import itertools
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
from scipy import stats as scipy_stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from moe_sched.benchmark.policies import get_dsl_policies
from moe_sched.compiler import compile_policy
from moe_sched.runtime.hooks import build_hook

TRACES_DIR = os.path.join(ROOT, "traces")
FIG_DIR = os.path.join(ROOT, "paper", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

DISPLAY = {
    "mixtral_sample": "Mixtral-8×7B",
    "deepseek_v2_lite_sample": "DeepSeek-V2-Lite",
}

N_LATENCY_RUNS = 30
N_BOOTSTRAP = 10_000
CI_LEVEL = 0.95
SEED = 42


def load_trace(path):
    lines = open(path).readlines()
    header = json.loads(lines[0])
    return header, [json.loads(l) for l in lines[1:]]


def load_all_traces():
    traces = {}
    for f in sorted(glob.glob(os.path.join(TRACES_DIR, "*.jsonl"))):
        header, data = load_trace(f)
        name = os.path.splitext(os.path.basename(f))[0]
        traces[name] = {"header": header, "data": data}
        print(f"  {name}: {len(data)} entries")
    return traces


def segment_by_token(entries, num_layers):
    """Group trace entries into per-token steps.

    Each token step = one token's entries across all layers.
    Returns list of lists, where each inner list is one token's entries.
    """
    segments = []
    current = []
    for e in entries:
        current.append(e)
        if len(current) == num_layers:
            segments.append(current)
            current = []
    if current:
        segments.append(current)
    return segments


# ═══════════════════════════════════════════════════════════════════════
# 1. Dispatch Latency with Confidence Intervals
# ═══════════════════════════════════════════════════════════════════════
def run_latency_analysis(traces):
    print("\n" + "=" * 70)
    print("1. DISPATCH LATENCY (30 runs → 95% CI)")
    print("=" * 70)

    policies = get_dsl_policies()
    results = {}

    for tname, tdata in traces.items():
        results[tname] = {}
        data = tdata["data"]
        label = DISPLAY.get(tname, tname)
        print(f"\n--- {label} ({len(data)} entries) ---")
        print(f"{'Policy':<25} {'Mean µs':>10} {'±95% CI':>10} {'Std':>10} {'P50':>10} {'P99':>10}")
        print("-" * 77)

        for pname, compiled in policies.items():
            run_means = []
            all_latencies = []
            for run_i in range(N_LATENCY_RUNS):
                hook = build_hook(compiled)
                t0 = time.perf_counter()
                for e in data:
                    hook.on_layer(
                        layer_idx=e["l"],
                        selected_experts=e["e"],
                        scores=e.get("s"),
                    )
                elapsed = time.perf_counter() - t0
                run_means.append((elapsed / len(data)) * 1e6)

            arr = np.array(run_means)
            mean = np.mean(arr)
            std = np.std(arr, ddof=1)
            sem = std / np.sqrt(len(arr))
            t_crit = scipy_stats.t.ppf((1 + CI_LEVEL) / 2, df=len(arr) - 1)
            ci = t_crit * sem
            p50 = np.median(arr)
            p99 = np.percentile(arr, 99)

            print(f"{pname:<25} {mean:>9.2f} {ci:>9.2f} {std:>9.2f} {p50:>9.2f} {p99:>9.2f}")
            results[tname][pname] = {
                "mean_us": round(float(mean), 2),
                "ci95_us": round(float(ci), 2),
                "std_us": round(float(std), 2),
                "p50_us": round(float(p50), 2),
                "p99_us": round(float(p99), 2),
                "runs": [round(float(x), 2) for x in arr],
            }

    return results


# ═══════════════════════════════════════════════════════════════════════
# 2. Bootstrap Hit Rate Confidence Intervals
# ═══════════════════════════════════════════════════════════════════════
def compute_per_token_hit_rates(trace_data, compiled, num_layers):
    """Run policy on trace and compute per-token hit rate.

    Returns array of per-token hit rates = hits / (hits + misses) per token.
    """
    hook = build_hook(compiled)
    segments = segment_by_token(trace_data, num_layers)
    per_token_hrs = []

    for seg in segments:
        snap_before = hook.stats_snapshot()["cache"]
        hits_before = snap_before["hits"]
        misses_before = snap_before["misses"]
        for e in seg:
            hook.on_layer(
                layer_idx=e["l"],
                selected_experts=e["e"],
                scores=e.get("s"),
            )
        snap_after = hook.stats_snapshot()["cache"]
        token_hits = snap_after["hits"] - hits_before
        token_misses = snap_after["misses"] - misses_before
        total = token_hits + token_misses
        per_token_hrs.append(token_hits / max(1, total))

    return np.array(per_token_hrs)


def bootstrap_ci(data, n_bootstrap=N_BOOTSTRAP, ci_level=CI_LEVEL, seed=SEED):
    """Compute bootstrap confidence interval for the mean."""
    rng = np.random.RandomState(seed)
    n = len(data)
    boot_means = np.array([
        np.mean(rng.choice(data, size=n, replace=True))
        for _ in range(n_bootstrap)
    ])
    alpha = (1 - ci_level) / 2
    lo = np.percentile(boot_means, alpha * 100)
    hi = np.percentile(boot_means, (1 - alpha) * 100)
    return float(np.mean(data)), float(lo), float(hi), boot_means


def run_bootstrap_analysis(traces):
    print("\n" + "=" * 70)
    print("2. BOOTSTRAP HIT RATE CIs (10K resamples)")
    print("=" * 70)

    policies = get_dsl_policies()
    results = {}
    per_token_data = {}  # saved for significance tests

    for tname, tdata in traces.items():
        results[tname] = {}
        per_token_data[tname] = {}
        data = tdata["data"]
        num_layers = tdata["header"]["num_layers"]
        label = DISPLAY.get(tname, tname)
        print(f"\n--- {label} ---")
        print(f"{'Policy':<25} {'Mean HR':>10} {'95% CI Lo':>10} {'95% CI Hi':>10} {'Width':>10}")
        print("-" * 67)

        for pname, compiled in policies.items():
            pt_hrs = compute_per_token_hit_rates(data, compiled, num_layers)
            per_token_data[tname][pname] = pt_hrs
            mean, lo, hi, _ = bootstrap_ci(pt_hrs)

            print(f"{pname:<25} {mean:>9.1%} {lo:>9.1%} {hi:>9.1%} {(hi-lo):>9.1%}")
            results[tname][pname] = {
                "mean": round(float(mean), 4),
                "ci95_lo": round(float(lo), 4),
                "ci95_hi": round(float(hi), 4),
                "ci_width": round(float(hi - lo), 4),
                "n_tokens": len(pt_hrs),
            }

    return results, per_token_data


# ═══════════════════════════════════════════════════════════════════════
# 3. Pairwise Significance Tests
# ═══════════════════════════════════════════════════════════════════════
def run_significance_tests(per_token_data):
    print("\n" + "=" * 70)
    print("3. PAIRWISE SIGNIFICANCE (Wilcoxon signed-rank)")
    print("=" * 70)

    results = {}

    for tname, policies_data in per_token_data.items():
        results[tname] = {}
        label = DISPLAY.get(tname, tname)
        print(f"\n--- {label} ---")

        policy_names = list(policies_data.keys())
        n = len(policy_names)

        # Print header
        header = f"{'':25}" + "".join(f"{p:>15}" for p in policy_names)
        print(header)
        print("-" * len(header))

        for i in range(n):
            row = f"{policy_names[i]:<25}"
            for j in range(n):
                if i == j:
                    row += f"{'---':>15}"
                else:
                    a = policies_data[policy_names[i]]
                    b = policies_data[policy_names[j]]
                    # Truncate to same length
                    min_len = min(len(a), len(b))
                    a, b = a[:min_len], b[:min_len]
                    if np.array_equal(a, b):
                        row += f"{'p=1.000':>15}"
                    else:
                        try:
                            stat, p_val = scipy_stats.wilcoxon(a, b, alternative="two-sided")
                            sig = ""
                            if p_val < 0.001:
                                sig = "***"
                            elif p_val < 0.01:
                                sig = "**"
                            elif p_val < 0.05:
                                sig = "*"
                            row += f"{'p='+f'{p_val:.3f}'+sig:>15}"
                        except ValueError:
                            row += f"{'n/a':>15}"
                results[tname][f"{policy_names[i]}_vs_{policy_names[j]}"] = {
                    "p_value": round(float(p_val), 6) if i != j else None,
                    "significant_05": bool(p_val < 0.05) if i != j else None,
                }
            print(row)

    return results


# ═══════════════════════════════════════════════════════════════════════
# 4. Per-Layer Hit Rate Distribution
# ═══════════════════════════════════════════════════════════════════════
def run_per_layer_analysis(traces):
    print("\n" + "=" * 70)
    print("4. PER-LAYER HIT RATE DISTRIBUTION")
    print("=" * 70)

    policies = get_dsl_policies()
    # Use lru_basic as representative
    target_policy = "lru_basic"
    compiled = policies[target_policy]

    fig, axes = plt.subplots(1, len(traces), figsize=(6 * len(traces), 4))
    if len(traces) == 1:
        axes = [axes]

    results = {}

    for ax, (tname, tdata) in zip(axes, traces.items()):
        data = tdata["data"]
        num_layers = tdata["header"]["num_layers"]
        label = DISPLAY.get(tname, tname)

        # Track per-layer hits/total-lookups
        layer_hits = defaultdict(int)
        layer_lookups = defaultdict(int)

        hook = build_hook(compiled)
        for e in data:
            snap_before = hook.stats_snapshot()["cache"]
            h_before = snap_before["hits"]
            m_before = snap_before["misses"]
            hook.on_layer(
                layer_idx=e["l"],
                selected_experts=e["e"],
                scores=e.get("s"),
            )
            snap_after = hook.stats_snapshot()["cache"]
            dh = snap_after["hits"] - h_before
            dm = snap_after["misses"] - m_before
            layer_hits[e["l"]] += dh
            layer_lookups[e["l"]] += dh + dm

        layers = sorted(layer_lookups.keys())
        hrs = [layer_hits[l] / max(1, layer_lookups[l]) for l in layers]

        ax.bar(layers, [hr * 100 for hr in hrs], color="#4C72B0", edgecolor="white", alpha=0.85)
        ax.set_xlabel("Layer Index", fontsize=10)
        ax.set_title(f"{label} ({target_policy})", fontsize=11)
        ax.set_ylim(0, 105)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        mean_hr = np.mean(hrs)
        std_hr = np.std(hrs)
        ax.axhline(y=mean_hr * 100, color="red", linestyle="--", linewidth=1,
                    label=f"mean={mean_hr:.1%} ± {std_hr:.1%}")
        ax.legend(fontsize=8)

        results[tname] = {
            "per_layer": {str(l): round(hr, 4) for l, hr in zip(layers, hrs)},
            "mean": round(float(mean_hr), 4),
            "std": round(float(std_hr), 4),
        }
        print(f"\n{label}: mean={mean_hr:.1%}, std={std_hr:.1%}, range=[{min(hrs):.1%}, {max(hrs):.1%}]")

    axes[0].set_ylabel("Hit Rate (%)", fontsize=10)
    fig.suptitle(f"Per-Layer Hit Rate Distribution ({target_policy})", fontsize=12)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "per_layer_hitrate.pdf")
    fig.savefig(path, dpi=300)
    print(f"\nSaved {path}")
    plt.close()

    return results


# ═══════════════════════════════════════════════════════════════════════
# 5. Generate CI bar chart figure for paper
# ═══════════════════════════════════════════════════════════════════════
def fig_hitrate_with_ci(bootstrap_results, traces):
    """Bar chart with error bars showing 95% CI on hit rate."""
    policies = list(get_dsl_policies().keys())
    colors = ["#4C72B0", "#C44E52"]

    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(policies))
    width = 0.35

    for i, tname in enumerate(traces.keys()):
        label = DISPLAY.get(tname, tname)
        means = []
        ci_lo = []
        ci_hi = []
        for pname in policies:
            r = bootstrap_results[tname][pname]
            means.append(r["mean"] * 100)
            ci_lo.append((r["mean"] - r["ci95_lo"]) * 100)
            ci_hi.append((r["ci95_hi"] - r["mean"]) * 100)

        offset = (i - (len(traces) - 1) / 2) * width
        ax.bar(x + offset, means, width * 0.9, label=label,
               color=colors[i % len(colors)], alpha=0.85,
               yerr=[ci_lo, ci_hi], capsize=3, error_kw={"linewidth": 1})

    ax.set_xlabel("Policy", fontsize=10)
    ax.set_ylabel("Hit Rate (%) with 95% CI", fontsize=10)
    ax.set_title("Cache Hit Rate with Bootstrap 95% Confidence Intervals", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(policies, rotation=15, ha="right", fontsize=9)
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "hitrate_with_ci.pdf")
    fig.savefig(path, dpi=300)
    print(f"Saved {path}")
    plt.close()


def fig_latency_with_ci(latency_results, traces):
    """Bar chart of dispatch latency with error bars."""
    policies = list(get_dsl_policies().keys())
    colors = ["#4C72B0", "#C44E52"]

    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(policies))
    width = 0.35

    for i, tname in enumerate(traces.keys()):
        label = DISPLAY.get(tname, tname)
        means = [latency_results[tname][p]["mean_us"] for p in policies]
        cis = [latency_results[tname][p]["ci95_us"] for p in policies]

        offset = (i - (len(traces) - 1) / 2) * width
        ax.bar(x + offset, means, width * 0.9, label=label,
               color=colors[i % len(colors)], alpha=0.85,
               yerr=cis, capsize=3, error_kw={"linewidth": 1})

    ax.set_xlabel("Policy", fontsize=10)
    ax.set_ylabel("Dispatch Time (µs/layer) ± 95% CI", fontsize=10)
    ax.set_title("Per-Layer Dispatch Overhead with 95% Confidence Intervals", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(policies, rotation=15, ha="right", fontsize=9)
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "latency_with_ci.pdf")
    fig.savefig(path, dpi=300)
    print(f"Saved {path}")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Loading traces...")
    traces = load_all_traces()

    # 1. Dispatch latency
    latency_results = run_latency_analysis(traces)

    # 2. Bootstrap hit rate CIs
    bootstrap_results, per_token_data = run_bootstrap_analysis(traces)

    # 3. Pairwise significance
    significance_results = run_significance_tests(per_token_data)

    # 4. Per-layer analysis
    per_layer_results = run_per_layer_analysis(traces)

    # 5. Generate figures with CIs
    print("\n" + "=" * 70)
    print("5. GENERATING FIGURES WITH CIs")
    print("=" * 70)
    fig_hitrate_with_ci(bootstrap_results, traces)
    fig_latency_with_ci(latency_results, traces)

    # Save all results
    all_results = {
        "dispatch_latency": latency_results,
        "bootstrap_hitrate": bootstrap_results,
        "significance": significance_results,
        "per_layer": per_layer_results,
        "config": {
            "n_latency_runs": N_LATENCY_RUNS,
            "n_bootstrap": N_BOOTSTRAP,
            "ci_level": CI_LEVEL,
            "seed": SEED,
        },
    }

    # Remove numpy arrays for JSON serialization
    for tname in latency_results:
        for pname in latency_results[tname]:
            latency_results[tname][pname].pop("runs", None)

    out_path = os.path.join(TRACES_DIR, "stats_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nAll results saved to {out_path}")
    print(f"Figures saved to {FIG_DIR}")
