"""Generate paper figures from Mixtral and DeepSeek trace data."""
import json
import os
import sys
import glob
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from moe_sched.compiler import compile_policy
from moe_sched.runtime.hooks import build_hook
from moe_sched.dsl import MoESched

TRACES_DIR = os.path.join(ROOT, "traces")
FIG_DIR = os.path.join(ROOT, "paper", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# Load all traces
traces = {}
for f in sorted(glob.glob(os.path.join(TRACES_DIR, "*.jsonl"))):
    lines = open(f).readlines()
    header = json.loads(lines[0])
    data = [json.loads(l) for l in lines[1:]]
    name = os.path.splitext(os.path.basename(f))[0]
    traces[name] = {"header": header, "data": data}
    print(f"Loaded {name}: {len(data)} entries ({header['model_name']})")

# Short display names
DISPLAY = {
    "mixtral_sample": "Mixtral-8×7B",
    "deepseek_v2_lite_sample": "DeepSeek-V2-Lite",
}


def _run_capacity_sweep(trace_data, caps, eviction="lru"):
    """Return list of hit rates for given capacities."""
    hit_rates = []
    for cap in caps:
        sched = MoESched()
        kw = dict(capacity=cap, eviction=eviction)
        if eviction == "lfu":
            kw["lfu_decay"] = 0.9
        ir = (
            sched.build(f"{eviction}_{cap}")
            .cache(**kw)
            .prefetch(strategy="none", budget=min(4, cap))
            .schedule(mode="gpu_only")
            .done()
        )
        hook = build_hook(compile_policy(ir))
        for e in trace_data:
            hook.on_layer(layer_idx=e["l"], selected_experts=e["e"], scores=e.get("s"))
        s = hook.stats_snapshot()
        h, m = s["cache"]["hits"], s["cache"]["misses"]
        hit_rates.append(h / max(1, h + m))
    return hit_rates


# ── Figure 1: Capacity sweep — both models ───────────────────────────
def fig_capacity_sweep():
    caps = [2, 4, 8, 16, 32]
    colors = ["#4C72B0", "#C44E52", "#55A868", "#8172B2"]
    fig, ax = plt.subplots(figsize=(7, 4))
    width = 0.35
    x = np.arange(len(caps))

    for i, (tname, tdata) in enumerate(traces.items()):
        hrs = _run_capacity_sweep(tdata["data"], caps)
        label = DISPLAY.get(tname, tname)
        offset = (i - (len(traces) - 1) / 2) * width
        bars = ax.bar(x + offset, [hr * 100 for hr in hrs], width * 0.9,
                      label=label, color=colors[i % len(colors)], alpha=0.85)
        for bar, hr in zip(bars, hrs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{hr:.0%}", ha="center", va="bottom", fontsize=7)

    ax.set_xlabel("Cache Capacity (experts)", fontsize=11)
    ax.set_ylabel("Hit Rate (%)", fontsize=11)
    ax.set_title("Cache Hit Rate vs. Capacity (LRU)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(caps)
    ax.set_ylim(0, 110)
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "capacity_sweep.pdf")
    fig.savefig(path, dpi=300)
    print(f"Saved {path}")
    plt.close()


# ── Figure 2: Rolling hit rate — side-by-side ────────────────────────
def fig_rolling_hit_rate():
    window = 500
    cap = 8  # same capacity for fair comparison
    fig, axes = plt.subplots(1, len(traces), figsize=(5.5 * len(traces), 3.5), sharey=True)
    if len(traces) == 1:
        axes = [axes]

    for ax, (tname, tdata) in zip(axes, traces.items()):
        label = DISPLAY.get(tname, tname)
        for ev in ["lru", "lfu"]:
            sched = MoESched()
            kw = dict(capacity=cap, eviction=ev)
            if ev == "lfu":
                kw["lfu_decay"] = 0.9
            ir = (
                sched.build(f"{ev}_{cap}")
                .cache(**kw)
                .prefetch(strategy="none", budget=min(4, cap))
                .schedule(mode="gpu_only")
                .done()
            )
            hook = build_hook(compile_policy(ir))
            hit_miss = []
            for e in tdata["data"]:
                s_before = hook.stats_snapshot()
                h_before = s_before["cache"]["hits"]
                hook.on_layer(layer_idx=e["l"], selected_experts=e["e"], scores=e.get("s"))
                s_after = hook.stats_snapshot()
                h_after = s_after["cache"]["hits"]
                hit_miss.append(1 if h_after > h_before else 0)

            cumsum = np.cumsum([0] + hit_miss)
            rolling_hr = (cumsum[window:] - cumsum[:-window]) / window
            ax.plot(range(window, len(hit_miss) + 1), rolling_hr * 100,
                    label=ev.upper(), linewidth=1.2)

        ax.set_xlabel("Dispatch Step", fontsize=10)
        ax.set_title(f"{label} (cap={cap})", fontsize=11)
        ax.legend(fontsize=8)
        ax.set_ylim(0, 105)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Rolling Hit Rate (%)", fontsize=10)
    fig.suptitle(f"Rolling Cache Hit Rate (window={window})", fontsize=12)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "rolling_hit_rate.pdf")
    fig.savefig(path, dpi=300)
    print(f"Saved {path}")
    plt.close()


# ── Figure 3: Ablation — side by side ────────────────────────────────
def fig_ablation():
    configs = [
        ("Cache\nonly", dict(capacity=8, eviction="lfu", lfu_decay=0.9),
         dict(strategy="none", budget=4), dict(mode="gpu_only")),
        ("+ Prefetch", dict(capacity=8, eviction="lfu", lfu_decay=0.9),
         dict(strategy="history", budget=4), dict(mode="gpu_only")),
        ("+ Scheduler", dict(capacity=8, eviction="lfu", lfu_decay=0.9),
         dict(strategy="history", budget=4), dict(mode="hybrid")),
        ("+ Triggers", dict(capacity=8, eviction="lfu", lfu_decay=0.9, ttl=50),
         dict(strategy="history", budget=4), dict(mode="hybrid")),
    ]
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]

    fig, axes = plt.subplots(1, len(traces), figsize=(5 * len(traces), 3.5), sharey=True)
    if len(traces) == 1:
        axes = [axes]

    for ax, (tname, tdata) in zip(axes, traces.items()):
        label = DISPLAY.get(tname, tname)
        abl_labels, abl_hrs = [], []
        for clabel, ckw, pkw, skw in configs:
            sched = MoESched()
            @sched.policy
            def _p(p, _c=ckw, _pk=pkw, _s=skw):
                p.cache(**_c)
                p.prefetch(**_pk)
                p.schedule(**_s)
            ir = sched.policies["_p"]
            hook = build_hook(compile_policy(ir))
            for e in tdata["data"]:
                hook.on_layer(layer_idx=e["l"], selected_experts=e["e"], scores=e.get("s"))
            s = hook.stats_snapshot()
            h, m = s["cache"]["hits"], s["cache"]["misses"]
            abl_labels.append(clabel)
            abl_hrs.append(h / max(1, h + m) * 100)

        bars = ax.bar(abl_labels, abl_hrs, color=colors, edgecolor="white")
        for bar, hr in zip(bars, abl_hrs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{hr:.1f}%", ha="center", va="bottom", fontsize=8)
        ax.set_title(f"{label} (cap=8)", fontsize=11)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Hit Rate (%)", fontsize=10)
    fig.suptitle("Ablation: Composition Axes", fontsize=12)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "ablation.pdf")
    fig.savefig(path, dpi=300)
    print(f"Saved {path}")
    plt.close()


# ── Figure 4: Expert activation frequency comparison ─────────────────
def fig_expert_frequency():
    fig, axes = plt.subplots(1, len(traces), figsize=(5.5 * len(traces), 3.5))
    if len(traces) == 1:
        axes = [axes]

    for ax, (tname, tdata) in zip(axes, traces.items()):
        label = DISPLAY.get(tname, tname)
        counts = Counter()
        for e in tdata["data"]:
            for eid in e["e"]:
                counts[eid] += 1
        num_experts = tdata["header"]["num_experts"]
        freqs = [counts.get(i, 0) for i in range(num_experts)]
        total = sum(freqs)
        pcts = [f / total * 100 for f in freqs]

        ax.bar(range(num_experts), pcts, color="#4C72B0", edgecolor="none", alpha=0.8)
        ax.set_xlabel("Expert ID", fontsize=10)
        ax.set_title(f"{label} ({num_experts} experts)", fontsize=11)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Activation Share (%)", fontsize=10)
    fig.suptitle("Expert Activation Frequency Distribution", fontsize=12)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "expert_frequency.pdf")
    fig.savefig(path, dpi=300)
    print(f"Saved {path}")
    plt.close()


# ── Figure 5: Dispatch overhead (from eval results) ──────────────────
def fig_overhead():
    # Load eval results
    results_by_trace = {}
    for f in glob.glob(os.path.join(TRACES_DIR, "eval_results_*.json")):
        with open(f) as fh:
            data = json.load(fh)
        tname = os.path.basename(f).replace("eval_results_", "").replace(".json", "")
        results_by_trace[tname] = data["policies"]

    if not results_by_trace:
        print("No eval results found, skipping overhead figure")
        return

    # Collect policy names from first trace
    policy_names = list(next(iter(results_by_trace.values())).keys())
    x = np.arange(len(policy_names))
    width = 0.35
    colors = ["#4C72B0", "#C44E52"]

    fig, ax = plt.subplots(figsize=(8, 3.5))
    for i, (tname, policies) in enumerate(results_by_trace.items()):
        label = DISPLAY.get(tname, tname)
        times = [policies[p]["us_per_layer"] for p in policy_names]
        offset = (i - (len(results_by_trace) - 1) / 2) * width
        ax.bar(x + offset, times, width * 0.9, label=label, color=colors[i % len(colors)], alpha=0.85)

    ax.set_xlabel("Policy", fontsize=10)
    ax.set_ylabel("Dispatch Time (\u00b5s/layer)", fontsize=10)
    ax.set_title("Per-Layer Dispatch Overhead", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(policy_names, rotation=15, ha="right", fontsize=9)
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "dispatch_overhead.pdf")
    fig.savefig(path, dpi=300)
    print(f"Saved {path}")
    plt.close()


# ── Figure 6: Capacity sweep line chart (both models) ────────────────
def fig_capacity_line():
    caps = [2, 4, 8, 16, 32]
    colors = ["#4C72B0", "#C44E52"]
    fig, ax = plt.subplots(figsize=(6, 4))

    for i, (tname, tdata) in enumerate(traces.items()):
        label = DISPLAY.get(tname, tname)
        hrs = _run_capacity_sweep(tdata["data"], caps)
        ax.plot(caps, [hr * 100 for hr in hrs], "o-", color=colors[i % len(colors)],
                label=label, linewidth=2, markersize=6)

    ax.set_xlabel("Cache Capacity (experts)", fontsize=11)
    ax.set_ylabel("Hit Rate (%)", fontsize=11)
    ax.set_title("Cache Hit Rate vs. Capacity (LRU)", fontsize=12)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "capacity_line.pdf")
    fig.savefig(path, dpi=300)
    print(f"Saved {path}")
    plt.close()


if __name__ == "__main__":
    fig_capacity_sweep()
    fig_capacity_line()
    fig_rolling_hit_rate()
    fig_ablation()
    fig_expert_frequency()
    fig_overhead()
    print(f"\nAll figures saved to {FIG_DIR}")
