# Copyright (c) 2026 Jesse Pokora — MIT License (see LICENSE)
"""Capacity sweep and eviction comparison on expert traces.

Usage:
    python run_sweep.py                         # Mixtral only
    python run_sweep.py --all                   # all traces
    python run_sweep.py traces/deepseek_v2_lite_sample.jsonl
"""
import json
import time
import sys
import os
import glob

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from moe_policylang.compiler import compile_policy
from moe_policylang.runtime.hooks import build_hook
from moe_policylang.dsl import MoEPolicyLang

TRACES_DIR = os.path.join(ROOT, "traces")


def load_trace(path):
    lines = open(path).readlines()
    header = json.loads(lines[0])
    return header, [json.loads(l) for l in lines[1:]]

def run_sweep(trace_path):
    header, trace = load_trace(trace_path)
    trace_name = os.path.splitext(os.path.basename(trace_path))[0]
    print(f"\n{'#' * 80}")
    print(f"Sweep: {header['model_name']} ({trace_name}, {len(trace)} entries)\n")

    # --- Capacity sweep ---
    print("Cache Capacity Sweep (LRU, no prefetch, gpu_only)")
    hdr = f"{'Cap':>5} {'Hits':>8} {'Misses':>8} {'Hit Rate':>10} {'Evictions':>10} {'us/layer':>10}"
    print(hdr)
    print("-" * len(hdr))

    sweep_results = []
    for cap in [2, 3, 4, 5, 6, 7, 8]:
        sched = MoEPolicyLang()
        ir = (
            sched.build(f"lru_{cap}")
            .cache(capacity=cap, eviction="lru")
            .prefetch(strategy="none", budget=min(4, cap))
            .schedule(mode="gpu_only")
            .done()
        )
        hook = build_hook(compile_policy(ir))
        t0 = time.perf_counter()
        for e in trace:
            hook.on_layer(layer_idx=e["l"], selected_experts=e["e"], scores=e.get("s"))
        elapsed = time.perf_counter() - t0
        us = (elapsed / len(trace)) * 1e6
        s = hook.stats_snapshot()
        h, m = s["cache"]["hits"], s["cache"]["misses"]
        hr = h / max(1, h + m)
        ev = s["cache"]["evictions"]
        print(f"{cap:>5} {h:>8} {m:>8} {hr:>9.1%} {ev:>10} {us:>9.1f}")
        sweep_results.append({"capacity": cap, "hit_rate": round(hr, 4), "evictions": ev})

    # --- Eviction comparison at cap=4 ---
    print()
    print("Eviction Strategy Comparison (capacity=4)")
    hdr2 = f"{'Eviction':>12} {'Hits':>8} {'Misses':>8} {'Hit Rate':>10} {'Evictions':>10}"
    print(hdr2)
    print("-" * len(hdr2))

    eviction_results = []
    for ev_name in ["lru", "lfu"]:
        sched = MoEPolicyLang()
        kw = dict(capacity=4, eviction=ev_name)
        if ev_name == "lfu":
            kw["lfu_decay"] = 0.9
        ir = (
            sched.build(f"{ev_name}_4")
            .cache(**kw)
            .prefetch(strategy="none")
            .schedule(mode="gpu_only")
            .done()
        )
        hook = build_hook(compile_policy(ir))
        for e in trace:
            hook.on_layer(layer_idx=e["l"], selected_experts=e["e"], scores=e.get("s"))
        s = hook.stats_snapshot()
        h, m = s["cache"]["hits"], s["cache"]["misses"]
        hr = h / max(1, h + m)
        evictions = s["cache"]["evictions"]
        print(f"{ev_name:>12} {h:>8} {m:>8} {hr:>9.1%} {evictions:>10}")
        eviction_results.append({"eviction": ev_name, "hit_rate": round(hr, 4), "evictions": evictions})

    # --- Ablation ---
    print()
    print("Ablation Study (capacity=4, LFU)")
    hdr3 = f"{'Config':<20} {'Hit Rate':>10} {'Evictions':>10}"
    print(hdr3)
    print("-" * len(hdr3))

    ablation_configs = [
        ("cache only",    dict(capacity=4, eviction="lfu", lfu_decay=0.9), dict(strategy="none"),            dict(mode="gpu_only")),
        ("+ prefetch",    dict(capacity=4, eviction="lfu", lfu_decay=0.9), dict(strategy="history", budget=2), dict(mode="gpu_only")),
        ("+ scheduler",   dict(capacity=4, eviction="lfu", lfu_decay=0.9), dict(strategy="history", budget=2), dict(mode="hybrid")),
        ("+ triggers",    dict(capacity=4, eviction="lfu", lfu_decay=0.9, ttl=50), dict(strategy="history", budget=2), dict(mode="hybrid")),
    ]

    ablation_results = []
    for label, cache_kw, pf_kw, sched_kw in ablation_configs:
        sched = MoEPolicyLang()

        @sched.policy
        def ablation_policy(p, _ck=cache_kw, _pk=pf_kw, _sk=sched_kw):
            p.cache(**_ck)
            p.prefetch(**_pk)
            p.schedule(**_sk)

        ir = sched.policies["ablation_policy"]
        hook = build_hook(compile_policy(ir))
        for e in trace:
            hook.on_layer(layer_idx=e["l"], selected_experts=e["e"], scores=e.get("s"))
        s = hook.stats_snapshot()
        h, m = s["cache"]["hits"], s["cache"]["misses"]
        hr = h / max(1, h + m)
        evictions = s["cache"]["evictions"]
        print(f"{label:<20} {hr:>9.1%} {evictions:>10}")
        ablation_results.append({"config": label, "hit_rate": round(hr, 4), "evictions": evictions})

    # Save
    out = os.path.join(TRACES_DIR, f"sweep_results_{trace_name}.json")
    with open(out, "w") as f:
        json.dump({"sweep": sweep_results, "eviction_cmp": eviction_results, "ablation": ablation_results}, f, indent=2)
    print(f"\nResults saved to {out}")
    return {"sweep": sweep_results, "eviction_cmp": eviction_results, "ablation": ablation_results}


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        run_sweep(os.path.join(TRACES_DIR, "mixtral_sample.jsonl"))
    elif args[0] == "--all":
        for f in sorted(glob.glob(os.path.join(TRACES_DIR, "*.jsonl"))):
            run_sweep(f)
    else:
        run_sweep(args[0])
