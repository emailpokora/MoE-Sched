"""Run all benchmark policies and autotuner against expert traces.

Usage:
    python run_eval.py                          # Mixtral only
    python run_eval.py --all                    # all traces in traces/
    python run_eval.py traces/deepseek_v2_lite_sample.jsonl  # specific trace
"""
import json
import time
import sys
import os
import glob

# Ensure moe_sched is importable
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from moe_sched.benchmark.policies import get_dsl_policies
from moe_sched.compiler import compile_policy
from moe_sched.runtime.hooks import build_hook
from moe_sched.autotuner import autotune

TRACES_DIR = os.path.join(ROOT, "traces")


def load_trace(path):
    with open(path) as f:
        lines = f.readlines()
    header = json.loads(lines[0])
    entries = [json.loads(l) for l in lines[1:]]
    return header, entries


def run_policies(trace_data):
    policies = get_dsl_policies()
    results = {}

    hdr = f"{'Policy':<25} {'Hits':>8} {'Misses':>8} {'Hit Rate':>10} {'Evictions':>10} {'us/layer':>10}"
    print(hdr)
    print("-" * len(hdr))

    for name, compiled in policies.items():
        hook = build_hook(compiled)
        t0 = time.perf_counter()
        for entry in trace_data:
            hook.on_layer(
                layer_idx=entry["l"],
                selected_experts=entry["e"],
                scores=entry.get("s"),
            )
        elapsed = time.perf_counter() - t0
        us_per = (elapsed / len(trace_data)) * 1e6

        stats = hook.stats_snapshot()
        hits = stats["cache"]["hits"]
        misses = stats["cache"]["misses"]
        total = hits + misses
        hr = hits / total if total > 0 else 0
        evictions = stats["cache"]["evictions"]

        print(f"{name:<25} {hits:>8} {misses:>8} {hr:>9.1%} {evictions:>10} {us_per:>9.1f}")
        results[name] = {
            "hits": hits,
            "misses": misses,
            "hit_rate": round(hr, 4),
            "evictions": evictions,
            "us_per_layer": round(us_per, 1),
        }

    return results


def run_autotuner(trace_data):
    print("\nAutotuner (default grid)...")
    t0 = time.perf_counter()
    best, top5 = autotune(trace_data, top_k=5, measure_latency=True)
    elapsed = time.perf_counter() - t0

    print(f"Completed in {elapsed:.1f}s")
    print(f"\n{'Rank':<6} {'Hit Rate':>10} {'Evictions':>10} {'us/layer':>10} {'Config'}")
    print("-" * 80)
    for i, r in enumerate(top5):
        cfg = f"cap={r.params['capacity']}, ev={r.params['eviction']}, pf={r.params['prefetch_strategy']}, sched={r.params['schedule_mode']}"
        print(f"{i+1:<6} {r.hit_rate:>9.1%} {r.evictions:>10} {r.dispatch_mean_us:>9.1f} {cfg}")

    return {
        "best": {
            "params": {k: str(v) for k, v in best.params.items()},
            "hit_rate": round(best.hit_rate, 4),
            "evictions": best.evictions,
            "us_per_layer": round(best.dispatch_mean_us, 1),
        },
        "top5": [
            {
                "params": {k: str(v) for k, v in r.params.items()},
                "hit_rate": round(r.hit_rate, 4),
                "evictions": r.evictions,
            }
            for r in top5
        ],
    }


def eval_trace(trace_path):
    header, trace_data = load_trace(trace_path)
    trace_name = os.path.splitext(os.path.basename(trace_path))[0]
    print(f"\n{'#' * 80}")
    print(f"Trace: {header['model_name']} ({trace_name})")
    print(f"Entries: {len(trace_data)}, Layers: {header['num_layers']}, Experts: {header['num_experts']}")
    print()

    print("=" * 73)
    print("BENCHMARK POLICIES")
    print("=" * 73)
    policy_results = run_policies(trace_data)

    print()
    print("=" * 80)
    print("AUTOTUNER")
    print("=" * 80)
    autotune_results = run_autotuner(trace_data)

    out_path = os.path.join(TRACES_DIR, f"eval_results_{trace_name}.json")
    with open(out_path, "w") as f:
        json.dump(
            {"header": header, "policies": policy_results, "autotuner": autotune_results},
            f, indent=2,
        )
    print(f"\nResults saved to {out_path}")
    return {"header": header, "policies": policy_results, "autotuner": autotune_results}


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        # Default: Mixtral
        eval_trace(os.path.join(TRACES_DIR, "mixtral_sample.jsonl"))
    elif args[0] == "--all":
        for f in sorted(glob.glob(os.path.join(TRACES_DIR, "*.jsonl"))):
            eval_trace(f)
    else:
        eval_trace(args[0])
