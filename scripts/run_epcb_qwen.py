#!/usr/bin/env python3
"""Offline EPCB design-space replay on the recorded Qwen trace.

Mirrors scripts/run_epcb_design_space.py but on the Qwen trace recorded by
bench_qwen_budget_sweep.py. Question: does the entropy-vs-uniform parity
seen in the live benchmark (qwen_ablation, qwen_budget_sweep) also hold in
deterministic trace replay? If yes, EPCB's entropy signal genuinely adds
nothing on Qwen routing patterns; the live-bench result is not noise.

Outputs:
    traces/qwen_epcb_design_space.json
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# Reuse all the signal functions and the run_design_space machinery
from scripts.run_epcb_design_space import (
    load_trace, compute_layer_frequencies,
    SIGNAL_FUNCTIONS, allocate_by_signal, run_with_allocation,
)
from moe_policylang.ir import PolicyIR, CacheIR, EvictionPolicy


def main():
    trace_name = "qwen1.5_moe_a2.7b_trace"
    print("=" * 70)
    print(f"EPCB DESIGN-SPACE on {trace_name}")
    print("=" * 70)

    header, data = load_trace(trace_name)
    num_layers = header["num_layers"]
    num_experts = header["num_experts"]
    top_k = header.get("top_k", 4)
    print(f"Trace: {len(data)} entries, {num_layers} layers, {num_experts} experts, top-{top_k}")
    print()

    probs = compute_layer_frequencies(data, num_layers, num_experts)

    # Diagnostic: per-layer Shannon entropy distribution
    shannon = SIGNAL_FUNCTIONS["Shannon entropy"](probs)
    ent_values = sorted(shannon.items())
    print("Per-layer Shannon entropy:")
    for l, h in ent_values:
        print(f"  L{l:>2}: H={h:.4f}")
    e_min = min(h for _, h in ent_values)
    e_max = max(h for _, h in ent_values)
    e_mean = sum(h for _, h in ent_values) / len(ent_values)
    print(f"  range: [{e_min:.4f}, {e_max:.4f}], spread: {e_max - e_min:.4f}, mean: {e_mean:.4f}")
    print(f"  (lower spread = entropy allocation closer to uniform)")
    print()

    # Match Qwen budget levels we used in live bench: 48, 72, 96, 144 total
    # Per-layer averages: 2, 3, 4, 6
    base_ir = PolicyIR(
        name="epcb_qwen",
        cache=CacheIR(capacity=4, eviction=EvictionPolicy.LFU, lfu_decay=0.9),
    )

    all_results = {}
    for total_budget in [48, 72, 96, 144]:
        avg = total_budget // num_layers
        min_cap = max(2, avg // 2)
        max_cap = min(num_experts, avg * 3)
        print(f"\n--- Total budget={total_budget} ({avg}/layer avg, min={min_cap}, max={max_cap}) ---")
        print(f"{'Signal':<22} {'Hit %':>8} {'Hits':>9} {'Misses':>9} {'Used':>6}  {'Caps (first 5)':<25}")
        print("-" * 90)

        budget_results = {}
        for signal_name, signal_fn in SIGNAL_FUNCTIONS.items():
            signals = signal_fn(probs)
            allocation = allocate_by_signal(signals, total_budget, min_cap, max_cap)
            result = run_with_allocation(base_ir, allocation, num_layers, num_experts, data)
            budget_results[signal_name] = result
            caps_preview = [allocation[l] for l in range(min(5, num_layers))]
            print(f"{signal_name:<22} {result['hit_rate']*100:>7.2f}% "
                  f"{result['hits']:>9,} {result['misses']:>9,} "
                  f"{result['total_budget_used']:>6}  {str(caps_preview):<25}")

        all_results[f"budget{total_budget}"] = budget_results

    # Save
    out_path = os.path.join(ROOT, "traces", "qwen_epcb_design_space.json")
    serializable = {}
    for budget_key, budget_results in all_results.items():
        serializable[budget_key] = {}
        for signal_name, result in budget_results.items():
            serializable[budget_key][signal_name] = {
                k: v for k, v in result.items()
            }
    serializable["_per_layer_entropy"] = {str(l): h for l, h in shannon.items()}
    serializable["_meta"] = {
        "trace": trace_name,
        "num_layers": num_layers,
        "num_experts": num_experts,
        "top_k": top_k,
        "trace_entries": len(data),
    }
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Final summary table
    print("\n" + "=" * 70)
    print("SUMMARY: EPCB on Qwen (offline trace replay)")
    print("=" * 70)
    print(f"{'Budget':<8}  {'Shannon':>8}  {'Uniform':>8}  {'Delta':>8}  {'Best signal':<24}")
    print("-" * 70)
    for budget_key, results in all_results.items():
        shannon_rate = results["Shannon entropy"]["hit_rate"] * 100
        uniform_rate = results["Uniform"]["hit_rate"] * 100
        delta = shannon_rate - uniform_rate
        best_name = max(results, key=lambda s: results[s]["hit_rate"])
        best_rate = results[best_name]["hit_rate"] * 100
        print(f"{budget_key:<8}  {shannon_rate:>7.2f}%  {uniform_rate:>7.2f}%  "
              f"{delta:>+7.2f}pp  {best_name} ({best_rate:.2f}%)")


if __name__ == "__main__":
    main()
