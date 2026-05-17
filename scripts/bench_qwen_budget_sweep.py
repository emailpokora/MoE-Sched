#!/usr/bin/env python3
"""Per-layer budget sweep + trace recording on Qwen1.5-MoE.

Two tasks in one model load (~67s saved):

  1. Record an expert routing trace from Qwen so offline EPCB replay can
     answer "does entropy-vs-uniform parity hold in trace replay?"

  2. Sweep per-layer budgets {48, 72, 144} x {uniform, entropy} to find
     where the throughput-vs-hit-rate trade-off flips. (budget=96 was
     covered in qwen_ablation_results.json; we don't re-run it.)

Trace -> traces/qwen1.5_moe_a2.7b_trace.jsonl
Sweep -> figures/qwen_budget_sweep_results.json

Usage:
    python scripts/bench_qwen_budget_sweep.py --runs 5
"""
import argparse
import gc
import json
import os
import statistics
import sys
import time

os.environ.setdefault("HF_HOME", "D:/hf_cache")
os.environ.setdefault("HF_HUB_CACHE", "D:/hf_cache/hub")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch
import psutil

MODEL_ID = "Qwen/Qwen1.5-MoE-A2.7B"
PROMPT = "Explain the key ideas behind mixture-of-experts models."
TRACE_PROMPTS = [
    "Explain the difference between LRU and LFU caching strategies in two sentences.",
    "What is a Mixture-of-Experts model and why is it useful?",
    "Describe how GPU memory management affects large language model serving.",
    "Write a short Python function that computes factorial recursively.",
]
TRACE_PATH = os.path.join(ROOT, "traces", "qwen1.5_moe_a2.7b_trace.jsonl")
SWEEP_OUT = os.path.join(ROOT, "figures", "qwen_budget_sweep_results.json")


def attach_policy(model, dsl):
    """Wrap moe_policylang.attach() to thread num_layers/num_experts
    through to per_layer policies (library's attach drops them)."""
    from moe_policylang.parser import parse_policy
    from moe_policylang.compiler import compile_policy
    from moe_policylang.runtime.hooks import build_hook
    from moe_policylang.integrations.accessors import auto_accessor
    from moe_policylang.integrations.weight_placement import WeightPlacementManager

    accessor = auto_accessor(model)
    ir = parse_policy(dsl)
    compiled = compile_policy(ir)
    hook = build_hook(
        compiled,
        num_layers=accessor.num_layers,
        num_experts=accessor.num_experts,
    )
    mgr = WeightPlacementManager(hook, accessor)
    mgr.attach()
    return mgr


def reset_stats(mgr):
    hook = mgr.hook
    if hasattr(hook, "_per_layer_hooks"):
        for layer_hook in hook._per_layer_hooks.values():
            layer_hook.cache.stats.hits = 0
            layer_hook.cache.stats.misses = 0
            layer_hook.cache.stats.evictions = 0
    elif hasattr(hook, "inner") and hasattr(hook.inner, "cache"):
        hook.inner.cache.stats.hits = 0
        hook.inner.cache.stats.misses = 0
        hook.inner.cache.stats.evictions = 0
    elif hasattr(hook, "cache"):
        hook.cache.stats.hits = 0
        hook.cache.stats.misses = 0
        hook.cache.stats.evictions = 0
    if hasattr(mgr, "stats"):
        mgr.stats.cpu_to_gpu_transfers = 0
        mgr.stats.gpu_to_cpu_transfers = 0
        mgr.stats.bytes_transferred = 0
        mgr.stats.transfer_time_s = 0.0


def run_once(model, tok, mgr, max_tokens, prompt):
    reset_stats(mgr)
    inp = tok(prompt, return_tensors="pt").to(model.device)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=max_tokens, do_sample=False)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    gen = out.shape[1] - inp["input_ids"].shape[1]
    stats = mgr.get_stats()
    return gen / elapsed, torch.cuda.memory_allocated() / 1e9, stats


def build_dsl(budget, mode, num_layers=24):
    """Build a per-layer DSL policy at a given total budget + allocation mode."""
    avg = budget // num_layers
    if mode == "uniform":
        # Lock both min and max to the average, and disable rebalancing entirely.
        # This produces uniform per-layer caches regardless of routing entropy.
        return (
            f"policy uniform_b{budget} {{ "
            f"cache {{ capacity = {avg}  eviction = lfu  frequency_decay = 0.9 }} "
            f"prefetch {{ strategy = history  budget = 2 }} "
            f"per_layer {{ "
            f"  allocation = entropy "
            f"  entropy_window = 200 "
            f"  min_capacity = {avg}  max_capacity = {avg} "
            f"  rebalance_interval = 999999 "
            f"  total_budget = {budget} "
            f"}} "
            f"}}"
        )
    elif mode == "entropy":
        # Wider min/max range so entropy can actually reallocate.
        lo = max(2, avg // 2)
        hi = min(num_layers * 2, avg * 3)  # cap at ~3x average
        return (
            f"policy entropy_b{budget} {{ "
            f"cache {{ capacity = {avg}  eviction = lfu  frequency_decay = 0.9 }} "
            f"prefetch {{ strategy = history  budget = 2 }} "
            f"per_layer {{ "
            f"  allocation = entropy "
            f"  entropy_window = 200 "
            f"  min_capacity = {lo}  max_capacity = {hi} "
            f"  rebalance_interval = 500 "
            f"  total_budget = {budget} "
            f"}} "
            f"}}"
        )
    else:
        raise ValueError(mode)


# ── trace recording ──────────────────────────────────────────────────

class TraceRecorder:
    def __init__(self, num_layers):
        self.num_layers = num_layers
        self.entries = []
        self._token_counter = 0
        self._recording = True

    def make_hook(self, layer_idx):
        def hook_fn(module, inp, out):
            if not self._recording:
                return
            # Qwen gate output: tuple where index 0 = topk_idx, 1 = topk_weights
            if not (isinstance(out, tuple) and len(out) >= 2):
                return
            expert_ids = out[0].detach().cpu()
            weights = out[1].detach().cpu()
            for tok_idx in range(expert_ids.shape[0]):
                self.entries.append({
                    "t": self._token_counter + tok_idx,
                    "l": layer_idx,
                    "e": [int(e) for e in expert_ids[tok_idx].tolist()],
                    "s": [round(float(s), 4) for s in weights[tok_idx].tolist()],
                })
            if layer_idx == 0:
                self._token_counter += expert_ids.shape[0]
        return hook_fn

    def stop(self):
        self._recording = False

    def save(self, path, model_name, num_layers, num_experts, top_k):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        header = {
            "model_name": model_name,
            "num_layers": num_layers,
            "num_experts": num_experts,
            "top_k": top_k,
            "total_entries": len(self.entries),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        with open(path, "w") as f:
            f.write(json.dumps(header) + "\n")
            for e in self.entries:
                f.write(json.dumps(e) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", "-n", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--trace-tokens", type=int, default=128,
                    help="Tokens per prompt during trace recording")
    ap.add_argument("--skip-trace", action="store_true")
    ap.add_argument("--skip-sweep", action="store_true")
    args = ap.parse_args()

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"HF cache: {os.environ.get('HF_HOME')}")
    print()

    import moe_policylang

    print("=" * 60)
    print("Loading Qwen1.5-MoE-A2.7B with expert-aware device map...")
    print("=" * 60)
    t_load = time.perf_counter()
    model, tok = moe_policylang.load_moe_model(MODEL_ID, trust_remote_code=True)
    print(f"  Skeleton: {torch.cuda.memory_allocated()/1e9:.2f} GB "
          f"(load took {time.perf_counter()-t_load:.1f}s)")
    print()

    # Detect layer / expert counts
    from moe_policylang.integrations.accessors import auto_accessor
    accessor = auto_accessor(model)
    num_layers = accessor.num_layers
    num_experts = accessor.num_experts
    print(f"Detected: {num_layers} layers, {num_experts} experts")
    print()

    # ── 1. Trace recording ─────────────────────────────────────────
    if not args.skip_trace:
        print("=" * 60)
        print("STAGE 1: Recording routing trace")
        print("=" * 60)
        # Use nocache policy for fastest generation
        mgr = attach_policy(model,
            "policy nocache { cache { capacity = 1  eviction = lru } "
            "prefetch { budget = 1 } }")

        # Attach trace hooks on gate modules
        gates = [model.model.layers[i].mlp.gate for i in range(num_layers)
                 if hasattr(model.model.layers[i].mlp, "gate")]
        print(f"  Found {len(gates)} gate modules")
        recorder = TraceRecorder(num_layers)
        handles = [g.register_forward_hook(recorder.make_hook(i))
                   for i, g in enumerate(gates)]

        # Warmup generation (not recorded — disable recording)
        recorder._recording = False
        for w in range(1):
            inp = tok(PROMPT, return_tensors="pt").to(model.device)
            with torch.no_grad():
                model.generate(**inp, max_new_tokens=32, do_sample=False)
        recorder._recording = True
        recorder._token_counter = 0
        recorder.entries.clear()

        t0 = time.perf_counter()
        for i, p in enumerate(TRACE_PROMPTS):
            inp = tok(p, return_tensors="pt").to(model.device)
            with torch.no_grad():
                model.generate(**inp, max_new_tokens=args.trace_tokens, do_sample=False)
            n = len(recorder.entries)
            dt = time.perf_counter() - t0
            print(f"  Prompt {i+1}/{len(TRACE_PROMPTS)}: {n} entries  ({dt:.1f}s)")
        recorder.stop()
        for h in handles:
            h.remove()

        # Get top_k from accessor or config
        top_k = getattr(model.config, "num_experts_per_tok", 4)
        recorder.save(TRACE_PATH, MODEL_ID, num_layers, num_experts, top_k)
        print(f"  Saved {len(recorder.entries)} entries to {TRACE_PATH}")
        mgr.detach()
        gc.collect()
        torch.cuda.empty_cache()
        print()

    # ── 2. Budget sweep ────────────────────────────────────────────
    if args.skip_sweep:
        return

    print("=" * 60)
    print("STAGE 2: Per-layer budget sweep")
    print("=" * 60)

    BUDGETS = [48, 72, 144]
    MODES = ["uniform", "entropy"]
    results = {
        "model": MODEL_ID,
        "gpu": torch.cuda.get_device_name(0),
        "num_layers": num_layers,
        "num_experts": num_experts,
        "runs": args.runs,
        "warmup": args.warmup,
        "max_tokens": args.max_tokens,
        "configs": {},
    }

    for budget in BUDGETS:
        for mode in MODES:
            name = f"{mode}_b{budget}"
            dsl = build_dsl(budget, mode, num_layers=num_layers)
            print(f"\n--- {name}  (avg={budget // num_layers}/layer) ---")

            mgr = attach_policy(model, dsl)

            tps_list, gpu_list, hr_list, xf_list = [], [], [], []
            last_stats = None
            for w in range(args.warmup):
                tps, gpu, stats = run_once(model, tok, mgr, args.max_tokens, PROMPT)
                hr = stats["policy"]["cache"]["hit_rate"]
                print(f"  warmup: {tps:.2f} tok/s  hit={hr:.1%}")

            for r in range(args.runs):
                tps, gpu, stats = run_once(model, tok, mgr, args.max_tokens, PROMPT)
                hr = stats["policy"]["cache"]["hit_rate"]
                xf = stats["placement"]["cpu_to_gpu_transfers"]
                tps_list.append(tps)
                gpu_list.append(gpu)
                hr_list.append(hr)
                xf_list.append(xf)
                last_stats = stats
                print(f"  run {r+1}/{args.runs}: {tps:.2f} tok/s  GPU={gpu:.1f}GB  "
                      f"hit={hr:.1%}  xfer={xf}")

            entry = {
                "dsl": dsl,
                "budget": budget,
                "mode": mode,
                "tps_values": tps_list,
                "tps_mean": statistics.mean(tps_list),
                "tps_std": statistics.stdev(tps_list) if len(tps_list) > 1 else 0.0,
                "tps_steady_mean": statistics.mean(tps_list[1:]) if len(tps_list) >= 2 else tps_list[0],
                "tps_steady_std": statistics.stdev(tps_list[1:]) if len(tps_list) >= 3 else 0.0,
                "gpu_gb_mean": statistics.mean(gpu_list),
                "hit_rate_mean": statistics.mean(hr_list),
                "transfers_mean": statistics.mean(xf_list),
            }
            if last_stats and "per_layer" in last_stats.get("policy", {}):
                entry["per_layer_capacities"] = last_stats["policy"].get("capacities")
            results["configs"][name] = entry

            mgr.detach()
            gc.collect()
            torch.cuda.empty_cache()

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Config':<20s}  {'tok/s (5)':<18s}  {'steady (2-5)':<18s}  {'Hit%':<6s}  {'Xfer':<6s}")
    print("-" * 80)
    for name, c in results["configs"].items():
        all_str = f"{c['tps_mean']:.2f} +/- {c['tps_std']:.2f}"
        st_str = f"{c['tps_steady_mean']:.2f} +/- {c['tps_steady_std']:.2f}"
        print(f"{name:<20s}  {all_str:<18s}  {st_str:<18s}  {c['hit_rate_mean']*100:5.1f}%  {int(c['transfers_mean']):>6d}")

    os.makedirs(os.path.dirname(SWEEP_OUT), exist_ok=True)
    with open(SWEEP_OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {SWEEP_OUT}")


if __name__ == "__main__":
    main()
