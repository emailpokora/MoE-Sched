#!/usr/bin/env python3
"""End-to-end EPCB validation on DeepSeek-V2-Lite (A100, fp16).

Closes the narrative gap from §5.6: the per-layer cache structural claim
(+14.7pp hit rate over shared cache, trace-replay) needs end-to-end
wall-clock validation on a model + hardware combination where:

  - The model is high-entropy-spread (DeepSeek-V2-Lite: ~2 nats inter-layer
    spread, vs Qwen's 0.158 nats), so EPCB's entropy signal differentiates.
  - VRAM headroom exceeds the aggregated per-layer cache, so per-layer
    dispatch isn't bottlenecked by the CUDA allocator hitting the ceiling
    (the Qwen-on-16GB failure mode).

DeepSeek-V2-Lite: 27 layers, 64 experts, top-6, ~16B total params (~32 GB fp16).
Fits with headroom on A100-40GB; comfortably on A100-80GB.

Configurations tested (matches paper Table 3 budgets 16/24/32/48 per layer):

  1. baseline_auto:    device_map='auto', no MoE-PolicyLang. Reference point.
  2. skeleton:         load_moe_model + cap=1 nocache. Isolates loading benefit.
  3. flat_cap32:       Shared cache, cap=32 LFU+history. Paper's strong baseline.
  4. per_layer_uniform: 32/layer x 27 layers = 864 total slots, uniform.
                       Tests the +14.7pp structural claim end-to-end.
  5. epcb_entropy:     Same 864 total, Shannon-entropy-allocated.
                       Tests the +2.2pp entropy claim end-to-end.

Optional larger budgets (--budget-large): runs config 4 and 5 at 1296 total
(48/layer avg) for a second data point on the structural claim.

Output: figures/deepseek_a100_results.json

==============================================================================
COLAB SETUP (Colab Pro / Pro+ with A100):

  # In a Colab cell, mount Drive and clone the repo:
  from google.colab import drive
  drive.mount('/content/drive')
  !git clone https://github.com/jesse-pokora/MoE-PolicyLang.git /content/repo
  %cd /content/repo
  !pip install -q torch transformers accelerate psutil lark

  # Point HF cache at a Drive folder so weights survive session restarts:
  import os
  os.environ['HF_HOME'] = '/content/drive/MyDrive/hf_cache'
  os.environ['HF_HUB_CACHE'] = '/content/drive/MyDrive/hf_cache/hub'

  # Run the benchmark:
  !python scripts/bench_deepseek_a100.py --runs 5

For the first run, model download takes ~5-10 min over Colab's network.
Subsequent runs reuse the cached weights.

==============================================================================
"""
import argparse
import gc
import json
import os
import statistics
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# Default HF cache to D: on local machines, /content/drive/... on Colab.
# Either can be overridden via the env vars before running.
if 'HF_HOME' not in os.environ:
    if os.path.exists('D:/hf_cache'):
        os.environ['HF_HOME'] = 'D:/hf_cache'
        os.environ['HF_HUB_CACHE'] = 'D:/hf_cache/hub'

import torch
import psutil

MODEL_ID = "deepseek-ai/DeepSeek-V2-Lite"
PROMPT = "Explain the key ideas behind mixture-of-experts models."
OUT_PATH = os.path.join(ROOT, "figures", "deepseek_a100_results.json")


def attach_policy(model, dsl):
    """Bypass moe_policylang.attach() to thread num_layers/num_experts
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
    if mgr is not None:
        reset_stats(mgr)
    inp = tok(prompt, return_tensors="pt").to(model.device)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=max_tokens, do_sample=False)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    gen = out.shape[1] - inp["input_ids"].shape[1]
    gpu_gb = torch.cuda.memory_allocated() / 1e9
    if mgr is None:
        return gen / elapsed, gpu_gb, None, 0
    stats = mgr.get_stats()
    hr = stats["policy"]["cache"]["hit_rate"]
    xf = stats["placement"]["cpu_to_gpu_transfers"]
    return gen / elapsed, gpu_gb, hr, xf


def measure(model, tok, mgr, label, max_tokens, warmup, runs):
    print(f"\n--- {label} ---")
    tps_list, gpu_list, hr_list, xf_list = [], [], [], []
    for w in range(warmup):
        tps, gpu, hr, xf = run_once(model, tok, mgr, max_tokens, PROMPT)
        print(f"  warmup {w+1}: {tps:.2f} tok/s  GPU={gpu:.1f}GB  hit={hr*100 if hr else 0:.1f}%")
    for r in range(runs):
        tps, gpu, hr, xf = run_once(model, tok, mgr, max_tokens, PROMPT)
        tps_list.append(tps); gpu_list.append(gpu)
        if hr is not None:
            hr_list.append(hr)
            xf_list.append(xf)
        print(f"  run {r+1}/{runs}: {tps:.2f} tok/s  GPU={gpu:.1f}GB  "
              f"hit={hr*100 if hr else 0:.1f}%  xfer={xf}")
    return {
        "tps_values": tps_list,
        "tps_mean": statistics.mean(tps_list),
        "tps_std": statistics.stdev(tps_list) if len(tps_list) > 1 else 0.0,
        "tps_steady_mean": statistics.mean(tps_list[1:]) if len(tps_list) >= 2 else tps_list[0],
        "tps_steady_std": statistics.stdev(tps_list[1:]) if len(tps_list) >= 3 else 0.0,
        "gpu_gb_mean": statistics.mean(gpu_list),
        "hit_rate_mean": statistics.mean(hr_list) if hr_list else None,
        "transfers_mean": statistics.mean(xf_list) if xf_list else 0,
    }


def build_per_layer_dsl(mode, total_budget, num_layers, num_experts):
    avg = total_budget // num_layers
    if mode == "uniform":
        return (
            f"policy uniform_b{total_budget} {{ "
            f"cache {{ capacity = {avg}  eviction = lfu  frequency_decay = 0.9 }} "
            f"prefetch {{ strategy = history  budget = 2 }} "
            f"per_layer {{ "
            f"  allocation = entropy "
            f"  entropy_window = 200 "
            f"  min_capacity = {avg}  max_capacity = {avg} "
            f"  rebalance_interval = 999999 "
            f"  total_budget = {total_budget} "
            f"}} "
            f"}}"
        )
    elif mode == "entropy":
        lo = max(2, avg // 2)
        hi = min(num_experts, avg * 3)
        return (
            f"policy entropy_b{total_budget} {{ "
            f"cache {{ capacity = {avg}  eviction = lfu  frequency_decay = 0.9 }} "
            f"prefetch {{ strategy = history  budget = 2 }} "
            f"per_layer {{ "
            f"  allocation = entropy "
            f"  entropy_window = 200 "
            f"  min_capacity = {lo}  max_capacity = {hi} "
            f"  rebalance_interval = 500 "
            f"  total_budget = {total_budget} "
            f"}} "
            f"}}"
        )
    raise ValueError(mode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", "-n", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--skip-baseline", action="store_true",
                    help="Skip device_map='auto' baseline (slow on A100 if it spills)")
    ap.add_argument("--budget-large", action="store_true",
                    help="Also run per-layer/EPCB at 1296 total (48/layer avg)")
    args = ap.parse_args()

    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    ram = psutil.virtual_memory()
    print(f"GPU: {gpu_name}  VRAM: {vram_gb:.1f} GB")
    print(f"RAM: {ram.total/1e9:.1f} GB total, {ram.available/1e9:.1f} GB available")
    print(f"HF cache: {os.environ.get('HF_HOME', '(default)')}")
    print(f"Model: {MODEL_ID}")
    print(f"Runs: {args.runs} (+{args.warmup} warmup), max_tokens: {args.max_tokens}")

    if vram_gb < 35:
        print(f"\nWARNING: GPU has {vram_gb:.0f} GB; DeepSeek-V2-Lite fp16 is ~32 GB.")
        print("Expert-aware loading still works (skeleton on GPU, experts on CPU)")
        print("but baseline_auto will OOM or thrash. Use --skip-baseline.")

    results = {
        "model": MODEL_ID,
        "gpu": gpu_name,
        "vram_gb": vram_gb,
        "runs": args.runs,
        "warmup": args.warmup,
        "max_tokens": args.max_tokens,
        "configs": {},
    }

    # ── Config 1: standard device_map='auto' baseline ─────────────────
    if not args.skip_baseline:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            print("\n" + "=" * 60)
            print("Loading baseline (device_map='auto')...")
            print("=" * 60)
            t0 = time.perf_counter()
            base_model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID, torch_dtype=torch.float16, device_map="auto",
                trust_remote_code=True,
            )
            base_model.eval()
            base_tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
            if base_tok.pad_token is None:
                base_tok.pad_token = base_tok.eos_token
            print(f"  loaded in {time.perf_counter()-t0:.1f}s, "
                  f"GPU={torch.cuda.memory_allocated()/1e9:.1f} GB")
            results["configs"]["baseline_auto"] = measure(
                base_model, base_tok, None, "baseline_auto",
                args.max_tokens, args.warmup, args.runs,
            )
            del base_model, base_tok
            gc.collect(); torch.cuda.empty_cache()
        except Exception as e:
            print(f"  baseline failed: {e}")
            results["configs"]["baseline_auto"] = {"error": str(e)}
            gc.collect(); torch.cuda.empty_cache()

    # ── Load model with expert-aware device map (reused for configs 2-N) ─
    import moe_policylang
    print("\n" + "=" * 60)
    print("Loading with expert-aware device map (load_moe_model)...")
    print("=" * 60)
    t0 = time.perf_counter()
    model, tok = moe_policylang.load_moe_model(MODEL_ID, trust_remote_code=True)
    print(f"  loaded in {time.perf_counter()-t0:.1f}s, skeleton "
          f"{torch.cuda.memory_allocated()/1e9:.2f} GB on GPU")

    from moe_policylang.integrations.accessors import auto_accessor
    accessor = auto_accessor(model)
    num_layers = accessor.num_layers
    num_experts = accessor.num_experts
    print(f"  Detected: {num_layers} layers, {num_experts} experts")
    results["num_layers"] = num_layers
    results["num_experts"] = num_experts
    results["skeleton_gpu_gb"] = torch.cuda.memory_allocated() / 1e9

    # ── Config 2: Skeleton-only (cap=1, isolates loading benefit) ──────
    dsl = (
        "policy skeleton { "
        "cache { capacity = 1  eviction = lru } "
        "prefetch { budget = 1 } "
        "}"
    )
    mgr = attach_policy(model, dsl)
    results["configs"]["skeleton_cap1"] = measure(
        model, tok, mgr, "skeleton_cap1",
        args.max_tokens, args.warmup, args.runs,
    )
    mgr.detach(); gc.collect(); torch.cuda.empty_cache()

    # ── Config 3: Flat cap=32 (paper's strong shared-cache baseline) ───
    dsl = (
        "policy flat_cap32 { "
        "cache { capacity = 32  eviction = lfu  frequency_decay = 0.9 } "
        "prefetch { strategy = history  budget = 4 } "
        "}"
    )
    mgr = attach_policy(model, dsl)
    results["configs"]["flat_cap32"] = measure(
        model, tok, mgr, "flat_cap32",
        args.max_tokens, args.warmup, args.runs,
    )
    mgr.detach(); gc.collect(); torch.cuda.empty_cache()

    # ── Config 4 & 5: per-layer uniform vs entropy at total=864 (=32/L) ─
    # This is the head-to-head: structural per-layer claim end-to-end.
    BUDGETS = [864]
    if args.budget_large:
        BUDGETS.append(1296)  # 48/layer avg
    for budget in BUDGETS:
        avg = budget // num_layers
        for mode in ("uniform", "entropy"):
            name = f"{mode}_b{budget}"
            dsl = build_per_layer_dsl(mode, budget, num_layers, num_experts)
            mgr = attach_policy(model, dsl)
            results["configs"][name] = measure(
                model, tok, mgr, f"{name} ({avg}/layer)",
                args.max_tokens, args.warmup, args.runs,
            )
            mgr.detach(); gc.collect(); torch.cuda.empty_cache()

    # ── Summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("SUMMARY: DeepSeek-V2-Lite end-to-end on", gpu_name)
    print("=" * 80)
    print(f"{'Config':<22s}  {'steady tok/s':<18s}  {'GPU (GB)':<10s}  {'Hit%':<7s}  {'xfer':<6s}")
    print("-" * 80)
    for name, c in results["configs"].items():
        if "error" in c:
            print(f"{name:<22s}  ERROR: {c['error']}")
            continue
        st = f"{c['tps_steady_mean']:.2f} +/- {c['tps_steady_std']:.2f}"
        hr = f"{c['hit_rate_mean']*100:5.1f}%" if c.get('hit_rate_mean') is not None else "  --- "
        xf = int(c.get('transfers_mean') or 0)
        print(f"{name:<22s}  {st:<18s}  {c['gpu_gb_mean']:<10.1f}  {hr:<7s}  {xf:<6d}")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {OUT_PATH}")

    # ── Quick interpretation guide for the user ──────────────────────
    print("\nInterpretation guide:")
    print("  - skeleton_cap1 vs baseline_auto:  measures loading-only speedup")
    print("    (replicates the +8.2x finding from Qwen on a second model)")
    print("  - flat_cap32 vs uniform_b864:  the structural per-layer claim")
    print("    (paper claims +14.7pp hit rate; here it's wall-clock)")
    print("  - uniform_b864 vs entropy_b864:  the entropy-signal claim")
    print("    (paper claims +2.2pp hit rate; here it's wall-clock)")
    print("  - All vs baseline_auto:  the headline end-to-end speedup")
    print("\nIf uniform_b864 > flat_cap32 in wall-clock, the structural claim")
    print("is validated end-to-end and the EPCB section comes back into the paper.")
    print("If entropy_b864 > uniform_b864, the entropy signal claim is also validated.")


if __name__ == "__main__":
    main()
