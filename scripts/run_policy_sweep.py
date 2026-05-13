#!/usr/bin/env python3
"""Policy sweep: auto-generates policies, benchmarks each, saves JSON.

Usage:
    python scripts/run_policy_sweep.py                  # OLMoE
    python scripts/run_policy_sweep.py --model mixtral   # Mixtral 4-bit
"""
import argparse, gc, json, os, sys, textwrap, time, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import moe_sched

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALIASES = {"olmoe": "allenai/OLMoE-1B-7B-0924",
           "mixtral": "mistralai/Mixtral-8x7B-Instruct-v0.1"}
QUANTIZE = {"mistralai/Mixtral-8x7B-Instruct-v0.1"}
PROMPTS = [
    "Explain quantum computing in simple terms.",
    "Write a Python function that sorts a list using merge sort.",
    "What are the main differences between TCP and UDP?",
    "Describe the process of photosynthesis step by step.",
]


def gen(model, tokenizer, prompts, max_tok):
    total = 0
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for p in prompts:
        inp = tokenizer(p, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=max_tok, do_sample=False)
        total += out.shape[1] - inp["input_ids"].shape[1]
    torch.cuda.synchronize()
    return total, time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="olmoe")
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    model_id = ALIASES.get(args.model, args.model)

    # Load
    kw = {"device_map": "auto"}
    if model_id in QUANTIZE:
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4")
    else:
        kw["torch_dtype"] = torch.float16
    model = AutoModelForCausalLM.from_pretrained(model_id, **kw)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    acc = moe_sched.integrations.auto_accessor(model)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9

    print(f"Model: {model_id}  GPU: {torch.cuda.get_device_name(0)} ({vram:.0f}GB)")
    print(f"Experts: {acc.num_experts}/layer x {len(acc.moe_layer_indices)} layers "
          f"({acc.expert_size_bytes()/1e6:.0f}MB each, top-{acc.top_k})\n")

    # Baseline
    base_tok, base_t = gen(model, tokenizer, PROMPTS, args.max_tokens)
    base_tps = base_tok / base_t
    print(f"{'Policy':<16} {'Cap':>4} {'Hit%':>6} {'CPU→GPU':>7} {'GPU→CPU':>7} "
          f"{'MB':>7} {'tok/s':>6} {'vs base':>7}")
    print(f"{'baseline':<16} {'—':>4} {'—':>6} {'—':>7} {'—':>7} "
          f"{'—':>7} {base_tps:>5.1f} {'1.00x':>7}")

    # Auto-generate + sweep
    policies = moe_sched.auto_policies(model)
    results = {}

    for name, dsl in policies.items():
        mgr = moe_sched.attach(model, dsl)
        tok_n, t = gen(model, tokenizer, PROMPTS, args.max_tokens)
        tps = tok_n / t
        s = mgr.get_stats()
        c, p = s["policy"]["cache"], s["placement"]

        print(f"{name:<16} {c.get('capacity','?'):>4} {c['hit_rate']:>5.1%} "
              f"{p['cpu_to_gpu_transfers']:>7} {p['gpu_to_cpu_transfers']:>7} "
              f"{p['bytes_transferred_mb']:>6.0f} {tps:>6.1f} "
              f"{tps/base_tps:>6.2f}x")

        results[name] = {
            "dsl": textwrap.dedent(dsl).strip(),
            "capacity": c.get("capacity"), "hit_rate": round(c["hit_rate"], 4),
            "hits": c["hits"], "misses": c["misses"], "evictions": c["evictions"],
            "cpu_to_gpu": p["cpu_to_gpu_transfers"],
            "gpu_to_cpu": p["gpu_to_cpu_transfers"],
            "mb": round(p["bytes_transferred_mb"], 1),
            "tps": round(tps, 1), "vs_base": round(tps / base_tps, 3),
        }
        mgr.detach(); gc.collect(); torch.cuda.empty_cache()

    best = max(results, key=lambda k: results[k]["tps"])
    worst = min(results, key=lambda k: results[k]["tps"])
    print(f"\nBest: {best} ({results[best]['tps']} tok/s)  "
          f"Worst: {worst} ({results[worst]['tps']} tok/s)  "
          f"Gap: {results[best]['tps']/results[worst]['tps']:.2f}x")

    out = args.output or os.path.join(ROOT, "traces", "policy_sweep.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({"model": model_id, "gpu": torch.cuda.get_device_name(0),
                    "vram_gb": round(vram, 1), "baseline_tps": round(base_tps, 1),
                    "policies": results}, f, indent=2)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
