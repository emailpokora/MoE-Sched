#!/usr/bin/env python3
"""Verify that expert offloading produces identical outputs.

Loads Qwen1.5-MoE-A2.7B two ways:
  1. Standard: device_map="auto" (baseline)
  2. MoE-PolicyLang: expert-aware loading + each policy config

Compares generated token IDs to confirm exact equivalence.
Results saved to figures/output_equivalence.json.

Usage (close IDE first to maximize free RAM):
    python scripts/verify_output_equivalence.py
    python scripts/verify_output_equivalence.py --max-tokens 32
    python scripts/verify_output_equivalence.py --prompts 3
"""
import argparse
import gc
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import psutil

MODEL_ID = "Qwen/Qwen1.5-MoE-A2.7B"
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")

PROMPTS = [
    "Explain the key ideas behind mixture-of-experts models.",
    "What is the capital of France and why is it historically significant?",
    "Write a Python function that computes the Fibonacci sequence.",
    "Describe the process of photosynthesis in simple terms.",
]

POLICIES = {
    "aggressive (cap=2, LRU)": (
        "policy a { cache { capacity = 2  eviction = lru } "
        "prefetch { strategy = history  budget = 1 } }"
    ),
    "balanced (cap=4, LFU)": (
        "policy b { cache { capacity = 4  eviction = lfu  frequency_decay = 0.9 } "
        "prefetch { strategy = history  budget = 2 } }"
    ),
    "generous (cap=8, LFU)": (
        "policy c { cache { capacity = 8  eviction = lfu  frequency_decay = 0.9 } "
        "prefetch { strategy = history  budget = 4 } }"
    ),
}


def generate(model, tok, prompt, max_tokens):
    """Generate with greedy decoding.  Returns (token_ids, text)."""
    inp = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inp, max_new_tokens=max_tokens, do_sample=False,
            temperature=1.0, top_p=1.0,
        )
    gen_ids = out[0, inp["input_ids"].shape[1]:].tolist()
    text = tok.decode(gen_ids, skip_special_tokens=True)
    return gen_ids, text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-tokens", type=int, default=32,
                    help="Tokens to generate per prompt (default: 32)")
    ap.add_argument("--prompts", type=int, default=len(PROMPTS),
                    help=f"Number of prompts to test (max {len(PROMPTS)})")
    args = ap.parse_args()

    prompts = PROMPTS[:args.prompts]

    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {gpu_name}  VRAM: {vram_gb:.1f} GB")
    print(f"RAM: {psutil.virtual_memory().available/1e9:.1f} GB available")
    print(f"Testing {len(prompts)} prompts × {1 + len(POLICIES)} configs")
    print(f"Max tokens: {args.max_tokens}")
    print()

    results = {
        "model": MODEL_ID,
        "gpu": gpu_name,
        "max_tokens": args.max_tokens,
        "num_prompts": len(prompts),
        "prompts": prompts,
        "configs": {},
        "equivalence": {},
    }

    # ── Step 1: Baseline generation ───────────────────────────────────
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("=" * 60)
    print("BASELINE: device_map='auto'")
    print("=" * 60)

    full_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map="auto",
        trust_remote_code=True)
    full_model.eval()
    btok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    btok.pad_token = btok.eos_token

    baseline_outputs = {}
    for i, prompt in enumerate(prompts):
        ids, text = generate(full_model, btok, prompt, args.max_tokens)
        baseline_outputs[i] = {"ids": ids, "text": text}
        print(f"  prompt {i+1}: {len(ids)} tokens")
        print(f"    → {text[:100]}...")

    results["configs"]["baseline"] = {
        "method": "device_map=auto",
        "outputs": {str(i): o["text"] for i, o in baseline_outputs.items()},
    }

    del full_model, btok
    gc.collect()
    torch.cuda.empty_cache()
    print()

    # ── Step 2: MoE-PolicyLang generation ─────────────────────────────
    import moe_policylang

    print("=" * 60)
    print("Loading with expert-aware device map...")
    print("=" * 60)
    model, tok = moe_policylang.load_moe_model(MODEL_ID, trust_remote_code=True)
    print(f"  Skeleton: {torch.cuda.memory_allocated()/1e9:.1f} GB on GPU")
    print()

    all_match = True

    for policy_name, dsl in POLICIES.items():
        print("=" * 60)
        print(f"POLICY: {policy_name}")
        print("=" * 60)

        mgr = moe_policylang.attach(model, dsl)

        policy_outputs = {}
        matches = {}
        for i, prompt in enumerate(prompts):
            ids, text = generate(model, tok, prompt, args.max_tokens)
            policy_outputs[i] = {"ids": ids, "text": text}

            baseline_ids = baseline_outputs[i]["ids"]
            match = ids == baseline_ids
            matches[i] = match

            status = "✓ MATCH" if match else "✗ MISMATCH"
            print(f"  prompt {i+1}: {status}")
            if not match:
                all_match = False
                # Show where they diverge
                min_len = min(len(ids), len(baseline_ids))
                for j in range(min_len):
                    if ids[j] != baseline_ids[j]:
                        print(f"    first diff at token {j}: "
                              f"baseline={baseline_ids[j]} vs policy={ids[j]}")
                        break
                if len(ids) != len(baseline_ids):
                    print(f"    length: baseline={len(baseline_ids)} vs "
                          f"policy={len(ids)}")
            else:
                print(f"    → {text[:80]}...")

        results["configs"][policy_name] = {
            "dsl": dsl,
            "outputs": {str(i): o["text"] for i, o in policy_outputs.items()},
            "matches": {str(i): m for i, m in matches.items()},
            "all_match": all(matches.values()),
        }

        mgr.detach()
        gc.collect()
        torch.cuda.empty_cache()
        print()

    # ── Summary ───────────────────────────────────────────────────────
    print("=" * 60)
    print("EQUIVALENCE SUMMARY")
    print("=" * 60)

    total_tests = 0
    total_pass = 0
    for config_name, config in results["configs"].items():
        if config_name == "baseline":
            continue
        matches = config.get("matches", {})
        n_match = sum(1 for v in matches.values() if v)
        n_total = len(matches)
        total_tests += n_total
        total_pass += n_match
        status = "ALL MATCH" if n_match == n_total else f"{n_match}/{n_total}"
        symbol = "✓" if n_match == n_total else "✗"
        print(f"  {symbol} {config_name}: {status}")

    results["equivalence"] = {
        "total_tests": total_tests,
        "total_pass": total_pass,
        "all_equivalent": total_pass == total_tests,
    }

    print()
    if total_pass == total_tests:
        print(f"  ✓ ALL {total_tests} TESTS PASSED — outputs are identical")
        print("    Expert offloading introduces no numerical divergence.")
    else:
        print(f"  ✗ {total_tests - total_pass}/{total_tests} MISMATCHES detected")
        print("    See details above. This may indicate non-determinism in")
        print("    the model or differences in device-map placement order.")

    # ── Save ──────────────────────────────────────────────────────────
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "output_equivalence.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")
    print("Done.")


if __name__ == "__main__":
    main()
