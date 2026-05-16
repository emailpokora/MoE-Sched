#!/usr/bin/env python3
"""Cold-start vs warm-state throughput analysis.

Measures per-token throughput over a single long generation to show:
1. Cache warmup phase (first N tokens with low hit rate)
2. Steady-state phase (cache saturated, high hit rate)
3. Transition point

Outputs a JSON with per-token timing suitable for plotting.

Usage:
    python scripts/bench_coldstart.py
    python scripts/bench_coldstart.py --max-tokens 256
"""

import argparse
import gc
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

MODEL_ID = "Qwen/Qwen1.5-MoE-A2.7B"
PROMPT = "Write a detailed tutorial on building neural network architectures from scratch, covering"
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")

POLICY_DSL = (
    "policy coldstart { "
    "cache { capacity = 8  eviction = lfu  frequency_decay = 0.9 } "
    "prefetch { strategy = history  budget = 4 } "
    "}"
)


def main():
    ap = argparse.ArgumentParser(description="Cold-start throughput analysis")
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--max-tokens", type=int, default=200,
                    help="Tokens to generate (longer = more steady-state data)")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA required.")
        sys.exit(1)

    import moe_policylang

    gpu_name = torch.cuda.get_device_name(0)
    print("=" * 70)
    print("Cold-Start Throughput Analysis")
    print("=" * 70)
    print(f"  Model:  {args.model}")
    print(f"  GPU:    {gpu_name}")
    print(f"  Tokens: {args.max_tokens}")
    print(f"  Policy: cap=8 LFU + history prefetch")
    print("=" * 70)

    # Load model
    print("\nLoading model...")
    model, tok = moe_policylang.load_moe_model(args.model, trust_remote_code=True)
    print(f"  Skeleton: {torch.cuda.memory_allocated() / 1e9:.1f} GB on GPU")

    # Attach policy
    mgr = moe_policylang.attach(model, POLICY_DSL)

    # Encode prompt
    inp = tok(PROMPT, return_tensors="pt").to(model.device)
    prompt_len = inp["input_ids"].shape[1]
    print(f"  Prompt length: {prompt_len} tokens")
    print(f"  Generating {args.max_tokens} tokens...\n")

    # Generate token-by-token to measure per-token latency
    # We'll use model.generate with a custom approach:
    # generate one token at a time via greedy loop
    token_times = []
    hit_rates = []
    cumulative_hits = 0
    cumulative_total = 0

    input_ids = inp["input_ids"].clone()
    past_key_values = None

    torch.cuda.synchronize()
    gen_start = time.perf_counter()

    with torch.no_grad():
        for i in range(args.max_tokens):
            # Reset per-token stats
            pre_hits = mgr.hook.cache.stats.hits
            pre_misses = mgr.hook.cache.stats.misses

            torch.cuda.synchronize()
            t0 = time.perf_counter()

            if past_key_values is None:
                outputs = model(input_ids=input_ids, use_cache=True)
            else:
                outputs = model(
                    input_ids=input_ids[:, -1:],
                    past_key_values=past_key_values,
                    use_cache=True,
                )

            torch.cuda.synchronize()
            t1 = time.perf_counter()

            # Greedy next token
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            past_key_values = outputs.past_key_values

            # Stats
            elapsed = t1 - t0
            token_times.append(elapsed)

            post_hits = mgr.hook.cache.stats.hits
            post_misses = mgr.hook.cache.stats.misses
            token_hits = post_hits - pre_hits
            token_misses = post_misses - pre_misses
            token_total = token_hits + token_misses
            cumulative_hits += token_hits
            cumulative_total += token_total

            if cumulative_total > 0:
                cum_hr = cumulative_hits / cumulative_total
            else:
                cum_hr = 0.0
            hit_rates.append(cum_hr)

            if (i + 1) % 20 == 0 or i == 0:
                tps = 1.0 / elapsed if elapsed > 0 else 0
                print(f"  token {i+1:>3d}: {tps:>6.1f} tok/s  "
                      f"latency={elapsed*1000:.1f}ms  "
                      f"cum_hit_rate={cum_hr:.1%}")

    torch.cuda.synchronize()
    total_time = time.perf_counter() - gen_start
    avg_tps = args.max_tokens / total_time

    # Compute phases
    # Warmup = tokens where hit_rate < 90% of final hit_rate
    final_hr = hit_rates[-1] if hit_rates else 0
    threshold = 0.9 * final_hr
    warmup_end = 0
    for i, hr in enumerate(hit_rates):
        if hr >= threshold:
            warmup_end = i
            break
    else:
        warmup_end = len(hit_rates)

    # Per-window throughput (windows of 10 tokens)
    window_size = 10
    window_tps = []
    for start in range(0, len(token_times), window_size):
        chunk = token_times[start:start + window_size]
        if chunk:
            window_tps.append(len(chunk) / sum(chunk))

    print(f"\n{'=' * 70}")
    print("RESULTS")
    print("=" * 70)
    print(f"  Total generation: {total_time:.1f}s for {args.max_tokens} tokens")
    print(f"  Average throughput: {avg_tps:.2f} tok/s")
    print(f"  Final hit rate: {final_hr:.1%}")
    print(f"  Warmup ends at token: ~{warmup_end}")

    if warmup_end > 0 and warmup_end < len(token_times):
        warmup_tps = len(token_times[:warmup_end]) / sum(token_times[:warmup_end])
        steady_tps = len(token_times[warmup_end:]) / sum(token_times[warmup_end:])
        print(f"  Warmup phase tok/s: {warmup_tps:.2f}")
        print(f"  Steady-state tok/s: {steady_tps:.2f}")
        print(f"  Speedup warm/cold: {steady_tps/warmup_tps:.2f}×")

    # First 10 vs last 10
    first10_tps = 10 / sum(token_times[:10]) if len(token_times) >= 10 else 0
    last10_tps = 10 / sum(token_times[-10:]) if len(token_times) >= 10 else 0
    print(f"\n  First 10 tokens: {first10_tps:.2f} tok/s")
    print(f"  Last 10 tokens:  {last10_tps:.2f} tok/s")
    print(f"  Ratio: {last10_tps/first10_tps:.2f}×" if first10_tps > 0 else "")

    # Save
    results = {
        "model": args.model,
        "gpu": gpu_name,
        "max_tokens": args.max_tokens,
        "prompt_length": prompt_len,
        "total_time_s": total_time,
        "avg_tps": avg_tps,
        "final_hit_rate": final_hr,
        "warmup_end_token": warmup_end,
        "per_token_latency_ms": [t * 1000 for t in token_times],
        "per_token_tps": [1.0 / t if t > 0 else 0 for t in token_times],
        "cumulative_hit_rate": hit_rates,
        "window_tps_10tok": window_tps,
    }

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "coldstart_throughput.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {out_path}")

    mgr.detach()


if __name__ == "__main__":
    main()
