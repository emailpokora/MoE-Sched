#!/usr/bin/env python3
"""Power/energy measurement during MoE inference.

Runs nvidia-smi as a sidecar process sampling power draw at 100ms intervals
during model inference, then reports:
  - Average power (W) during generation
  - Peak power (W)
  - Total energy (J) = integral of power over time
  - Energy per token (J/tok)
  - Comparison between offloaded (MoE-PolicyLang) and baseline modes

Usage:
    python scripts/bench_power.py
    python scripts/bench_power.py --max-tokens 128 --runs 3
"""

import argparse
import gc
import json
import os
import statistics
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import psutil

MODEL_ID = "Qwen/Qwen1.5-MoE-A2.7B"
PROMPT = "Explain the key ideas behind mixture-of-experts models in detail."
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")

POLICY_DSL = (
    "policy power_bench { "
    "cache { capacity = 8  eviction = lfu  frequency_decay = 0.9 } "
    "prefetch { strategy = history  budget = 4 } "
    "}"
)


class PowerMonitor:
    """Background thread that samples GPU power via nvidia-smi."""

    def __init__(self, gpu_index: int = 0, interval_ms: int = 100):
        self.gpu_index = gpu_index
        self.interval_ms = interval_ms
        self.samples = []  # list of (timestamp, power_watts)
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        """Start power sampling in background."""
        self._stop.clear()
        self.samples = []
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop sampling and return collected data."""
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        return self.samples

    def _sample_loop(self):
        """Poll nvidia-smi for power.draw."""
        while not self._stop.is_set():
            try:
                result = subprocess.run(
                    ["nvidia-smi",
                     f"--id={self.gpu_index}",
                     "--query-gpu=power.draw",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=2.0
                )
                if result.returncode == 0:
                    power_w = float(result.stdout.strip())
                    self.samples.append((time.perf_counter(), power_w))
            except (subprocess.TimeoutExpired, ValueError, FileNotFoundError):
                pass
            self._stop.wait(self.interval_ms / 1000.0)

    def summary(self):
        """Compute power/energy statistics from samples."""
        if len(self.samples) < 2:
            return {"error": "insufficient samples", "n_samples": len(self.samples)}

        powers = [s[1] for s in self.samples]
        timestamps = [s[0] for s in self.samples]

        # Total energy via trapezoidal integration
        total_energy_j = 0.0
        for i in range(1, len(self.samples)):
            dt = timestamps[i] - timestamps[i - 1]
            avg_power = (powers[i] + powers[i - 1]) / 2.0
            total_energy_j += avg_power * dt

        duration_s = timestamps[-1] - timestamps[0]
        return {
            "n_samples": len(self.samples),
            "duration_s": round(duration_s, 2),
            "avg_power_w": round(statistics.mean(powers), 1),
            "peak_power_w": round(max(powers), 1),
            "min_power_w": round(min(powers), 1),
            "total_energy_j": round(total_energy_j, 1),
        }


def run_inference(model, tok, max_tokens):
    """Run inference, return (generated_tokens, elapsed_s)."""
    inp = tok(PROMPT, return_tensors="pt").to(model.device)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=max_tokens, do_sample=False)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    gen_tokens = out.shape[1] - inp["input_ids"].shape[1]
    return gen_tokens, elapsed


def main():
    ap = argparse.ArgumentParser(description="Power/energy measurement")
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--runs", "-n", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--sample-interval-ms", type=int, default=100)
    ap.add_argument("--skip-baseline", action="store_true")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA required.")
        sys.exit(1)

    # Check nvidia-smi availability
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=power.draw",
                           "--format=csv,noheader,nounits"],
                          capture_output=True, text=True, timeout=5)
        if r.returncode != 0:
            print("ERROR: nvidia-smi power query failed.")
            sys.exit(1)
        print(f"Current GPU power: {r.stdout.strip()} W")
    except FileNotFoundError:
        print("ERROR: nvidia-smi not found. Install NVIDIA drivers.")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    tdp_w = None  # Could query from nvidia-smi
    print("=" * 70)
    print("Power/Energy Benchmark")
    print("=" * 70)
    print(f"  Model:     {args.model}")
    print(f"  GPU:       {gpu_name}")
    print(f"  Tokens:    {args.max_tokens}")
    print(f"  Runs:      {args.warmup}w + {args.runs}r per mode")
    print(f"  Sampling:  {args.sample_interval_ms}ms intervals")
    print("=" * 70)

    results = {
        "model": args.model,
        "gpu": gpu_name,
        "max_tokens": args.max_tokens,
        "sample_interval_ms": args.sample_interval_ms,
        "modes": {},
    }

    import moe_policylang

    # ── MoE-PolicyLang offloaded mode ─────────────────────────────────
    print("\n[1] Loading model (expert-offloaded)...")
    model, tok = moe_policylang.load_moe_model(args.model, trust_remote_code=True)
    mgr = moe_policylang.attach(model, POLICY_DSL)
    skeleton_gb = torch.cuda.memory_allocated() / 1e9
    print(f"  Skeleton: {skeleton_gb:.1f} GB on GPU")

    # Warmup
    for w in range(args.warmup):
        gen_tok, elapsed = run_inference(model, tok, args.max_tokens)
        print(f"  warmup {w+1}: {gen_tok/elapsed:.2f} tok/s")

    # Measured runs with power monitoring
    offload_runs = []
    print(f"\n  Measuring ({args.runs} runs)...")
    for i in range(args.runs):
        monitor = PowerMonitor(interval_ms=args.sample_interval_ms)
        monitor.start()
        gen_tok, elapsed = run_inference(model, tok, args.max_tokens)
        samples = monitor.stop()
        power = monitor.summary()
        tps = gen_tok / elapsed
        energy_per_tok = power["total_energy_j"] / gen_tok if "total_energy_j" in power else 0

        run_data = {
            "tps": tps,
            "elapsed_s": elapsed,
            "gen_tokens": gen_tok,
            **power,
            "energy_per_token_j": round(energy_per_tok, 3),
        }
        offload_runs.append(run_data)
        print(f"  run {i+1}/{args.runs}: {tps:.2f} tok/s  "
              f"avg={power.get('avg_power_w', '?')}W  "
              f"peak={power.get('peak_power_w', '?')}W  "
              f"energy={power.get('total_energy_j', '?')}J  "
              f"J/tok={energy_per_tok:.3f}")

    results["modes"]["offloaded"] = {
        "runs": offload_runs,
        "avg_tps": statistics.mean([r["tps"] for r in offload_runs]),
        "avg_power_w": statistics.mean([r.get("avg_power_w", 0) for r in offload_runs]),
        "avg_energy_per_token_j": statistics.mean([r["energy_per_token_j"] for r in offload_runs]),
        "total_energy_j_mean": statistics.mean([r.get("total_energy_j", 0) for r in offload_runs]),
    }

    mgr.detach()
    del model, tok
    gc.collect()
    torch.cuda.empty_cache()

    # ── Baseline (device_map=auto) ────────────────────────────────────
    if not args.skip_baseline:
        print(f"\n[2] Loading model (device_map='auto', baseline)...")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        try:
            model_bl = AutoModelForCausalLM.from_pretrained(
                args.model, torch_dtype=torch.float16, device_map="auto",
                trust_remote_code=True)
            model_bl.eval()
            tok_bl = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
            tok_bl.pad_token = tok_bl.eos_token

            # Warmup
            for w in range(args.warmup):
                gen_tok, elapsed = run_inference(model_bl, tok_bl, args.max_tokens)
                print(f"  warmup {w+1}: {gen_tok/elapsed:.2f} tok/s")

            # Measured
            baseline_runs = []
            print(f"\n  Measuring ({args.runs} runs)...")
            for i in range(args.runs):
                monitor = PowerMonitor(interval_ms=args.sample_interval_ms)
                monitor.start()
                gen_tok, elapsed = run_inference(model_bl, tok_bl, args.max_tokens)
                samples = monitor.stop()
                power = monitor.summary()
                tps = gen_tok / elapsed
                energy_per_tok = power["total_energy_j"] / gen_tok if "total_energy_j" in power else 0

                run_data = {
                    "tps": tps,
                    "elapsed_s": elapsed,
                    "gen_tokens": gen_tok,
                    **power,
                    "energy_per_token_j": round(energy_per_tok, 3),
                }
                baseline_runs.append(run_data)
                print(f"  run {i+1}/{args.runs}: {tps:.2f} tok/s  "
                      f"avg={power.get('avg_power_w', '?')}W  "
                      f"peak={power.get('peak_power_w', '?')}W  "
                      f"energy={power.get('total_energy_j', '?')}J  "
                      f"J/tok={energy_per_tok:.3f}")

            results["modes"]["baseline"] = {
                "runs": baseline_runs,
                "avg_tps": statistics.mean([r["tps"] for r in baseline_runs]),
                "avg_power_w": statistics.mean([r.get("avg_power_w", 0) for r in baseline_runs]),
                "avg_energy_per_token_j": statistics.mean([r["energy_per_token_j"] for r in baseline_runs]),
                "total_energy_j_mean": statistics.mean([r.get("total_energy_j", 0) for r in baseline_runs]),
            }

            del model_bl, tok_bl
        except Exception as e:
            print(f"  Baseline FAILED: {e}")
            results["modes"]["baseline"] = {"error": str(e)}

        gc.collect()
        torch.cuda.empty_cache()

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("POWER/ENERGY SUMMARY")
    print("=" * 70)
    print(f"{'Mode':<20s}  {'tok/s':<10s}  {'Avg W':<8s}  {'Peak W':<8s}  {'J/tok':<8s}  {'Total J':<8s}")
    print("-" * 70)

    for mode_name, mode_data in results["modes"].items():
        if "error" in mode_data:
            print(f"  {mode_name:<18s}  FAILED: {mode_data['error']}")
            continue
        print(f"  {mode_name:<18s}  "
              f"{mode_data['avg_tps']:<10.2f}  "
              f"{mode_data['avg_power_w']:<8.1f}  "
              f"{statistics.mean([r.get('peak_power_w', 0) for r in mode_data['runs']]):<8.1f}  "
              f"{mode_data['avg_energy_per_token_j']:<8.3f}  "
              f"{mode_data['total_energy_j_mean']:<8.1f}")

    if "baseline" in results["modes"] and "offloaded" in results["modes"]:
        bl = results["modes"]["baseline"]
        of = results["modes"]["offloaded"]
        if "error" not in bl and "error" not in of:
            print(f"\n  Energy efficiency: offloaded uses "
                  f"{of['avg_energy_per_token_j']/bl['avg_energy_per_token_j']:.2f}× "
                  f"energy per token vs baseline")
            print(f"  For 1-hour workload:")
            bl_tokens = bl["avg_tps"] * 3600
            of_tokens = of["avg_tps"] * 3600
            bl_energy_kwh = bl["avg_power_w"] * 1 / 1000
            of_energy_kwh = of["avg_power_w"] * 1 / 1000
            print(f"    Baseline: {bl_tokens:.0f} tokens, {bl_energy_kwh:.2f} kWh")
            print(f"    Offloaded: {of_tokens:.0f} tokens, {of_energy_kwh:.2f} kWh")

    # Save
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "power_benchmark.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
