# Baselines

Real MoE serving systems for apples-to-apples comparison against MoE-Sched.

## Setup

### vLLM

```bash
pip install vllm
# Requires CUDA 11.8+ and ~50GB disk for Mixtral weights
```

### MoE-Infinity

```bash
pip install moe-infinity
# Or clone: git clone https://github.com/TorchMoE/MoE-Infinity
```

### DeepSpeed-MoE (optional)

```bash
pip install deepspeed
```

## Running

```bash
# After Phase 3 implementation:
python -m baselines.vllm_baseline --model mistralai/Mixtral-8x7B-v0.1 --prompts traces/sharegpt_sample.jsonl
python -m baselines.moe_infinity_baseline --model mistralai/Mixtral-8x7B-v0.1 --prompts traces/sharegpt_sample.jsonl
```

## Notes

- All baselines must run on **identical hardware** and **identical prompts** for fair comparison.
- Record GPU memory, tok/s, TTFT, and (if accessible) internal cache hit rates.
- Results should be exported in a format compatible with `moe_sched.benchmark.metrics.MetricsSummary`.
