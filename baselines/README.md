# Baselines

> Part of [MoE-Sched](../README.md) by **Jesse Pokora** &middot; [MIT License](../LICENSE)

Hand-coded and third-party MoE serving baselines for comparison against
MoE-Sched DSL-specified policies.

## Setup

### vLLM

```bash
pip install vllm   # Requires CUDA 11.8+
```

### MoE-Infinity

```bash
pip install moe-infinity
```

### DeepSpeed-MoE (optional)

```bash
pip install deepspeed
```

## Running

```bash
python -m baselines.vllm_baseline \
    --model mistralai/Mixtral-8x7B-v0.1 \
    --prompts traces/sharegpt_sample.jsonl

python -m baselines.moe_infinity_baseline \
    --model mistralai/Mixtral-8x7B-v0.1 \
    --prompts traces/sharegpt_sample.jsonl
```

## Comparison Methodology

- All baselines run on **identical hardware** and **identical prompts**.
- Metrics recorded: GPU memory, tokens/sec, time-to-first-token, and
  internal cache hit rates (where accessible).
- Results are exported in a format compatible with
  `moe_sched.benchmark.metrics.MetricsSummary`.
