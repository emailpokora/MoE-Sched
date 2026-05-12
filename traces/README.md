# Expert Activation Traces

This directory stores recorded expert activation traces from real MoE models.

## Files (to be generated)

| File | Model | Workload | Size |
|------|-------|----------|------|
| `mixtral_sharegpt.jsonl` | Mixtral-8x7B | ShareGPT 100 prompts | ~50MB |
| `mixtral_lmsys.jsonl` | Mixtral-8x7B | LMSYS-Chat subset | ~50MB |
| `mixtral_longcontext.jsonl` | Mixtral-8x7B | Long-context (32K+) | ~200MB |
| `deepseek_sharegpt.jsonl` | DeepSeek-V2-Lite | ShareGPT 100 prompts | ~100MB |
| `deepseek_lmsys.jsonl` | DeepSeek-V2-Lite | LMSYS-Chat subset | ~100MB |

## How to generate

```bash
# Requires GPU and model weights
python scripts/record_traces.py --model mistralai/Mixtral-8x7B-Instruct-v0.1 \
    --prompts evaluation/workloads/sharegpt_sample.json \
    --output traces/mixtral_sharegpt.jsonl
```

## Note

Trace files are large and should NOT be committed to git.
Add to `.gitignore`: `traces/*.jsonl`
