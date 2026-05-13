# Expert Activation Traces

> Part of [MoE-PolicyLang](../README.md) by **Jesse Pokora** &middot; [MIT License](../LICENSE)

This directory stores recorded expert activation traces and experiment results.

## Trace Files

| File | Model | Description |
|------|-------|-------------|
| `mixtral_sample.jsonl` | Mixtral-8×7B | 8 experts, 32 layers, top-2 routing |
| `deepseek_v2_lite_sample.jsonl` | DeepSeek-V2-Lite | 64 experts, 27 layers, top-6 routing |

## Experiment Results

| File | Description |
|------|-------------|
| `eval_results.json` | Cross-policy evaluation (both models) |
| `eval_results_mixtral_sample.json` | Mixtral-specific evaluation |
| `eval_results_deepseek_v2_lite_sample.json` | DeepSeek-specific evaluation |
| `sweep_results.json` | Capacity sweep results |
| `deepseek_strategy_results.json` | DeepSeek caching strategy comparison |
| `stats_results.json` | Statistical analysis (bootstrap CIs, Wilcoxon) |
| `constrained_e2e_results.json` | Live OLMoE inference on RTX 5080 |

## Recording New Traces

```bash
python scripts/record_traces.py \
    --model mistralai/Mixtral-8x7B-Instruct-v0.1 \
    --prompts evaluation/workloads/sharegpt_sample.json \
    --output traces/mixtral_sharegpt.jsonl
```

Requires a GPU and model weights. See `evaluation/workloads/README.md` for
trace format details.

## Note

Large `.jsonl` trace files are excluded from version control via `.gitignore`.
