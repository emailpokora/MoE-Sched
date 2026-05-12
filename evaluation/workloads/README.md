# Workloads

> Part of [MoE-Sched](../../README.md) by **Jesse Pokora** &middot; [MIT License](../../LICENSE)

Workload definitions and trace sources for MoE-Sched benchmarks.

## Trace Sources

- **ShareGPT** — ~90K real conversations ([HuggingFace](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered))
- **LMSYS-Chat-1M** — 1M diverse conversations ([HuggingFace](https://huggingface.co/datasets/lmsys/lmsys-chat-1m)), subset to 1,000
- **Long-context** — Synthetic 32K+ token sequences from concatenated ShareGPT turns

## Trace Format (`.jsonl`)

```json
{"model_name": "Mixtral-8x7B", "num_layers": 32, "num_experts": 8, "top_k": 2, "num_entries": 1600}
{"t": 0, "l": 0, "e": [3, 7], "s": [0.82, 0.18]}
{"t": 0, "l": 1, "e": [1, 5], "s": [0.71, 0.29]}
```

| Field | Description |
|-------|-------------|
| `t` | Token index |
| `l` | Layer index |
| `e` | Selected expert IDs |
| `s` | Router softmax scores (optional) |

## Recording Traces

```bash
python scripts/record_traces.py \
    --model mistralai/Mixtral-8x7B-Instruct-v0.1 \
    --prompts evaluation/workloads/sharegpt_sample.json \
    --output traces/mixtral_sharegpt.jsonl \
    --max-tokens 128
```

Requires a GPU and model weights.
