# Real Workloads

## Trace Sources

### ShareGPT
- Download: https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered
- ~90K conversations, variable length
- Use `scripts/record_traces.py` to convert to expert activation traces

### LMSYS-Chat-1M
- Download: https://huggingface.co/datasets/lmsys/lmsys-chat-1m
- 1M conversations with diverse prompt types
- Subset to first 1000 for tractable experiments

### Long-context
- Synthetic: concatenate multiple ShareGPT turns into 32K+ token sequences
- Tests prefetcher and trigger behavior under sustained access

## Trace Format

Traces are stored as `.jsonl` files with the following format:

```json
{"model_name": "Mixtral-8x7B", "num_layers": 32, "num_experts": 8, "top_k": 2, "num_entries": 1600}
{"t": 0, "l": 0, "e": [3, 7], "s": [0.82, 0.18]}
{"t": 0, "l": 1, "e": [1, 5], "s": [0.71, 0.29]}
...
```

Where:
- `t` = token index
- `l` = layer index
- `e` = selected expert IDs
- `s` = router softmax scores (optional)

## Recording Traces

```bash
python scripts/record_traces.py \
    --model mistralai/Mixtral-8x7B-Instruct-v0.1 \
    --prompts evaluation/workloads/sharegpt_sample.json \
    --output traces/mixtral_sharegpt.jsonl \
    --max-tokens 128
```
