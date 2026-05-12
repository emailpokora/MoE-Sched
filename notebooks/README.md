# Notebooks

> Part of [MoE-Sched](../README.md) by **Jesse Pokora** &middot; [MIT License](../LICENSE)

Jupyter notebooks for trace recording, profiling, and evaluation.
GPU experiments run on **Google Colab** with an A100 runtime.

## Notebooks

| Notebook | Purpose |
|----------|---------|
| `01_trace_recording.ipynb` | Load Mixtral-8×7B, record expert activation traces |
| `02_profile_dispatch.ipynb` | Profile Python vs. Cython dispatch overhead |
| `03_baselines.ipynb` | vLLM and MoE-Infinity baseline comparison |
| `04_full_evaluation.ipynb` | Full policy × workload × model evaluation |
| `05_deepseek_traces.ipynb` | DeepSeek-V2-Lite trace recording |
| `06_e2e_throughput.ipynb` | End-to-end throughput benchmark (hooks vs. vanilla) |

## Setup

1. Upload `moe_sched/` package to Colab (or mount from Google Drive)
2. Each notebook installs its own dependencies in the first cell
3. Results are saved to Google Drive for local analysis

## Google Drive Layout

```
My Drive/moe-sched-paper/
├── moe_sched/          # Package source
├── traces/             # Recorded traces (.jsonl)
├── results/            # Experiment outputs (JSON, PDF)
└── checkpoints/        # Model cache
```
