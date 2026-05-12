# Colab Notebooks

All GPU experiments run on **Google Colab** with A100 runtime.

## Notebooks

| Notebook | Status | Purpose |
|----------|--------|---------|
| `01_trace_recording.ipynb` | ✅ | Load Mixtral, record expert activation traces |
| `02_profile_dispatch.ipynb` | ✅ | Profile Python vs Cython (component + full hook) dispatch |
| `03_baselines.ipynb` | 🔄 | vLLM baseline; MoE-Infinity placeholder |
| `04_full_evaluation.ipynb` | ✅ | Full policy × workload × model evaluation |
| `05_deepseek_traces.ipynb` | ✅ | DeepSeek-V2-Lite trace recording |

## Setup

1. Upload `moe_sched/` package to Colab (or mount from Google Drive)
2. Each notebook installs its own dependencies in the first cell
3. Results are saved to Google Drive for local analysis

## Google Drive Layout

```
My Drive/moe-sched-paper/
├── moe_sched/          # Package source (zip uploaded from local)
├── traces/             # Recorded traces (.jsonl)
├── results/            # Experiment outputs (JSON, PDF figures)
└── checkpoints/        # Model cache (avoid re-downloading)
```
