# Notebooks

> Part of [MoE-PolicyLang](../README.md) by **Jesse Pokora** &middot; [MIT License](../LICENSE)

Jupyter notebooks for trace recording, profiling, and evaluation.
GPU experiments run on **Google Colab** with an A100 runtime.

## Notebooks

| Notebook | Purpose |
|----------|---------|
| `trace_recording.ipynb` | Load Mixtral-8×7B, record expert activation traces |
| `profile_dispatch.ipynb` | Profile Python vs. Cython dispatch overhead |
| `full_evaluation.ipynb` | Full policy × workload × model evaluation |
| `deepseek_traces.ipynb` | DeepSeek-V2-Lite trace recording and strategy analysis |

## Setup

1. Upload `moe_policylang/` package to Colab (or mount from Google Drive)
2. Each notebook installs its own dependencies in the first cell
3. Results are saved to Google Drive for local analysis

## Google Drive Layout

```
My Drive/moe-policylang-paper/
├── moe_policylang/     # Package source
├── traces/             # Recorded traces (.jsonl)
├── results/            # Experiment outputs (JSON, PDF)
└── checkpoints/        # Model cache
```

## Reproducing on Local Hardware

The live inference experiments run Qwen1.5-MoE-A2.7B with physical
expert offloading on a consumer GPU — no Colab required.

### Requirements

- **GPU**: NVIDIA GPU with ≥ 16 GB VRAM (tested on RTX 5080 Laptop)
- **RAM**: ≥ 32 GB system memory
- **Disk**: ~30 GB for model weights (downloaded automatically)
- **Python**: 3.10+
- **CUDA**: 11.8+ with compatible PyTorch

### Steps

1. **Install dependencies**

   ```bash
   pip install -e ".[gpu,eval]"
   ```

2. **(Optional) Redirect model cache to a larger drive**

   ```bash
   # Windows
   set HF_HOME=D:\hf_cache

   # Linux / macOS
   export HF_HOME=/path/to/large/drive/hf_cache
   ```

3. **Run the experiments**

   ```bash
   # Full demo with figures
   python scripts/run_qwen_moe_demo.py

   # Multi-run benchmark (mean ± std, n=3)
   python scripts/bench_qwen_multirun.py --runs 3

   # Output equivalence verification
   python scripts/verify_output_equivalence.py
   ```

   On first run, Qwen1.5-MoE-A2.7B weights (~28.6 GB fp16) are
   downloaded from HuggingFace. Subsequent runs use the local cache.

4. **View results**

   - `figures/qwen_multirun_results.json` — throughput mean ± std
   - `figures/output_equivalence.json` — bit-identical output verification
   - `figures/policy_sweep_qwen.pdf` — throughput/VRAM/hit-rate tradeoff
   - `figures/vram_comparison_qwen.pdf` — VRAM comparison chart
