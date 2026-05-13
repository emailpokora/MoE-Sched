# Notebooks

> Part of [MoE-PolicyLang](../README.md) by **Jesse Pokora** &middot; [MIT License](../LICENSE)

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

1. Upload `moe_policylang/` package to Colab (or mount from Google Drive)
2. Each notebook installs its own dependencies in the first cell
3. Results are saved to Google Drive for local analysis

## Google Drive Layout

```
My Drive/moe-policylang-paper/
├── moe_policylang/          # Package source
├── traces/             # Recorded traces (.jsonl)
├── results/            # Experiment outputs (JSON, PDF)
└── checkpoints/        # Model cache
```

## Reproducing on Local Hardware

The live inference experiment (`scripts/run_constrained_e2e.py`) runs
OLMoE-1B-7B with MoE-PolicyLang hooks on a consumer GPU — no Colab required.

### Requirements

- **GPU**: NVIDIA GPU with ≥ 16 GB VRAM (tested on RTX 5080 Laptop)
- **RAM**: ≥ 16 GB system memory
- **Disk**: ~14 GB for model weights (downloaded automatically)
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

3. **Run the experiment**

   ```bash
   python scripts/run_constrained_e2e.py
   ```

   On first run, OLMoE-1B-7B weights (~14 GB) are downloaded from
   HuggingFace. Subsequent runs use the local cache.

4. **View results**

   - `traces/constrained_e2e_results.json` — raw metrics (tok/s, cache
     hits/misses, hit rate, dispatch latency per policy)
   - `paper/figures/constrained_throughput.pdf` — bar chart comparing
     throughput and hit rate across policies

### Expected Output

```
Policy           tok/s   Hit Rate  Peak GB
vanilla           39.2        N/A     14.0
naive_c4          34.6       2.4%     14.0
lru_c16           34.7      26.3%     14.0
lfu_hist_c16      33.8      27.1%     14.0
epcb_c16          33.6      47.3%     14.0
```

Numbers will vary by hardware. The key result is EPCB's ~1.8× hit-rate
improvement over LRU at equal cache capacity (16 experts).
