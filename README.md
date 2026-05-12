# MoE-Sched — Conference Paper

Systems conference paper (MLSys / EuroSys / ASPLOS target) presenting
MoE-Sched, a domain-specific language for Mixture-of-Experts scheduling.

## Project Structure

```
MoE-Sched/
├── moe_sched/                  # Core DSL package
│   ├── grammar.lark            # Lark grammar definition
│   ├── parser.py               # Grammar → AST
│   ├── ir.py                   # Intermediate representation (PolicyIR)
│   ├── validator.py            # IR validation
│   ├── compiler.py             # IR → CompiledPolicy
│   ├── dsl.py                  # High-level DSL API
│   ├── adaptive.py             # Adaptive policy support (adapt blocks)
│   ├── autotuner.py            # Grid-search autotuner
│   ├── cli.py                  # CLI entry point (moe-sched command)
│   ├── benchmark/              # Benchmarking harness + metrics
│   ├── runtime/
│   │   ├── hooks.py            # PolicyHook — dispatch orchestrator
│   │   ├── cache.py            # LRU/LFU/Score/FreqThreshold caches
│   │   ├── scheduler.py        # GPU-only/CPU-fallback/Hybrid schedulers
│   │   ├── prefetch.py         # Affinity/History/Lookahead prefetchers
│   │   ├── per_layer.py        # Per-layer entropy-adaptive caching
│   │   ├── monitor.py          # Runtime metrics monitoring
│   │   ├── triggers.py         # Adaptive policy triggers
│   │   └── _fast/              # Cython accelerated versions
│   │       ├── _cache.pyx      # LRU/LFU cache
│   │       ├── _scheduler.pyx  # Scheduler components
│   │       └── _hooks.pyx      # Full dispatch loop
│   └── integrations/           # Model hookup
│       ├── huggingface.py      # HuggingFace integration
│       ├── trace_recorder.py   # Expert activation recording
│       ├── weight_placement.py # Weight placement strategies
│       └── mock_moe.py         # Mock model for testing
├── examples/                   # .moe policy files + demo scripts
│   └── cross_system_policies/  # vLLM/MoE-Infinity/ExpertFlow in DSL
├── evaluation/                 # Experiment framework
│   ├── configs/                # YAML experiment configs
│   └── workloads/              # ShareGPT sample workloads
├── paper/                      # LaTeX source (10 pages, ACM sigconf)
│   ├── main.tex                # Full paper
│   ├── semantics.tex           # Formal semantics appendix
│   ├── references.bib          # Bibliography
│   └── figures/                # Generated figures (PDF)
├── scripts/                    # Analysis & evaluation scripts
│   ├── run_eval.py             # Policy evaluation on traces
│   ├── run_sweep.py            # Capacity sweep + ablation
│   ├── run_stats.py            # Statistical analysis (CIs, Wilcoxon)
│   ├── run_deepseek_strategies.py  # DeepSeek caching strategies
│   ├── generate_figures.py     # Figure generation
│   ├── profile_dispatch.py     # Dispatch overhead profiling
│   ├── record_traces.py        # Trace recording helper
│   ├── run_all_experiments.py  # Full experiment orchestration
│   └── verify_cross_system.py  # Cross-system policy verification
├── notebooks/                  # Colab notebooks (A100 GPU)
│   ├── 00_full_pipeline.ipynb      # End-to-end pipeline
│   ├── 01_trace_recording.ipynb    # Record expert activation traces
│   ├── 02_profile_dispatch.ipynb   # Profile dispatch overhead + Cython
│   ├── 03_baselines.ipynb          # vLLM + MoE-Infinity baselines
│   ├── 04_full_evaluation.ipynb    # Full policy × workload evaluation
│   ├── 05_deepseek_traces.ipynb    # DeepSeek trace recording
│   └── 06_e2e_throughput.ipynb     # End-to-end throughput measurement
├── traces/                     # Expert activation traces (.jsonl)
├── results/                    # Benchmark outputs
├── tests/                      # Test suite
├── baselines/                  # Hand-coded baseline implementations
├── docs/                       # Documentation (MANUAL.md)
├── pyproject.toml              # Dependencies + project metadata
├── setup.py                    # Package build (find_packages)
└── setup_cython.py             # Cython extension build
```

## Quick Start

```bash
pip install -e ".[dev]"
python -m pytest tests/ -q
```

Expected: **362 passed, 3 skipped** (skipped = Cython tests, require `python setup_cython.py build_ext --inplace`)

## Building the Paper

```bash
cd paper
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

Produces `main.pdf` (10 pages, ACM sigconf format).

## Phase Status

| Phase | Status | Description |
|-------|--------|-------------|
| 1 — Model Integration | ⬜ Pending | Mixtral + DeepSeek traces on A100 |
| 2 — Cython Fast Path | ⬜ Pending | Cache + scheduler + full dispatch loop |
| 3 — Baselines | ⬜ Pending | vLLM comparison; MoE-Infinity baseline |
| 4 — Evaluation | ⬜ Pending | Capacity sweeps, ablation, statistical analysis |
| 5 — DSL Extensions | ✅ Done | Adaptive policies, autotuner, per-layer |
| 6 — Paper | ⬜ Pending | 10 pages, ACM sigconf |

## Workflow

**Local (Windows, no GPU):** DSL development, trace analysis, paper writing, tests
```bash
pip install -e ".[dev]"
python -m pytest tests/ -q
python scripts/run_stats.py          # Statistical analysis
python scripts/generate_figures.py   # Regenerate figures
```

**Colab (A100):** All GPU experiments — trace recording, dispatch profiling, baselines
1. Zip and upload: `zip -r moe_sched.zip moe_sched/` → Google Drive `moe-sched-paper/`
2. Open notebooks from `notebooks/` in Colab
3. Select A100 runtime
4. Results save to Google Drive → download locally for paper

## Colab Notebooks

| Notebook | Purpose |
|----------|---------|
| `00_full_pipeline.ipynb` | End-to-end pipeline (all steps) |
| `01_trace_recording.ipynb` | Load Mixtral/DeepSeek, record expert activation traces |
| `02_profile_dispatch.ipynb` | Profile Python vs Cython (component + full hook) dispatch |
| `03_baselines.ipynb` | vLLM + MoE-Infinity baseline comparison |
| `04_full_evaluation.ipynb` | Complete policy × workload × model evaluation |
| `05_deepseek_traces.ipynb` | DeepSeek-V2-Lite trace recording |
| `06_e2e_throughput.ipynb` | End-to-end throughput measurement |

## Key Results

- **Mixtral:** 99.2–99.7% hit rate with basic LRU (8 experts, top-2 = easy caching)
- **DeepSeek:** 48.6% baseline → **64.5%** with entropy-adaptive per-layer caching (+15.9pp at same budget)
- **Dispatch overhead:** 6–47 µs Python, <3.2% of MoE forward-pass time on A100
- **Code reduction:** 16–22× fewer lines vs hand-coded policies
- **Statistical rigor:** Bootstrap 95% CIs, Wilcoxon signed-rank tests (p < 0.001)
