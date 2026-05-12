# MoE-Sched

**A domain-specific language for Mixture-of-Experts scheduling policies.**

> Author: **Jesse Pokora** &middot; License: [MIT](LICENSE)

MoE-Sched lets you declaratively specify expert caching, prefetching, and
CPU-GPU scheduling strategies for MoE model inference.  Write a concise
`.moe` policy file (or use the Python eDSL), and MoE-Sched compiles it into
efficient runtime dispatch hooks that integrate with HuggingFace Transformers.

## Key Results

| Metric | Value |
|--------|-------|
| Dispatch overhead (A100) | < 3.2 % of MoE forward-pass time |
| Dispatch overhead (RTX 5080 laptop) | < 14 % during live inference |
| Code reduction vs. hand-coded systems | 13–36 × fewer lines |
| EPCB hit-rate improvement (OLMoE, cap=16) | **47.3 %** vs. 26.3 % LRU |

Evaluated on **Mixtral-8×7B**, **DeepSeek-V2-Lite**, and **OLMoE-1B-7B**.

## Installation

```bash
pip install -e ".[dev]"
```

Optional Cython acceleration (reduces dispatch latency to < 10 µs/layer):

```bash
python setup_cython.py build_ext --inplace
```

## Quick Start

### Standalone `.moe` policy file

```text
policy lfu_history {
    cache {
        capacity = 16
        eviction = lfu
        frequency_decay = 0.9
    }
    prefetch {
        strategy = history
        budget = 4
    }
    schedule { mode = hybrid }
}
```

```bash
moe-sched validate examples/lfu_policy.moe
```

### Python eDSL

```python
from moe_sched.dsl import MoESched

sched = MoESched()

@sched.policy
def lfu_history(p):
    p.cache(capacity=16, eviction="lfu", lfu_decay=0.9)
    p.prefetch(strategy="history", budget=4)
    p.schedule(mode="hybrid")
```

### Live inference with hooks

```python
from moe_sched.compiler import compile_policy
from moe_sched.runtime.hooks import build_hook

compiled = compile_policy(policy_ir)
hook = build_hook(compiled)

# Attach to any HuggingFace MoE model
for layer_idx, gate_module in enumerate(moe_gates):
    gate_module.register_forward_hook(make_dispatch_hook(hook, layer_idx))
```

## Architecture

```
moe_sched/
├── grammar.lark           # Lark LALR grammar (62 productions)
├── parser.py              # Grammar → AST
├── ir.py                  # Intermediate representation (PolicyIR)
├── validator.py           # 17 semantic validation rules
├── compiler.py            # IR → CompiledPolicy
├── dsl.py                 # Decorator / fluent-builder eDSL
├── adaptive.py            # Adaptive policies (adapt blocks)
├── autotuner.py           # Grid-search policy autotuner
├── cli.py                 # CLI: validate, compile, run
├── benchmark/             # Benchmarking harness + workloads
├── runtime/
│   ├── hooks.py           # PolicyHook — 5-step dispatch protocol
│   ├── cache.py           # LRU / LFU / Score / FreqThreshold
│   ├── scheduler.py       # GPU-only / CPU-fallback / Hybrid
│   ├── prefetch.py        # Affinity / History / Lookahead
│   ├── per_layer.py       # EPCB — entropy-proportional caching
│   ├── monitor.py         # Rolling-window metrics
│   ├── triggers.py        # Memory-pressure & TTL triggers
│   └── _fast/             # Cython-accelerated paths
└── integrations/
    ├── huggingface.py     # HuggingFace Transformers hooks
    ├── trace_recorder.py  # Expert activation trace recording
    └── weight_placement.py
```

## Supported Models

| Model | Experts × Layers | Routing | Tested on |
|-------|-----------------|---------|-----------|
| Mixtral-8×7B-Instruct | 8 × 32 | top-2 | A100-80 GB |
| DeepSeek-V2-Lite | 64 × 27 | top-6 | A100-80 GB |
| OLMoE-1B-7B | 64 × 16 | top-8 | RTX 5080 (16 GB) |

## Running Experiments

```bash
# Offline trace replay (works on any machine)
python scripts/run_eval.py
python scripts/run_sweep.py
python scripts/run_deepseek_strategies.py

# Live inference on consumer GPU
python scripts/run_constrained_e2e.py

# Generate all paper figures
python scripts/generate_figures.py
```

## Tests

```bash
python -m pytest tests/ -q
```

362 tests covering parsing, validation, compilation, runtime dispatch,
adaptive policies, and integration hooks.

## Documentation

See [`docs/MANUAL.md`](docs/MANUAL.md) for the full language reference,
runtime API, and policy authoring guide.

## Citation

If you use MoE-Sched in your research, please cite:

```bibtex
@article{pokora2026moesched,
  title={MoE-Sched: A Domain-Specific Language for Mixture-of-Experts Scheduling Policies},
  author={Pokora, Jesse},
  year={2026}
}
```

## License

This project is licensed under the [MIT License](LICENSE).

Copyright (c) 2026 Jesse Pokora
