"""Real system baselines for comparison.

This package wraps external MoE serving systems to extract comparable
metrics (cache hit rate, throughput, latency) on identical workloads.

Baselines:
    - vLLM (default expert offloading)
    - MoE-Infinity (sparsity-aware caching)
    - DeepSpeed-MoE (optional, expert parallelism)
"""
