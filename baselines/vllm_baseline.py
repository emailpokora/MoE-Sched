"""vLLM baseline: measure expert caching performance under vLLM's default policy.

Phase 3: Run identical prompts through vLLM with Mixtral and extract
cache hit rates, throughput, and latency for comparison.

Status: STUB — implement after Phase 1 + 2.

Requirements:
    pip install vllm

TODO:
    - Set up vLLM with Mixtral-8x7B in offline mode
    - Instrument vLLM's expert cache (if accessible) or measure externally
    - Run ShareGPT traces, collect tok/s, TTFT, memory usage
    - Export results in same MetricsSummary format as MoE-Sched benchmark
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class VLLMResult:
    """Metrics from a vLLM baseline run."""

    model_name: str
    num_prompts: int
    tokens_per_second: float = 0.0
    time_to_first_token_ms: float = 0.0
    peak_gpu_memory_gb: float = 0.0
    # vLLM doesn't directly expose cache hit rate, so this may be N/A
    cache_hit_rate: Optional[float] = None


def run_vllm_baseline(
    model_name: str = "mistralai/Mixtral-8x7B-v0.1",
    prompts: Optional[List[str]] = None,
    max_tokens: int = 128,
    gpu_memory_utilization: float = 0.9,
) -> VLLMResult:
    """Run vLLM inference and collect baseline metrics.

    TODO (Phase 3):
        - Initialize vLLM LLM engine with specified model
        - Run prompts through the engine
        - Measure throughput and latency
        - Extract GPU memory usage
    """
    raise NotImplementedError(
        "Phase 3 not yet implemented. "
        "See conference-paper/PROGRESS.md for status."
    )
