"""Automatic policy generation based on model architecture and hardware.

Given a HuggingFace MoE model, inspects its structure and available GPU
memory to generate a set of sensible DSL policies — or a single
recommended one.

Usage:
    import moe_sched

    # Generate policies tuned to this model + GPU
    policies = moe_sched.auto_policies(model)

    # Or just attach the best one directly
    mgr = moe_sched.auto_attach(model)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    import torch.nn as nn


def auto_policies(model: "nn.Module", gpu_device: int = 0) -> dict[str, str]:
    """Generate DSL policy strings tuned to a specific model and GPU.

    Inspects the model's MoE architecture (num_experts, top_k, expert
    size) and the GPU's available memory to compute cache capacities
    that make physical sense.

    Returns:
        Dict mapping policy name to DSL source string.
    """
    from moe_sched.integrations.accessors import auto_accessor

    accessor = auto_accessor(model)
    n = accessor.num_experts
    k = accessor.top_k
    expert_bytes = accessor.expert_size_bytes()
    expert_mb = expert_bytes / 1e6

    # Available GPU memory for expert cache
    props = torch.cuda.get_device_properties(gpu_device)
    total_vram = props.total_memory / 1e9
    allocated = torch.cuda.memory_allocated(gpu_device) / 1e9
    # Reserve 20% headroom for activations/KV cache
    available_gb = max(0, (total_vram - allocated) * 0.8)
    max_experts_fit = int(available_gb * 1e9 / expert_bytes) if expert_bytes > 0 else n

    # Clamp capacities to [top_k, num_experts]
    def clamp(c):
        return max(k, min(c, n))

    cap_small = clamp(max(k, n // 8))
    cap_medium = clamp(max(k * 2, n // 4))
    cap_large = clamp(max(k * 4, n // 2))
    cap_hw = clamp(min(max_experts_fit, n))  # hardware-limited

    policies = {}

    policies["aggressive"] = f"""
        policy aggressive {{
            cache {{ capacity = {cap_small}  eviction = lru }}
        }}
    """

    policies["balanced"] = f"""
        policy balanced {{
            cache {{
                capacity        = {cap_medium}
                eviction        = lfu
                frequency_decay = 0.9
            }}
            prefetch {{ strategy = history  budget = {min(k, 4)} }}
        }}
    """

    policies["conservative"] = f"""
        policy conservative {{
            cache {{
                capacity        = {cap_large}
                eviction        = lfu
                frequency_decay = 0.95
            }}
            prefetch {{ strategy = history  budget = {min(k, 6)} }}
        }}
    """

    policies["hw_limit"] = f"""
        policy hw_limit {{
            cache {{
                capacity        = {cap_hw}
                eviction        = lfu
                frequency_decay = 0.9
            }}
            prefetch {{ strategy = history  budget = {min(k, 4)} }}
        }}
    """

    return policies


def auto_attach(model: "nn.Module", strategy: str = "balanced", gpu_device: int = 0):
    """Generate a policy for this model and attach it in one call.

    Args:
        model: HuggingFace MoE model.
        strategy: One of 'aggressive', 'balanced', 'conservative', 'hw_limit'.
        gpu_device: GPU index.

    Returns:
        WeightPlacementManager with the auto-generated policy attached.
    """
    from moe_sched.integrations import attach

    policies = auto_policies(model, gpu_device=gpu_device)
    if strategy not in policies:
        valid = ", ".join(sorted(policies.keys()))
        raise ValueError(f"Unknown strategy '{strategy}'. Choose from: {valid}")

    return attach(model, policies[strategy], gpu_device=gpu_device)
