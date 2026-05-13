"""MoE-PolicyLang experts backend for HuggingFace Transformers.

Registers a ``"moe_policylang"`` experts implementation via HF's official
``ExpertsInterface`` API.  When active, expert weights live on CPU and
are copied to GPU on-demand based on the DSL policy's cache decisions.

This is NOT monkey-patching — it uses the same extension mechanism that
HF's own ``batched_mm`` and ``grouped_mm`` backends use.

Integration flow:
    1. ``attach()`` registers the backend and sets
       ``config._experts_implementation = "moe_policylang"``
    2. Each MoE layer's ``Experts.forward()`` dispatches to our backend
    3. Our backend:
       a. Feeds expert selections to the PolicyHook
       b. Copies needed expert slices from CPU → GPU
       c. Runs computation on GPU using the standard eager loop
       d. Eviction callback (from cache) frees GPU copies
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from moe_policylang.integrations.weight_placement import WeightPlacementManager


def _moe_policylang_experts_forward(
    self: nn.Module,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    """Policy-controlled expert offloading forward.

    This replaces the default eager loop with one that:
    - Feeds router decisions to the MoE-PolicyLang policy
    - Copies only needed expert weight slices from CPU to GPU
    - Runs computation identically to HF's eager implementation
    """
    ctx = self._moe_policylang_ctx
    mgr: WeightPlacementManager = ctx["mgr"]
    layer_idx: int = ctx["layer_idx"]
    gpu_device: torch.device = ctx["gpu_device"]
    expert_bytes: int = ctx["expert_bytes"]

    # -- 1. Feed expert selections to the policy --
    mgr._current_layer_idx = layer_idx
    for token_idx in range(top_k_index.shape[0]):
        selected = top_k_index[token_idx].cpu().tolist()
        scores = [float(w) for w in top_k_weights[token_idx].cpu()]
        mgr.hook.on_layer(
            layer_idx, selected,
            scores=scores,
            expert_size_gb=expert_bytes / 1e9,
        )

    # -- 2. Determine unique experts needed and load to GPU --
    needed_experts = top_k_index.unique().cpu().tolist()
    gpu_cache = ctx["gpu_cache"]  # {expert_id: {"gate_up": Tensor, "down": Tensor}}

    for eid in needed_experts:
        if eid not in gpu_cache:
            t0 = time.perf_counter()
            gpu_cache[eid] = {
                "gate_up": self.gate_up_proj[eid].to(gpu_device, non_blocking=True),
                "down": self.down_proj[eid].to(gpu_device, non_blocking=True),
            }
            torch.cuda.synchronize(gpu_device)
            elapsed = time.perf_counter() - t0
            mgr.stats.cpu_to_gpu_transfers += 1
            mgr.stats.bytes_transferred += expert_bytes
            mgr.stats.transfer_time_s += elapsed
            mgr._on_gpu.add((layer_idx, eid))

    # -- 3. Compute (same as HF eager loop) --
    final_hidden_states = torch.zeros_like(hidden_states)
    expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts)
    expert_mask = expert_mask.permute(2, 1, 0)  # [num_experts, top_k, tokens]
    expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

    for expert_idx in expert_hit:
        expert_idx = expert_idx[0].item()
        if expert_idx == self.num_experts:
            continue

        top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
        current_state = hidden_states[token_idx]

        # Use GPU-cached expert weights
        cached = gpu_cache[expert_idx]
        gate, up = F.linear(current_state, cached["gate_up"]).chunk(2, dim=-1)
        current_hidden_states = self.act_fn(gate) * up
        current_hidden_states = F.linear(current_hidden_states, cached["down"])
        current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
        final_hidden_states.index_add_(
            0, token_idx, current_hidden_states.to(final_hidden_states.dtype)
        )

    mgr.stats.forward_calls += 1
    return final_hidden_states


def _evict_from_gpu_cache(ctx: dict, expert_id: int) -> None:
    """Called by the cache eviction callback to free GPU memory."""
    gpu_cache = ctx["gpu_cache"]
    if expert_id in gpu_cache:
        del gpu_cache[expert_id]
        # The tensors are freed when garbage collected


def register_backend() -> None:
    """Register the 'moe_policylang' experts backend with HuggingFace."""
    try:
        from transformers.integrations.moe import ExpertsInterface
        ExpertsInterface.register("moe_policylang", _moe_policylang_experts_forward)
    except ImportError:
        raise ImportError(
            "HuggingFace Transformers >= 4.51 required for experts backend. "
            "Install with: pip install transformers>=4.51"
        )


def install_backend(
    model: nn.Module,
    mgr: "WeightPlacementManager",
) -> list[dict]:
    """Activate the moe_policylang backend on a loaded model.

    1. Moves expert weight tensors to CPU
    2. Attaches per-layer context to each Experts module
    3. Sets config._experts_implementation = "moe_policylang"

    Returns the list of per-layer context dicts (for cleanup).
    """
    register_backend()

    accessor = mgr.accessor
    gpu_device = mgr.gpu_device
    contexts = []

    # Find all Experts modules and attach context
    layers = accessor._layers
    for layer_idx in accessor.moe_layer_indices:
        moe_block = accessor._get_moe_block(layer_idx)
        experts_mod = moe_block.experts

        # Move expert weights to CPU
        experts_mod.gate_up_proj.data = experts_mod.gate_up_proj.data.cpu()
        experts_mod.down_proj.data = experts_mod.down_proj.data.cpu()

        # Per-expert byte size
        gate_up_bytes = experts_mod.gate_up_proj[0].numel() * experts_mod.gate_up_proj[0].element_size()
        down_bytes = experts_mod.down_proj[0].numel() * experts_mod.down_proj[0].element_size()
        expert_bytes = gate_up_bytes + down_bytes

        ctx = {
            "mgr": mgr,
            "layer_idx": layer_idx,
            "gpu_device": gpu_device,
            "expert_bytes": expert_bytes,
            "gpu_cache": {},  # {expert_id: {"gate_up": Tensor, "down": Tensor}}
        }
        experts_mod._moe_policylang_ctx = ctx
        contexts.append(ctx)

    # Install eviction callback: when the policy cache evicts an expert,
    # free its GPU tensor copies.  We override the method directly on the
    # cache so it fires for every eviction.
    def _evict_handler(expert_id: int):
        # Evict from ALL layers' GPU caches — the global cache doesn't
        # track which layer an expert was loaded for.
        evicted_any = False
        for ctx in contexts:
            if expert_id in ctx["gpu_cache"]:
                _evict_from_gpu_cache(ctx, expert_id)
                key = (ctx["layer_idx"], expert_id)
                mgr._on_gpu.discard(key)
                evicted_any = True
        if evicted_any:
            mgr.stats.gpu_to_cpu_transfers += 1

    cache = mgr.hook.cache
    if hasattr(cache, 'on_evict'):
        cache.on_evict = _evict_handler
    if hasattr(cache, 'primary') and hasattr(cache.primary, 'on_evict'):
        cache.primary.on_evict = _evict_handler
    if hasattr(cache, 'secondary') and hasattr(cache.secondary, 'on_evict'):
        cache.secondary.on_evict = _evict_handler

    # Activate the backend
    model.config._experts_implementation = "moe_policylang"

    return contexts


def uninstall_backend(model: nn.Module, contexts: list[dict]) -> None:
    """Revert to default experts implementation."""
    model.config._experts_implementation = "eager"
    for ctx in contexts:
        ctx["gpu_cache"].clear()
    # Expert weights remain on CPU — caller can move back if needed
