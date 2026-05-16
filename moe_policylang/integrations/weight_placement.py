"""Policy-controlled expert placement for HuggingFace MoE models.

Hooks into each MoE layer's router/gate so the DSL policy controls
which expert weights live on GPU vs CPU.  HuggingFace's own execution
kernels handle the actual inference — we only move weights.

Usage:
    import moe_policylang

    sched = moe_policylang.MoEPolicyLang()

    @sched.policy
    def my_policy(p):
        p.cache(capacity=32, eviction="lfu")
        p.prefetch(strategy="history", budget=4)

    mgr = moe_policylang.attach(model, my_policy)
    output = model.generate(...)  # MoE-PolicyLang manages expert placement
    print(mgr.get_stats())
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from moe_policylang.runtime.hooks import PolicyHook
from moe_policylang.runtime.scheduler import ExecutionDevice


# ---------------------------------------------------------------------------
# Expert accessor — model-specific, user-provided
# ---------------------------------------------------------------------------

class ExpertAccessor(ABC):
    """Abstract interface for accessing expert weights in any MoE model.

    Subclass this for each model architecture. The accessor knows:
      - How many layers and experts exist
      - How to get/set expert weight tensors
      - How to run a single expert's forward pass
    """

    @property
    @abstractmethod
    def num_layers(self) -> int:
        ...

    @property
    @abstractmethod
    def num_experts(self) -> int:
        ...

    @abstractmethod
    def get_expert_params(self, layer_idx: int, expert_idx: int) -> List[torch.Tensor]:
        """Return the weight tensors for one expert (may be multiple params)."""
        ...

    @abstractmethod
    def set_expert_device(self, layer_idx: int, expert_idx: int, device: torch.device) -> None:
        """Move one expert's weights to the specified device."""
        ...

    @abstractmethod
    def get_expert_device(self, layer_idx: int, expert_idx: int) -> torch.device:
        """Return the current device of an expert's weights."""
        ...

    @abstractmethod
    def expert_forward(
        self, layer_idx: int, expert_idx: int, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """Run one expert's forward pass on the given hidden states."""
        ...

    @abstractmethod
    def get_router(self, layer_idx: int) -> nn.Module:
        """Return the router/gate module for a layer (for hooking)."""
        ...

    @abstractmethod
    def run_router(
        self, layer_idx: int, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the router and return (expert_indices, expert_weights).

        expert_indices: [batch*seq, top_k] int tensor
        expert_weights: [batch*seq, top_k] float tensor
        """
        ...

    def expert_size_bytes(self, layer_idx: int = 0, expert_idx: int = 0) -> int:
        """Return size of one expert in bytes."""
        return sum(p.numel() * p.element_size()
                   for p in self.get_expert_params(layer_idx, expert_idx))


# ---------------------------------------------------------------------------
# Placement stats
# ---------------------------------------------------------------------------

@dataclass
class PlacementStats:
    """Tracks physical tensor movement during inference."""
    cpu_to_gpu_transfers: int = 0
    gpu_to_cpu_transfers: int = 0
    bytes_transferred: int = 0
    transfer_time_s: float = 0.0
    forward_calls: int = 0

    @property
    def avg_transfer_us(self) -> float:
        if self.cpu_to_gpu_transfers == 0:
            return 0.0
        return (self.transfer_time_s / self.cpu_to_gpu_transfers) * 1e6


# ---------------------------------------------------------------------------
# Weight Placement Manager
# ---------------------------------------------------------------------------

class WeightPlacementManager:
    """Connects MoE-PolicyLang policy decisions to real tensor movement.

    This is the mechanism layer. The PolicyHook (policy layer) decides
    which experts should be cached; this class executes those decisions
    by calling the ExpertAccessor to move weights between devices.

    When ``async_transfers=True``, expert weight copies use a dedicated
    CUDA stream so transfers overlap with compute.  Prefetched experts
    (predicted by the policy's prefetcher) are transferred asynchronously
    during the current layer's forward pass.
    """

    def __init__(
        self,
        hook: PolicyHook,
        accessor: ExpertAccessor,
        gpu_device: int | torch.device = 0,
        async_transfers: bool = False,
    ):
        self.hook = hook
        self.accessor = accessor
        self.gpu_device = torch.device(f"cuda:{gpu_device}" if isinstance(gpu_device, int) else gpu_device)
        self.stats = PlacementStats()
        self.async_transfers = async_transfers
        # Track which (layer, expert) pairs are on GPU
        self._on_gpu: set[tuple[int, int]] = set()
        # Current layer being processed (set by forward_layer / callers)
        self._current_layer_idx: int = 0

        # Async transfer manager (created lazily if async_transfers=True)
        self._atm = None
        if async_transfers and torch.cuda.is_available():
            from moe_policylang.integrations.async_transfer import AsyncTransferManager
            self._atm = AsyncTransferManager(self.gpu_device)

        # Install eviction callback on the cache so physical GPU memory
        # is freed when the policy evicts an expert — O(1) per eviction.
        self._install_eviction_callback()

    def _install_eviction_callback(self) -> None:
        """Wire the cache's eviction events to physical GPU→CPU movement."""
        cache = self.hook.cache
        if hasattr(cache, 'on_evict'):
            cache.on_evict = self._on_cache_evict
        # FallbackCache: install on both tiers
        if hasattr(cache, 'primary') and hasattr(cache.primary, 'on_evict'):
            cache.primary.on_evict = self._on_cache_evict
        if hasattr(cache, 'secondary') and hasattr(cache.secondary, 'on_evict'):
            cache.secondary.on_evict = self._on_cache_evict

    def _on_cache_evict(self, expert_id: int) -> None:
        """Callback fired by the cache when it evicts an expert."""
        self._evict_to_cpu(self._current_layer_idx, expert_id)

    def offload_experts_to_cpu(self) -> None:
        """Move all expert weights to CPU. Call before inference.

        Uses the accessor's offload_source_to_cpu() if available, which
        moves the underlying source tensors (3D indexed or 2D fused) to
        CPU so subsequent transfers are real CPU→GPU copies.
        """
        if hasattr(self.accessor, 'offload_source_to_cpu'):
            self.accessor.offload_source_to_cpu()
        else:
            for layer_idx in range(self.accessor.num_layers):
                for expert_idx in range(self.accessor.num_experts):
                    self.accessor.set_expert_device(layer_idx, expert_idx, torch.device("cpu"))
        self._on_gpu.clear()

    def _ensure_on_gpu(self, layer_idx: int, expert_idx: int) -> None:
        """Move expert to GPU if not already there.

        If async transfers are enabled, checks whether the expert was
        already prefetched asynchronously before falling back to a
        synchronous copy.
        """
        key = (layer_idx, expert_idx)
        if key in self._on_gpu:
            return

        # Check async manager for pre-fetched expert
        if self._atm is not None:
            gpu_tensors = self._atm.ensure_ready(layer_idx, expert_idx)
            if gpu_tensors is not None:
                # Expert was async-transferred — just mark it on GPU
                self._on_gpu.add(key)
                self.stats.cpu_to_gpu_transfers += 1
                self.stats.bytes_transferred += self.accessor.expert_size_bytes(layer_idx, expert_idx)
                return

        # Synchronous fallback
        t0 = time.perf_counter()
        self.accessor.set_expert_device(layer_idx, expert_idx, self.gpu_device)
        torch.cuda.synchronize(self.gpu_device)
        elapsed = time.perf_counter() - t0
        self._on_gpu.add(key)
        self.stats.cpu_to_gpu_transfers += 1
        self.stats.bytes_transferred += self.accessor.expert_size_bytes(layer_idx, expert_idx)
        self.stats.transfer_time_s += elapsed

    def _evict_to_cpu(self, layer_idx: int, expert_idx: int) -> None:
        """Move expert back to CPU to free GPU memory."""
        key = (layer_idx, expert_idx)
        if key not in self._on_gpu:
            return
        self.accessor.set_expert_device(layer_idx, expert_idx, torch.device("cpu"))
        self._on_gpu.discard(key)
        self.stats.gpu_to_cpu_transfers += 1
        # Also evict from async transfer cache
        if self._atm is not None:
            self._atm.evict(layer_idx, expert_idx)

    def attach(self) -> list:
        """Hook into the model's MoE layers.

        Automatically selects the best integration strategy:

        **HF Experts Backend** (Mixtral, OLMoE, DeepSeek — 3D tensor experts):
          - Registers ``"moe_policylang"`` via HF's official ``ExpertsInterface``
          - Expert weights live on CPU; slices copied to GPU on-demand
          - Eviction callback frees GPU copies automatically
          - HF dispatches to our backend — no monkey-patching

        **ModuleList** (older models with per-expert nn.Module):
          - Gate hooks intercept router decisions
          - Individual expert modules moved between CPU/GPU
          - HF's own execution loop handles the rest

        **Observation mode** (fallback):
          - Gate hooks observe routing decisions
          - Policy tracks cache stats for evaluation

        Returns list of hook handles / contexts for cleanup.
        """
        self._handles = []
        self._backend_contexts = []
        expert_sz = self.accessor.expert_size_bytes() / 1e9

        # Strategy 1: HF Experts Backend (3D tensor models — most modern MoE)
        if self.accessor._expert_type == "indexed":
            try:
                from moe_policylang.integrations.hf_experts_backend import install_backend
                self._backend_contexts = install_backend(self.accessor._model, self)
                self._mode = "backend"
                return self._backend_contexts
            except (ImportError, AttributeError):
                pass  # Fall through to observation mode

        # Strategy 2: ModuleList — per-expert device movement + gate hooks
        if self.accessor._expert_type == "modulelist":
            self.offload_experts_to_cpu()
            for layer_idx in self.accessor.moe_layer_indices:
                gate = self.accessor.get_router(layer_idx)
                h = gate.register_forward_hook(
                    self._make_gate_hook(layer_idx, expert_sz, physical=True)
                )
                self._handles.append(h)
            self._mode = "modulelist"
            return self._handles

        # Strategy 3: Observation mode (gate hooks, no physical movement)
        for layer_idx in self.accessor.moe_layer_indices:
            gate = self.accessor.get_router(layer_idx)
            h = gate.register_forward_hook(
                self._make_gate_hook(layer_idx, expert_sz, physical=False)
            )
            self._handles.append(h)
        self._mode = "observe"
        return self._handles

    def detach(self) -> None:
        """Remove all hooks and restore original behavior."""
        for h in getattr(self, '_handles', []):
            h.remove()
        self._handles = []
        if getattr(self, '_backend_contexts', None):
            from moe_policylang.integrations.hf_experts_backend import uninstall_backend
            uninstall_backend(self.accessor._model, self._backend_contexts)
            self._backend_contexts = []

    def _make_gate_hook(self, layer_idx: int, expert_size_gb: float, physical: bool):
        """Create a forward hook for one layer's router/gate.

        The hook fires after the router computes logits but before the
        MoE block uses those logits to dispatch tokens to experts.
        This is the correct integration point: we know which experts
        are needed and can move them before HF accesses them.
        """
        def gate_hook(module, inputs, output):
            # Parse router output — models return different formats:
            #   OLMoE gate: (logits, top_k_weights, top_k_indices)
            #   Mixtral gate: logits tensor
            #   Generic: logits or (logits, ...)
            if isinstance(output, tuple) and len(output) >= 3:
                # Pre-computed routing: (logits, weights, indices)
                weights = output[1]  # [batch*seq, top_k]
                indices = output[2]  # [batch*seq, top_k]
            elif isinstance(output, tuple):
                logits = output[0]
                if not isinstance(logits, torch.Tensor):
                    return output
                top_k = self.accessor.top_k
                routing_weights = torch.softmax(logits.view(-1, logits.shape[-1]), dim=-1)
                weights, indices = torch.topk(routing_weights, top_k, dim=-1)
            elif isinstance(output, torch.Tensor):
                logits = output
                top_k = self.accessor.top_k
                routing_weights = torch.softmax(logits.view(-1, logits.shape[-1]), dim=-1)
                weights, indices = torch.topk(routing_weights, top_k, dim=-1)
            else:
                return output

            # Set current layer for eviction callback
            self._current_layer_idx = layer_idx

            # Feed each token's expert selections to the policy
            for token_idx in range(indices.shape[0]):
                selected = indices[token_idx].cpu().tolist()
                scores = [float(w) for w in weights[token_idx].cpu()]
                self.hook.on_layer(
                    layer_idx, selected,
                    scores=scores,
                    expert_size_gb=expert_size_gb,
                )

            if physical:
                # Move all needed experts to GPU before HF's forward uses them
                needed = set(indices.cpu().flatten().tolist())
                for eid in needed:
                    self._ensure_on_gpu(layer_idx, eid)

            # Async prefetch: start transferring predicted experts for next layer
            if self._atm is not None:
                # The prefetcher already ran during on_layer() calls above;
                # use the last token's dispatch plan for predictions.
                next_layer = layer_idx + 1
                if next_layer < self.accessor.num_layers:
                    last_selected = indices[-1].cpu().tolist()
                    predicted = self.hook.prefetcher.predict(layer_idx, last_selected)
                    self.start_async_prefetch(next_layer, predicted)

            self.stats.forward_calls += 1
            return output

        return gate_hook

    def start_async_prefetch(
        self,
        layer_idx: int,
        expert_ids: Sequence[int],
    ) -> None:
        """Begin async transfer of predicted experts for an upcoming layer.

        Called after on_layer() returns a prefetch list.  The transfers
        run on a dedicated CUDA stream while the current layer's expert
        forward passes execute on the default stream.
        """
        if self._atm is None:
            return
        for eid in expert_ids:
            key = (layer_idx, eid)
            if key in self._on_gpu:
                continue
            # Get CPU tensors to transfer
            try:
                params = self.accessor.get_expert_params(layer_idx, eid)
                cpu_tensors = {f"param_{i}": p for i, p in enumerate(params)}
                size_bytes = sum(p.numel() * p.element_size() for p in params)
                self._atm.start_transfer(layer_idx, eid, cpu_tensors, size_bytes)
            except Exception:
                pass  # Skip on error — sync fallback handles it

    def get_stats(self) -> dict:
        """Combined policy + placement stats."""
        policy_stats = self.hook.stats_snapshot()
        result = {
            "policy": policy_stats,
            "placement": {
                "cpu_to_gpu_transfers": self.stats.cpu_to_gpu_transfers,
                "gpu_to_cpu_transfers": self.stats.gpu_to_cpu_transfers,
                "bytes_transferred_mb": round(self.stats.bytes_transferred / 1e6, 1),
                "transfer_time_s": round(self.stats.transfer_time_s, 4),
                "avg_transfer_us": round(self.stats.avg_transfer_us, 1),
                "forward_calls": self.stats.forward_calls,
            },
        }
        if self._atm is not None:
            result["async"] = self._atm.stats.to_dict()
        return result
