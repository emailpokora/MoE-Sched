"""Generic auto-detected ExpertAccessor for any HuggingFace MoE model.

The user never writes model-specific code — ``auto_accessor(model)`` inspects
the model's structure at runtime and returns a single ``GenericMoEAccessor``
that handles both expert storage patterns:

  - **ModuleList**: experts are ``nn.Module`` objects in a list → move with ``.to()``
  - **Fused tensors**: all experts are packed into one weight matrix → slice per-expert

Detection is fully automatic based on the model's module tree and config.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from moe_sched.integrations.weight_placement import ExpertAccessor


# ---------------------------------------------------------------------------
# MoE layer detection
# ---------------------------------------------------------------------------

def _find_layers(model: nn.Module):
    """Locate the transformer layer list in a HuggingFace model."""
    base = getattr(model, "model", model)
    layers = getattr(base, "layers", None)
    if layers is None:
        layers = getattr(getattr(model, "transformer", None), "h", None)
    return layers


def _find_moe_block(layer: nn.Module) -> Optional[nn.Module]:
    """Find the MoE sub-module within a transformer layer."""
    for attr in ("mlp", "block_sparse_moe", "moe"):
        candidate = getattr(layer, attr, None)
        if candidate is not None and (
            hasattr(candidate, "gate") or hasattr(candidate, "experts")
        ):
            return candidate
    return None


def _find_gate(moe_block: nn.Module) -> Optional[nn.Module]:
    """Find the router/gate module within a MoE block."""
    for attr in ("gate", "router", "gate_proj"):
        gate = getattr(moe_block, attr, None)
        if gate is not None and isinstance(gate, nn.Module):
            return gate
    return None


def _detect_expert_type(moe_block: nn.Module) -> str:
    """Detect whether experts are stored as ModuleList, 3D indexed, or 2D fused.

    Returns:
        'modulelist' — experts are nn.ModuleList, individually moveable
        'indexed'    — expert weights are 3D tensors [num_experts, ...], index by expert_id
        'fused'      — expert weights are 2D fused tensors, need row slicing
    """
    experts_attr = getattr(moe_block, "experts", None)
    if experts_attr is not None and isinstance(experts_attr, nn.ModuleList):
        return "modulelist"
    # Check for 3D indexed patterns (e.g., gate_up_proj: [num_experts, out, in])
    sub = getattr(moe_block, "experts", moe_block)
    for attr_name in ("gate_up_proj", "gate_proj"):
        param = getattr(sub, attr_name, None)
        if param is not None and hasattr(param, "dim") and param.dim() == 3:
            return "indexed"
    # Check for 2D fused patterns (gate_proj: [num_experts * intermediate, hidden])
    for attr_name in ("gate_proj", "up_proj"):
        proj = getattr(sub, attr_name, None)
        if proj is not None and hasattr(proj, "weight") and proj.weight.dim() == 2:
            return "fused"
    if hasattr(moe_block, "gate_proj"):
        return "fused"
    return "unknown"


def _infer_fused_intermediate(moe_block: nn.Module, num_experts: int) -> int:
    """Infer per-expert intermediate size from fused weight tensor shape."""
    experts_sub = getattr(moe_block, "experts", moe_block)
    for name in ("gate_proj", "up_proj"):
        proj = getattr(experts_sub, name, None)
        if proj is not None and hasattr(proj, "weight"):
            total_rows = proj.weight.shape[0]
            return total_rows // num_experts
    return 0


def _detect_indexed_weight_names(moe_block: nn.Module) -> list[str]:
    """Find the 3D parameter names in the experts sub-module."""
    sub = getattr(moe_block, "experts", moe_block)
    names = []
    for name, param in sub.named_parameters(recurse=False):
        if param.dim() == 3:
            names.append(name)
    return names


# ---------------------------------------------------------------------------
# Auto-detection entry point
# ---------------------------------------------------------------------------

def auto_accessor(model: nn.Module) -> "GenericMoEAccessor":
    """Inspect any HuggingFace MoE model and return a ready-to-use accessor.

    Automatically detects:
      - Transformer layer list location
      - MoE block location within each layer
      - Router/gate module
      - Expert storage type (ModuleList vs fused tensors)
      - Expert count, top-k, intermediate size

    Raises ValueError if the model structure is not recognized.
    """
    config = getattr(model, "config", None)
    layers = _find_layers(model)
    if layers is None:
        raise ValueError(
            f"Cannot locate transformer layers in {type(model).__name__}. "
            f"Expected model.model.layers or model.transformer.h"
        )

    # Find which layers are MoE (not all layers need be — e.g., Jamba)
    moe_layer_indices = []
    moe_block = None
    for i, layer in enumerate(layers):
        block = _find_moe_block(layer)
        if block is not None:
            moe_layer_indices.append(i)
            if moe_block is None:
                moe_block = block

    if moe_block is None:
        raise ValueError(
            f"No MoE block found in {type(model).__name__}. "
            f"Searched for .mlp, .block_sparse_moe, .moe with .gate or .experts"
        )

    # Detect shared experts (DeepSeek V2/V3 pattern — always active, not cacheable)
    has_shared_experts = hasattr(moe_block, "shared_experts")

    # Detect expert storage type
    expert_type = _detect_expert_type(moe_block)

    # Get num_experts from config (try multiple field names)
    num_experts = (
        getattr(config, "num_experts", None)
        or getattr(config, "num_local_experts", None)
    )
    if num_experts is None and expert_type == "modulelist":
        num_experts = len(moe_block.experts)
    if num_experts is None:
        raise ValueError("Cannot determine num_experts from model config")

    top_k = (
        getattr(config, "num_experts_per_tok", None)
        or getattr(config, "num_experts_per_token", None)
        or 2  # safe default
    )

    # For fused tensors, infer intermediate size
    intermediate = None
    if expert_type == "fused":
        intermediate = _infer_fused_intermediate(moe_block, num_experts)
        if intermediate == 0:
            raise ValueError("Cannot infer per-expert intermediate size from fused weights")

    # For indexed tensors, discover weight names
    indexed_weight_names = None
    if expert_type == "indexed":
        indexed_weight_names = _detect_indexed_weight_names(moe_block)
        if not indexed_weight_names:
            raise ValueError("Detected indexed expert type but found no 3D weight tensors")

    return GenericMoEAccessor(
        model=model,
        layers=layers,
        num_experts=num_experts,
        top_k=top_k,
        expert_type=expert_type,
        fused_intermediate=intermediate,
        indexed_weight_names=indexed_weight_names,
        moe_layer_indices=moe_layer_indices,
        has_shared_experts=has_shared_experts,
    )


# ---------------------------------------------------------------------------
# Generic accessor — handles both ModuleList and fused tensors
# ---------------------------------------------------------------------------

class GenericMoEAccessor(ExpertAccessor):
    """Architecture-agnostic expert accessor.

    Handles two expert storage patterns transparently:

      - **ModuleList**: ``moe_block.experts[i]`` is an ``nn.Module`` with its
        own parameters.  Movement is simply ``.to(device)``.
      - **Fused tensors**: All experts are packed into shared weight matrices
        (e.g., ``gate_proj.weight`` of shape ``[num_experts * intermediate, hidden]``).
        Individual experts are contiguous row slices.

    Created automatically by ``auto_accessor(model)`` — users never
    instantiate this directly.
    """

    def __init__(
        self,
        model: nn.Module,
        layers: nn.ModuleList,
        num_experts: int,
        top_k: int,
        expert_type: str,
        fused_intermediate: Optional[int] = None,
        indexed_weight_names: Optional[list[str]] = None,
        moe_layer_indices: Optional[list[int]] = None,
        has_shared_experts: bool = False,
    ):
        self._model = model
        self._layers = layers
        self._num_experts_val = num_experts
        self._top_k = top_k
        self._expert_type = expert_type
        self._fused_intermediate = fused_intermediate
        self._indexed_weight_names = indexed_weight_names or []
        self._moe_layer_indices = moe_layer_indices or list(range(len(layers)))
        self._has_shared_experts = has_shared_experts

        # For non-modulelist: GPU cache of expert weights
        self._gpu_expert_weights: dict[tuple[int, int], dict[str, torch.Tensor]] = {}
        self._on_gpu: set[tuple[int, int]] = set()

    @property
    def num_layers(self) -> int:
        return len(self._moe_layer_indices)

    @property
    def moe_layer_indices(self) -> list[int]:
        """Indices of layers that actually contain MoE blocks."""
        return self._moe_layer_indices

    @property
    def has_shared_experts(self) -> bool:
        """Whether the model has shared experts (e.g., DeepSeek V2/V3)."""
        return self._has_shared_experts

    @property
    def num_experts(self) -> int:
        return self._num_experts_val

    @property
    def top_k(self) -> int:
        return self._top_k

    def _get_moe_block(self, layer_idx: int) -> nn.Module:
        return _find_moe_block(self._layers[layer_idx])

    # -- ModuleList helpers --

    def _get_expert_module(self, layer_idx: int, expert_idx: int) -> nn.Module:
        moe = self._get_moe_block(layer_idx)
        return moe.experts[expert_idx]

    # -- Fused tensor helpers --

    def _expert_slice(self, expert_idx: int) -> slice:
        start = expert_idx * self._fused_intermediate
        end = start + self._fused_intermediate
        return slice(start, end)

    def _get_fused_experts_module(self, layer_idx: int) -> nn.Module:
        moe = self._get_moe_block(layer_idx)
        experts_sub = getattr(moe, "experts", moe)
        return experts_sub

    def offload_source_to_cpu(self) -> None:
        """Move the underlying expert weight tensors to CPU.

        For indexed/fused types, this moves the source 3D/2D tensors
        themselves to CPU so that subsequent _ensure_on_gpu calls perform
        real CPU→GPU transfers.
        """
        for layer_idx in self._moe_layer_indices:
            if self._expert_type == "modulelist":
                for eid in range(self._num_experts_val):
                    self._get_expert_module(layer_idx, eid).to("cpu")
            else:
                mod = self._get_fused_experts_module(layer_idx)
                mod.to("cpu")
        self._on_gpu.clear()
        self._gpu_expert_weights.clear()

    @property
    def gpu_cached_experts(self) -> set[tuple[int, int]]:
        """Return the set of (layer, expert) pairs currently on GPU."""
        return set(self._on_gpu)

    # -- ExpertAccessor interface --

    def get_expert_params(self, layer_idx: int, expert_idx: int) -> List[torch.Tensor]:
        if self._expert_type == "modulelist":
            expert = self._get_expert_module(layer_idx, expert_idx)
            return [p for p in expert.parameters()]
        elif self._expert_type == "indexed":
            mod = self._get_fused_experts_module(layer_idx)
            return [getattr(mod, name)[expert_idx] for name in self._indexed_weight_names]
        else:
            mod = self._get_fused_experts_module(layer_idx)
            s = self._expert_slice(expert_idx)
            params = []
            for name in ("gate_proj", "up_proj"):
                w = getattr(mod, name).weight
                params.append(w[s, :])
            w_down = getattr(mod, "down_proj").weight
            params.append(w_down[:, s])
            return params

    def set_expert_device(self, layer_idx: int, expert_idx: int, device: torch.device) -> None:
        key = (layer_idx, expert_idx)

        if self._expert_type == "modulelist":
            expert = self._get_expert_module(layer_idx, expert_idx)
            expert.to(device)
            if device.type == "cuda":
                self._on_gpu.add(key)
            else:
                self._on_gpu.discard(key)
        else:
            # Indexed or fused: manage extracted copies
            if device.type == "cpu":
                self._on_gpu.discard(key)
                self._gpu_expert_weights.pop(key, None)
            elif device.type == "cuda" and key not in self._on_gpu:
                mod = self._get_fused_experts_module(layer_idx)
                gpu_weights = {}
                if self._expert_type == "indexed":
                    for name in self._indexed_weight_names:
                        gpu_weights[name] = getattr(mod, name)[expert_idx].to(device, non_blocking=True)
                else:
                    s = self._expert_slice(expert_idx)
                    for name in ("gate_proj", "up_proj"):
                        w = getattr(mod, name).weight
                        gpu_weights[name] = w[s, :].to(device, non_blocking=True)
                    w_down = getattr(mod, "down_proj").weight
                    gpu_weights["down_proj"] = w_down[:, s].to(device, non_blocking=True)
                self._gpu_expert_weights[key] = gpu_weights
                self._on_gpu.add(key)

    def get_expert_device(self, layer_idx: int, expert_idx: int) -> torch.device:
        key = (layer_idx, expert_idx)
        if self._expert_type == "modulelist":
            expert = self._get_expert_module(layer_idx, expert_idx)
            return next(expert.parameters()).device
        else:
            if key in self._on_gpu:
                return torch.device("cuda:0")
            return torch.device("cpu")

    def expert_forward(
        self, layer_idx: int, expert_idx: int, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        if self._expert_type == "modulelist":
            expert = self._get_expert_module(layer_idx, expert_idx)
            return expert(hidden_states)
        else:
            key = (layer_idx, expert_idx)
            if key in self._gpu_expert_weights:
                w = self._gpu_expert_weights[key]
            else:
                # Fallback: extract on the fly
                device = hidden_states.device
                w = {}
                mod = self._get_fused_experts_module(layer_idx)
                if self._expert_type == "indexed":
                    for name in self._indexed_weight_names:
                        w[name] = getattr(mod, name)[expert_idx].to(device)
                else:
                    s = self._expert_slice(expert_idx)
                    w["gate_proj"] = getattr(mod, "gate_proj").weight[s, :].to(device)
                    w["up_proj"] = getattr(mod, "up_proj").weight[s, :].to(device)
                    w["down_proj"] = getattr(mod, "down_proj").weight[:, s].to(device)
            return self._compute_expert_output(w, hidden_states)

    def _compute_expert_output(
        self, w: dict[str, torch.Tensor], hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """Compute SwiGLU expert output from weight dict, regardless of naming."""
        if "gate_up_proj" in w:
            # Fused gate+up: [2*intermediate, hidden] or [out, in]
            gu = hidden_states @ w["gate_up_proj"].t()
            half = gu.shape[-1] // 2
            gate_out = gu[..., :half]
            up_out = gu[..., half:]
        elif "gate_proj" in w and "up_proj" in w:
            gate_out = hidden_states @ w["gate_proj"].t()
            up_out = hidden_states @ w["up_proj"].t()
        else:
            raise ValueError(f"Unrecognized expert weight names: {list(w.keys())}")

        activated = torch.nn.functional.silu(gate_out) * up_out
        return activated @ w["down_proj"].t()

    def get_router(self, layer_idx: int) -> nn.Module:
        moe = self._get_moe_block(layer_idx)
        return _find_gate(moe)

    def run_router(
        self, layer_idx: int, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        gate = self.get_router(layer_idx)
        flat = hidden_states.view(-1, hidden_states.shape[-1])
        gate_out = gate(flat)

        # Handle different gate output formats
        if isinstance(gate_out, tuple):
            if len(gate_out) >= 3:
                # (logits, weights, indices) — OLMoE style
                return gate_out[2], gate_out[1]
            elif len(gate_out) >= 2:
                # (indices, weights) — Qwen style
                return gate_out[0], gate_out[1]
            gate_out = gate_out[0]

        # Raw logits — compute topk ourselves
        weights, indices = torch.topk(gate_out, k=self._top_k, dim=-1)
        weights = torch.softmax(weights.float(), dim=-1).to(hidden_states.dtype)
        return indices, weights

    def describe(self) -> str:
        """Human-readable description of what was auto-detected."""
        sparse = f', moe_layers={len(self._moe_layer_indices)}/{len(self._layers)}' if len(self._moe_layer_indices) != len(self._layers) else ''
        shared = ', shared_experts=yes' if self._has_shared_experts else ''
        return (
            f"GenericMoEAccessor("
            f"layers={self.num_layers}, "
            f"experts={self.num_experts}, "
            f"top_k={self._top_k}, "
            f"type={self._expert_type}"
            f"{f', intermediate={self._fused_intermediate}' if self._fused_intermediate else ''}"
            f"{f', weights={self._indexed_weight_names}' if self._indexed_weight_names else ''}"
            f"{sparse}{shared}"
            f")"
        )
