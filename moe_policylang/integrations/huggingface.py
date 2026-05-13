"""HuggingFace Transformers integration for MoE-Sched policy hooks.

This module installs a ``PolicyHook`` into a HuggingFace MoE model
(Mixtral, Qwen1.5-MoE, DeepSeek-V3, ...) by monkey-patching each MoE layer's
forward method to invoke the hook on the router-selected experts.

The integration is deliberately best-effort and **optional** — the core
MoE-Sched pipeline (DSL → IR → compile → hook) is framework-agnostic and
does not depend on torch or transformers.  This module only activates when
a caller explicitly imports it and has those packages installed.

Typical usage:

    from transformers import AutoModelForCausalLM
    from moe_sched import parse_file, compile_policy
    from moe_sched.runtime.hooks import build_hook
    from moe_sched.integrations.huggingface import install_policy_hook

    [ir] = parse_file("examples/lru_policy.moe")
    hook = build_hook(compile_policy(ir))

    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen1.5-MoE-A2.7B", ...)
    install_policy_hook(model, hook)

    # Now every MoE layer forward invokes the hook and surfaces stats via
    # hook.stats_snapshot().
"""

from __future__ import annotations

from typing import Any, List, Optional

from moe_sched.runtime.hooks import PolicyHook

# Names of attributes on each MoE layer that expose router output.  We search
# these in order; the first match wins.  Extend for new model families.
_ROUTER_ATTRS = ("mlp", "block_sparse_moe", "moe", "experts")


def install_policy_hook(
    model: Any,
    hook: PolicyHook,
    layer_attr_candidates: Optional[List[str]] = None,
) -> List[int]:
    """Monkey-patch each detected MoE layer to invoke ``hook`` on forward.

    Args:
        model: A HuggingFace ``PreTrainedModel`` with MoE layers.
        hook: The policy hook to invoke.
        layer_attr_candidates: Override the default set of attribute names to
            search on each layer when locating the MoE sub-module.

    Returns:
        The list of layer indices that were successfully instrumented.

    Notes:
        * This function is a *scaffold* for Week 3.  Full integration requires
          model-specific router-output extraction (Mixtral exposes
          ``router_logits``; Qwen2MoE exposes ``expert_mask`` + ``experts``).
        * The scaffold validates that the model has iterable layers, locates
          MoE sub-modules, and installs a pre-forward wrapper that captures
          the selected-experts tensor and calls ``hook.on_layer``.
        * For heavy end-to-end testing, prefer ``moe_sched.integrations.mock_moe``.
    """
    try:
        import torch  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "install_policy_hook requires torch.  Install torch+transformers "
            "or use moe_sched.integrations.mock_moe for framework-free testing."
        ) from e

    candidates = list(layer_attr_candidates or _ROUTER_ATTRS)

    # Locate the layer list.  Most HF causal-LM models expose it at one of:
    #   model.model.layers      (LLaMA/Qwen family)
    #   model.transformer.h     (GPT family)
    base = getattr(model, "model", model)
    layers = getattr(base, "layers", None) or getattr(
        getattr(model, "transformer", None), "h", None
    )
    if layers is None:
        raise RuntimeError(
            "Could not locate transformer layer list on the provided model. "
            "Pass the layer list explicitly or extend _ROUTER_ATTRS."
        )

    instrumented: List[int] = []
    for layer_idx, layer in enumerate(layers):
        moe = None
        for attr in candidates:
            if hasattr(layer, attr):
                candidate = getattr(layer, attr)
                # Heuristic: a MoE block exposes either a gate/router module
                # or an experts ModuleList.
                if hasattr(candidate, "gate") or hasattr(candidate, "experts"):
                    moe = candidate
                    break
        if moe is None:
            continue

        _wrap_moe_forward(moe, hook, layer_idx)
        instrumented.append(layer_idx)

    if not instrumented:
        raise RuntimeError(
            "No MoE layers were detected.  Either this model has no MoE layers "
            "or its layer attribute naming differs from the supported set: "
            f"{candidates}"
        )

    return instrumented


def _wrap_moe_forward(moe_module: Any, hook: PolicyHook, layer_idx: int) -> None:
    """Replace ``moe_module.forward`` with a wrapper that calls the hook.

    The wrapper's responsibilities:
      1. Invoke the original forward to obtain router logits / selected experts.
      2. Extract the top-k expert indices for each token.
      3. Call ``hook.on_layer(layer_idx, selected_experts, scores=...)``.
      4. Return the original output unchanged.

    Full router-output extraction is model-specific and is left as follow-up
    work for Week 4 integration polish.
    """
    original_forward = moe_module.forward

    def wrapped_forward(*args, **kwargs):
        output = original_forward(*args, **kwargs)
        selected = _extract_selected_experts(moe_module, output)
        if selected is not None:
            # Flatten token dimension: HF returns (batch*seq, top_k) indices.
            for row in selected:
                hook.on_layer(layer_idx=layer_idx, selected_experts=row)
        return output

    moe_module.forward = wrapped_forward


def _extract_selected_experts(moe_module: Any, output: Any) -> Optional[List[List[int]]]:
    """Best-effort router-output extractor.

    Supports common shapes:
      * Mixtral: output is (hidden_states, router_logits); experts are the
        top-k argmax of router_logits.
      * Qwen2MoE: moe_module.last_expert_ids attribute (set inside forward).

    Returns a list of per-token expert-id lists, or None if extraction failed.
    """
    # Try a side-channel attribute first.
    for attr in ("last_expert_ids", "_last_selected_experts"):
        if hasattr(moe_module, attr):
            val = getattr(moe_module, attr)
            if val is None:
                continue
            return _tensor_to_python_list(val)

    # Fall back to parsing the forward output.
    if isinstance(output, tuple) and len(output) >= 2:
        router_logits = output[1]
        top_k = getattr(moe_module, "top_k", None) or 2
        try:
            import torch  # local import to keep the scaffold torch-optional at import time
            if hasattr(router_logits, "topk"):
                _, idx = router_logits.topk(top_k, dim=-1)
                return _tensor_to_python_list(idx)
        except Exception:
            return None

    return None


def _tensor_to_python_list(t: Any) -> List[List[int]]:
    """Convert a (N, K) tensor or ndarray to a list-of-lists of ints."""
    if hasattr(t, "tolist"):
        data = t.tolist()
    else:
        data = list(t)
    if data and isinstance(data[0], (int, float)):
        return [[int(x) for x in data]]
    return [[int(x) for x in row] for row in data]
