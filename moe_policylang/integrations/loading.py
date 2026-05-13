"""Expert-aware model loading for memory-constrained GPUs.

When a MoE model exceeds available VRAM, standard ``device_map="auto"`` fails
because it doesn't understand the MoE structure — it treats expert layers the
same as attention layers.

This module provides ``load_moe_model()``, which builds a custom device_map
that places the model *skeleton* (embeddings, attention, norms) on GPU and
*expert weights* on CPU.  MoE-PolicyLang then manages expert placement at
runtime via the DSL policy.

This is the same approach used by vLLM, MoE-Infinity, and DeepSpeed-MoE —
integrate expert offloading into the loading path rather than treating it
as a post-load optimization.

Usage:
    import moe_policylang
    from moe_policylang.integrations.loading import load_moe_model

    model, tokenizer = load_moe_model("mistralai/Mixtral-8x7B-Instruct-v0.1")
    mgr = moe_policylang.attach(model, policy_dsl)
    output = model.generate(...)
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch


def _get_module_names(model_config) -> dict:
    """Infer module naming conventions from model config/architecture type."""
    arch = getattr(model_config, "model_type", "")
    # All supported HF MoE architectures share similar naming
    return {
        "embed": "model.embed_tokens",
        "norm": "model.norm",
        "lm_head": "lm_head",
        "layer_prefix": "model.layers",
    }


def _is_expert_key(key: str) -> bool:
    """Check if a state_dict key belongs to an expert module.

    Expert keys contain patterns like:
      - .block_sparse_moe.experts.  (Mixtral)
      - .mlp.experts.               (OLMoE, Qwen)
      - .moe.experts.               (generic)
    Non-expert MoE components (gate/router) stay on GPU.
    """
    expert_markers = [
        ".block_sparse_moe.experts.",
        ".mlp.experts.",
        ".moe.experts.",
        # Indexed expert tensors (e.g., .experts.gate_up_proj)
        ".experts.gate_up_proj",
        ".experts.down_proj",
        ".experts.gate_proj",
        ".experts.up_proj",
    ]
    return any(marker in key for marker in expert_markers)


def build_expert_device_map(
    model_id: str,
    gpu_device: int = 0,
    *,
    dtype: torch.dtype = torch.float16,
    quantization_config: Any = None,
    trust_remote_code: bool = False,
) -> Dict[str, Any]:
    """Build a device_map that places expert weights on CPU, everything else on GPU.

    Args:
        model_id: HuggingFace model ID or local path.
        gpu_device: CUDA device index.
        dtype: Weight dtype (used for size estimation).
        quantization_config: Optional BitsAndBytesConfig for quantized loading.
        trust_remote_code: Whether to trust remote code for model config.

    Returns:
        A dict mapping module names to devices, suitable for
        ``from_pretrained(device_map=...)``.
    """
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(
        model_id, trust_remote_code=trust_remote_code
    )

    names = _get_module_names(config)
    device_map = {}

    # Get the state dict keys to map
    from transformers import AutoModelForCausalLM

    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(
            config, trust_remote_code=trust_remote_code
        )

    for key, param in model.state_dict().items():
        if _is_expert_key(key):
            device_map[key] = "cpu"
        else:
            device_map[key] = gpu_device

    del model
    return device_map


def load_moe_model(
    model_id: str,
    *,
    gpu_device: int = 0,
    dtype: torch.dtype = torch.float16,
    quantization_config: Any = None,
    trust_remote_code: bool = False,
    tokenizer_id: Optional[str] = None,
) -> Tuple[Any, Any]:
    """Load a MoE model with expert weights on CPU, skeleton on GPU.

    This is the recommended loading path for MoE models that exceed GPU VRAM.
    After loading, use ``moe_policylang.attach(model, policy)`` to enable
    policy-driven expert caching.

    Args:
        model_id: HuggingFace model ID or local path.
        gpu_device: CUDA device index.
        dtype: Weight dtype (float16 or bfloat16).
        quantization_config: Optional BitsAndBytesConfig for quantized loading.
        trust_remote_code: Whether to trust remote code.
        tokenizer_id: Override tokenizer ID (defaults to model_id).

    Returns:
        (model, tokenizer) tuple, ready for ``moe_policylang.attach()``.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Building expert-aware device map for {model_id}...")
    device_map = build_expert_device_map(
        model_id,
        gpu_device=gpu_device,
        dtype=dtype,
        quantization_config=quantization_config,
        trust_remote_code=trust_remote_code,
    )

    gpu_keys = sum(1 for v in device_map.values() if v == gpu_device)
    cpu_keys = sum(1 for v in device_map.values() if v == "cpu")
    print(f"  {gpu_keys} params on GPU, {cpu_keys} expert params on CPU")

    kw: dict[str, Any] = {
        "device_map": device_map,
        "low_cpu_mem_usage": True,
        "trust_remote_code": trust_remote_code,
    }

    if quantization_config is not None:
        kw["quantization_config"] = quantization_config
    else:
        kw["torch_dtype"] = dtype

    print(f"Loading model...")
    model = AutoModelForCausalLM.from_pretrained(model_id, **kw)
    model.eval()

    gpu_gb = torch.cuda.memory_allocated(gpu_device) / 1e9
    print(f"  Loaded: {gpu_gb:.1f} GB on GPU (experts on CPU)")

    tok_id = tokenizer_id or model_id
    tokenizer = AutoTokenizer.from_pretrained(tok_id, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer
