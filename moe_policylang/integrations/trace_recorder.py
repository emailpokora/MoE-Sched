"""Record real expert activation traces from HuggingFace MoE models.

Phase 1: Captures per-token, per-layer expert selections from real model
inference and saves them as .jsonl files for reproducible replay in
benchmarks and evaluation.

Status: STUB — implement as part of Phase 1.

Usage (planned):
    from moe_policylang.integrations.trace_recorder import record_trace
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x7B-v0.1", ...)
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-v0.1")

    trace = record_trace(model, tokenizer, prompts=["Hello, world!"], output_path="traces/mixtral_sharegpt.jsonl")
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Sequence


@dataclass
class TraceEntry:
    """One layer's expert selection for one token."""

    token_idx: int
    layer_idx: int
    selected_experts: List[int]
    router_scores: Optional[List[float]] = None


@dataclass
class TraceRecording:
    """Complete trace from a model inference run."""

    model_name: str
    num_layers: int
    num_experts: int
    top_k: int
    entries: List[TraceEntry] = field(default_factory=list)

    def save(self, path: Path) -> None:
        """Save trace as newline-delimited JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            # Header line
            f.write(json.dumps({
                "model_name": self.model_name,
                "num_layers": self.num_layers,
                "num_experts": self.num_experts,
                "top_k": self.top_k,
                "num_entries": len(self.entries),
            }) + "\n")
            for entry in self.entries:
                f.write(json.dumps({
                    "t": entry.token_idx,
                    "l": entry.layer_idx,
                    "e": entry.selected_experts,
                    "s": entry.router_scores,
                }) + "\n")

    @classmethod
    def load(cls, path: Path) -> "TraceRecording":
        """Load a previously recorded trace."""
        path = Path(path)
        with open(path) as f:
            header = json.loads(f.readline())
            entries = []
            for line in f:
                d = json.loads(line)
                entries.append(TraceEntry(
                    token_idx=d["t"],
                    layer_idx=d["l"],
                    selected_experts=d["e"],
                    router_scores=d.get("s"),
                ))
        rec = cls(
            model_name=header["model_name"],
            num_layers=header["num_layers"],
            num_experts=header["num_experts"],
            top_k=header["top_k"],
            entries=entries,
        )
        return rec


def record_trace(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    output_path: Optional[str] = None,
    max_new_tokens: int = 128,
) -> TraceRecording:
    """Record expert activation traces from a real HuggingFace MoE model.

    TODO (Phase 1):
        - Hook into model's MoE layers to capture router outputs
        - Run inference on each prompt
        - Collect per-token, per-layer expert selections
        - Optionally save to output_path
    """
    raise NotImplementedError(
        "Phase 1 not yet implemented. "
        "See conference-paper/PROGRESS.md for status."
    )
