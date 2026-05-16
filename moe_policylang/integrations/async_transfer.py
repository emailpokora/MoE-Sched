"""Async expert weight transfers using CUDA streams.

Overlaps CPU→GPU expert weight copies with MoE layer computation by
running transfers on a dedicated CUDA stream.  The default (compute)
stream executes expert forward passes; transfers happen concurrently.

Architecture:
    Layer L forward (compute stream)
    ├── Expert computation on GPU
    └── Meanwhile: transfer stream copies L+1's prefetched experts

    Before using expert E at layer L+1:
    └── Wait on transfer event for E (if still in-flight)

This is the mechanism that ExpertFlow, Fiddler, and MoE-Infinity all
implement internally.  MoE-PolicyLang makes it policy-driven: the
prefetcher predicts which experts to transfer, and this module executes
those predictions asynchronously.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Set, Tuple

import torch


@dataclass
class AsyncTransferStats:
    """Tracks async transfer performance."""
    async_transfers: int = 0
    sync_transfers: int = 0
    prefetch_hits: int = 0       # expert was already transferred when needed
    sync_waits: int = 0          # had to wait for in-flight transfer
    transfer_time_s: float = 0.0
    bytes_transferred: int = 0

    @property
    def overlap_ratio(self) -> float:
        """Fraction of transfers that completed before the expert was needed."""
        total = self.async_transfers + self.sync_transfers
        if total == 0:
            return 0.0
        return self.prefetch_hits / total

    def to_dict(self) -> dict:
        total = self.async_transfers + self.sync_transfers
        return {
            "async_transfers": self.async_transfers,
            "sync_transfers": self.sync_transfers,
            "prefetch_hits": self.prefetch_hits,
            "sync_waits": self.sync_waits,
            "overlap_ratio": round(self.overlap_ratio, 3),
            "transfer_time_s": round(self.transfer_time_s, 4),
            "bytes_transferred_mb": round(self.bytes_transferred / 1e6, 1),
        }


class AsyncTransferManager:
    """Manages asynchronous CPU→GPU expert weight transfers via CUDA streams.

    Usage::

        atm = AsyncTransferManager(gpu_device=torch.device("cuda:0"))

        # During layer L's compute, start prefetching L+1's experts
        for eid in predicted_experts:
            atm.start_transfer(layer_idx, eid, weight_tensors, size_bytes)

        # Before using expert at layer L+1, ensure it's ready
        gpu_weights = atm.ensure_ready(layer_idx, eid)
        if gpu_weights is None:
            # Not prefetched — do synchronous transfer
            ...

    The manager maintains a transfer stream separate from the default
    compute stream.  Transfers use ``non_blocking=True`` on the transfer
    stream, and an event is recorded after each transfer completes.
    ``ensure_ready()`` synchronizes only the specific event for the
    requested expert, minimizing pipeline stalls.
    """

    def __init__(self, gpu_device: torch.device):
        self.gpu_device = gpu_device
        self.stats = AsyncTransferStats()

        # Dedicated stream for CPU→GPU copies
        self._transfer_stream = torch.cuda.Stream(device=gpu_device)

        # In-flight transfers: (layer, expert) → {event, gpu_tensors}
        self._in_flight: Dict[Tuple[int, int], _InFlightTransfer] = {}

        # Completed transfers ready for use: (layer, expert) → gpu_tensors
        self._ready: Dict[Tuple[int, int], Dict[str, torch.Tensor]] = {}

    def start_transfer(
        self,
        layer_idx: int,
        expert_idx: int,
        cpu_tensors: Dict[str, torch.Tensor],
        size_bytes: int,
    ) -> None:
        """Begin async transfer of expert weights from CPU to GPU.

        Args:
            layer_idx: Layer index.
            expert_idx: Expert index.
            cpu_tensors: Dict mapping weight name → CPU tensor
                (e.g., {"gate_up": tensor, "down": tensor}).
            size_bytes: Total bytes being transferred (for stats).
        """
        key = (layer_idx, expert_idx)

        # Already in-flight or ready — skip
        if key in self._in_flight or key in self._ready:
            return

        t0 = time.perf_counter()

        # Transfer on the dedicated stream
        gpu_tensors = {}
        with torch.cuda.stream(self._transfer_stream):
            for name, cpu_tensor in cpu_tensors.items():
                gpu_tensors[name] = cpu_tensor.to(
                    self.gpu_device, non_blocking=True
                )

        # Record event on transfer stream — compute stream can wait on this
        event = torch.cuda.Event()
        self._transfer_stream.record_event(event)

        elapsed = time.perf_counter() - t0
        self._in_flight[key] = _InFlightTransfer(
            gpu_tensors=gpu_tensors,
            event=event,
        )
        self.stats.async_transfers += 1
        self.stats.bytes_transferred += size_bytes
        self.stats.transfer_time_s += elapsed

    def ensure_ready(
        self, layer_idx: int, expert_idx: int
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Get GPU tensors for an expert, waiting on transfer if needed.

        Returns:
            Dict of GPU tensors if the expert was prefetched (async or
            already complete), or None if it was never started.
        """
        key = (layer_idx, expert_idx)

        # Already completed
        if key in self._ready:
            self.stats.prefetch_hits += 1
            return self._ready[key]

        # In-flight — wait for it
        if key in self._in_flight:
            transfer = self._in_flight.pop(key)
            if not transfer.event.query():
                # Transfer not yet complete — must wait
                transfer.event.synchronize()
                self.stats.sync_waits += 1
            else:
                self.stats.prefetch_hits += 1
            self._ready[key] = transfer.gpu_tensors
            return transfer.gpu_tensors

        # Not prefetched at all
        return None

    def mark_ready(
        self,
        layer_idx: int,
        expert_idx: int,
        gpu_tensors: Dict[str, torch.Tensor],
    ) -> None:
        """Register a synchronously-transferred expert as ready."""
        key = (layer_idx, expert_idx)
        self._ready[key] = gpu_tensors
        self.stats.sync_transfers += 1

    def evict(self, layer_idx: int, expert_idx: int) -> None:
        """Remove an expert from the async cache (on eviction)."""
        key = (layer_idx, expert_idx)
        self._in_flight.pop(key, None)
        self._ready.pop(key, None)

    def evict_expert_all_layers(self, expert_id: int) -> None:
        """Remove an expert from all layers (for global cache eviction)."""
        for key in list(self._in_flight.keys()):
            if key[1] == expert_id:
                self._in_flight.pop(key, None)
        for key in list(self._ready.keys()):
            if key[1] == expert_id:
                self._ready.pop(key, None)

    def clear(self) -> None:
        """Clear all transfers and ready buffers."""
        self._in_flight.clear()
        self._ready.clear()

    def sync_all(self) -> None:
        """Wait for all in-flight transfers to complete."""
        for key, transfer in list(self._in_flight.items()):
            transfer.event.synchronize()
            self._ready[key] = transfer.gpu_tensors
        self._in_flight.clear()


@dataclass
class _InFlightTransfer:
    """Tracks a single in-flight async transfer."""
    gpu_tensors: Dict[str, torch.Tensor]
    event: torch.cuda.Event
