"""Prefetch engines for MoE-Sched runtime."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PrefetchStats:
    issued: int = 0
    useful: int = 0  # prefetched expert was actually used

    @property
    def accuracy(self) -> float:
        return self.useful / self.issued if self.issued > 0 else 0.0


class NullPrefetcher:
    """No-op prefetcher (strategy=NONE)."""

    def __init__(self) -> None:
        self.stats = PrefetchStats()

    def predict(self, layer_idx: int, selected_experts: list[int]) -> list[int]:
        return []

    def report_usage(self, expert_id: int) -> None:
        pass


class AffinityPrefetcher:
    """Prefetch based on inter-layer co-activation affinity matrix."""

    def __init__(
        self,
        affinity: dict[tuple[int, int], dict[int, float]] | None = None,
        threshold: float = 0.3,
        budget: int = 4,
    ):
        self.affinity = affinity or {}
        self.threshold = threshold
        self.budget = budget
        self.stats = PrefetchStats()
        self._pending: set[int] = set()

    def predict(self, layer_idx: int, selected_experts: list[int]) -> list[int]:
        predicted: set[int] = set()
        for eid in selected_experts:
            key = (layer_idx, eid)
            if key in self.affinity:
                for next_eid, prob in self.affinity[key].items():
                    if prob >= self.threshold:
                        predicted.add(next_eid)
        result = list(predicted)[: self.budget]
        self._pending.update(result)
        self.stats.issued += len(result)
        return result

    def report_usage(self, expert_id: int) -> None:
        if expert_id in self._pending:
            self.stats.useful += 1
            self._pending.discard(expert_id)


class HistoryPrefetcher:
    """Prefetch based on recent expert activation history."""

    def __init__(self, window: int = 50, budget: int = 4):
        self.window = window
        self.budget = budget
        self.history: list[set[int]] = []
        self.stats = PrefetchStats()
        self._pending: set[int] = set()

    def predict(self, layer_idx: int, selected_experts: list[int]) -> list[int]:
        self.history.append(set(selected_experts))
        if len(self.history) > self.window:
            self.history = self.history[-self.window :]

        freq: dict[int, int] = {}
        for activation_set in self.history:
            for eid in activation_set:
                freq[eid] = freq.get(eid, 0) + 1

        current = set(selected_experts)
        candidates = sorted(
            ((eid, cnt) for eid, cnt in freq.items() if eid not in current),
            key=lambda x: -x[1],
        )
        result = [eid for eid, _ in candidates[: self.budget]]
        self._pending.update(result)
        self.stats.issued += len(result)
        return result

    def report_usage(self, expert_id: int) -> None:
        if expert_id in self._pending:
            self.stats.useful += 1
            self._pending.discard(expert_id)


class LookaheadPrefetcher:
    """Per-layer pattern-matching prefetcher.

    Tracks a sliding window of activations *per layer index* and, given the
    current layer ``L``, predicts the most-frequently-activated experts for
    layers ``L+1 .. L+lookahead``.  This exploits the observation (ProMoE,
    ExpertFlow) that per-layer expert distributions are partially stable
    across tokens, making position-based prediction effective on its own.
    """

    def __init__(
        self,
        lookahead: int = 1,
        budget: int = 4,
        history_window: int = 50,
    ):
        self.lookahead = max(1, lookahead)
        self.budget = budget
        self.window = history_window
        self._per_layer: dict[int, list[set[int]]] = {}
        self.stats = PrefetchStats()
        self._pending: set[int] = set()

    def _record(self, layer_idx: int, selected: list[int]) -> None:
        buf = self._per_layer.setdefault(layer_idx, [])
        buf.append(set(selected))
        if len(buf) > self.window:
            del buf[: len(buf) - self.window]

    def _top_for_layer(self, layer_idx: int, exclude: set[int]) -> list[int]:
        buf = self._per_layer.get(layer_idx, [])
        if not buf:
            return []
        freq: dict[int, int] = {}
        for s in buf:
            for e in s:
                freq[e] = freq.get(e, 0) + 1
        ranked = sorted(
            ((e, c) for e, c in freq.items() if e not in exclude),
            key=lambda x: -x[1],
        )
        return [e for e, _ in ranked]

    def predict(self, layer_idx: int, selected_experts: list[int]) -> list[int]:
        self._record(layer_idx, selected_experts)

        predicted: list[int] = []
        seen: set[int] = set(selected_experts)
        for step in range(1, self.lookahead + 1):
            for eid in self._top_for_layer(layer_idx + step, exclude=seen):
                if len(predicted) >= self.budget:
                    break
                predicted.append(eid)
                seen.add(eid)
            if len(predicted) >= self.budget:
                break

        self._pending.update(predicted)
        self.stats.issued += len(predicted)
        return predicted

    def report_usage(self, expert_id: int) -> None:
        if expert_id in self._pending:
            self.stats.useful += 1
            self._pending.discard(expert_id)
