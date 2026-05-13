"""Metrics monitoring for MoE-Sched runtime."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass
class MonitorSnapshot:
    """A point-in-time snapshot of monitored metrics."""
    hit_rate: float = 0.0
    latency_ms: float = 0.0
    memory_gb: float = 0.0
    access_count: int = 0


class Monitor:
    """Rolling-window metrics monitor."""

    def __init__(self, metrics: list[str], window: int = 100, log_interval: int = 50):
        self.tracked_metrics = set(metrics)
        self.window = window
        self.log_interval = log_interval

        self._hits: deque[bool] = deque(maxlen=window)
        self._latencies: deque[float] = deque(maxlen=window)
        self._access_count = 0
        self._log: list[MonitorSnapshot] = []

    def record_access(self, hit: bool, latency_ms: float = 0.0) -> None:
        self._access_count += 1
        if "hit_rate" in self.tracked_metrics:
            self._hits.append(hit)
        if "latency" in self.tracked_metrics:
            self._latencies.append(latency_ms)
        if self._access_count % self.log_interval == 0:
            self._log.append(self.snapshot())

    def snapshot(self) -> MonitorSnapshot:
        hr = sum(self._hits) / len(self._hits) if self._hits else 0.0
        lat = sum(self._latencies) / len(self._latencies) if self._latencies else 0.0
        return MonitorSnapshot(
            hit_rate=hr,
            latency_ms=lat,
            access_count=self._access_count,
        )

    @property
    def history(self) -> list[MonitorSnapshot]:
        return list(self._log)
