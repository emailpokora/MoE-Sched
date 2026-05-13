"""Tests for the runtime metrics monitor."""

import pytest

from moe_policylang.runtime.monitor import Monitor, MonitorSnapshot


class TestMonitorSnapshot:
    def test_defaults(self):
        s = MonitorSnapshot()
        assert s.hit_rate == 0.0
        assert s.latency_ms == 0.0
        assert s.access_count == 0


class TestMonitor:
    def test_hit_rate_tracking(self):
        m = Monitor(metrics=["hit_rate"], window=10)
        m.record_access(hit=True)
        m.record_access(hit=True)
        m.record_access(hit=False)
        snap = m.snapshot()
        assert snap.hit_rate == pytest.approx(2 / 3)

    def test_latency_tracking(self):
        m = Monitor(metrics=["latency"], window=10)
        m.record_access(hit=True, latency_ms=10.0)
        m.record_access(hit=False, latency_ms=50.0)
        snap = m.snapshot()
        assert snap.latency_ms == pytest.approx(30.0)

    def test_window_rolls(self):
        m = Monitor(metrics=["hit_rate"], window=5)
        for _ in range(5):
            m.record_access(hit=True)
        assert m.snapshot().hit_rate == pytest.approx(1.0)
        for _ in range(5):
            m.record_access(hit=False)
        assert m.snapshot().hit_rate == pytest.approx(0.0)

    def test_log_interval(self):
        m = Monitor(metrics=["hit_rate"], window=100, log_interval=10)
        for _ in range(25):
            m.record_access(hit=True)
        # 25 accesses / interval 10 = 2 snapshots logged (at 10 and 20)
        assert len(m.history) == 2

    def test_empty_monitor(self):
        m = Monitor(metrics=["hit_rate"], window=10)
        snap = m.snapshot()
        assert snap.hit_rate == 0.0
        assert snap.access_count == 0

    def test_only_tracks_requested_metrics(self):
        m = Monitor(metrics=["hit_rate"], window=10)
        m.record_access(hit=True, latency_ms=99.0)
        snap = m.snapshot()
        # Latency not tracked
        assert snap.latency_ms == 0.0

    def test_access_count_increments(self):
        m = Monitor(metrics=[], window=10)
        for _ in range(7):
            m.record_access(hit=True)
        assert m.snapshot().access_count == 7
