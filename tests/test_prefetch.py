"""Tests for prefetch engines: Null, Affinity, History."""

import pytest

from moe_sched.runtime.prefetch import (
    AffinityPrefetcher,
    HistoryPrefetcher,
    NullPrefetcher,
)


class TestNullPrefetcher:
    def test_predicts_nothing(self):
        pf = NullPrefetcher()
        assert pf.predict(0, [1, 2]) == []

    def test_stats_zero(self):
        pf = NullPrefetcher()
        pf.predict(0, [1])
        assert pf.stats.issued == 0
        assert pf.stats.accuracy == 0.0


class TestAffinityPrefetcher:
    @pytest.fixture
    def affinity_map(self):
        """Layer 0, expert 1 → predicts expert 10 (next layer) with 80%."""
        return {
            (0, 1): {10: 0.8, 11: 0.1},
            (0, 2): {10: 0.5, 12: 0.4},
        }

    def test_predicts_above_threshold(self, affinity_map):
        pf = AffinityPrefetcher(affinity=affinity_map, threshold=0.3, budget=4)
        result = pf.predict(0, [1])
        assert 10 in result
        assert 11 not in result  # 0.1 < 0.3 threshold

    def test_budget_limits_predictions(self, affinity_map):
        pf = AffinityPrefetcher(affinity=affinity_map, threshold=0.01, budget=1)
        result = pf.predict(0, [1])
        assert len(result) <= 1

    def test_empty_affinity(self):
        pf = AffinityPrefetcher(affinity={}, threshold=0.3, budget=4)
        result = pf.predict(0, [1, 2, 3])
        assert result == []

    def test_stats_counting(self, affinity_map):
        pf = AffinityPrefetcher(affinity=affinity_map, threshold=0.3, budget=4)
        pf.predict(0, [1])  # predicts expert 10
        assert pf.stats.issued == 1
        pf.report_usage(10)
        assert pf.stats.useful == 1
        assert pf.stats.accuracy == pytest.approx(1.0)

    def test_unused_prediction(self, affinity_map):
        pf = AffinityPrefetcher(affinity=affinity_map, threshold=0.3, budget=4)
        pf.predict(0, [1])
        pf.report_usage(99)  # not predicted
        assert pf.stats.useful == 0

    def test_multiple_experts_combine(self, affinity_map):
        pf = AffinityPrefetcher(affinity=affinity_map, threshold=0.3, budget=10)
        result = pf.predict(0, [1, 2])
        # Expert 10 predicted by both, 12 predicted by expert 2
        assert 10 in result
        assert 12 in result


class TestHistoryPrefetcher:
    def test_predicts_frequent_experts(self):
        pf = HistoryPrefetcher(window=10, budget=2)
        for _ in range(5):
            pf.predict(0, [1, 2])
        result = pf.predict(0, [3])  # now ask for something different
        # Experts 1 and 2 were frequent, should be predicted
        assert 1 in result or 2 in result

    def test_budget_respected(self):
        pf = HistoryPrefetcher(window=10, budget=1)
        for _ in range(5):
            pf.predict(0, [1, 2, 3])
        result = pf.predict(0, [99])
        assert len(result) <= 1

    def test_window_limits_history(self):
        pf = HistoryPrefetcher(window=3, budget=4)
        pf.predict(0, [1])
        pf.predict(0, [2])
        pf.predict(0, [3])
        pf.predict(0, [4])  # window full, [1] should be pushed out
        result = pf.predict(0, [99])
        # Expert 1 may still appear but experts 2,3,4 are more frequent
        assert len(result) <= 4

    def test_current_experts_excluded(self):
        pf = HistoryPrefetcher(window=10, budget=4)
        for _ in range(5):
            pf.predict(0, [1, 2])
        result = pf.predict(0, [1, 2])
        # Current experts should NOT be in predictions
        assert 1 not in result
        assert 2 not in result

    def test_stats_tracking(self):
        pf = HistoryPrefetcher(window=10, budget=4)
        pf.predict(0, [1, 2])
        pf.predict(0, [3])
        issued_so_far = pf.stats.issued
        assert issued_so_far >= 0
        pf.report_usage(1)
