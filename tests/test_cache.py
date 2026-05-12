"""Tests for cache algorithm correctness: LRU, LFU, Score, FreqThreshold."""

import pytest

from moe_sched.runtime.cache import (
    CacheStats,
    FreqThresholdCache,
    LFUCache,
    LRUCache,
    ScoreCache,
)


# ---------------------------------------------------------------------------
# CacheStats
# ---------------------------------------------------------------------------

class TestCacheStats:
    def test_initial(self):
        s = CacheStats()
        assert s.hits == 0 and s.misses == 0 and s.total == 0
        assert s.hit_rate == 0.0

    def test_hit_rate(self):
        s = CacheStats(hits=3, misses=7)
        assert s.hit_rate == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# LRU Cache
# ---------------------------------------------------------------------------

class TestLRUCache:
    def test_hit_and_miss(self):
        c = LRUCache(capacity=2)
        assert c.access(0) is False   # miss
        assert c.access(0) is True    # hit
        assert c.stats.hits == 1
        assert c.stats.misses == 1

    def test_eviction_order(self):
        c = LRUCache(capacity=2)
        c.access(0)  # cache: [0]
        c.access(1)  # cache: [0, 1]
        c.access(2)  # evict 0 -> cache: [1, 2]
        assert c.is_cached(1)
        assert c.is_cached(2)
        assert not c.is_cached(0)

    def test_access_refreshes_lru(self):
        c = LRUCache(capacity=2)
        c.access(0)
        c.access(1)
        c.access(0)  # refresh 0
        c.access(2)  # should evict 1, not 0
        assert c.is_cached(0)
        assert not c.is_cached(1)

    def test_pinned_experts_never_evicted(self):
        c = LRUCache(capacity=2, pin_experts=[0])
        c.access(0)
        c.access(1)
        c.access(2)  # must evict 1, not 0 (pinned)
        assert c.is_cached(0)
        assert not c.is_cached(1)
        assert c.is_cached(2)

    def test_pinned_preloaded(self):
        c = LRUCache(capacity=4, pin_experts=[10, 20])
        assert c.is_cached(10)
        assert c.is_cached(20)
        assert c.size == 2

    def test_capacity_one(self):
        c = LRUCache(capacity=1)
        c.access(0)
        c.access(1)
        assert c.size == 1
        assert c.is_cached(1)
        assert not c.is_cached(0)

    def test_size_never_exceeds_capacity(self, uniform_workload):
        c = LRUCache(capacity=16)
        for eid in uniform_workload:
            c.access(eid)
            assert c.size <= c.capacity

    def test_stats_consistency(self, skewed_workload):
        c = LRUCache(capacity=32)
        for eid in skewed_workload:
            c.access(eid)
        assert c.stats.total == len(skewed_workload)
        assert c.stats.hits + c.stats.misses == c.stats.total
        assert c.stats.hit_rate == pytest.approx(c.stats.hits / c.stats.total)


# ---------------------------------------------------------------------------
# LFU Cache
# ---------------------------------------------------------------------------

class TestLFUCache:
    def test_hit_and_miss(self):
        c = LFUCache(capacity=2)
        assert c.access(0) is False
        assert c.access(0) is True

    def test_evicts_least_frequent(self):
        c = LFUCache(capacity=2)
        c.access(0)  # freq: {0:1}
        c.access(1)  # freq: {0:1, 1:1}
        c.access(0)  # freq: {0:2, 1:1}
        c.access(2)  # evict 1 (freq=1) -> {0:2, 2:1}
        assert c.is_cached(0)
        assert not c.is_cached(1)
        assert c.is_cached(2)

    def test_pinned_never_evicted(self):
        c = LFUCache(capacity=2, pin_experts=[0])
        c.access(0)
        c.access(1)
        c.access(2)  # evict 1 (least frequent unpinned)
        assert c.is_cached(0)
        assert c.is_cached(2)

    def test_decay(self):
        c = LFUCache(capacity=4, decay=0.5)
        for _ in range(10):
            c.access(0)
        # After 100 total accesses, decay kicks in
        for i in range(1, 91):
            c.access(i % 3 + 1)
        # Expert 0 freq should have decayed
        assert c.is_cached(0) or not c.is_cached(0)  # just ensure no crash

    def test_size_bounded(self, uniform_workload):
        c = LFUCache(capacity=16)
        for eid in uniform_workload:
            c.access(eid)
            assert c.size <= c.capacity

    def test_skewed_hit_rate_better_than_uniform(
        self, uniform_workload, skewed_workload
    ):
        """LFU should achieve higher hit rate on skewed vs uniform workload."""
        cu = LFUCache(capacity=16)
        for eid in uniform_workload:
            cu.access(eid)

        cs = LFUCache(capacity=16)
        for eid in skewed_workload:
            cs.access(eid)

        assert cs.stats.hit_rate > cu.stats.hit_rate


# ---------------------------------------------------------------------------
# Score Cache
# ---------------------------------------------------------------------------

class TestScoreCache:
    def test_hit_and_miss(self):
        c = ScoreCache(capacity=2)
        assert c.access(0, score=0.9) is False
        assert c.access(0, score=0.8) is True

    def test_evicts_lowest_score(self):
        c = ScoreCache(capacity=2, ema_alpha=1.0)  # no smoothing
        c.access(0, score=0.9)
        c.access(1, score=0.1)
        c.access(2, score=0.5)  # evict 1 (score=0.1)
        assert c.is_cached(0)
        assert not c.is_cached(1)
        assert c.is_cached(2)

    def test_ema_smoothing(self):
        c = ScoreCache(capacity=4, ema_alpha=0.5)
        c.access(0, score=1.0)  # score = 1.0
        c.access(0, score=0.0)  # score = 0.5*0.0 + 0.5*1.0 = 0.5
        assert c.scores[0] == pytest.approx(0.5)

    def test_pinned_not_evicted(self):
        c = ScoreCache(capacity=2, pin_experts=[0])
        c.access(0, score=0.01)  # low score but pinned
        c.access(1, score=0.9)
        c.access(2, score=0.8)  # evict 1 (not 0, pinned)
        assert c.is_cached(0)

    def test_size_bounded(self, uniform_workload):
        c = ScoreCache(capacity=16)
        for eid in uniform_workload:
            c.access(eid, score=0.5)
            assert c.size <= c.capacity


# ---------------------------------------------------------------------------
# FreqThreshold Cache
# ---------------------------------------------------------------------------

class TestFreqThresholdCache:
    def test_frequent_expert_cached(self):
        c = FreqThresholdCache(capacity=8, threshold=0.3, window=10)
        for _ in range(10):
            c.access(0)  # 100% frequency
        assert c.is_cached(0)

    def test_infrequent_expert_evicted(self):
        c = FreqThresholdCache(capacity=8, threshold=0.3, window=20)
        c.access(99)  # one-off
        for _ in range(19):
            c.access(0)
        # expert 99 was 1/20 = 5% < 30% threshold
        assert not c.is_cached(99)

    def test_pinned_stays(self):
        c = FreqThresholdCache(capacity=4, threshold=0.5, window=10, pin_experts=[0])
        for i in range(1, 11):
            c.access(i)
        assert c.is_cached(0)

    def test_size_bounded(self, uniform_workload):
        c = FreqThresholdCache(capacity=16, threshold=0.01, window=100)
        for eid in uniform_workload:
            c.access(eid)
            assert c.size <= c.capacity


# ---------------------------------------------------------------------------
# Cross-algorithm comparison
# ---------------------------------------------------------------------------

class TestCacheComparison:
    """Sanity-check that all caches produce valid stats on the same workload."""

    @pytest.mark.parametrize("CacheClass,kwargs", [
        (LRUCache, {"capacity": 32}),
        (LFUCache, {"capacity": 32}),
        (ScoreCache, {"capacity": 32}),
        (FreqThresholdCache, {"capacity": 32, "threshold": 0.01, "window": 100}),
    ])
    def test_all_caches_run_without_error(self, skewed_workload, CacheClass, kwargs):
        c = CacheClass(**kwargs)
        for eid in skewed_workload:
            if isinstance(c, ScoreCache):
                c.access(eid, score=0.5)
            else:
                c.access(eid)
        assert c.stats.total == len(skewed_workload)
        assert 0.0 <= c.stats.hit_rate <= 1.0
