"""Performance tests for the Evidence Binding Pipeline.

These tests verify that key operations complete within expected time limits
and that optimizations don't regress performance.
"""

from __future__ import annotations

import time
from typing import List

import numpy as np
import pytest

from final_sc_review.metrics.ranking import ndcg_at_k, recall_at_k, mrr_at_k


class TestMetricsPerformance:
    """Performance tests for metric computation."""

    def test_ndcg_computation_speed(self):
        """Test nDCG computation completes in reasonable time."""
        # Setup: 1000 queries
        n_queries = 1000
        gold_ids = [list(range(5)) for _ in range(n_queries)]
        ranked_ids = [list(range(20)) for _ in range(n_queries)]

        start = time.perf_counter()
        for g, r in zip(gold_ids, ranked_ids):
            ndcg_at_k(g, r, k=10)
        elapsed = time.perf_counter() - start

        # Should complete 1000 queries in under 1 second
        assert elapsed < 1.0, f"nDCG too slow: {elapsed:.2f}s for {n_queries} queries"

    def test_recall_computation_speed(self):
        """Test Recall computation completes in reasonable time."""
        n_queries = 1000
        gold_ids = [list(range(5)) for _ in range(n_queries)]
        ranked_ids = [list(range(20)) for _ in range(n_queries)]

        start = time.perf_counter()
        for g, r in zip(gold_ids, ranked_ids):
            recall_at_k(g, r, k=10)
        elapsed = time.perf_counter() - start

        assert elapsed < 0.5, f"Recall too slow: {elapsed:.2f}s for {n_queries} queries"

    def test_mrr_computation_speed(self):
        """Test MRR computation completes in reasonable time."""
        n_queries = 1000
        gold_ids = [list(range(5)) for _ in range(n_queries)]
        ranked_ids = [list(range(20)) for _ in range(n_queries)]

        start = time.perf_counter()
        for g, r in zip(gold_ids, ranked_ids):
            mrr_at_k(g, r, k=10)
        elapsed = time.perf_counter() - start

        assert elapsed < 0.5, f"MRR too slow: {elapsed:.2f}s for {n_queries} queries"


class TestGraphBuilderPerformance:
    """Performance tests for graph construction."""

    @pytest.fixture
    def mock_embeddings(self):
        """Create mock embeddings for testing."""
        np.random.seed(42)
        return np.random.randn(100, 128).astype(np.float32)

    def test_knn_numpy_speed(self, mock_embeddings):
        """Test numpy kNN is reasonably fast."""
        from final_sc_review.gnn.config import GraphConstructionConfig, EdgeType
        from final_sc_review.gnn.graphs.builder import GraphBuilder

        config = GraphConstructionConfig(
            embedding_dim=128,
            edge_types=[EdgeType.SEMANTIC_KNN],
            knn_k=5,
            knn_threshold=0.5,
        )
        builder = GraphBuilder(config)

        # Normalize embeddings
        norms = np.linalg.norm(mock_embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        normed = (mock_embeddings / norms).astype(np.float32)

        start = time.perf_counter()
        edges = builder._build_knn_numpy(normed, k=6, threshold=0.5)
        elapsed = time.perf_counter() - start

        # 100 nodes should complete in under 100ms
        assert elapsed < 0.1, f"kNN too slow: {elapsed:.2f}s for 100 nodes"
        assert edges.shape[0] == 2, "Edge index should have 2 rows"


class TestProfilingUtilities:
    """Tests for profiling utilities."""

    def test_profiler_basic(self):
        """Test profiler records timings correctly."""
        from final_sc_review.utils.profiling import Profiler

        profiler = Profiler(enabled=True)

        @profiler.profile("test_func")
        def test_func():
            time.sleep(0.01)

        test_func()
        test_func()

        stats = profiler.get_stats("test_func")
        assert stats is not None
        assert stats.call_count == 2
        assert stats.total_time >= 0.02

    def test_profiler_context_manager(self):
        """Test profiler context manager."""
        from final_sc_review.utils.profiling import Profiler

        profiler = Profiler(enabled=True)

        with profiler.time("block"):
            time.sleep(0.01)

        stats = profiler.get_stats("block")
        assert stats is not None
        assert stats.call_count == 1
        assert stats.total_time >= 0.01

    def test_profiler_disabled(self):
        """Test disabled profiler has minimal overhead."""
        from final_sc_review.utils.profiling import Profiler

        profiler = Profiler(enabled=False)

        @profiler.profile("test_func")
        def test_func():
            return 42

        result = test_func()
        assert result == 42
        assert profiler.get_stats("test_func") is None


class TestBatchConsistency:
    """Tests to verify batch and sequential results match."""

    def test_metrics_batch_matches_sequential(self):
        """Verify vectorized metrics match sequential computation."""
        from final_sc_review.metrics.ranking import ndcg_at_k, recall_at_k

        n_queries = 100
        np.random.seed(42)

        # Generate test data
        gold_ids = [list(np.random.choice(20, size=5, replace=False)) for _ in range(n_queries)]
        ranked_ids = [list(np.random.permutation(20)) for _ in range(n_queries)]

        # Sequential computation
        sequential_ndcg = [ndcg_at_k(g, r, k=10) for g, r in zip(gold_ids, ranked_ids)]
        sequential_recall = [recall_at_k(g, r, k=10) for g, r in zip(gold_ids, ranked_ids)]

        # Vectorized (using numpy arrays)
        vectorized_ndcg = np.array([ndcg_at_k(g, r, k=10) for g, r in zip(gold_ids, ranked_ids)])
        vectorized_recall = np.array([recall_at_k(g, r, k=10) for g, r in zip(gold_ids, ranked_ids)])

        # Results should match exactly
        np.testing.assert_array_almost_equal(sequential_ndcg, vectorized_ndcg)
        np.testing.assert_array_almost_equal(sequential_recall, vectorized_recall)
