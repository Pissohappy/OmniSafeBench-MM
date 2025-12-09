"""
Test batch processing strategies
"""

import pytest
from unittest.mock import Mock, patch
import logging

from pipeline.batch_strategies import (
    BatchProcessingStrategy,
    SequentialStrategy,
    ParallelStrategy,
    BatchedStrategy,
    BatchStrategyFactory,
)


class TestBatchProcessingStrategies:
    """Test batch processing strategies"""

    def test_sequential_strategy(self):
        """Test sequential processing strategy"""
        strategy = SequentialStrategy()

        # Mock processing function
        process_func = Mock(side_effect=lambda x: {"value": x * 2})

        items = [1, 2, 3, 4, 5]
        results = strategy.process(items, process_func)

        assert len(results) == 5
        assert results[0]["value"] == 2
        assert results[1]["value"] == 4
        assert results[2]["value"] == 6
        assert results[3]["value"] == 8
        assert results[4]["value"] == 10

        # Verify processing function was called correctly
        assert process_func.call_count == 5
        process_func.assert_any_call(1)
        process_func.assert_any_call(2)
        process_func.assert_any_call(3)
        process_func.assert_any_call(4)
        process_func.assert_any_call(5)

    def test_sequential_strategy_with_failures(self):
        """Test failure handling in sequential processing strategy"""
        strategy = SequentialStrategy()

        # Mock processing function, third item will fail
        def process_func(x):
            if x == 3:
                raise ValueError("Processing failed")
            return {"value": x * 2}

        items = [1, 2, 3, 4, 5]

        # Use logger
        logger = logging.getLogger(__name__)
        results = strategy.process(items, process_func, logger=logger)

        # Only 4 items succeeded (except 3)
        assert len(results) == 4
        assert results[0]["value"] == 2
        assert results[1]["value"] == 4
        assert results[2]["value"] == 8  # Skipped 3, so it's 4*2
        assert results[3]["value"] == 10

    def test_parallel_strategy(self):
        """Test parallel processing strategy"""
        strategy = ParallelStrategy()

        # Mock processing function
        process_func = Mock(side_effect=lambda x: {"value": x * 2})

        items = [1, 2, 3, 4, 5]
        results = strategy.process(items, process_func, max_workers=2)

        assert len(results) == 5

        # Result order may differ, but content should be consistent
        result_values = {r["value"] for r in results}
        expected_values = {2, 4, 6, 8, 10}
        assert result_values == expected_values

        # Verify processing function was called correctly 5 times
        assert process_func.call_count == 5

    def test_batched_strategy(self):
        """Test batched processing strategy"""
        # Use batch size of 2
        strategy = BatchedStrategy(batch_size=2)

        # Mock processing function
        process_func = Mock(side_effect=lambda x: {"value": x * 2})

        items = [1, 2, 3, 4, 5]
        results = strategy.process(items, process_func)

        assert len(results) == 5
        assert results[0]["value"] == 2
        assert results[1]["value"] == 4
        assert results[2]["value"] == 6
        assert results[3]["value"] == 8
        assert results[4]["value"] == 10

        # Verify processing function was called correctly
        assert process_func.call_count == 5

    def test_batched_strategy_with_inner_strategy(self):
        """Test batched processing strategy with inner strategy"""
        # Create inner strategy (sequential strategy)
        inner_strategy = SequentialStrategy()
        strategy = BatchedStrategy(batch_size=2, inner_strategy=inner_strategy)

        process_func = Mock(side_effect=lambda x: {"value": x * 2})

        items = [1, 2, 3, 4, 5]
        results = strategy.process(items, process_func)

        assert len(results) == 5
        # Verify inner strategy is used
        assert isinstance(strategy.inner_strategy, SequentialStrategy)

    def test_strategy_names(self):
        """Test strategy names"""
        sequential = SequentialStrategy()
        parallel = ParallelStrategy()
        batched = BatchedStrategy()

        assert sequential.get_name() == "SequentialStrategy"
        assert parallel.get_name() == "ParallelStrategy"
        assert batched.get_name() == "BatchedStrategy"


class TestBatchStrategyFactory:
    """Test batch processing strategy factory"""

    def test_create_sequential_strategy(self):
        """Test creating sequential strategy"""
        strategy = BatchStrategyFactory.create_strategy("sequential")
        assert isinstance(strategy, SequentialStrategy)

    def test_create_parallel_strategy(self):
        """Test creating parallel strategy"""
        strategy = BatchStrategyFactory.create_strategy("parallel")
        assert isinstance(strategy, ParallelStrategy)

    def test_create_batched_strategy(self):
        """Test creating batched strategy"""
        strategy = BatchStrategyFactory.create_strategy("batched", batch_size=5)
        assert isinstance(strategy, BatchedStrategy)
        assert strategy.batch_size == 5

    def test_create_batched_strategy_with_inner_strategy(self):
        """Test creating batched strategy with inner strategy"""
        strategy = BatchStrategyFactory.create_strategy(
            "batched", batch_size=3, inner_strategy="parallel"
        )
        assert isinstance(strategy, BatchedStrategy)
        assert strategy.batch_size == 3
        assert isinstance(strategy.inner_strategy, ParallelStrategy)

    def test_create_invalid_strategy(self):
        """Test creating invalid strategy"""
        with pytest.raises(ValueError) as exc_info:
            BatchStrategyFactory.create_strategy("invalid_strategy")

        assert "Invalid strategy name" in str(exc_info.value)

    def test_list_strategies(self):
        """Test listing all strategies"""
        strategies = BatchStrategyFactory.list_strategies()

        assert "sequential" in strategies
        assert "parallel" in strategies
        assert "batched" in strategies
        assert len(strategies) >= 3

    def test_register_custom_strategy(self):
        """Test registering custom strategy"""

        # Create custom strategy class
        class CustomStrategy(BatchProcessingStrategy):
            def process(self, items, process_func, max_workers=None, **kwargs):
                return [{"custom": True}]

        # Register custom strategy
        BatchStrategyFactory.register_strategy("custom", CustomStrategy)

        # Verify can create custom strategy
        strategy = BatchStrategyFactory.create_strategy("custom")
        assert isinstance(strategy, CustomStrategy)

        # Verify custom strategy is in list
        strategies = BatchStrategyFactory.list_strategies()
        assert "custom" in strategies


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
