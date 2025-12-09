"""
Batch processing strategy abstraction - provides configurable batch processing strategies
"""

from abc import ABC, abstractmethod
from typing import List, Callable, Any, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging


class BatchProcessingStrategy(ABC):
    """Abstract base class for batch processing strategies"""

    @abstractmethod
    def process(
        self,
        items: List[Any],
        process_func: Callable[[Any], Dict],
        max_workers: Optional[int] = None,
        **kwargs,
    ) -> List[Dict]:
        """
        Process a list of items

        Args:
            items: List of items to process
            process_func: Function to process a single item
            max_workers: Maximum number of worker threads (if applicable)
            **kwargs: Other parameters

        Returns:
            List of processing results
        """
        pass

    def get_name(self) -> str:
        """Get strategy name"""
        return self.__class__.__name__


class ParallelStrategy(BatchProcessingStrategy):
    """Parallel processing strategy - uses thread pool for parallel processing, supports batch processing"""

    def __init__(self, batch_size: int = 0):
        """
        Initialize parallel processing strategy

        Args:
            batch_size: Batch size. If > 0, process in batches with parallel processing within each batch; if <= 0, process all items in parallel at once
        """
        self.batch_size = batch_size

    def process(
        self,
        items: List[Any],
        process_func: Callable[[Any], Dict],
        max_workers: Optional[int] = None,
        **kwargs,
    ) -> List[Dict]:
        """Process items in parallel"""
        logger = kwargs.get("logger", logging.getLogger(__name__))
        max_workers = max_workers or min(4, len(items))

        # If batch_size <= 0, process all items at once
        if self.batch_size <= 0:
            logger.debug(
                f"Using parallel processing strategy, {max_workers} worker threads, processing {len(items)} items"
            )
            return self._process_parallel(items, process_func, max_workers, logger)

        # If batch_size > 0, process in batches
        logger.debug(
            f"Using batched parallel processing strategy, batch size: {self.batch_size}, {max_workers} worker threads, total items: {len(items)}"
        )

        results = []
        total_batches = (len(items) + self.batch_size - 1) // self.batch_size

        for i in range(0, len(items), self.batch_size):
            batch = items[i : i + self.batch_size]
            batch_num = i // self.batch_size + 1

            logger.debug(
                f"Processing batch {batch_num}/{total_batches}, contains {len(batch)} items"
            )

            batch_results = self._process_parallel(
                batch, process_func, max_workers, logger
            )
            results.extend(batch_results)

            logger.debug(
                f"Batch {batch_num}/{total_batches} completed, processed {len(batch_results)} items"
            )

        return results

    def _process_parallel(
        self,
        items: List[Any],
        process_func: Callable[[Any], Dict],
        max_workers: int,
        logger: logging.Logger,
    ) -> List[Dict]:
        """Process a batch of items in parallel"""
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_item = {
                executor.submit(process_func, item): item for item in items
            }

            # Collect results
            for future in as_completed(future_to_item):
                item = future_to_item[future]
                try:
                    result = future.result()
                    if result is not None:
                        results.append(result)
                except Exception as e:
                    logger.error(f"Failed to process item {item}: {e}")

        return results


class BatchedStrategy(BatchProcessingStrategy):
    """Batched processing strategy - divides items into batches, each batch uses parallel processing"""

    def __init__(self, batch_size: int = 10):
        """
        Initialize batched processing strategy

        Args:
            batch_size: Batch size
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")
        self.batch_size = batch_size

    def process(
        self,
        items: List[Any],
        process_func: Callable[[Any], Dict],
        max_workers: Optional[int] = None,
        **kwargs,
    ) -> List[Dict]:
        """Process items in batches, each batch uses parallel processing"""
        logger = kwargs.get("logger", logging.getLogger(__name__))
        max_workers = max_workers or min(4, len(items))

        logger.debug(
            f"Using batched processing strategy, batch size: {self.batch_size}, {max_workers} worker threads, total items: {len(items)}"
        )

        results = []
        total_batches = (len(items) + self.batch_size - 1) // self.batch_size

        for i in range(0, len(items), self.batch_size):
            batch = items[i : i + self.batch_size]
            batch_num = i // self.batch_size + 1

            logger.debug(
                f"Processing batch {batch_num}/{total_batches}, contains {len(batch)} items"
            )

            # Use parallel processing within each batch
            batch_results = self._process_batch_parallel(
                batch, process_func, max_workers, logger
            )
            results.extend(batch_results)

            logger.debug(
                f"Batch {batch_num}/{total_batches} completed, processed {len(batch_results)} items"
            )

        return results

    def _process_batch_parallel(
        self,
        items: List[Any],
        process_func: Callable[[Any], Dict],
        max_workers: int,
        logger: logging.Logger,
    ) -> List[Dict]:
        """Process a batch of items in parallel"""
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_item = {
                executor.submit(process_func, item): item for item in items
            }

            # Collect results
            for future in as_completed(future_to_item):
                item = future_to_item[future]
                try:
                    result = future.result()
                    if result is not None:
                        results.append(result)
                except Exception as e:
                    logger.error(f"Failed to process item {item}: {e}")

        return results


# Strategy factory
class BatchStrategyFactory:
    """Batch processing strategy factory"""

    _strategies = {
        "parallel": ParallelStrategy,
        "batched": BatchedStrategy,
    }

    @classmethod
    def create_strategy(
        cls, strategy_name: str = "parallel", **kwargs
    ) -> BatchProcessingStrategy:
        """
        Create batch processing strategy

        Args:
            strategy_name: Strategy name, optional values: "parallel", "batched"
            **kwargs: Strategy initialization parameters
                - batch_size: Batch size (required, must be > 0)

        Returns:
            Batch processing strategy instance

        Raises:
            ValueError: If strategy name is invalid or parameters are invalid
        """
        if strategy_name not in cls._strategies:
            raise ValueError(
                f"Invalid strategy name: {strategy_name}, available strategies: {list(cls._strategies.keys())}"
            )

        strategy_class = cls._strategies[strategy_name]

        # All strategies require batch_size parameter
        batch_size = kwargs.pop("batch_size", None)
        if batch_size is None:
            raise ValueError(
                f"Strategy {strategy_name} requires batch_size parameter, but it was not provided"
            )

        # For parallel strategy, batch_size <= 0 means process all items at once
        # For batched strategy, batch_size must be > 0
        if strategy_name == "batched" and batch_size <= 0:
            raise ValueError("batched strategy's batch_size must be greater than 0")

        return strategy_class(batch_size=batch_size)

    @classmethod
    def register_strategy(cls, name: str, strategy_class):
        """Register custom strategy"""
        if not issubclass(strategy_class, BatchProcessingStrategy):
            raise ValueError("Strategy class must inherit from BatchProcessingStrategy")
        cls._strategies[name] = strategy_class

    @classmethod
    def list_strategies(cls) -> List[str]:
        """List all available strategies"""
        return list(cls._strategies.keys())
