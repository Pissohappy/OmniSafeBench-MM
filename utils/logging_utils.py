"""
Logging utility functions
"""

import logging
import time
from functools import wraps
from typing import Callable, Any


def setup_logging(level: int = logging.INFO, log_file: str = None) -> None:
    """Setup logging configuration"""
    handlers = [logging.StreamHandler()]

    if log_file:
        handlers.append(logging.FileHandler(log_file))

    # Improved log format, includes code location information
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s() - %(message)s"
    )

    for handler in handlers:
        handler.setFormatter(formatter)

    logging.basicConfig(level=level, handlers=handlers)


def get_logger(name: str) -> logging.Logger:
    """Get logger with specified name"""
    return logging.getLogger(name)


def log_execution_time(func: Callable) -> Callable:
    """Decorator to record function execution time"""

    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        logger = get_logger(func.__module__)
        start_time = time.time()

        logger.info(f"Starting execution: {func.__name__}")

        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(
                f"Completed execution: {func.__name__}, took {execution_time:.2f} seconds"
            )
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(
                f"Execution failed: {func.__name__}, took {execution_time:.2f} seconds, error: {e}"
            )
            raise

    return wrapper


def log_errors(func: Callable) -> Callable:
    """Decorator to record errors and stack traces

    Usage example:
        @log_errors
        def my_function():
            # Function code
            pass
    """

    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        logger = get_logger(func.__module__)

        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Use logger.exception to automatically record stack trace
            logger.exception(f"Function {func.__name__} execution failed: {e}")
            raise

    return wrapper


def log_with_context(context: str = None):
    """Error logging decorator with context

    Usage example:
        @log_with_context("Processing user request")
        def process_request():
            pass

        @log_with_context("Processing data")
        def process_data(self, data):
            pass
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Use the module name where the function is located
            logger = get_logger(func.__module__)
            func_name = func.__name__
            context_str = f" [{context}]" if context else ""

            logger.debug(f"Starting execution{context_str}: {func_name}")

            try:
                result = func(*args, **kwargs)
                logger.debug(f"Completed execution{context_str}: {func_name}")
                return result
            except Exception as e:
                logger.exception(
                    f"Execution failed{context_str}: {func_name}, error: {e}"
                )
                raise

        return wrapper

    return decorator
