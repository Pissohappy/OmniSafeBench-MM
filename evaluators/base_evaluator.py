"""
Evaluator base class - new architecture implementation
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any

from core.data_formats import ModelResponse, EvaluationResult


class BaseEvaluator(ABC):
    """Evaluator abstract base class"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        self.name = self.__class__.__name__.lower()

    @abstractmethod
    def evaluate_response(
        self, model_response: ModelResponse, **kwargs
    ) -> EvaluationResult:
        """
        Evaluate model response

        Args:
            model_response: Model response
            **kwargs: Additional parameters

        Returns:
            Evaluation result
        """
        pass
