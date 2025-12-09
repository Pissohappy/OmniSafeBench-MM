"""
Defense method base class - new architecture implementation
Supports two defense modes:
1. Pre-processing defense: modify input before model generation (apply_defense)
2. Post-processing defense: modify output after model generation (post_process_response)
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple

from core.data_formats import TestCase


class BaseDefense(ABC):
    """Abstract base class for defense methods"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        self.name = self.__class__.__name__.replace("Defense", "").lower()

    @abstractmethod
    def apply_defense(self, test_case: TestCase, **kwargs) -> TestCase:
        """
        Apply defense method to test case (pre-processing)
        Modify input before model generates response

        Args:
            test_case: Original test case
            **kwargs: Additional parameters

        Returns:
            Test case after applying defense
        """
        pass

    def create_defended_case(
        self,
        test_case: TestCase,
        defended_prompt: str = None,
        defended_image_path: str = None,
        metadata: Dict = None,
    ) -> TestCase:
        """Create test case after applying defense"""
        return TestCase(
            test_case_id=test_case.test_case_id,
            prompt=defended_prompt,
            image_path=defended_image_path,
            metadata={
                **test_case.metadata,
                "defense_method": self.name,
                "defended_prompt": defended_prompt,
                "defended_image_path": defended_image_path,
                **(metadata or {}),
            },
        )
