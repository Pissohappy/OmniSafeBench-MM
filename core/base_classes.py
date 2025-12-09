"""
Core abstract class definitions
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple, Type, TypeVar
import logging

from .data_formats import TestCase, ModelResponse, EvaluationResult
from dataclasses import fields, is_dataclass, MISSING

T = TypeVar("T")


class BaseComponent(ABC):
    """Component base class, provides common configuration and logging handling logic"""

    # Configuration class, subclasses can override
    CONFIG_CLASS: Optional[Type] = None

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize base component

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.name = self.__class__.__name__

        # Initialize logger (using the module name where component is located)
        self.logger = logging.getLogger(self.__class__.__module__)

        # Automatically handle configuration conversion
        self._process_config(config)

    def _process_config(self, config: Dict[str, Any] = None) -> None:
        """Process configuration, convert to configuration object or keep as dictionary"""
        config = config or {}

        if self.CONFIG_CLASS is not None and is_dataclass(self.CONFIG_CLASS):
            # Convert configuration dictionary to configuration object
            cfg_dict = config
            allowed_fields = {f.name for f in fields(self.CONFIG_CLASS)}
            filtered = {k: v for k, v in cfg_dict.items() if k in allowed_fields}
            self.cfg = self.CONFIG_CLASS(**filtered)
        else:
            # No configuration class or not a dataclass, use dictionary directly
            self.cfg = config

        # Validate configuration
        self.validate_config()

    def validate_config(self) -> None:
        """Validate configuration, subclasses can override this method to add custom validation logic

        Default implementation checks required fields (if configuration class has required fields)
        """
        if hasattr(self, "cfg") and is_dataclass(self.cfg):
            # Check required fields
            for field in fields(self.cfg.__class__):
                if field.default is MISSING and field.default_factory is MISSING:
                    # This is a required field (no default value or default factory)
                    if (
                        not hasattr(self.cfg, field.name)
                        or getattr(self.cfg, field.name) is None
                    ):
                        raise ValueError(
                            f"Configuration field '{field.name}' is required but not set"
                        )
        elif hasattr(self, "cfg") and isinstance(self.cfg, dict):
            # For dictionary configuration, can add dictionary-specific validation
            pass
        # Subclasses can add more validation logic


class BaseAttack(BaseComponent, ABC):
    """Attack method base class (enhanced version)"""

    def __init__(
        self, config: Dict[str, Any] = None, output_image_dir: Optional[str] = None
    ):
        """
        Initialize attack method

        Args:
            config: Configuration dictionary, will be automatically converted to configuration object
            output_image_dir: Output image directory path
        """
        # Call parent class initialization (including logging and configuration handling)
        super().__init__(config)

        # Set output image directory
        from pathlib import Path

        self.output_image_dir = Path(output_image_dir) if output_image_dir else None

        # Determine if local model needs to be loaded
        self.load_model = self._determine_load_model()

    def _determine_load_model(self) -> bool:
        """Determine if local model needs to be loaded

        Only decide based on the load_model field in configuration
        """
        # Check if load_model field exists in configuration
        if hasattr(self, "cfg"):
            config_obj = self.cfg
        else:
            config_obj = self.config

        # Only check load_model configuration item
        if isinstance(config_obj, dict):
            return config_obj.get("load_model", False)
        elif hasattr(config_obj, "load_model"):
            return getattr(config_obj, "load_model", False)

        return False

    @abstractmethod
    def generate_test_case(
        self, original_prompt: str, image_path: str, case_id: str, **kwargs
    ) -> TestCase:
        """
        Generate test case

        Args:
            original_prompt: Original prompt
            image_path: Image path
            case_id: Test case ID
            **kwargs: Other parameters

        Returns:
            TestCase: Generated test case
        """
        pass

    def __str__(self):
        return f"{self.name}(config={self.config})"


class BaseModel(BaseComponent, ABC):
    """Model base class (enhanced version)"""

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize model

        Args:
            config: Configuration dictionary, will be automatically converted to configuration object
        """
        # Call parent class initialization (including logging and configuration processing)
        super().__init__(config)

    @abstractmethod
    def generate_response(self, test_case: TestCase, **kwargs) -> ModelResponse:
        """
        Generate model response

        Args:
            test_case: Test case
            **kwargs: Other parameters

        Returns:
            ModelResponse: Model response
        """
        pass

    def __str__(self):
        return f"{self.name}(config={self.config})"


class BaseDefense(BaseComponent, ABC):
    """Defense method base class (enhanced version)"""

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize defense method

        Args:
            config: Configuration dictionary, will be automatically converted to configuration object
        """
        # Call parent class initialization (including logging and configuration handling)
        super().__init__(config)

        # Determine if local model needs to be loaded
        self.load_model = self._determine_load_model()

    def _determine_load_model(self) -> bool:
        """Determine if local model needs to be loaded

        Only based on the load_model field in configuration
        """
        # Check if load_model field exists in configuration
        if hasattr(self, "cfg"):
            config_obj = self.cfg
        else:
            config_obj = self.config

        # Only check load_model configuration item
        if isinstance(config_obj, dict):
            return config_obj.get("load_model", False)
        elif hasattr(config_obj, "load_model"):
            return getattr(config_obj, "load_model", False)

        return False

    @abstractmethod
    def apply_defense(self, test_case: TestCase, **kwargs) -> TestCase:
        """
        Apply defense method

        Args:
            test_case: Original test case
            **kwargs: Other parameters

        Returns:
            TestCase: Test case after applying defense
        """
        pass

    def __str__(self):
        return f"{self.name}(config={self.config})"


class BaseEvaluator(BaseComponent, ABC):
    """Evaluator base class (enhanced version)"""

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize evaluator

        Args:
            config: Configuration dictionary, will be automatically converted to configuration object
        """
        # Call parent class initialization (including logging and configuration handling)
        super().__init__(config)

    @abstractmethod
    def evaluate(self, model_response: ModelResponse, **kwargs) -> EvaluationResult:
        """
        Evaluate model response

        Args:
            model_response: Model response
            **kwargs: Other parameters

        Returns:
            EvaluationResult: Evaluation result
        """
        pass

    def __str__(self):
        return f"{self.name}(config={self.config})"


# Note: Old architecture's Registry class and global registry instance have been removed
# Please use UNIFIED_REGISTRY from core.unified_registry
