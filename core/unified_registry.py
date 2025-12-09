"""
Unified registry system - merged version, removed redundant code
"""

from typing import Dict, Any, Optional, Type, List
import logging
import importlib
from typing import Set

from .base_classes import BaseAttack, BaseModel, BaseDefense, BaseEvaluator


class UnifiedRegistry:
    """Unified registry, only retains lazy mapping/explicit registration, no longer uses decorator paths"""

    def __init__(self):
        self.attack_registry: Dict[str, Type["BaseAttack"]] = {}
        self.model_registry: Dict[str, Type["BaseModel"]] = {}
        self.defense_registry: Dict[str, Type["BaseDefense"]] = {}
        self.evaluator_registry: Dict[str, Type["BaseEvaluator"]] = {}

        self.logger = logging.getLogger(__name__)
        # Imported module cache to avoid duplicate imports
        self._imported_modules: Set[str] = set()

    def register_attack(self, name: str, attack_class: Type["BaseAttack"]) -> None:
        """Register attack method"""
        if name in self.attack_registry:
            self.logger.warning(
                f"Attack method '{name}' already exists, will be overwritten"
            )
        self.attack_registry[name] = attack_class
        self.logger.debug(f"Registered attack method: {name}")

    def register_model(self, name: str, model_class: Type["BaseModel"]) -> None:
        """Register model"""
        if name in self.model_registry:
            self.logger.warning(f"Model '{name}' already exists, will be overwritten")
        self.model_registry[name] = model_class
        self.logger.debug(f"Registered model: {name}")

    def register_defense(self, name: str, defense_class: Type["BaseDefense"]) -> None:
        """Register defense method"""
        if name in self.defense_registry:
            self.logger.warning(
                f"Defense method '{name}' already exists, will be overwritten"
            )
        self.defense_registry[name] = defense_class
        self.logger.debug(f"Registered defense method: {name}")

    def register_evaluator(
        self, name: str, evaluator_class: Type["BaseEvaluator"]
    ) -> None:
        """Register evaluator"""
        if name in self.evaluator_registry:
            self.logger.warning(
                f"Evaluator '{name}' already exists, will be overwritten"
            )
        self.evaluator_registry[name] = evaluator_class
        self.logger.debug(f"Registered evaluator: {name}")

    def get_attack(self, name: str) -> Optional[Type["BaseAttack"]]:
        """Get attack method class"""
        if name in self.attack_registry:
            return self.attack_registry[name]

        # Get mapping information from registry_init
        try:
            from .registry_init import initialize_registry_lazy_imports

            mappings = initialize_registry_lazy_imports()
            if name in mappings["attacks"]:
                module_path, class_name = mappings["attacks"][name]
                module = importlib.import_module(module_path)
                cls = getattr(module, class_name)
                # Register to cache
                self.attack_registry[name] = cls
                self.logger.debug(
                    f"Successfully imported attack method from mapping: {name}"
                )
                return cls
        except (ImportError, AttributeError) as e:
            self.logger.debug(
                f"Unable to import attack method '{name}' from mapping: {e}"
            )
            return None
        except Exception as e:
            self.logger.error(
                f"Unknown error occurred while importing attack method '{name}': {e}"
            )
            return None

        self.logger.warning(f"Attack method '{name}' is not defined in mapping")
        return None

    def get_model(self, name: str) -> Optional[Type["BaseModel"]]:
        """Get model class"""
        if name in self.model_registry:
            return self.model_registry[name]

        # Get mapping information from registry_init
        try:
            from .registry_init import initialize_registry_lazy_imports

            mappings = initialize_registry_lazy_imports()
            if name in mappings["models"]:
                module_path, class_name = mappings["models"][name]
                module = importlib.import_module(module_path)
                cls = getattr(module, class_name)
                self.model_registry[name] = cls
                self.logger.debug(f"Successfully imported model from mapping: {name}")
                return cls
        except (ImportError, AttributeError) as e:
            self.logger.debug(f"Unable to import model '{name}' from mapping: {e}")
            return None
        except Exception as e:
            self.logger.error(
                f"Unknown error occurred while importing model '{name}': {e}"
            )
            return None

        self.logger.warning(f"Model '{name}' is not defined in mapping")
        return None

    def get_defense(self, name: str) -> Optional[Type["BaseDefense"]]:
        """Get defense method class"""
        if name in self.defense_registry:
            return self.defense_registry[name]

        # Get mapping information from registry_init
        try:
            from .registry_init import initialize_registry_lazy_imports

            mappings = initialize_registry_lazy_imports()
            if name in mappings["defenses"]:
                module_path, class_name = mappings["defenses"][name]
                module = importlib.import_module(module_path)
                cls = getattr(module, class_name)
                self.defense_registry[name] = cls
                self.logger.debug(
                    f"Successfully imported defense method from mapping: {name}"
                )
                return cls
        except (ImportError, AttributeError) as e:
            self.logger.debug(
                f"Unable to import defense method '{name}' from mapping: {e}"
            )
            return None
        except Exception as e:
            self.logger.error(
                f"Unknown error occurred while importing defense method '{name}': {e}"
            )
            return None

        self.logger.warning(f"Defense method '{name}' is not defined in mapping")
        return None

    def get_evaluator(self, name: str) -> Optional[Type["BaseEvaluator"]]:
        """Get evaluator class"""
        if name in self.evaluator_registry:
            return self.evaluator_registry[name]

        # Get mapping information from registry_init
        try:
            from .registry_init import initialize_registry_lazy_imports

            mappings = initialize_registry_lazy_imports()
            if name in mappings["evaluators"]:
                module_path, class_name = mappings["evaluators"][name]
                module = importlib.import_module(module_path)
                cls = getattr(module, class_name)
                self.evaluator_registry[name] = cls
                self.logger.debug(
                    f"Successfully imported evaluator from mapping: {name}"
                )
                return cls
        except (ImportError, AttributeError) as e:
            self.logger.debug(f"Unable to import evaluator '{name}' from mapping: {e}")
            return None
        except Exception as e:
            self.logger.error(
                f"Unknown error occurred while importing evaluator '{name}': {e}"
            )
            return None

        self.logger.warning(f"Evaluator '{name}' is not defined in mapping")
        return None

    def create_attack(
        self, name: str, config: Dict[str, Any] = None, output_image_dir: str = None
    ) -> Optional["BaseAttack"]:
        """Create attack method instance

        Args:
            name: Attack method name
            config: Configuration dictionary
            output_image_dir: Output image directory path
        """
        attack_class = self.get_attack(name)
        if attack_class:
            try:
                return attack_class(config=config, output_image_dir=output_image_dir)
            except Exception as e:
                self.logger.error(f"Failed to create attack method '{name}': {e}")
        return None

    def create_model(
        self, name: str, config: Dict[str, Any] = None
    ) -> Optional["BaseModel"]:
        """Create model instance"""
        # Get provider information from configuration
        if config is None:
            config = {}

        provider = config.get("provider", name)
        model_class = self.get_model(provider)

        if model_class:
            try:
                # Extract parameters from configuration
                model_name = config.get("model_name", name)
                api_key = config.get("api_key", "")
                base_url = config.get("base_url", None)

                # Pass appropriate parameters based on provider type
                if provider in ["openai", "qwen", "vllm", "any"]:
                    # OpenAI-compatible models need model_name, api_key, base_url
                    return model_class(
                        model_name=model_name, api_key=api_key, base_url=base_url
                    )
                elif provider in ["google", "anthropic", "doubao", "mistral"]:
                    # Other models only need model_name and api_key
                    return model_class(model_name=model_name, api_key=api_key)
                else:
                    # Default case
                    return model_class(model_name=model_name, api_key=api_key)
            except Exception as e:
                self.logger.error(
                    f"Failed to create model '{name}' (provider: {provider}): {e}"
                )
        else:
            self.logger.error(f"Model provider not found: {provider}")
        return None

    def create_defense(
        self, name: str, config: Dict[str, Any] = None
    ) -> Optional["BaseDefense"]:
        """Create defense method instance"""
        defense_class = self.get_defense(name)
        if defense_class:
            try:
                return defense_class(config=config)
            except Exception as e:
                self.logger.error(f"Failed to create defense method '{name}': {e}")
        return None

    def create_evaluator(
        self, name: str, config: Dict[str, Any] = None
    ) -> Optional["BaseEvaluator"]:
        """Create evaluator instance"""
        evaluator_class = self.get_evaluator(name)
        if evaluator_class:
            try:
                return evaluator_class(config=config)
            except Exception as e:
                self.logger.error(f"Failed to create evaluator '{name}': {e}")
        return None

    def list_attacks(self) -> List[str]:
        """List all registered attack methods"""
        return sorted(self.attack_registry.keys())

    def list_models(self) -> List[str]:
        """List all registered models"""
        return sorted(self.model_registry.keys())

    def list_defenses(self) -> List[str]:
        """List all registered defense methods"""
        return sorted(self.defense_registry.keys())

    def list_evaluators(self) -> List[str]:
        """List all registered evaluators"""
        return sorted(self.evaluator_registry.keys())

    def validate_attack(self, name: str) -> bool:
        """Validate if attack method exists"""
        return name in self.attack_registry

    def validate_model(self, name: str) -> bool:
        """Validate if model exists"""
        return name in self.model_registry

    def validate_defense(self, name: str) -> bool:
        """Validate if defense method exists"""
        return name in self.defense_registry

    def validate_evaluator(self, name: str) -> bool:
        """Validate if evaluator exists"""
        return name in self.evaluator_registry

    def get_component_summary(self) -> Dict[str, List[str]]:
        """Get component summary"""
        return {
            "attacks": self.list_attacks(),
            "defenses": self.list_defenses(),
            "models": self.list_models(),
            "evaluators": self.list_evaluators(),
        }

    def initialize_components(self) -> Dict[str, List[str]]:
        """Initialize all components and return summary (based on lazy mapping, dynamically imported when accessed)"""
        summary = self.get_component_summary()
        self.logger.info("Component initialization completed")
        self.logger.info(f"Available attack methods: {len(summary['attacks'])}")
        self.logger.info(f"Available defense methods: {len(summary['defenses'])}")
        self.logger.info(f"Available models: {len(summary['models'])}")
        self.logger.info(f"Available evaluators: {len(summary['evaluators'])}")

        return summary


# Global registry instance
UNIFIED_REGISTRY = UnifiedRegistry()
