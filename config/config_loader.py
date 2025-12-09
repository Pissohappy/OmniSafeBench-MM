"""
1. general_config.yaml - General experiment configuration
2. model_config.yaml - Model detailed configuration
3. attacks/ - Attack method configuration
4. defenses/ - Defense method configuration
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Union, List, Optional
import copy


from core.data_formats import PipelineConfig


class ConfigLoader:
    """Configuration loader"""

    def __init__(self, config_dir: str = "config"):
        """
        Initialize configuration loader

        Args:
            config_dir: Configuration directory path
        """
        self.config_dir = Path(config_dir)

        # Cache configurations
        self._general_config = None
        self._model_config = None
        self._attack_configs = {}
        self._defense_configs = {}

    def load_general_config(
        self, config_file: str = "general_config.yaml"
    ) -> Dict[str, Any]:
        """
        Load general configuration file

        Args:
            config_file: General configuration file name

        Returns:
            General configuration dictionary
        """
        if self._general_config is not None:
            return self._general_config

        config_path = self.config_dir / config_file
        if not config_path.exists():
            raise FileNotFoundError(
                f"General configuration file does not exist: {config_path}"
            )

        self._general_config = self._load_yaml_file(config_path)
        return self._general_config

    def load_model_config(
        self, config_file: str = "model_config.yaml"
    ) -> Dict[str, Any]:
        """
        Load model configuration file

        Args:
            config_file: Model configuration file name

        Returns:
            Model configuration dictionary
        """
        if self._model_config is not None:
            return self._model_config

        config_path = self.config_dir / config_file
        if not config_path.exists():
            raise FileNotFoundError(
                f"Model configuration file does not exist: {config_path}"
            )

        self._model_config = self._load_yaml_file(config_path)
        return self._model_config

    def load_attack_config(self, attack_name: str) -> Dict[str, Any]:
        """
        Load attack method configuration

        Args:
            attack_name: Attack method name

        Returns:
            Attack configuration dictionary
        """
        if attack_name in self._attack_configs:
            return self._attack_configs[attack_name]

        config_path = self.config_dir / "attacks" / f"{attack_name}.yaml"
        if not config_path.exists():
            # Try .json format
            config_path = self.config_dir / "attacks" / f"{attack_name}.json"

        if not config_path.exists():
            raise FileNotFoundError(
                f"Attack configuration file does not exist: {attack_name}"
            )

        config = self._load_config_file(config_path)
        self._attack_configs[attack_name] = config
        return config

    def load_defense_config(self, defense_name: str) -> Dict[str, Any]:
        """
        Load defense method configuration

        Args:
            defense_name: Defense method name

        Returns:
            Defense configuration dictionary
        """
        if defense_name == "None":
            return {"name": "None", "description": "No defense", "parameters": {}}

        if defense_name in self._defense_configs:
            return self._defense_configs[defense_name]

        config_path = self.config_dir / "defenses" / f"{defense_name}.yaml"
        if not config_path.exists():
            # Try .json format
            config_path = self.config_dir / "defenses" / f"{defense_name}.json"

        if not config_path.exists():
            raise FileNotFoundError(
                f"Defense configuration file does not exist: {defense_name}"
            )

        config = self._load_config_file(config_path)
        self._defense_configs[defense_name] = config
        return config

    def load_all_configs(
        self, general_config_file: str = "general_config.yaml"
    ) -> PipelineConfig:
        """
        Load all configurations and merge into PipelineConfig

        Args:
            general_config_file: General configuration file name

        Returns:
            PipelineConfig object
        """
        # Load general configuration
        general_config = self.load_general_config(general_config_file)

        # Load model configuration
        model_config = self.load_model_config()

        # Build complete configuration dictionary
        full_config = self._build_full_config(general_config, model_config)

        # Convert to PipelineConfig object
        return PipelineConfig.from_dict(full_config)

    def _build_full_config(
        self, general_config: Dict[str, Any], model_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Build complete configuration dictionary

        Args:
            general_config: General configuration
            model_config: Model configuration

        Returns:
            Complete configuration dictionary
        """
        # Deep copy general configuration as base
        full_config = copy.deepcopy(general_config)

        # Process test case generation configuration
        if "test_case_generation" in full_config:
            test_case_cfg = full_config["test_case_generation"]

            # Load attack method configurations
            if "attacks" in test_case_cfg:
                attack_names = test_case_cfg["attacks"]
                attack_params = test_case_cfg.get("attack_params", {}) or {}

                # Merge attack configurations
                merged_attack_params = {}
                for attack_name in attack_names:
                    try:
                        attack_config = self.load_attack_config(attack_name)
                        base_params = attack_config.get("parameters", {})

                        # If there are override parameters in general config, merge them
                        if attack_name in attack_params:
                            merged_attack_params[attack_name] = self._deep_merge(
                                base_params, attack_params[attack_name]
                            )
                        else:
                            merged_attack_params[attack_name] = base_params
                    except FileNotFoundError:
                        print(
                            f"Warning: Attack configuration file does not exist: {attack_name}"
                        )
                        continue

                test_case_cfg["attack_params"] = merged_attack_params

        # Process response generation configuration
        if "response_generation" in full_config:
            response_cfg = full_config["response_generation"]

            # Process model configuration
            if "models" in response_cfg:
                model_names = response_cfg["models"]
                model_overrides = response_cfg.get("model_params", {}) or {}

                # Merge model configurations
                merged_model_params = {}
                for model_name in model_names:
                    model_info = self._find_model_config(model_name, model_config)
                    if model_info:
                        # If there are override parameters in general config, merge them
                        if model_name in model_overrides:
                            merged_model_params[model_name] = self._deep_merge(
                                model_info, model_overrides[model_name]
                            )
                        else:
                            merged_model_params[model_name] = model_info
                    else:
                        print(
                            f"Warning: Model configuration does not exist: {model_name}"
                        )
                        # Add empty configuration to avoid subsequent errors
                        merged_model_params[model_name] = {}

                response_cfg["model_params"] = merged_model_params

            # Process defense configuration
            if "defenses" in response_cfg:
                defense_names = response_cfg["defenses"]
                defense_overrides = response_cfg.get("defense_params", {}) or {}

                # Merge defense configurations
                merged_defense_params = {}
                for defense_name in defense_names:
                    if defense_name == "None":
                        merged_defense_params[defense_name] = {}
                        continue

                    try:
                        defense_config = self.load_defense_config(defense_name)
                        base_params = defense_config.get("parameters", {})

                        # If there are override parameters in general config, merge them
                        if defense_name in defense_overrides:
                            merged_defense_params[defense_name] = self._deep_merge(
                                base_params, defense_overrides[defense_name]
                            )
                        else:
                            merged_defense_params[defense_name] = base_params
                    except FileNotFoundError:
                        print(
                            f"Warning: Defense configuration file does not exist: {defense_name}"
                        )
                        continue

                response_cfg["defense_params"] = merged_defense_params

        # Process evaluation configuration
        if "evaluation" in full_config:
            eval_cfg = full_config["evaluation"]

            # Process evaluator model configuration
            if "evaluator_params" in eval_cfg:
                for evaluator_name, evaluator_params in eval_cfg[
                    "evaluator_params"
                ].items():
                    if "model" in evaluator_params:
                        model_name = evaluator_params["model"]
                        model_info = self._find_model_config(model_name, model_config)
                        if model_info:
                            # Merge into evaluator parameters
                            evaluator_params.update(model_info)

        return full_config

    def _find_model_config(
        self, model_name: str, model_config: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Find specified model in model configuration (supports new provider structure)

        Args:
            model_name: Model name
            model_config: Model configuration dictionary

        Returns:
            Model configuration information, returns None if not found
        """
        # Check if it's the defaults section
        if model_name == "defaults" or model_name in model_config.get("defaults", {}):
            return None  # defaults is not a specific model configuration

        # Support new provider structure
        if "providers" in model_config:
            # Traverse all providers to find model
            for provider_name, provider_config in model_config["providers"].items():
                if (
                    "models" in provider_config
                    and model_name in provider_config["models"]
                ):
                    # Get model configuration
                    model_info = provider_config["models"][model_name].copy()

                    # Inherit provider-level configuration
                    if "api_key" in provider_config:
                        model_info.setdefault("api_key", provider_config["api_key"])
                    if "base_url" in provider_config:
                        model_info.setdefault("base_url", provider_config["base_url"])

                    # Set provider information
                    model_info["provider"] = provider_name

                    return model_info

        # Backward compatibility: directly find model (old flat structure)
        elif model_name in model_config:
            return model_config[model_name]

        return None

    def _deep_merge(
        self, base: Dict[str, Any], override: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Deep merge two dictionaries

        Args:
            base: Base dictionary
            override: Override dictionary

        Returns:
            Merged dictionary
        """
        result = copy.deepcopy(base)

        for key, value in override.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value

        return result

    def _load_yaml_file(self, file_path: Path) -> Dict[str, Any]:
        """Load YAML file"""
        with open(file_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _load_json_file(self, file_path: Path) -> Dict[str, Any]:
        """Load JSON file"""
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_config_file(self, file_path: Path) -> Dict[str, Any]:
        """Load configuration file based on file extension"""
        suffix = file_path.suffix.lower()

        if suffix == ".yaml" or suffix == ".yml":
            return self._load_yaml_file(file_path)
        elif suffix == ".json":
            return self._load_json_file(file_path)
        else:
            raise ValueError(f"Unsupported configuration file format: {suffix}")


def load_config(config_file: str = "config/general_config.yaml") -> PipelineConfig:
    """
    Configuration loading function

    Args:
        config_file: Configuration file path, can be full path or relative path
                    Function will automatically split into config directory path and general config file name

    Returns:
        PipelineConfig object
    """
    # Convert configuration file path to Path object
    config_path = Path(config_file)

    # Get configuration directory path (directory where config file is located)
    config_dir = str(config_path.parent)

    # Get general configuration file name
    general_config_file = config_path.name

    # Create configuration loader and load all configurations
    loader = ConfigLoader(config_dir)
    return loader.load_all_configs(general_config_file)


def validate_config(config: PipelineConfig) -> bool:
    """
    Validate configuration validity

    Args:
        config: PipelineConfig object

    Returns:
        Whether configuration is valid
    """
    # Basic validation
    if not config.output_dir:
        print("Error: Output directory is not set")
        return False

    # Test case generation configuration validation
    test_case_cfg = config.test_case_generation
    if not test_case_cfg.get("input", {}).get("behaviors_file"):
        print("Error: Harmful behavior file is not set")
        return False

    # No longer check image_dir, as image paths are now read directly from behavior data

    if not test_case_cfg.get("attacks"):
        print("Error: Attack methods are not set")
        return False

    # Response generation configuration validation
    response_cfg = config.response_generation
    if not response_cfg.get("models"):
        print("Error: Models are not set")
        return False

    # Check if model configurations are complete
    model_params = response_cfg.get("model_params", {})
    for model_name in response_cfg.get("models", []):
        if model_name not in model_params:
            print(f"Warning: Model '{model_name}' configuration does not exist")

    return True


# Global model configuration lookup function
def get_model_config(
    model_name: str, config_dir: str = "config"
) -> Optional[Dict[str, Any]]:
    """
    Global function: Find model configuration by model name

    Args:
        model_name: Model name
        config_dir: Configuration directory path, defaults to "config"

    Returns:
        Model configuration dictionary, returns None if not found
    """
    try:
        loader = ConfigLoader(config_dir)
        model_config = loader.load_model_config()
        return loader._find_model_config(model_name, model_config)
    except Exception as e:
        print(f"Warning: Failed to find model configuration {model_name}: {e}")
        return None
