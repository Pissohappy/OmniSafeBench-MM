"""
Llama-Guard-4 defense method - Meta's safety model
Implementation based on original Llama-Guard-4 logic: uses specialized safety model for input safety checks
"""

import os
import logging
from typing import Dict, Any, Optional, Tuple, List

from .base_defense import BaseDefense
from core.data_formats import TestCase
from core.unified_registry import UNIFIED_REGISTRY
from .utils import generate_output
from config.config_loader import get_model_config


class LlamaGuard4Defense(BaseDefense):

    def apply_defense(self, test_case: TestCase, **kwargs) -> TestCase:
        target_model_name = self.config["target_model_name"]
        # Get model parameters from config
        # Use global function to get model config
        model_config = get_model_config("llama_guard_4")
        if not model_config:
            # If global lookup fails, try to get from local config
            model_params = self.config.get("model_params", {})
            model_config = model_params.get("llama_guard_4", {})
        defense_client = UNIFIED_REGISTRY.create_model(target_model_name, model_config)

        attack_image_path = test_case.image_path
        attack_prompt = test_case.prompt

        judge_response = generate_output(
            attack_image_path,  # Llama-Guard-4 requires image input
            attack_prompt,
            defense_client,
            max_new_tokens=10,
        )
        # print("judge_response:", judge_response)

        defended_case = self.create_defended_case(
            test_case=test_case,
            defended_prompt=attack_prompt,
            defended_image_path=attack_image_path,
            metadata={"should_return_default": "unsafe" in judge_response.lower()},
        )
        return defended_case
