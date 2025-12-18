from typing import Dict, Any, Tuple, Optional, List

from core.base_classes import BaseDefense
from core.data_formats import TestCase

from core.unified_registry import UNIFIED_REGISTRY
from .utils import generate_output
from config.config_loader import get_model_config


class LlavaGuardDefense(BaseDefense):
    """LlavaGuard defense method - defense system specifically designed for LLaVA models"""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)

    def apply_defense(self, test_case: TestCase, **kwargs) -> TestCase:
        policy_file = self.config["policy"]
        with open(policy_file, "r", encoding="utf-8") as f:
            policy = f.read()

        attack_image_path = test_case.image_path

        defense_model_name = self.config["defense_model_name"]
        # Use global function to get model config
        model_config = get_model_config(defense_model_name)
        if not model_config:
            # If global lookup fails, try to get from local config
            model_params = self.config.get("model_params", {})
            model_config = model_params.get(defense_model_name, {})

        defense_client = UNIFIED_REGISTRY.create_model(defense_model_name, model_config)

        judge_response = generate_output(
            attack_image_path,
            policy,
            defense_client,
            temperature=0.2,
            max_new_tokens=500,
            top_p=0.95,
        )
        if "unsafe" in judge_response.lower():
            return self.block_input(
                test_case,
                defended_prompt=test_case.metadata.get("jailbreak_prompt", test_case.prompt),
                defended_image_path=test_case.image_path,
            )

        return self.create_defended_case(
            test_case=test_case,
            defended_prompt=test_case.metadata.get("jailbreak_prompt", test_case.prompt),
            defended_image_path=test_case.image_path,
        )
