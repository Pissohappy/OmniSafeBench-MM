"""
CoCA defense method - new architecture implementation
"""

from typing import Dict, Any

from core.base_classes import BaseDefense
from core.data_formats import TestCase
import torch

from transformers import AutoProcessor, LlavaForConditionalGeneration

from .utils import generate_with_coca


class CoCADefense(BaseDefense):
    """CoCA defense method"""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        # Get parameters from config

    def apply_defense(self, test_case: TestCase, **kwargs) -> TestCase:

        # CoCA uses the victim model itself, so its path is the victim model's path
        coca_model_path = self.config["llava-1.5-7b-hf"]

        # Read CoCA parameters from JSON config
        safety_principle = self.config["safety_principle"]
        alpha = self.config["default_alpha"]

        # Load model using path read from JSON
        model = LlavaForConditionalGeneration.from_pretrained(
            coca_model_path, torch_dtype=torch.float16, device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(coca_model_path)
        print("Model and processor loaded.")

        # Call CoCA generation function
        output = generate_with_coca(
            case=test_case,
            model=model,
            processor=processor,
            safety_principle=safety_principle,
            alpha=alpha,
            max_new_tokens=1000,
        )
        return self.reply_directly(test_case, response_text=output)
