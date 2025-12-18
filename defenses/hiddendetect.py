"""
HiddenDetect defense method - defense based on hidden detection
Implementation based on original HiddenDetect logic: uses specialized model to detect harmful responses and intercept
"""

import logging
from typing import Dict, Any, Tuple, Optional

from core.base_classes import BaseDefense
from core.data_formats import TestCase
from .utils import score_and_generate_hiddendetect
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
import torch


class HiddenDetectDefense(BaseDefense):
    """HiddenDetect defense method - defense based on hidden detection"""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)

        # --- 1. Load model and config (execute only once) ---
        hiddendetect_model_path = self.config["llava-v1.6-vicuna-7b"]

        self.HIDDENDETECT_THRESHOLD = self.config["default_threshold"]

        # Load model, tokenizer, image processor
        print("Loading HiddenDetect model (once only)...")
        model_name = get_model_name_from_path(hiddendetect_model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = (
            load_pretrained_model(
                model_path=hiddendetect_model_path,
                model_base=None,
                model_name=model_name,
                device_map="auto",
                torch_dtype=torch.float16,
            )
        )
        print("Model loaded.")

        # Pre-compute refusal vectors (from hiddendetect_test)
        self.refusal_lst = [
            "Sorry",
            "sorry",
            "unfortunately",
            "unfortunate",
            "sadly",
            "explicit",
            "deadly",
            "crim",
            "criminal",
            "illegal",
            "dangerous",
            "harmful",
            "warning",
            "alarm",
            "caution",
            "shame",
            "conspiracy",
            "Subject",
            "contrary",
            "shouldn",
        ]
        self.refusal_token_ids = torch.zeros(self.tokenizer.vocab_size)
        for t in self.refusal_lst:
            token_id = self.tokenizer.encode(t, add_special_tokens=False)
            if token_id:
                self.refusal_token_ids[token_id[0]] = 1.0

    def apply_defense(self, test_case: TestCase, **kwargs) -> TestCase:

        sample = {
            "txt": test_case.prompt,
            "img": test_case.image_path,
        }

        score, output = score_and_generate_hiddendetect(
            sample=sample,
            model=self.model,
            tokenizer=self.tokenizer,
            image_processor=self.image_processor,
            refusal_token_ids=self.refusal_token_ids,
            threshold=self.HIDDENDETECT_THRESHOLD,
            max_new_tokens=1000,
        )
        if output == "I'm sorry, I can't assist with that.":
            return self.block_input(
                test_case,
                default_response=output,
                defended_prompt=test_case.prompt,
                defended_image_path=test_case.image_path,
            )

        # If not blocked, we can directly return the generated safe output
        return self.reply_directly(
            test_case,
            response_text=output,
            defended_prompt=test_case.prompt,
            defended_image_path=test_case.image_path,
            metadata={"hiddendetect_score": score},
        )
