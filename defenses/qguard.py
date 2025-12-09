"""
QGuard defense method - Qwen safety protection
QGuard defense method, safety protection specifically designed for Qwen models
"""

from typing import Dict, Any, Tuple, Optional, List

from .base_defense import BaseDefense
from core.data_formats import TestCase

from .qguard_utils.vision import load_images_batch
from .qguard_utils.pipeline import (
    evaluate_prompt_with_pagerank,
    DEFAULT_ALLOWED_GROUPS,
)
from transformers import AutoModel, AutoTokenizer
import torch
import json


class QGuardDefense(BaseDefense):
    """QGuard defense method - Qwen safety protection"""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        qguard_path = self.config["QGuard_model"]
        # --- Model and data initialization (execute outside loop) ---
        self.qguard_model = AutoModel.from_pretrained(
            qguard_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            # use_flash_attn=True, # Can be kept if environment supports Flash Attention
            device_map="auto",
        ).eval()

        self.qguard_tokenizer = AutoTokenizer.from_pretrained(
            qguard_path,
            trust_remote_code=True,
            use_fast=False,
        )

        if self.qguard_tokenizer.pad_token_id is None:
            self.qguard_tokenizer.pad_token = self.qguard_tokenizer.eos_token
            self.qguard_tokenizer.pad_token_id = self.qguard_tokenizer.eos_token_id

    def apply_defense(self, test_case: TestCase, **kwargs) -> TestCase:

        qguard_threshold = self.config["default_threshold"]
        qguard_question_path = self.config["default_guard_question"]

        with open(qguard_question_path, "r", encoding="utf-8") as f:
            guard_qs: Dict[str, List[str]] = json.load(f)

        attack_image_path = test_case.image_path
        attack_prompt = test_case.prompt

        images_tensor = load_images_batch(
            [attack_image_path], input_size=448, max_num=12
        )
        judge_result = evaluate_prompt_with_pagerank(
            model=self.qguard_model,
            tokenizer=self.qguard_tokenizer,
            prompt=attack_prompt,
            guard_questions=guard_qs,
            allowed_groups=DEFAULT_ALLOWED_GROUPS,
            threshold=qguard_threshold,
            images_tensor=images_tensor,
        )

        # print("judge_result:", judge_result["prediction"])
        is_harmful = judge_result["prediction"]

        defended_case = self.create_defended_case(
            test_case=test_case,
            defended_prompt=test_case.metadata["jailbreak_prompt"],
            defended_image_path=test_case.image_path,
            metadata={"should_return_default": is_harmful.strip().lower() == "harmful"},
        )
        return defended_case
