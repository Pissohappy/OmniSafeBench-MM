"""
JailGuard defense method - multi-layer defense system specifically for jailbreak attacks
Implementation based on original JailGuard logic: multi-layer detection (keywords, patterns, semantics, context)
"""

import logging
import re
import os
from typing import Dict, Any, Tuple, Optional, List
from PIL import Image
from io import BytesIO
from core.unified_registry import UNIFIED_REGISTRY
from .base_defense import BaseDefense
from core.data_formats import TestCase
from .JailGuard_utils.utils import update_divergence, detect_attack
from .JailGuard_utils.augmentations import (
    get_method_text,
    get_method_image,
)
from .utils import generate_output

import nltk
import spacy
from config.config_loader import get_model_config


class JailGuardDefense(BaseDefense):
    """JailGuard defense method - multi-layer defense system specifically for jailbreak attacks"""

    def apply_defense(self, test_case: TestCase, **kwargs) -> TestCase:
        mutator = self.config["default_mutator"]
        number = self.config["default_number"]
        threshold = self.config["default_threshold"]

        # Get model parameters from config

        target_model_name = self.config.get("target_model_name")
        model_config = get_model_config(target_model_name)
        target_model = (
            UNIFIED_REGISTRY.create_model(target_model_name, model_config)
            if target_model_name
            else None
        )

        # print("Downloading/loading NLTK stopwords and Spacy model (once only)...")
        nltk.download("stopwords")
        metric = spacy.load("en_core_web_md")

        temp_image_base_dir = os.path.join(
            self.config["output_dir"], "temp_jailguard_images"
        )
        os.makedirs(temp_image_base_dir, exist_ok=True)

        idx = test_case.test_case_id
        attack_image_path = test_case.image_path
        attack_prompt = test_case.prompt
        rewrite_attack_prompt_list = []
        rewrite_attack_image_list = []
        rewrite_output_list = []

        case_temp_dir = os.path.join(temp_image_base_dir, f"idx_{idx}")
        os.makedirs(case_temp_dir, exist_ok=True)
        temp_image_paths_to_clean = []

        for i in range(number):
            # print(f"  - Processing perturbation {i+1}/{number} for idx {idx}")
            text_method = get_method_text(mutator)
            image_method = get_method_image(mutator)
            rewrite_attack_prompt = text_method(text_list=[attack_prompt])[0]
            rewrite_attack_prompt_list.append(rewrite_attack_prompt)

            attack_image = Image.open(attack_image_path)
            if attack_image.mode != "RGB":
                attack_image = attack_image.convert("RGB")
            jpg_buffer = BytesIO()
            attack_image.save(jpg_buffer, format="JPEG", quality=100)
            jpg_buffer.seek(0)
            attack_image = Image.open(jpg_buffer)
            rewrite_attack_image = image_method(img=attack_image)
            rewrite_attack_image_list.append(rewrite_attack_image)

            temp_image_filename = f"idx_{idx}_perturb_{i}.jpg"
            rewrite_attack_image_path = os.path.join(case_temp_dir, temp_image_filename)
            rewrite_attack_image.save(
                rewrite_attack_image_path, format="JPEG", quality=100
            )
            temp_image_paths_to_clean.append(rewrite_attack_image_path)

            rewrite_output = generate_output(
                rewrite_attack_image_path,
                rewrite_attack_prompt,
                target_model,
            )
            # =========================================================

            rewrite_output_list.append(rewrite_output)

        max_div, jailbreak_keywords = update_divergence(
            rewrite_output_list, metric=metric
        )
        # print("max_div:", max_div, type(max_div))
        # print("jailbreak_keywords:", jailbreak_keywords)
        # print("threshold:", threshold)

        # Clean up temporary files
        for path in temp_image_paths_to_clean:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except OSError:
                pass
        try:
            if not os.listdir(case_temp_dir):
                os.rmdir(case_temp_dir)
        except OSError:
            pass
        detection_result = detect_attack(max_div, jailbreak_keywords, threshold)

        defended_case = self.create_defended_case(
            test_case=test_case,
            defended_prompt=test_case.prompt,
            defended_image_path=test_case.image_path,
            metadata={"should_return_default": detection_result == True},
        )

        return defended_case
