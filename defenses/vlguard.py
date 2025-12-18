import logging
import torch
from typing import Dict, Any, Tuple, Optional
from PIL import Image

from core.base_classes import BaseDefense
from core.data_formats import TestCase

import torch
from llava.model.builder import load_pretrained_model as load_llava_model
from llava.conversation import conv_templates
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from llava.mm_utils import tokenizer_image_token

# https://github.com/haotian-liu/LLaVA
# After downloading, place the llava folder in the project root directory


class VLGuardDefense(BaseDefense):

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        vlguard_model_path = self.config["llava-v1.5-7b-vlguard"]

        print(f"VLGuard model path: {vlguard_model_path}")

        model_name = "llava"
        self.tokenizer, self.model, self.image_processor, self.context_len = (
            load_llava_model(
                model_path=vlguard_model_path,
                model_base=None,
                model_name=model_name,
                device_map="auto",
                torch_dtype=torch.float16,
            )
        )

    def apply_defense(self, test_case: TestCase, **kwargs) -> TestCase:
        attack_image_path = test_case.image_path
        attack_prompt = test_case.prompt

        # Load and preprocess image
        image = Image.open(attack_image_path).convert("RGB")
        image_tensor = (
            self.image_processor.preprocess([image], return_tensors="pt")[
                "pixel_values"
            ]
            .to(torch.float16)
            .cuda()
        )

        # Build conversation prompt conforming to LLaVA v1.5
        if self.model.config.mm_use_im_start_end:
            inp = (
                DEFAULT_IM_START_TOKEN
                + DEFAULT_IMAGE_TOKEN
                + DEFAULT_IM_END_TOKEN
                + "\n"
                + attack_prompt
            )
        else:
            inp = DEFAULT_IMAGE_TOKEN + "\n" + attack_prompt

        conv_mode = "llava_v1"
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        full_prompt = conv.get_prompt()

        # Tokenize
        input_ids = (
            tokenizer_image_token(
                full_prompt,
                self.tokenizer,
                IMAGE_TOKEN_INDEX,
                return_tensors="pt",
            )
            .unsqueeze(0)
            .cuda()
        )

        # Generate model output
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0),
                do_sample=False,
                temperature=self.config["temperature"],
                max_new_tokens=self.config["max_tokens"],
                min_new_tokens=1,
            )

        # Decode output
        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        return self.reply_directly(test_case, response_text=outputs)
