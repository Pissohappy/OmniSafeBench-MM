"""
CIDER defense method - Content Integrity Defense
CIDER (Content Integrity Defense with Enhanced Robustness) defense method
"""

from typing import Dict, Any, Tuple, Optional
from pathlib import Path

from .base_defense import BaseDefense
from core.data_formats import TestCase
import argparse
from .cider_utils import defence, QApair


class CIDERDefense(BaseDefense):
    """CIDER (Content Integrity Defense with Enhanced Robustness) defense method"""

    def apply_defense(self, test_case: TestCase, **kwargs) -> TestCase:

        attack_image_path = test_case.image_path
        attack_prompt = test_case.prompt

        # Ensure output_dir exists
        output_dir = Path(self.config.get("output_dir", "./output"))
        output_dir.mkdir(parents=True, exist_ok=True)
        tmp_dir = output_dir / "tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        text_file = self.config["data_path"]

        CIDER_args = argparse.Namespace(
            tempdir=str(tmp_dir),  # e.g., "./tmp"
            denoiser="diffusion",  # or "dncnn"
            cuda=self.config.get("cuda", 0),  # GPU device
            text_file=text_file,  # Attack text file
            pair_mode="injection",  # or "injection"
            threshold=self.config["default_threshold"],  # Default threshold
            denoiser_path=self.config["denoiser_diffusion_path"],
            encoder_path=self.config["llava_path"],
        )
        denoised_image_paths, refuse = defence(
            imgpth=[attack_image_path], args=CIDER_args
        )
        pair = QApair(
            attack_prompt,
            denoised_image_paths[0],
            refuse[0],
            ans=None,
            behav="CIDER_defense",
        )

        defended_case = self.create_defended_case(
            test_case=test_case,
            defended_prompt=test_case.prompt,
            defended_image_path=(test_case.image_path),
            metadata={"should_return_default": pair.refuse == True},
        )
        return defended_case
