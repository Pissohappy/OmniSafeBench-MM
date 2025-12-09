"""
Uniguard defense method - unified safety protection
Implementation based on original Uniguard logic: adds text and image safety patches
"""

import os
import logging
import tempfile
import uuid
import threading
import time
from typing import Dict, Any, Optional, Tuple
from PIL import Image
import numpy as np

from .base_defense import BaseDefense
from core.data_formats import TestCase
from .utils import add_patch_to_image_universal


class UniguardDefense(BaseDefense):
    """Uniguard defense method - unified safety protection (thread-safe version)"""

    def __init__(self, config):
        super().__init__(config)
        # Temporary file tracking dictionary: {thread_id: [file_path_list]}
        self._temp_files = {}
        self._temp_files_lock = threading.Lock()

    def _generate_unique_filename(self, extension=".png"):
        """Generate unique filename"""
        # Use thread ID, timestamp and UUID to ensure uniqueness
        thread_id = threading.get_ident()
        timestamp = int(time.time() * 1000)  # Millisecond timestamp
        unique_id = uuid.uuid4().hex[:8]  # 8-digit UUID
        return f"temp_{thread_id}_{timestamp}_{unique_id}{extension}"

    def _register_temp_file(self, filepath):
        """Register temporary file for later cleanup"""
        thread_id = threading.get_ident()
        with self._temp_files_lock:
            if thread_id not in self._temp_files:
                self._temp_files[thread_id] = []
            self._temp_files[thread_id].append(filepath)
            self.logger.debug(
                f"Registered temporary file: {filepath} (thread: {thread_id})"
            )

    def _cleanup_thread_temp_files(self, thread_id=None):
        """Clean up temporary files for specified thread"""
        if thread_id is None:
            thread_id = threading.get_ident()

        with self._temp_files_lock:
            if thread_id in self._temp_files:
                files_to_remove = self._temp_files[thread_id]
                removed_count = 0

                for filepath in files_to_remove:
                    try:
                        if os.path.exists(filepath):
                            os.remove(filepath)
                            self.logger.debug(f"Cleaned up temporary file: {filepath}")
                            removed_count += 1
                        else:
                            self.logger.debug(
                                f"Temporary file does not exist: {filepath}"
                            )
                    except Exception as e:
                        self.logger.warning(
                            f"Failed to clean up temporary file {filepath}: {e}"
                        )

                # Clear file list for this thread
                self._temp_files[thread_id] = []
                self.logger.info(
                    f"Cleaned up {removed_count} temporary files (thread: {thread_id})"
                )

    def _cleanup_all_temp_files(self):
        """Clean up all temporary files"""
        with self._temp_files_lock:
            total_removed = 0
            for thread_id in list(self._temp_files.keys()):
                files_to_remove = self._temp_files[thread_id]
                removed_count = 0

                for filepath in files_to_remove:
                    try:
                        if os.path.exists(filepath):
                            os.remove(filepath)
                            removed_count += 1
                    except Exception as e:
                        self.logger.warning(
                            f"Failed to clean up temporary file {filepath}: {e}"
                        )

                total_removed += removed_count
                self._temp_files[thread_id] = []

            if total_removed > 0:
                self.logger.info(f"Cleaned up {total_removed} temporary files")

    def __del__(self):
        """Destructor: clean up all temporary files"""
        try:
            self._cleanup_all_temp_files()
        except Exception as e:
            # Avoid raising exceptions in destructor
            pass

    def apply_defense(self, test_case: TestCase, **kwargs) -> TestCase:
        """Apply defense method - thread-safe version"""

        try:
            default_image_patch_type = self.config["default_image_patch_type"]
            default_text_patch_type = self.config["default_text_patch_type"]

            image_patch_path = self.config[default_image_patch_type]
            text_patch = self.config[default_text_patch_type]

            attack_image_path = test_case.image_path
            attack_prompt = test_case.prompt

            # Apply text patch
            if "optimized" in default_text_patch_type:
                attack_prompt = text_patch + "\n" + attack_prompt
            elif "heuristic" in default_text_patch_type:
                attack_prompt = attack_prompt + "\n" + text_patch

            self.logger.info(f"Applied text safety patch: {default_text_patch_type}")
            self.logger.debug(f"Processed prompt: {attack_prompt}")

            # Apply image patch
            attack_image_with_defense, _ = add_patch_to_image_universal(
                attack_image_path, image_patch_path, "cuda"
            )

            if attack_image_with_defense is None:
                self.logger.error(
                    "Image patch application failed, using original image"
                )
                # Use original image
                defended_case = self.create_defended_case(
                    test_case=test_case,
                    defended_prompt=attack_prompt,
                    defended_image_path=attack_image_path,
                    metadata={"image_patch_failed": True},
                )
                return defended_case

            # Generate unique temporary filename
            unique_filename = self._generate_unique_filename(".png")

            # Ensure output directory exists
            output_dir = self.config.get("output_dir", ".")
            os.makedirs(output_dir, exist_ok=True)

            attack_image_with_defense_path = os.path.join(output_dir, unique_filename)

            # Save temporary file
            try:
                attack_image_with_defense.save(
                    attack_image_with_defense_path, format="PNG", quality=100
                )
                self.logger.debug(
                    f"Saved temporary image: {attack_image_with_defense_path}"
                )

                # Register temporary file for cleanup
                self._register_temp_file(attack_image_with_defense_path)

            except Exception as e:
                self.logger.error(f"Failed to save temporary image: {e}")
                # Use original image when save fails
                defended_case = self.create_defended_case(
                    test_case=test_case,
                    defended_prompt=attack_prompt,
                    defended_image_path=attack_image_path,
                    metadata={"temp_file_save_failed": True, "error": str(e)},
                )
                return defended_case

            # Create test case after applying defense
            defended_case = self.create_defended_case(
                test_case=test_case,
                defended_prompt=attack_prompt,
                defended_image_path=attack_image_with_defense_path,
                metadata={
                    "temp_file": unique_filename,
                    "image_patch_type": default_image_patch_type,
                    "text_patch_type": default_text_patch_type,
                },
            )

            return defended_case

        except Exception as e:
            self.logger.error(f"Failed to apply Uniguard defense: {e}")
            # Return original test case on error
            return self.create_defended_case(
                test_case=test_case,
                defended_prompt=test_case.prompt,
                defended_image_path=test_case.image_path,
                metadata={"defense_failed": True, "error": str(e)},
            )

    def cleanup(self):
        """Manually clean up temporary files for current thread"""
        self._cleanup_thread_temp_files()
