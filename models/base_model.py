from abc import ABC, abstractmethod
from typing import List, Optional, Union, Dict, Any
import backoff
from core.base_classes import BaseModel as CoreBaseModel
from core.data_formats import TestCase, ModelResponse


class BaseModel(CoreBaseModel):
    """Base class for all model implementations."""

    API_RETRY_SLEEP = 10
    API_ERROR_OUTPUT = "$ERROR$"
    API_CONTENT_REJECTION_OUTPUT = (
        "[ERROR] Prompt detected as harmful content, refusing to answer"
    )
    API_QUERY_SLEEP = 0.5
    API_MAX_RETRY = 3
    API_TIMEOUT = 600

    def __init__(self, model_name: str, api_key: str = None, base_url: str = None):
        # Call parent class __init__, pass empty configuration
        super().__init__(config={})
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url
        self.model_type = self._determine_model_type()

    def _determine_model_type(self):
        """Determine model type: api (API call) or local (local loading)"""
        # Default implementation: determine based on whether api_key exists
        # Subclasses can override this method
        if self.api_key or self.base_url:
            return "api"
        else:
            return "local"

    def _retry_with_backoff(self, func, *args, **kwargs):
        """Execute function with retry logic and exponential backoff using backoff library."""

        @backoff.on_exception(
            backoff.expo,
            Exception,
            max_tries=self.API_MAX_RETRY,
            max_time=self.API_TIMEOUT * 2,
            on_backoff=lambda details: print(
                f"Attempt {details['tries']} failed: {details['exception'].__class__.__name__}: {details['exception']}"
            ),
            on_giveup=lambda details: print(
                f"Final attempt failed after {details['tries']} tries: {details['exception'].__class__.__name__}: {details['exception']}"
            ),
        )
        def _execute():
            return func(*args, **kwargs)

        try:
            return _execute()
        except Exception as e:
            raise

    @abstractmethod
    def _generate_single(
        self,
        messages: List[dict],
        **kwargs,
    ) -> str:
        """Generate response for a single prompt."""
        pass

    @abstractmethod
    def _generate_stream(
        self,
        messages: List[dict],
        **kwargs,
    ):
        """Generate streaming response for a single prompt."""
        pass

    def generate_response(self, test_case: TestCase, **kwargs) -> ModelResponse:
        """
        Generate model response

        Args:
            test_case: Test case
            **kwargs: Other parameters

        Returns:
            ModelResponse: Model response
        """
        # Convert TestCase to message list
        messages = self._test_case_to_messages(test_case)

        # Generate response
        response = self.generate(messages, **kwargs)

        response_text = response.choices[0].message.content

        return ModelResponse(
            test_case_id=test_case.test_case_id,
            model_response=response_text,
            model_name=self.model_name,
            metadata=test_case.metadata,
        )

    def generate_responses_batch(
        self, test_cases: List[TestCase], **kwargs
    ) -> List[ModelResponse]:
        """
        Batch generate model responses (for locally loaded models)

        Args:
            test_cases: List of test cases
            **kwargs: Other parameters

        Returns:
            List[ModelResponse]: List of model responses
        """
        # Default implementation: loop calling single generation
        # Local models should override this method to implement true batch inference
        responses = []
        for test_case in test_cases:
            response = self.generate_response(test_case, **kwargs)
            responses.append(response)
        return responses

    def _test_case_to_messages(self, test_case: TestCase) -> List[Dict[str, Any]]:
        """Convert TestCase to message list"""
        messages = []

        # If there is an image, add image message
        if test_case.image_path:
            try:
                # Check if image file exists
                import os
                from pathlib import Path

                image_path = Path(test_case.image_path)

                # Load image and encode as base64
                from PIL import Image
                import base64
                from io import BytesIO

                # Open image
                image = Image.open(image_path)

                # Encode image as base64
                buffered = BytesIO()
                image.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

                # Create data URL
                data_url = f"data:image/png;base64,{img_str}"

                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": test_case.prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": data_url},
                            },
                        ],
                    }
                )

            except Exception as e:
                self.logger.error(f"Error processing image: {e}")
                # Fallback to plain text on error
                messages.append({"role": "user", "content": test_case.prompt})
        else:
            # Plain text message
            messages.append({"role": "user", "content": test_case.prompt})

        return messages

    def generate(
        self,
        messages: Union[List[dict], List[List[dict]]],
        use_tqdm: bool = False,
        stream: bool = False,
        **kwargs,
    ) -> Union[str, List[str]]:
        """Generate responses for multiple prompts.

        Args:
            messages: Single message list or list of message lists
            use_tqdm: Whether to show progress bar
            stream: Whether to use streaming output
            **kwargs: Additional model-specific parameters

        Returns:
            Single response string or list of generated responses
        """
        if isinstance(messages, list) and all(
            isinstance(msg, dict) for msg in messages
        ):
            if stream:
                return self._generate_stream(messages, **kwargs)
            else:
                return self._generate_single(messages, **kwargs)

        if use_tqdm:
            from tqdm import tqdm

            messages = tqdm(messages)

        if stream:
            # For multiple messages with streaming, return a list of generators
            return [self._generate_stream(msg_set, **kwargs) for msg_set in messages]
        else:
            return [self._generate_single(msg_set, **kwargs) for msg_set in messages]
