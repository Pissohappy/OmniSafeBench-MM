from typing import List, Optional
from .base_model import BaseModel


class OpenAIModel(BaseModel):
    """OpenAI model implementation using OpenAI API."""

    def __init__(self, model_name: str, api_key: str, base_url: Optional[str] = None):
        super().__init__(model_name=model_name, api_key=api_key, base_url=base_url)

        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "OpenAI package is required for GPT models. "
                "Install it with: pip install openai"
            )

        self.client = OpenAI(
            api_key=api_key, timeout=self.API_TIMEOUT, base_url=base_url
        )

    def _generate_single(
        self,
        messages: List[dict],
        **kwargs,
    ) -> str:
        """Generate response for a single prompt."""

        def _api_call():
            # Create a copy of kwargs to avoid modifying the original
            api_kwargs = kwargs.copy()
            # Use provided model name or fall back to instance model_name
            model = api_kwargs.pop("model", self.model_name)
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    **api_kwargs,
                )
                return response

            except Exception as e:
                # Check for content policy violations in GPT models
                error_str = str(e).lower()
                print("Error during API call:", error_str)
                content_keywords = [
                    "content policy",
                    "invalid",
                    "safety",
                    "harmful",
                    "unsafe",
                    "violation",
                    "moderation",
                    "data_inspection_failed",
                    "inappropriate",
                    "invalid_prompt",
                    "limited access",
                ]
                if any(keyword in error_str for keyword in content_keywords):
                    print("✓ Content rejection triggered")
                    return self.API_CONTENT_REJECTION_OUTPUT
                print("✗ No content keywords matched, raising exception")
                raise e

        return self._retry_with_backoff(_api_call)

    def _generate_stream(
        self,
        messages: List[dict],
        **kwargs,
    ):
        """Generate streaming response for a single prompt."""

        def _api_call():
            # Create a copy of kwargs to avoid modifying the original
            api_kwargs = kwargs.copy()
            # Use provided model name or fall back to instance model_name
            model = api_kwargs.pop("model", self.model_name)
            try:
                stream = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    stream=True,
                    **api_kwargs,
                )
                return stream
            except Exception as e:
                # Check for content policy violations in GPT models
                error_str = str(e).lower()
                content_keywords = [
                    "content policy",
                    "safety",
                    "harmful",
                    "unsafe",
                    "violation",
                    "moderation",
                    "data_inspection_failed",
                    "inappropriate content",
                ]
                if any(keyword in error_str for keyword in content_keywords):
                    # Return a generator that yields the content rejection placeholder
                    def error_generator():
                        yield self.API_CONTENT_REJECTION_OUTPUT

                    return error_generator()
                # Handle BadRequestError specifically
                if (
                    "badrequesterror" in error_str
                    and "data_inspection_failed" in error_str
                ):

                    def error_generator():
                        yield self.API_CONTENT_REJECTION_OUTPUT

                    return error_generator()
                raise

        try:
            stream = self._retry_with_backoff(_api_call)
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
        except Exception:
            yield self.API_ERROR_OUTPUT
