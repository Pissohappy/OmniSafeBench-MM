from typing import List
from .base_model import BaseModel


class DoubaoModel(BaseModel):
    """Doubao model implementation using ByteDance API."""

    def __init__(self, model_name: str, api_key: str):
        super().__init__(model_name, api_key)
        try:
            from volcenginesdkarkruntime import Ark
        except ImportError:
            raise ImportError(
                "ByteDance package is required for ByteDance models. "
                "Install it with: pip install volcenginesdkarkruntime"
            )
        self.client = Ark(api_key=api_key)
        self.model_name = model_name

    def _generate_single(self, messages: List[dict], **kwargs):
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
                # Check for content policy violations in ByteDance models
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
                    return self.API_CONTENT_REJECTION_OUTPUT
                raise

        return self._retry_with_backoff(_api_call)

    def _generate_stream(self, messages: List[dict], **kwargs):
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
                # Check for content policy violations in ByteDance models
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
