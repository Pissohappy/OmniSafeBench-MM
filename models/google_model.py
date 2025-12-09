from typing import List
from .base_model import BaseModel


class GoogleModel(BaseModel):
    """Google model implementation using Google Generative AI API."""

    default_output = "I'm sorry, but I cannot assist with that request."

    def __init__(self, model_name: str, api_key: str, base_url: str = None) -> None:
        super().__init__(model_name=model_name, api_key=api_key, base_url=base_url)

        try:
            from google import genai
        except ImportError:
            raise ImportError(
                "Google Generative AI package is required for Gemini models. "
                "Install it with: pip install google-generativeai"
            )
        self.model = genai.Client(api_key=api_key)

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
                response = self.model.models.generate_content(
                    model=model, contents=messages, **api_kwargs
                )
                return response
            except Exception as e:
                # Check for content policy violations in Gemini models
                error_str = str(e).lower()
                content_keywords = [
                    "content policy",
                    "safety",
                    "harmful",
                    "unsafe",
                    "violation",
                    "moderation",
                    "blocked",
                    "data_inspection_failed",
                    "inappropriate content",
                ]
                if any(keyword in error_str for keyword in content_keywords):
                    return self.API_CONTENT_REJECTION_OUTPUT
                # Handle BadRequestError specifically
                if (
                    "badrequesterror" in error_str
                    and "data_inspection_failed" in error_str
                ):
                    return self.API_CONTENT_REJECTION_OUTPUT
                raise

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
            # Gemini streaming API
            response = self.model.models.generate_content(
                model=model, contents=messages, stream=True, **api_kwargs
            )
            return response

        try:
            response = self._retry_with_backoff(_api_call)
            for chunk in response:
                if chunk.text:
                    yield chunk.text
        except Exception:
            yield self.API_ERROR_OUTPUT
