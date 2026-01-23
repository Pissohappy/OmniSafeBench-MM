from typing import List
from .base_model import BaseModel


class GoogleModel(BaseModel):
    """Google model implementation using Google Generative AI API."""

    default_output = "I'm sorry, but I cannot assist with that request."

    # Google-specific content policy keywords
    PROVIDER_SPECIFIC_KEYWORDS = [
        "blocked",
    ]

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
                if self._is_content_policy_rejection(e):
                    return self._handle_content_rejection()
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
            yield self._handle_api_error()
