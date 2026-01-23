from typing import List, Optional
from .base_model import BaseModel


class OpenAIModel(BaseModel):
    """OpenAI model implementation using OpenAI API."""

    # OpenAI-specific content policy keywords
    PROVIDER_SPECIFIC_KEYWORDS = [
        "invalid",
        "inappropriate",
        "invalid_prompt",
        "limited access",
    ]

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
                print("Error during API call:", str(e).lower())
                if self._is_content_policy_rejection(e):
                    print("Content rejection triggered")
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
            try:
                stream = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    stream=True,
                    **api_kwargs,
                )
                return stream
            except Exception as e:
                if self._is_content_policy_rejection(e):
                    return self._handle_content_rejection_stream()
                raise

        try:
            stream = self._retry_with_backoff(_api_call)
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
        except Exception:
            yield self._handle_api_error()
