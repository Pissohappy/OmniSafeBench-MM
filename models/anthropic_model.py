from typing import List, Union
from .base_model import BaseModel


class AnthropicModel(BaseModel):
    """Anthropic model implementation using Anthropic API."""

    default_output = "I'm sorry, but I cannot assist with that request."

    # Anthropic-specific content policy keywords
    PROVIDER_SPECIFIC_KEYWORDS = [
        "output blocked by content filtering policy",
    ]

    def __init__(self, model_name: str, api_key: str) -> None:
        super().__init__(model_name, api_key)

        try:
            from anthropic import Anthropic
        except ImportError:
            raise ImportError(
                "Anthropic package is required for Claude models. "
                "Install it with: pip install anthropic"
            )

        self.client = Anthropic(api_key=api_key)

    def _generate_single(
        self,
        messages: Union[List[dict], List[List[dict]]],
        **kwargs,
    ) -> str:
        """Generate response for a single prompt."""

        def _api_call():
            try:
                # Create a copy of kwargs to avoid modifying the original
                api_kwargs = kwargs.copy()
                # Use provided model name or fall back to instance model_name
                model = api_kwargs.pop("model", self.model_name)

                # Convert messages to Claude format
                claude_messages = []
                for msg in messages:
                    if msg["role"] == "system":
                        # System messages are handled separately in Claude
                        continue
                    elif msg["role"] in ["user", "assistant"]:
                        claude_messages.append(
                            {"role": msg["role"], "content": msg["content"]}
                        )

                # Use the newer messages API for Claude 3 models
                response = self.client.messages.create(
                    model=model,
                    messages=claude_messages,
                    **api_kwargs,
                )
                return response
            except Exception as e:
                if self._is_content_policy_rejection(e):
                    return self._handle_content_rejection()
                raise

        return self._retry_with_backoff(_api_call)

    def _generate_stream(
        self,
        messages: Union[List[dict], List[List[dict]]],
        **kwargs,
    ):
        """Generate streaming response for a single prompt."""

        def _api_call():
            try:
                # Create a copy of kwargs to avoid modifying the original
                api_kwargs = kwargs.copy()
                # Use provided model name or fall back to instance model_name
                model = api_kwargs.pop("model", self.model_name)

                # Convert messages to Claude format
                claude_messages = []
                for msg in messages:
                    if msg["role"] == "system":
                        # System messages are handled separately in Claude
                        continue
                    elif msg["role"] in ["user", "assistant"]:
                        claude_messages.append(
                            {"role": msg["role"], "content": msg["content"]}
                        )

                # Use the newer messages API for Claude 3 models with streaming
                stream = self.client.messages.create(
                    model=model,
                    messages=claude_messages,
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
                if hasattr(chunk, "delta") and chunk.delta:
                    if hasattr(chunk.delta, "text") and chunk.delta.text:
                        yield chunk.delta.text
                elif hasattr(chunk, "completion") and chunk.completion:
                    yield chunk.completion
        except Exception:
            yield self._handle_api_error()
