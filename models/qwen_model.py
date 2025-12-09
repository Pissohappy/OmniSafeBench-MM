from typing import List, Optional
from .openai_model import OpenAIModel


class QwenModel(OpenAIModel):
    """Qwen model implementation using OpenAI-compatible API."""

    def __init__(self, model_name: str, api_key: str, base_url: Optional[str] = None):
        # Qwen uses OpenAI-compatible API
        super().__init__(model_name, api_key, base_url)
