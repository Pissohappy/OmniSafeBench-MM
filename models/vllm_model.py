from typing import List, Optional
from .openai_model import OpenAIModel


class VLLMModel(OpenAIModel):
    """vLLM model implementation using OpenAI-compatible API."""

    def __init__(self, model_name: str, api_key: str, base_url: Optional[str] = None):
        # vLLM uses OpenAI-compatible API
        super().__init__(model_name, api_key, base_url)
