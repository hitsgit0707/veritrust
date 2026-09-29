"""Factory module for creating and configuring LLM providers."""

from typing import Optional
from backend.app.core.config import Settings, get_settings
from backend.app.core.llm.base import BaseLLMProvider
from backend.app.core.llm.mock_provider import MockLLMProvider
from backend.app.core.llm.gemini_provider import GeminiProvider
from backend.app.core.llm.openai_provider import OpenAIProvider
from backend.app.core.llm.ollama_provider import OllamaProvider


def get_llm_provider(
    provider_type: Optional[str] = None,
    settings: Optional[Settings] = None,
) -> BaseLLMProvider:
    """Instantiate and return the appropriate LLM provider based on settings or explicit override."""
    app_settings = settings or get_settings()
    selected = (provider_type or app_settings.LLM_PROVIDER).lower().strip()

    if selected == "mock":
        return MockLLMProvider()
    elif selected == "gemini":
        return GeminiProvider(
            api_key=app_settings.GEMINI_API_KEY,
            model_name=app_settings.GEMINI_MODEL,
        )
    elif selected == "openai":
        return OpenAIProvider(
            api_key=app_settings.OPENAI_API_KEY,
            model_name=app_settings.OPENAI_MODEL,
        )
    elif selected == "ollama":
        return OllamaProvider(
            base_url=app_settings.OLLAMA_BASE_URL,
            model_name=app_settings.OLLAMA_MODEL,
        )
    else:
        raise ValueError(
            f"Unsupported LLM provider '{selected}'. "
            f"Allowed options are: 'mock', 'gemini', 'openai', 'ollama'."
        )
