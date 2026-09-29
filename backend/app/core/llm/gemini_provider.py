"""Google Gemini LLM Provider using the official Google GenAI SDK."""

import json
from typing import Optional, Type, TypeVar
from pydantic import BaseModel
from backend.app.core.config import get_settings
from backend.app.core.llm.base import BaseLLMProvider

T = TypeVar("T", bound=BaseModel)


class GeminiProvider(BaseLLMProvider):
    """LLM provider for Google Gemini models."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        settings = get_settings()
        self.api_key = api_key or settings.GEMINI_API_KEY
        self.model_name = model_name or settings.GEMINI_MODEL
        self._client = None

    def _get_client(self):
        """Lazily initialize Google GenAI client to avoid failing import if key is missing."""
        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY is not configured. "
                "Please add GEMINI_API_KEY to your .env file or set LLM_PROVIDER=mock."
            )
        if self._client is None:
            try:
                from google import genai
                self._client = genai.Client(api_key=self.api_key)
            except ImportError as e:
                raise ImportError(
                    "google-genai package is not installed. Run: pip install google-genai"
                ) from e
        return self._client

    async def generate_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
    ) -> str:
        """Generate unstructured text from Gemini."""
        client = self._get_client()
        from google.genai import types

        config = types.GenerateContentConfig(
            temperature=temperature,
            system_instruction=system_prompt if system_prompt else None,
        )

        response = await client.aio.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=config,
        )
        return response.text or ""

    async def generate_structured(
        self,
        prompt: str,
        response_model: Type[T],
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
    ) -> T:
        """Generate structured Pydantic object from Gemini with schema validation."""
        client = self._get_client()
        from google.genai import types

        config = types.GenerateContentConfig(
            temperature=temperature,
            system_instruction=system_prompt if system_prompt else None,
            response_mime_type="application/json",
            response_schema=response_model,
        )

        response = await client.aio.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=config,
        )

        text = response.text or "{}"
        return response_model.model_validate_json(text)
