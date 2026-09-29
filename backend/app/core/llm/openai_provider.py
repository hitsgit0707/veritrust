"""OpenAI LLM Provider using the official OpenAI Python SDK."""

import json
from typing import Optional, Type, TypeVar
from pydantic import BaseModel
from backend.app.core.config import get_settings
from backend.app.core.llm.base import BaseLLMProvider

T = TypeVar("T", bound=BaseModel)


class OpenAIProvider(BaseLLMProvider):
    """LLM provider for OpenAI models (GPT-4o, GPT-4o-mini, etc.)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        settings = get_settings()
        self.api_key = api_key or settings.OPENAI_API_KEY
        self.model_name = model_name or settings.OPENAI_MODEL
        self._client = None

    def _get_client(self):
        """Lazily initialize OpenAI AsyncClient."""
        if not self.api_key:
            raise ValueError(
                "OPENAI_API_KEY is not configured. "
                "Please add OPENAI_API_KEY to your .env file or set LLM_PROVIDER=mock."
            )
        if self._client is None:
            try:
                from openai import AsyncOpenAI
                self._client = AsyncOpenAI(api_key=self.api_key)
            except ImportError as e:
                raise ImportError(
                    "openai package is not installed. Run: pip install openai"
                ) from e
        return self._client

    async def generate_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
    ) -> str:
        """Generate unstructured text from OpenAI."""
        client = self._get_client()
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = await client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
        )
        return response.choices[0].message.content or ""

    async def generate_structured(
        self,
        prompt: str,
        response_model: Type[T],
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
    ) -> T:
        """Generate structured Pydantic object from OpenAI using Structured Outputs."""
        client = self._get_client()
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        completion = await client.beta.chat.completions.parse(
            model=self.model_name,
            messages=messages,
            response_format=response_model,
            temperature=temperature,
        )
        parsed = completion.choices[0].message.parsed
        if parsed is None:
            raise ValueError("OpenAI failed to return parsed structured output.")
        return parsed
