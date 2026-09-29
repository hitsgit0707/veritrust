"""Local Ollama LLM Provider using HTTP API."""

import json
from typing import Optional, Type, TypeVar
import httpx
from pydantic import BaseModel
from backend.app.core.config import get_settings
from backend.app.core.llm.base import BaseLLMProvider

T = TypeVar("T", bound=BaseModel)


class OllamaProvider(BaseLLMProvider):
    """LLM provider for locally hosted Ollama instances."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        settings = get_settings()
        self.base_url = (base_url or settings.OLLAMA_BASE_URL).rstrip("/")
        self.model_name = model_name or settings.OLLAMA_MODEL

    async def generate_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
    ) -> str:
        """Generate unstructured text via Ollama /api/chat."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                response = await client.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
                return data.get("message", {}).get("content", "")
            except httpx.ConnectError as e:
                raise ConnectionError(
                    f"Could not connect to Ollama at {self.base_url}. "
                    "Make sure Ollama is running locally, or switch LLM_PROVIDER=mock."
                ) from e

    async def generate_structured(
        self,
        prompt: str,
        response_model: Type[T],
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
    ) -> T:
        """Generate structured Pydantic object via Ollama JSON format."""
        schema_json = json.dumps(response_model.model_json_schema())
        system_instruction = (
            f"{system_prompt or ''}\n"
            f"You MUST respond ONLY with valid JSON conforming to this JSON Schema:\n{schema_json}"
        ).strip()

        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": prompt},
        ]

        payload = {
            "model": self.model_name,
            "messages": messages,
            "format": "json",
            "stream": False,
            "options": {"temperature": temperature},
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                response = await client.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
                content = data.get("message", {}).get("content", "{}")
                return response_model.model_validate_json(content)
            except httpx.ConnectError as e:
                raise ConnectionError(
                    f"Could not connect to Ollama at {self.base_url}. "
                    "Make sure Ollama is running locally, or switch LLM_PROVIDER=mock."
                ) from e
