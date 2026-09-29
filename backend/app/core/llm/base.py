"""Base abstract class for LLM providers."""

import asyncio
from abc import ABC, abstractmethod
from typing import Optional, Type, TypeVar
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class BaseLLMProvider(ABC):
    """Abstract base class defining the interface for all LLM providers."""

    @abstractmethod
    async def generate_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
    ) -> str:
        """Generate unstructured text response from the LLM."""
        pass

    @abstractmethod
    async def generate_structured(
        self,
        prompt: str,
        response_model: Type[T],
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
    ) -> T:
        """Generate a strictly typed, structured Pydantic object from the LLM."""
        pass

    def generate_text_sync(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
    ) -> str:
        """Synchronous wrapper for generate_text."""
        return asyncio.run(
            self.generate_text(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
            )
        )

    def generate_structured_sync(
        self,
        prompt: str,
        response_model: Type[T],
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
    ) -> T:
        """Synchronous wrapper for generate_structured."""
        return asyncio.run(
            self.generate_structured(
                prompt=prompt,
                response_model=response_model,
                system_prompt=system_prompt,
                temperature=temperature,
            )
        )
