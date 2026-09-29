"""Application configuration module using Pydantic Settings."""

from functools import lru_cache
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Application settings
    APP_NAME: str = "VeriTrust AI"
    APP_ENV: str = "development"
    API_PORT: int = 8000

    # LLM Provider selection: 'mock', 'gemini', 'openai', 'ollama'
    LLM_PROVIDER: str = "mock"

    # Google Gemini Settings (Optional for mock/ollama)
    GEMINI_API_KEY: Optional[str] = None
    GEMINI_MODEL: str = "gemini-2.5-flash"

    # OpenAI Settings (Optional for mock/ollama)
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-4o-mini"

    # Ollama Settings (Optional for mock/cloud)
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.2:3b"

    # RAG & Knowledge Base Settings
    CHROMA_PERSIST_DIR: str = "./data/chroma_db"
    CHROMA_COLLECTION_NAME: str = "company_knowledge_base"
    # Configurable reference similarity threshold (initial test value, not a hard barrier)
    SIMILARITY_THRESHOLD: float = 0.7

    # Guardrail Workflow Settings
    MAX_CORRECTION_ATTEMPTS: int = 3
    JUDGE_SCORE_THRESHOLD: int = 80
    SAFE_FALLBACK_RESPONSE: str = (
        "We are currently unable to verify the policy details required to answer your request with full certainty. "
        "A customer service representative has been notified to assist you."
    )

    # Database
    DATABASE_URL: str = "sqlite+aiosqlite:///./data/veritrust.db"


@lru_cache
def get_settings() -> Settings:
    """Return a cached singleton instance of the application settings."""
    return Settings()
