"""Unit tests for Stage 1: LLM Provider Abstraction, Selection, and Mock Provider."""

import pytest
from typing import List, Optional
from pydantic import BaseModel

from backend.app.core.config import Settings
from backend.app.core.llm import (
    BaseLLMProvider,
    MockLLMProvider,
    GeminiProvider,
    OpenAIProvider,
    OllamaProvider,
    get_llm_provider,
)


# Test Pydantic schemas mimicking agents
class MockMakerOutput(BaseModel):
    draft_answer: str
    sources: List[str]
    confidence: float


class MockJudgeIssue(BaseModel):
    issue_type: str
    description: str
    severity: str
    claim_text: str
    evidence_ref: str


class MockJudgeOutput(BaseModel):
    approved: bool
    score: int
    issues: List[MockJudgeIssue]


class MockCorrectionOutput(BaseModel):
    corrected_answer: str
    explanation: str


# ==============================================================================
# 1. Provider Selection / Factory Tests
# ==============================================================================

def test_provider_factory_mock():
    """Verify factory returns MockLLMProvider for 'mock'."""
    provider = get_llm_provider("mock")
    assert isinstance(provider, MockLLMProvider)


def test_provider_factory_gemini():
    """Verify factory returns GeminiProvider for 'gemini'."""
    provider = get_llm_provider("gemini")
    assert isinstance(provider, GeminiProvider)
    assert provider.model_name == "gemini-2.5-flash"


def test_provider_factory_openai():
    """Verify factory returns OpenAIProvider for 'openai'."""
    provider = get_llm_provider("openai")
    assert isinstance(provider, OpenAIProvider)
    assert provider.model_name == "gpt-4o-mini"


def test_provider_factory_ollama():
    """Verify factory returns OllamaProvider for 'ollama'."""
    provider = get_llm_provider("ollama")
    assert isinstance(provider, OllamaProvider)
    assert provider.model_name == "llama3.2:3b"


def test_provider_factory_invalid():
    """Verify factory raises ValueError for invalid provider names."""
    with pytest.raises(ValueError, match="Unsupported LLM provider 'unknown_provider'"):
        get_llm_provider("unknown_provider")


def test_provider_factory_respects_settings():
    """Verify factory respects Settings object passed to it."""
    custom_settings = Settings(LLM_PROVIDER="mock")
    provider = get_llm_provider(settings=custom_settings)
    assert isinstance(provider, MockLLMProvider)


# ==============================================================================
# 2. MockLLMProvider Deterministic Behavior Tests (Zero API Key / Offline)
# ==============================================================================

@pytest.mark.asyncio
async def test_mock_provider_text_generation():
    """Verify text generation returns expected deterministic text and records history."""
    provider = MockLLMProvider()
    res1 = await provider.generate_text("What is your refund policy?")
    assert "refund" in res1.lower()

    res2 = await provider.generate_text("General inquiry without keywords")
    assert "deterministic mock response" in res2

    assert len(provider.call_history) == 2


def test_mock_provider_sync_wrapper():
    """Verify synchronous wrapper works correctly."""
    provider = MockLLMProvider()
    res = provider.generate_text_sync("Tell me about warranty")
    assert "warranty" in res.lower()


@pytest.mark.asyncio
async def test_mock_maker_structured_output():
    """Verify Mock provider returns valid structured output for Maker Agent."""
    provider = MockLLMProvider()
    result = await provider.generate_structured(
        prompt="Draft an answer for: Can I return after 45 days?",
        response_model=MockMakerOutput,
    )
    assert isinstance(result, MockMakerOutput)
    assert "45 days" in result.draft_answer
    assert len(result.sources) > 0
    assert 0.0 <= result.confidence <= 1.0


@pytest.mark.asyncio
async def test_mock_judge_adversarial_cases():
    """Verify Mock provider implements the required adversarial judge cases deterministically."""
    provider = MockLLMProvider()

    # Case 1: Accurate Answer -> Must APPROVE
    res_case1 = await provider.generate_structured(
        prompt="Judge this answer: Refunds are available within 30 days of purchase.",
        response_model=MockJudgeOutput,
    )
    assert res_case1.approved is True
    assert res_case1.score >= 80
    assert len(res_case1.issues) == 0

    # Case 2: Contradiction (45 days vs 30 days) -> Must REJECT
    res_case2 = await provider.generate_structured(
        prompt="Judge this answer: Refunds are processed within 45 days of purchase.",
        response_model=MockJudgeOutput,
    )
    assert res_case2.approved is False
    assert res_case2.score < 80
    assert any(i.issue_type == "CONTRADICTION" for i in res_case2.issues)

    # Case 3: Fabricated Policy (Premium free replacement) -> Must REJECT
    res_case3 = await provider.generate_structured(
        prompt="Judge this answer: Premium users receive free replacement under VIP care.",
        response_model=MockJudgeOutput,
    )
    assert res_case3.approved is False
    assert res_case3.score < 80
    assert any(i.issue_type == "FABRICATED_POLICY" for i in res_case3.issues)

    # Case 4: Numerical / Period Drift (2 years vs 1 year) -> Must REJECT
    res_case4 = await provider.generate_structured(
        prompt="Judge this answer: Devices are covered by a 2-year warranty.",
        response_model=MockJudgeOutput,
    )
    assert res_case4.approved is False
    assert res_case4.score < 80
    assert any(i.issue_type == "UNSUPPORTED_NUMERICAL" for i in res_case4.issues)

    # Case 5: Confidence Decoupling (Same-day claims with wrong facts) -> Must REJECT
    res_case5 = await provider.generate_structured(
        prompt="Judge this answer: All items are delivered same-day anywhere in the country.",
        response_model=MockJudgeOutput,
    )
    assert res_case5.approved is False
    assert any(i.issue_type == "CONTRADICTION" for i in res_case5.issues)

    # Case 6: Three failed attempts / unresolvable claims -> Must REJECT
    res_case6 = await provider.generate_structured(
        prompt="Judge this answer: Unresolvable claim after retry_count: 3",
        response_model=MockJudgeOutput,
    )
    assert res_case6.approved is False
    assert any(i.issue_type == "MISSING_EVIDENCE" for i in res_case6.issues)


@pytest.mark.asyncio
async def test_mock_corrector_structured_output():
    """Verify Mock provider returns valid structured output for Correction Agent."""
    provider = MockLLMProvider()
    result = await provider.generate_structured(
        prompt="Correct this rejected answer: refund within 45 days",
        response_model=MockCorrectionOutput,
    )
    assert isinstance(result, MockCorrectionOutput)
    assert "30 days" in result.corrected_answer
    assert "45 days" in result.explanation


@pytest.mark.asyncio
async def test_mock_overrides():
    """Verify explicit mock response overrides work as expected."""
    provider = MockLLMProvider()

    # Text override
    provider.set_mock_response("Custom injected text")
    text_res = await provider.generate_text("Any query")
    assert text_res == "Custom injected text"

    # Structured override
    custom_obj = MockMakerOutput(draft_answer="Custom draft", sources=["s1"], confidence=0.77)
    provider.set_structured_response(custom_obj)
    struct_res = await provider.generate_structured("Any prompt", MockMakerOutput)
    assert struct_res.draft_answer == "Custom draft"
    assert struct_res.confidence == 0.77

    # Clear overrides
    provider.clear_overrides()
    text_res_after = await provider.generate_text("Any query")
    assert text_res_after == "This is a deterministic mock response from VeriTrust AI."


# ==============================================================================
# 3. Optional External API Key Validation Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_gemini_missing_api_key_raises_informative_error():
    """Verify GeminiProvider raises clear ValueError if called without API key."""
    provider = GeminiProvider(api_key=None)
    with pytest.raises(ValueError, match="GEMINI_API_KEY is not configured"):
        await provider.generate_text("Hello Gemini")


@pytest.mark.asyncio
async def test_openai_missing_api_key_raises_informative_error():
    """Verify OpenAIProvider raises clear ValueError if called without API key."""
    provider = OpenAIProvider(api_key=None)
    with pytest.raises(ValueError, match="OPENAI_API_KEY is not configured"):
        await provider.generate_text("Hello OpenAI")
