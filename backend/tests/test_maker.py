"""Unit and integration tests for Stage 4: Maker Agent."""

import os
import shutil
import tempfile
import pytest
from httpx import ASGITransport, AsyncClient

from backend.app.agents.maker import MakerAgent, MakerResult
from backend.app.core.llm.mock_provider import MockLLMProvider
from backend.app.main import app
from backend.app.rag.service import RAGService
from backend.app.rag.vector_store import ChromaVectorStore, RetrievedChunk
from backend.app.schemas.compliance import MakerOutput


@pytest.fixture
def populated_rag_service():
    """Create an isolated temporary ChromaDB populated with sample_data."""
    temp_dir = tempfile.mkdtemp(prefix="veritrust_test_maker_")
    store = ChromaVectorStore(persist_dir=temp_dir, collection_name="test_maker_kb")
    service = RAGService(vector_store=store)

    sample_dir = os.path.join(os.path.dirname(__file__), "..", "..", "sample_data")
    service.ingest_directory(sample_dir)

    yield service
    shutil.rmtree(temp_dir, ignore_errors=True)


# ==============================================================================
# 1. Grounded Drafting & Evidence Retrieval Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_maker_grounded_answer(populated_rag_service):
    """Verify Maker drafts a response directly grounded in retrieved knowledge."""
    mock_llm = MockLLMProvider()
    maker = MakerAgent(llm_provider=mock_llm, rag_service=populated_rag_service)

    result = await maker.generate_draft("What is the warranty period for hardware?")

    assert isinstance(result, MakerResult)
    assert len(result.retrieved_evidence) > 0
    assert "1-year" in result.draft_answer.lower()
    assert 0.0 <= result.confidence <= 1.0


@pytest.mark.asyncio
async def test_maker_sources_correspond_to_retrieved_evidence(populated_rag_service):
    """Verify cited sources correspond to actual chunks retrieved."""
    mock_llm = MockLLMProvider()
    maker = MakerAgent(llm_provider=mock_llm, rag_service=populated_rag_service)

    result = await maker.generate_draft("What is your refund policy window?")

    assert len(result.sources) > 0
    valid_sources = {c.source for c in result.retrieved_evidence} | {c.chunk_id for c in result.retrieved_evidence}

    for source in result.sources:
        assert source in valid_sources, f"Source '{source}' not in retrieved chunks!"


@pytest.mark.asyncio
async def test_maker_source_filtering_rejects_hallucinated_sources():
    """Verify MakerAgent filters out sources invented by the LLM that were not retrieved."""
    mock_llm = MockLLMProvider()
    # Mock output that hallucinates a fabricated source
    mock_llm.set_structured_response(
        MakerOutput(
            draft_answer="Refunds are 30 days.",
            sources=["invented_fake_source.txt#chunk_99", "legit_source.txt#chunk_0"],
            confidence=0.9,
        )
    )

    class DummyRAGService:
        def retrieve(self, query, top_k=4):
            return [
                RetrievedChunk(
                    chunk_id="doc_1#chunk_0",
                    text="Refunds are 30 days.",
                    document_id="doc_1",
                    filename="legit_source.txt",
                    source="legit_source.txt#chunk_0",
                    similarity_score=0.9,
                    distance=0.1,
                )
            ]

    maker = MakerAgent(llm_provider=mock_llm, rag_service=DummyRAGService())
    result = await maker.generate_draft("What is refund window?")

    assert "legit_source.txt#chunk_0" in result.sources
    assert "invented_fake_source.txt#chunk_99" not in result.sources


# ==============================================================================
# 2. Insufficient Evidence Handling
# ==============================================================================

@pytest.mark.asyncio
async def test_maker_cautious_on_insufficient_evidence():
    """Verify Maker produces a cautious response when no evidence exists rather than fabricating."""
    mock_llm = MockLLMProvider()

    class EmptyRAGService:
        def retrieve(self, query, top_k=4):
            return []

    maker = MakerAgent(llm_provider=mock_llm, rag_service=EmptyRAGService())
    result = await maker.generate_draft("Do you accept payment in gold bullion or bitcoin?")

    assert len(result.retrieved_evidence) == 0
    assert len(result.sources) == 0
    assert "unable to answer" in result.draft_answer.lower() or "could not be verified" in result.draft_answer.lower()
    assert result.confidence <= 0.40


# ==============================================================================
# 3. Adversarial / Overconfident Wrong Outputs (For Stage 5 Judge rejection)
# ==============================================================================

@pytest.mark.asyncio
async def test_maker_can_produce_wrong_high_confidence_for_judge_test(populated_rag_service):
    """Verify Maker can return an overconfident wrong answer for later Stage 5 Judge interception."""
    mock_llm = MockLLMProvider()
    maker = MakerAgent(llm_provider=mock_llm, rag_service=populated_rag_service)

    # Trigger Adversarial Case 5
    result = await maker.generate_draft(
        question="Can I get same-day delivery?",
        adversarial_mode="case 5",
    )

    assert result.confidence == 0.99
    assert "same-day" in result.draft_answer.lower()
    # Maker output should preserve this draft so Judge in Stage 5 will reject it despite the 0.99 confidence!


# ==============================================================================
# 4. API Endpoint Integration
# ==============================================================================

@pytest.mark.asyncio
async def test_maker_api_draft_endpoint():
    """Verify POST /api/v1/maker/draft generates structured output via HTTP."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        payload = {"question": "What is the warranty coverage period?", "top_k": 2}
        response = await client.post("/api/v1/maker/draft", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert data["question"] == "What is the warranty coverage period?"
        assert "draft_answer" in data
        assert isinstance(data["sources"], list)
        assert 0.0 <= data["confidence"] <= 1.0
        assert "retrieved_evidence" in data
