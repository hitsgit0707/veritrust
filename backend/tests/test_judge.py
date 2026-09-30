"""Unit and integration tests for Stage 5: Judge Agent."""

import os
import shutil
import tempfile
import pytest
from httpx import ASGITransport, AsyncClient

from backend.app.agents.judge import JudgeAgent, JudgeResult
from backend.app.agents.maker import MakerAgent, MakerResult
from backend.app.core.llm.mock_provider import MockLLMProvider
from backend.app.main import app
from backend.app.rag.service import RAGService
from backend.app.rag.vector_store import ChromaVectorStore, RetrievedChunk
from backend.app.schemas.compliance import IssueItem, JudgeOutput


@pytest.fixture
def populated_rag_service():
    """Create an isolated temporary ChromaDB populated with sample_data."""
    temp_dir = tempfile.mkdtemp(prefix="veritrust_test_judge_")
    store = ChromaVectorStore(persist_dir=temp_dir, collection_name="test_judge_kb")
    service = RAGService(vector_store=store)

    sample_dir = os.path.join(os.path.dirname(__file__), "..", "..", "sample_data")
    service.ingest_directory(sample_dir)

    yield service
    shutil.rmtree(temp_dir, ignore_errors=True)


# ==============================================================================
# 1. Grounded Approval Test
# ==============================================================================

@pytest.mark.asyncio
async def test_judge_approves_grounded_answer(populated_rag_service):
    """Verify Judge approves an accurate draft fully supported by knowledge-base evidence."""
    mock_llm = MockLLMProvider()
    judge = JudgeAgent(llm_provider=mock_llm)

    question = "What is the warranty period for hardware?"
    chunks = populated_rag_service.retrieve(query=question, top_k=2)

    draft_answer = (
        "According to our company warranty policy, all hardware products carry a standard "
        "1-year limited warranty from the confirmed date of purchase."
    )
    sources = [chunks[0].source]

    result = await judge.evaluate(
        question=question,
        draft_answer=draft_answer,
        retrieved_evidence=chunks,
        sources=sources,
        confidence=0.95,
    )

    assert isinstance(result, JudgeResult)
    assert result.approved is True
    assert result.score >= 80
    assert len(result.issues) == 0


# ==============================================================================
# 2. Contradiction Rejection Test
# ==============================================================================

@pytest.mark.asyncio
async def test_judge_rejects_contradiction(populated_rag_service):
    """Verify Judge detects factual contradictions between draft and knowledge base."""
    mock_llm = MockLLMProvider()
    judge = JudgeAgent(llm_provider=mock_llm)

    question = "What is your return policy timeframe?"
    chunks = populated_rag_service.retrieve(query="refund policy", top_k=2)

    # Draft contradicts verified 30-day policy
    draft_answer = "Refunds are accepted within 45 days with original receipt."
    sources = [chunks[0].source]

    result = await judge.evaluate(
        question=question,
        draft_answer=draft_answer,
        retrieved_evidence=chunks,
        sources=sources,
        confidence=0.88,
    )

    assert result.approved is False
    assert result.score < 80
    assert any(i.issue_type == "CONTRADICTION" for i in result.issues)


# ==============================================================================
# 3. Fabricated Policy Rejection Test
# ==============================================================================

@pytest.mark.asyncio
async def test_judge_rejects_fabricated_policy(populated_rag_service):
    """Verify Judge catches fabricated policies explicitly negated by evidence."""
    mock_llm = MockLLMProvider()
    judge = JudgeAgent(llm_provider=mock_llm)

    question = "Do premium users get free replacement?"
    chunks = populated_rag_service.retrieve(query="warranty replacement tiers", top_k=2)

    # Hallucinated policy that company explicitly does not operate
    draft_answer = "Premium users receive free replacement anytime under our VIP care policy."
    sources = [chunks[0].source]

    result = await judge.evaluate(
        question=question,
        draft_answer=draft_answer,
        retrieved_evidence=chunks,
        sources=sources,
        confidence=0.85,
    )

    assert result.approved is False
    assert result.score < 80
    assert any(i.issue_type == "FABRICATED_POLICY" for i in result.issues)


# ==============================================================================
# 4. Numerical Drift Rejection Test
# ==============================================================================

@pytest.mark.asyncio
async def test_judge_rejects_unsupported_numerical_claim(populated_rag_service):
    """Verify Judge detects numerical deviations (e.g. 2 years vs 1 year warranty)."""
    mock_llm = MockLLMProvider()
    judge = JudgeAgent(llm_provider=mock_llm)

    question = "What is the hardware warranty duration?"
    chunks = populated_rag_service.retrieve(query=question, top_k=2)

    # Inaccurate duration
    draft_answer = "All devices are covered by a 2-year comprehensive hardware warranty."
    sources = [chunks[0].source]

    result = await judge.evaluate(
        question=question,
        draft_answer=draft_answer,
        retrieved_evidence=chunks,
        sources=sources,
        confidence=0.90,
    )

    assert result.approved is False
    assert any(i.issue_type in ("UNSUPPORTED_NUMERICAL", "CONTRADICTION") for i in result.issues)


# ==============================================================================
# 5. Maker Confidence Independence Test
# ==============================================================================

@pytest.mark.asyncio
async def test_judge_ignores_maker_confidence(populated_rag_service):
    """Verify Judge rejects incorrect drafts even when Maker reports 0.99 confidence."""
    mock_llm = MockLLMProvider()
    judge = JudgeAgent(llm_provider=mock_llm)

    question = "Can I get same-day delivery?"
    chunks = populated_rag_service.retrieve(query=question, top_k=2)

    # Maker claims 0.99 confidence on wrong fact
    draft_answer = "All items are delivered same-day anywhere in the country."
    sources = [chunks[0].source]

    result = await judge.evaluate(
        question=question,
        draft_answer=draft_answer,
        retrieved_evidence=chunks,
        sources=sources,
        confidence=0.99,  # Overconfident Maker
    )

    # Judge must NOT trust Maker confidence
    assert result.approved is False
    assert result.score < 80
    assert any(i.issue_type == "CONTRADICTION" for i in result.issues)


# ==============================================================================
# 6. Missing Evidence Rejection Test
# ==============================================================================

@pytest.mark.asyncio
async def test_judge_rejects_missing_evidence():
    """Verify Judge flags claims that have no supporting evidence in the knowledge base."""
    mock_llm = MockLLMProvider()
    judge = JudgeAgent(llm_provider=mock_llm)

    question = "Do you accept payment in gold bullion and bitcoin?"
    draft_answer = "Payment accepted in gold bullion or bitcoin with zero processing fees."
    retrieved_chunks = []  # No evidence exists

    result = await judge.evaluate(
        question=question,
        draft_answer=draft_answer,
        retrieved_evidence=retrieved_chunks,
        sources=[],
        confidence=0.30,
    )

    assert result.approved is False
    assert result.score < 80
    assert any(i.issue_type in ("MISSING_EVIDENCE", "UNSUPPORTED_FACT") for i in result.issues)


# ==============================================================================
# 7. Source Mismatch Detection Test
# ==============================================================================

@pytest.mark.asyncio
async def test_judge_detects_source_mismatch(populated_rag_service):
    """Verify Judge detects citations of sources not in the retrieved evidence."""
    mock_llm = MockLLMProvider()
    judge = JudgeAgent(llm_provider=mock_llm)

    question = "What is the warranty period?"
    chunks = populated_rag_service.retrieve(query=question, top_k=2)

    draft_answer = "Hardware products have a 1-year limited warranty."
    # Fabricated / mismatched source citation
    mismatched_sources = ["fabricated_source.txt#chunk_99"]

    result = await judge.evaluate(
        question=question,
        draft_answer=draft_answer,
        retrieved_evidence=chunks,
        sources=mismatched_sources,
        confidence=0.95,
    )

    assert result.approved is False
    assert any(i.issue_type == "SOURCE_MISMATCH" for i in result.issues)
    mismatch_issue = next(i for i in result.issues if i.issue_type == "SOURCE_MISMATCH")
    assert "fabricated_source.txt#chunk_99" in (mismatch_issue.source or mismatch_issue.description)


# ==============================================================================
# 8. Schema & MakerResult Integration Test
# ==============================================================================

@pytest.mark.asyncio
async def test_judge_result_schema_validation(populated_rag_service):
    """Verify JudgeResult Pydantic schema validation, attributes, and MakerResult evaluation."""
    mock_llm = MockLLMProvider()
    maker = MakerAgent(llm_provider=mock_llm, rag_service=populated_rag_service)
    judge = JudgeAgent(llm_provider=mock_llm)

    # 1. Maker generates grounded draft
    maker_result = await maker.generate_draft("What is the warranty period for hardware?")
    assert isinstance(maker_result, MakerResult)

    # 2. Judge evaluates MakerResult directly
    judge_result = await judge.evaluate_maker_result(maker_result)

    assert isinstance(judge_result, JudgeResult)
    assert isinstance(judge_result, JudgeOutput)
    assert isinstance(judge_result.approved, bool)
    assert isinstance(judge_result.score, int)
    assert 0 <= judge_result.score <= 100
    assert isinstance(judge_result.issues, list)

    # Check model serialization
    dumped = judge_result.model_dump()
    assert "approved" in dumped
    assert "score" in dumped
    assert "issues" in dumped
    assert dumped["approved"] is True

    # Check issue structure schema with a rejected case
    bad_result = await judge.evaluate(
        question="Can I return after 45 days?",
        draft_answer="Refunds are accepted within 45 days with original receipt.",
        retrieved_evidence=maker_result.retrieved_evidence,
        sources=maker_result.sources,
        confidence=0.88,
    )
    assert len(bad_result.issues) > 0
    issue = bad_result.issues[0]
    assert isinstance(issue, IssueItem)
    assert issue.issue_type == "CONTRADICTION"
    assert issue.claim != ""
    assert issue.evidence != ""


# ==============================================================================
# 9. API Endpoint Integration Test
# ==============================================================================

@pytest.mark.asyncio
async def test_judge_api_endpoint():
    """Verify POST /api/v1/judge endpoint validates a draft via HTTP."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        payload = {
            "question": "What is the warranty coverage period?",
            "draft_answer": "Hardware products include a standard 1-year limited warranty.",
            "sources": ["warranty_policy.txt#chunk_0"],
            "confidence": 0.95,
            "retrieved_evidence": [
                {
                    "chunk_id": "doc_1#chunk_0",
                    "text": "All hardware products carry a standard 1-year limited warranty.",
                    "document_id": "doc_1",
                    "filename": "warranty_policy.txt",
                    "source": "warranty_policy.txt#chunk_0",
                    "similarity_score": 0.95,
                    "distance": 0.05,
                }
            ],
        }
        response = await client.post("/api/v1/judge", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["approved"] is True
        assert data["score"] >= 80
        assert data["issues"] == []
