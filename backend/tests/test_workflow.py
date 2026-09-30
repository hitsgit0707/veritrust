"""Unit and integration tests for Stage 6: LangGraph Compliance Workflow."""

import os
import shutil
import tempfile
import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select

from backend.app.core.llm.mock_provider import MockLLMProvider
from backend.app.db.models import AuditTrace, Hallucination, Metric, Request
from backend.app.db.session import async_session_factory, init_db
from backend.app.graph.workflow import (
    SAFE_FALLBACK_ANSWER,
    WorkflowState,
    create_compliance_workflow,
    run_compliance_workflow,
)
from backend.app.main import app
from backend.app.rag.service import RAGService
from backend.app.rag.vector_store import ChromaVectorStore


@pytest.fixture
def populated_rag_service():
    """Create an isolated temporary ChromaDB populated with sample_data."""
    temp_dir = tempfile.mkdtemp(prefix="veritrust_test_wf_rag_")
    store = ChromaVectorStore(persist_dir=temp_dir, collection_name="test_wf_kb")
    service = RAGService(vector_store=store)

    sample_dir = os.path.join(os.path.dirname(__file__), "..", "..", "sample_data")
    service.ingest_directory(sample_dir)

    yield service
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
async def db_session():
    """Provide an isolated test database session."""
    await init_db()
    async with async_session_factory() as session:
        yield session


# ==============================================================================
# 1. Grounded Immediate Approval
# ==============================================================================

@pytest.mark.asyncio
async def test_workflow_approves_grounded_answer(populated_rag_service):
    """Case 1: Accurate draft is immediately approved with 0 correction cycles."""
    mock_llm = MockLLMProvider()
    question = "What is the warranty period for hardware?"

    final_state = await run_compliance_workflow(
        question=question,
        llm_provider=mock_llm,
        rag_service=populated_rag_service,
    )

    assert final_state["status"] == "APPROVED"
    assert final_state["correction_attempts"] == 0
    assert "1-year" in final_state["final_answer"].lower()
    assert len(final_state["sources"]) > 0
    assert final_state["judge_result"]["approved"] is True
    assert final_state["judge_result"]["score"] >= 80


# ==============================================================================
# 2. Contradiction Correction
# ==============================================================================

@pytest.mark.asyncio
async def test_workflow_corrects_contradictory_answer(populated_rag_service):
    """Case 2: 45-day refund claim is rejected by Judge and corrected to 30 days."""
    mock_llm = MockLLMProvider()
    question = "Can I return a product after 45 days?"

    final_state = await run_compliance_workflow(
        question=question,
        adversarial_mode="case 2",
        llm_provider=mock_llm,
        rag_service=populated_rag_service,
    )

    assert final_state["status"] == "APPROVED"
    assert final_state["correction_attempts"] == 1
    assert "30 days" in final_state["final_answer"].lower()
    assert "45 days" not in final_state["final_answer"].lower()
    assert final_state["judge_result"]["approved"] is True


# ==============================================================================
# 3. Fabricated Policy Removal
# ==============================================================================

@pytest.mark.asyncio
async def test_workflow_corrects_fabricated_policy(populated_rag_service):
    """Case 3: Fabricated VIP policy is rejected and rewritten without the hallucination."""
    mock_llm = MockLLMProvider()
    question = "What is the replacement policy for VIP customers?"

    final_state = await run_compliance_workflow(
        question=question,
        adversarial_mode="case 3",
        llm_provider=mock_llm,
        rag_service=populated_rag_service,
    )

    assert final_state["status"] == "APPROVED"
    assert final_state["correction_attempts"] == 1
    assert "1-year" in final_state["final_answer"].lower() or "standard" in final_state["final_answer"].lower()
    assert final_state["judge_result"]["approved"] is True


# ==============================================================================
# 4. Numerical Contradiction Correction
# ==============================================================================

@pytest.mark.asyncio
async def test_workflow_corrects_numerical_contradiction(populated_rag_service):
    """Case 4: 2-year warranty draft is rejected and corrected to 1-year per policy."""
    mock_llm = MockLLMProvider()
    question = "What is the hardware warranty coverage period?"

    final_state = await run_compliance_workflow(
        question=question,
        adversarial_mode="case 4",
        llm_provider=mock_llm,
        rag_service=populated_rag_service,
    )

    assert final_state["status"] == "APPROVED"
    assert final_state["correction_attempts"] == 1
    assert "1-year" in final_state["final_answer"].lower()
    assert "2-year" not in final_state["final_answer"].lower()
    assert final_state["judge_result"]["approved"] is True


# ==============================================================================
# 5. Maker Confidence Independence
# ==============================================================================

@pytest.mark.asyncio
async def test_workflow_ignores_maker_confidence(populated_rag_service):
    """Case 5: Maker 0.99 confidence does not bypass Judge; correction triggers."""
    mock_llm = MockLLMProvider()
    question = "Can I get same-day delivery?"

    final_state = await run_compliance_workflow(
        question=question,
        adversarial_mode="case 5",
        llm_provider=mock_llm,
        rag_service=populated_rag_service,
    )

    # Maker claimed 0.99 confidence on same-day delivery
    assert final_state["maker_confidence"] == 0.99
    # Correction still triggered despite 0.99 confidence
    assert final_state["correction_attempts"] >= 1
    assert final_state["status"] == "APPROVED"
    assert "3 to 5 business days" in final_state["final_answer"].lower()
    assert final_state["judge_result"]["approved"] is True


# ==============================================================================
# 6. Hard Maximum Retry Rule (3 Attempts -> Block)
# ==============================================================================

@pytest.mark.asyncio
async def test_workflow_blocks_after_three_failed_corrections(populated_rag_service):
    """Case 6: Persistent unresolvable claims are blocked after exactly 3 corrections."""
    mock_llm = MockLLMProvider()
    question = "Can I pay in bitcoin or dogecoin with zero fees?"

    final_state = await run_compliance_workflow(
        question=question,
        adversarial_mode="persistent_failure",
        llm_provider=mock_llm,
        rag_service=populated_rag_service,
    )

    assert final_state["status"] == "BLOCKED"
    assert final_state["correction_attempts"] == 3
    assert final_state["final_answer"] == SAFE_FALLBACK_ANSWER
    # Ensure unverified claim was never returned
    assert "bitcoin" not in final_state["final_answer"].lower()


# ==============================================================================
# 7. Final Response Safety Invariant
# ==============================================================================

@pytest.mark.asyncio
async def test_workflow_never_returns_unvalidated_answer(populated_rag_service):
    """Verify customer answer is NEVER delivered unless judge approved == True."""
    mock_llm = MockLLMProvider()
    question = "Unresolvable test question"

    final_state = await run_compliance_workflow(
        question=question,
        adversarial_mode="persistent_failure",
        llm_provider=mock_llm,
        rag_service=populated_rag_service,
    )

    # Invariant: If not approved, must equal SAFE_FALLBACK_ANSWER
    if final_state["status"] != "APPROVED":
        assert final_state["final_answer"] == SAFE_FALLBACK_ANSWER
        assert final_state["judge_result"]["approved"] is False


# ==============================================================================
# 8. Database Audit Trail Persistence
# ==============================================================================

@pytest.mark.asyncio
async def test_workflow_records_audit_trace(populated_rag_service, db_session):
    """Verify execution history is fully persisted to Request, Metric, Hallucination, and AuditTrace."""
    mock_llm = MockLLMProvider()
    question = "Can I return a product after 45 days?"

    final_state = await run_compliance_workflow(
        question=question,
        adversarial_mode="case 2",
        llm_provider=mock_llm,
        rag_service=populated_rag_service,
        db_session=db_session,
    )

    req_id = final_state["request_id"]
    stmt = select(Request).where(Request.id == req_id)
    result = await db_session.execute(stmt)
    req = result.scalar_one_or_none()

    assert req is not None
    assert req.question == question
    assert req.status == "APPROVED"
    assert req.retry_count == 1
    assert req.approved is True

    # Audit traces reconstruct full sequence
    step_names = [t.step_name for t in req.audit_traces]
    assert "RETRIEVAL" in step_names
    assert "MAKER" in step_names
    assert "JUDGE_0" in step_names
    assert "CORRECTOR_1" in step_names
    assert "JUDGE_1" in step_names
    assert "FINALIZE" in step_names

    # Metric recorded
    assert req.metric is not None
    assert req.metric.retry_count == 1

    # Hallucinations recorded
    assert len(req.hallucinations) > 0
    assert any(h.issue_type == "CONTRADICTION" for h in req.hallucinations)


# ==============================================================================
# 9. Customer-Facing POST /api/v1/ask Endpoint
# ==============================================================================

@pytest.mark.asyncio
async def test_ask_endpoint_runs_complete_workflow():
    """Verify POST /api/v1/ask executes the complete workflow over HTTP."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        payload = {"question": "What is the warranty coverage period?"}
        response = await client.post("/api/v1/ask", json=payload)

        assert response.status_code == 200
        data = response.json()

        assert "request_id" in data
        assert data["question"] == "What is the warranty coverage period?"
        assert data["status"] in ("APPROVED", "BLOCKED")
        assert "answer" in data
        assert isinstance(data["sources"], list)
        assert data["correction_attempts"] >= 0
        assert data["approved"] is True
        assert "1-year" in data["answer"].lower()


# ==============================================================================
# 10. Correction Attempts State Tracking
# ==============================================================================

@pytest.mark.asyncio
async def test_workflow_state_tracks_correction_attempts(populated_rag_service):
    """Verify WorkflowState accurately counts correction iterations at each step."""
    mock_llm = MockLLMProvider()

    # 1. 0 attempts for grounded query
    state_0 = await run_compliance_workflow(
        question="What is the warranty period for hardware?",
        llm_provider=mock_llm,
        rag_service=populated_rag_service,
    )
    assert state_0["correction_attempts"] == 0

    # 2. 1 attempt for single correction
    state_1 = await run_compliance_workflow(
        question="What is refund window?",
        adversarial_mode="case 2",
        llm_provider=mock_llm,
        rag_service=populated_rag_service,
    )
    assert state_1["correction_attempts"] == 1

    # 3. Exactly 3 attempts for persistent failure
    state_3 = await run_compliance_workflow(
        question="Persistent query",
        adversarial_mode="persistent_failure",
        llm_provider=mock_llm,
        rag_service=populated_rag_service,
    )
    assert state_3["correction_attempts"] == 3
    assert state_3["status"] == "BLOCKED"


# ==============================================================================
# 11. Out-of-Scope Grounding Regression Test
# ==============================================================================

@pytest.mark.asyncio
async def test_workflow_out_of_scope_query_team_india(populated_rag_service):
    """
    Regression Test: An out-of-scope question ('Who is captain of team India?')
    must not produce a refund-policy response, must not cite unrelated refund sources,
    and must not approve an unsupported answer.
    """
    mock_llm = MockLLMProvider()
    question = "Who is captain of team India?"

    final_state = await run_compliance_workflow(
        question=question,
        llm_provider=mock_llm,
        rag_service=populated_rag_service,
    )

    final_answer = final_state.get("final_answer", "")
    sources = final_state.get("sources", [])

    # 1. Final answer does not contain the refund-policy answer
    assert "refund" not in final_answer.lower()
    assert "30 days" not in final_answer.lower()
    assert "return" not in final_answer.lower()

    # 2. No unrelated refund source is cited (and no irrelevant sources cited)
    assert not any("refund" in s.lower() for s in sources)
    assert len(sources) == 0

    # 3. The workflow does not approve an unsupported answer (returns cautious unable-to-answer)
    cautious_keywords = ["unable to answer", "could not be verified", "insufficient", "cannot answer"]
    assert any(k in final_answer.lower() for k in cautious_keywords)

    # 4. If an unsupported draft answer (e.g. refund claim or cricket claim) is evaluated,
    # the Judge Agent must NOT approve it.
    from backend.app.agents.judge import JudgeAgent
    judge = JudgeAgent(llm_provider=mock_llm)
    unsupported_eval = await judge.evaluate(
        question=question,
        draft_answer="According to company policy, refunds are available within 30 days of purchase.",
        retrieved_evidence=final_state.get("retrieved_evidence", []),
        sources=["product_catalog.csv#chunk_0"],
    )
    assert unsupported_eval.approved is False
    assert any(i.issue_type in ("MISSING_EVIDENCE", "CONTRADICTION") for i in unsupported_eval.issues)

