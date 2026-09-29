"""API and Schema tests for Stage 2: FastAPI skeleton, Health check, CORS, and Pydantic schemas."""

import pytest
from datetime import datetime, timezone
from httpx import ASGITransport, AsyncClient
from pydantic import ValidationError

from backend.app.main import app
from backend.app.schemas import (
    AskRequest,
    ComplianceResponse,
    HealthResponse,
    IssueItem,
    JudgeOutput,
    MakerOutput,
)


# ==============================================================================
# 1. FastAPI Endpoint Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_root_endpoint():
    """Verify root GET / returns application information and docs links."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "VeriTrust AI"
        assert data["status"] == "online"
        assert data["docs"] == "/docs"
        assert data["health"] == "/health"


@pytest.mark.asyncio
async def test_health_check_endpoint():
    """Verify GET /health returns 200 with database connectivity and system status."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        health = HealthResponse.model_validate(data)
        assert health.status == "healthy"
        assert health.database == "connected"
        assert health.app_name == "VeriTrust AI"
        assert health.llm_provider in ("mock", "gemini", "openai", "ollama")


@pytest.mark.asyncio
async def test_cors_headers():
    """Verify CORS headers are returned for cross-origin requests."""
    transport = ASGITransport(app=app)
    headers = {"Origin": "http://localhost:5173"}
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/", headers=headers)
        assert response.status_code == 200
        # When allow_credentials=True, Starlette echoes the requested Origin header
        assert response.headers.get("access-control-allow-origin") in ("http://localhost:5173", "*")
        assert response.headers.get("access-control-allow-credentials") == "true"


# ==============================================================================
# 2. Pydantic v2 Schema Validation Tests
# ==============================================================================

def test_ask_request_validation():
    """Verify AskRequest validation boundaries."""
    # Valid
    req = AskRequest(question="What is the return window?")
    assert req.question == "What is the return window?"

    # Empty question should fail
    with pytest.raises(ValidationError):
        AskRequest(question="")

    # Question exceeding 2000 chars should fail
    with pytest.raises(ValidationError):
        AskRequest(question="x" * 2001)


def test_maker_output_validation():
    """Verify MakerOutput confidence score constraints."""
    # Valid
    m = MakerOutput(draft_answer="Draft", sources=["src1"], confidence=0.85)
    assert m.confidence == 0.85

    # Confidence > 1.0 should fail
    with pytest.raises(ValidationError):
        MakerOutput(draft_answer="Draft", sources=[], confidence=1.5)

    # Confidence < 0.0 should fail
    with pytest.raises(ValidationError):
        MakerOutput(draft_answer="Draft", sources=[], confidence=-0.1)


def test_judge_output_validation():
    """Verify JudgeOutput score bounds."""
    # Valid
    j = JudgeOutput(approved=True, score=90, issues=[])
    assert j.score == 90
    assert j.approved is True

    # Score > 100 should fail
    with pytest.raises(ValidationError):
        JudgeOutput(approved=True, score=101, issues=[])

    # Score < 0 should fail
    with pytest.raises(ValidationError):
        JudgeOutput(approved=True, score=-5, issues=[])


def test_compliance_response_serialization():
    """Verify ComplianceResponse schema serializes correctly."""
    now = datetime.now(timezone.utc)
    resp = ComplianceResponse(
        request_id="req-12345",
        question="Can I return after 45 days?",
        status="BLOCKED",
        approved=False,
        final_answer="Unable to verify policy.",
        judge_score=35,
        retry_count=3,
        latency_ms=450.2,
        issues=[
            IssueItem(
                issue_type="CONTRADICTION",
                description="45 days contradicts 30 days policy.",
                severity="CRITICAL",
                claim_text="Return within 45 days",
                evidence_ref="Refunds are available within 30 days.",
            )
        ],
        created_at=now,
    )
    dumped = resp.model_dump()
    assert dumped["request_id"] == "req-12345"
    assert dumped["status"] == "BLOCKED"
    assert dumped["approved"] is False
    assert len(dumped["issues"]) == 1
    assert dumped["issues"][0]["issue_type"] == "CONTRADICTION"
