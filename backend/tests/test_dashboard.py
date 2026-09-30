"""Tests for the dashboard metrics and audit endpoints (Stage 7 additions)."""

import pytest
from httpx import ASGITransport, AsyncClient

from backend.app.main import app


@pytest.mark.asyncio
async def test_metrics_endpoint_returns_structure():
    """GET /api/v1/metrics returns expected DashboardMetrics schema fields."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/v1/metrics")
    assert response.status_code == 200
    data = response.json()

    required_keys = {
        "total_requests",
        "approved_count",
        "blocked_count",
        "corrected_count",
        "total_hallucinations_detected",
        "approval_rate",
        "avg_correction_attempts",
        "recent_requests",
    }
    for key in required_keys:
        assert key in data, f"Missing key: {key}"

    assert isinstance(data["total_requests"], int)
    assert isinstance(data["approved_count"], int)
    assert isinstance(data["blocked_count"], int)
    assert isinstance(data["corrected_count"], int)
    assert isinstance(data["approval_rate"], float)
    assert 0.0 <= data["approval_rate"] <= 1.0
    assert isinstance(data["recent_requests"], list)


@pytest.mark.asyncio
async def test_metrics_counts_are_consistent():
    """approved_count + blocked_count must equal total_requests."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/v1/metrics")
    assert response.status_code == 200
    d = response.json()
    assert d["approved_count"] + d["blocked_count"] == d["total_requests"]


@pytest.mark.asyncio
async def test_audit_endpoint_404_for_unknown_id():
    """GET /api/v1/audit/{id} returns 404 for a non-existent request ID."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/v1/audit/nonexistent-request-id-xyz")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_ask_then_audit_round_trip():
    """
    Submit a question via /api/v1/ask, then verify the request_id
    is retrievable via /api/v1/audit/{id} with correct structure.
    """
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Submit compliance request
        ask_resp = await client.post(
            "/api/v1/ask",
            json={"question": "What is the hardware warranty period?"},
        )
        assert ask_resp.status_code == 200
        ask_data = ask_resp.json()
        request_id = ask_data["request_id"]
        assert request_id

        # Retrieve audit trace
        audit_resp = await client.get(f"/api/v1/audit/{request_id}")
        assert audit_resp.status_code == 200
        audit = audit_resp.json()

        assert audit["request_id"] == request_id
        assert audit["question"] == "What is the hardware warranty period?"
        assert audit["status"] in ("APPROVED", "BLOCKED")
        assert isinstance(audit["audit_traces"], list)
        assert len(audit["audit_traces"]) > 0, "Audit trail should not be empty"
        assert isinstance(audit["hallucinations"], list)


@pytest.mark.asyncio
async def test_metrics_updates_after_ask():
    """Metrics total_requests increases after a /api/v1/ask call."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Get baseline
        m1 = (await client.get("/api/v1/metrics")).json()
        before = m1["total_requests"]

        # Submit a request
        await client.post(
            "/api/v1/ask",
            json={"question": "What is the refund window?"},
        )

        # Metrics should increment
        m2 = (await client.get("/api/v1/metrics")).json()
        after = m2["total_requests"]

    assert after == before + 1, f"Expected {before + 1} requests, got {after}"


@pytest.mark.asyncio
async def test_ui_route_serves_html():
    """GET /ui serves the dashboard HTML file (index.html)."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/ui")
    assert response.status_code == 200
    content_type = response.headers.get("content-type", "")
    assert "text/html" in content_type, f"Expected HTML, got: {content_type}"
    assert b"VeriTrust" in response.content
