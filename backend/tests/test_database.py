"""Database tests for Stage 2: SQLite, SQLAlchemy ORM models, and relationships."""

import uuid
import pytest
import pytest_asyncio
from datetime import datetime, timezone
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from backend.app.db.models import Base, Request, Hallucination, Metric, AuditTrace
from backend.app.db.session import init_db


@pytest_asyncio.fixture
async def test_db_session():
    """Create an isolated in-memory SQLite database session for unit tests."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
    )
    async with engine.begin() as conn:
        # Enforce foreign keys in SQLite
        await conn.execute(select(1))
        await conn.run_sync(Base.metadata.create_all)

    session_maker = async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    async with session_maker() as session:
        yield session

    await engine.dispose()


@pytest.mark.asyncio
async def test_init_db_creates_all_tables():
    """Verify init_db initializes the database without errors."""
    await init_db()


@pytest.mark.asyncio
async def test_create_request_record(test_db_session: AsyncSession):
    """Verify creating and retrieving a Request record."""
    req_id = str(uuid.uuid4())
    req = Request(
        id=req_id,
        question="What is the refund policy?",
        maker_output={"draft_answer": "Refunds are processed within 30 days.", "confidence": 0.95},
        judge_score=95,
        approved=True,
        status="APPROVED",
        retry_count=0,
        final_answer="Refunds are processed within 30 days.",
        latency_ms=125.4,
    )
    test_db_session.add(req)
    await test_db_session.commit()

    # Query back
    result = await test_db_session.execute(select(Request).where(Request.id == req_id))
    fetched = result.scalar_one_or_none()
    assert fetched is not None
    assert fetched.id == req_id
    assert fetched.approved is True
    assert fetched.status == "APPROVED"
    assert fetched.maker_output["confidence"] == 0.95


@pytest.mark.asyncio
async def test_request_with_hallucinations_relationship(test_db_session: AsyncSession):
    """Verify Hallucination records attach to Request and can be queried via relationship."""
    req_id = str(uuid.uuid4())
    req = Request(
        id=req_id,
        question="Can I return after 45 days?",
        approved=False,
        status="BLOCKED",
        retry_count=3,
        final_answer="Unable to verify.",
    )
    test_db_session.add(req)
    await test_db_session.flush()

    # Add 2 hallucination issues
    h1 = Hallucination(
        request_id=req_id,
        iteration=0,
        issue_type="CONTRADICTION",
        description="Draft claims 45 days; policy says 30 days.",
        severity="CRITICAL",
        claim_text="Return allowed after 45 days",
        evidence_ref="Refunds are available within 30 days.",
    )
    h2 = Hallucination(
        request_id=req_id,
        iteration=1,
        issue_type="FABRICATED_POLICY",
        description="Draft invented VIP replacement policy.",
        severity="CRITICAL",
        claim_text="VIP free replacement",
        evidence_ref="NONE",
    )
    test_db_session.add_all([h1, h2])
    await test_db_session.commit()

    # Query request with selectinload
    result = await test_db_session.execute(select(Request).where(Request.id == req_id))
    fetched = result.scalar_one()
    assert len(fetched.hallucinations) == 2
    types = {h.issue_type for h in fetched.hallucinations}
    assert types == {"CONTRADICTION", "FABRICATED_POLICY"}


@pytest.mark.asyncio
async def test_request_with_metric_and_audit_traces(test_db_session: AsyncSession):
    """Verify Metric and AuditTrace records link correctly to Request."""
    req_id = str(uuid.uuid4())
    req = Request(
        id=req_id,
        question="What is the shipping timeframe?",
        approved=True,
        status="APPROVED",
    )
    test_db_session.add(req)
    await test_db_session.flush()

    # Metric
    metric = Metric(
        request_id=req_id,
        latency_ms=210.5,
        accuracy_score=96.0,
        approval_score=95,
        retry_count=1,
        hallucination_count=1,
    )
    test_db_session.add(metric)

    # Sequential Audit Traces
    trace1 = AuditTrace(
        request_id=req_id,
        step_index=0,
        step_name="RETRIEVAL",
        payload={"chunks_found": 3},
    )
    trace2 = AuditTrace(
        request_id=req_id,
        step_index=1,
        step_name="MAKER",
        payload={"draft": "Shipping is 3-5 days."},
    )
    trace3 = AuditTrace(
        request_id=req_id,
        step_index=2,
        step_name="JUDGE_0",
        payload={"score": 95, "approved": True},
    )
    test_db_session.add_all([trace1, trace2, trace3])
    await test_db_session.commit()

    # Query back
    result = await test_db_session.execute(select(Request).where(Request.id == req_id))
    fetched = result.scalar_one()

    assert fetched.metric is not None
    assert fetched.metric.latency_ms == 210.5
    assert len(fetched.audit_traces) == 3
    assert [t.step_name for t in fetched.audit_traces] == ["RETRIEVAL", "MAKER", "JUDGE_0"]


@pytest.mark.asyncio
async def test_cascade_deletion(test_db_session: AsyncSession):
    """Verify deleting a Request cascades to its hallucinations, metric, and audit traces."""
    req_id = str(uuid.uuid4())
    req = Request(id=req_id, question="Test cascade")
    test_db_session.add(req)
    await test_db_session.flush()

    h = Hallucination(
        request_id=req_id,
        iteration=0,
        issue_type="CONTRADICTION",
        description="Desc",
        claim_text="Claim",
    )
    m = Metric(request_id=req_id, latency_ms=10.0)
    t = AuditTrace(request_id=req_id, step_index=0, step_name="TEST", payload={})
    test_db_session.add_all([h, m, t])
    await test_db_session.commit()

    # Delete Request
    await test_db_session.delete(req)
    await test_db_session.commit()

    # Verify children are deleted
    h_res = await test_db_session.execute(select(Hallucination).where(Hallucination.request_id == req_id))
    assert len(h_res.scalars().all()) == 0

    m_res = await test_db_session.execute(select(Metric).where(Metric.request_id == req_id))
    assert len(m_res.scalars().all()) == 0

    t_res = await test_db_session.execute(select(AuditTrace).where(AuditTrace.request_id == req_id))
    assert len(t_res.scalars().all()) == 0
